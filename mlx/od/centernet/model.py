import torch
import torch.nn as nn
from torchvision import models
from fastai.vision.learner import (
    create_body, apply_init)
from fastai.vision.models import DynamicUnet
from fastai.layers import SequentialEx
from torchvision import models

from mlx.od.centernet.decoder import decode
from mlx.od.centernet.encoder import encode
from mlx.od.centernet.loss import loss
from mlx.od.centernet.utils import get_positions, prob2logit

class MyFPN(nn.Module):
    def __init__(self, backbone_arch, out_channels=256, pretrained=True):
        # Assumes backbone_arch in is Resnet family.
        super().__init__()

        # Strides of cells in each level of the pyramid. Should be in
        # descending order.
        self.strides = [32, 16, 8, 4]

        # Setup bottom-up backbone and hooks to capture output of stages.
        # Assumes there is layer1, 2, 3, 4, which is true for Resnets.
        backbone = getattr(models, backbone_arch)(pretrained=pretrained)
        self.backbone_out = {}

        def make_save_output(layer_name):
            def save_output(layer, input, output):
                self.backbone_out[layer_name] = output
            return save_output

        backbone.layer1.register_forward_hook(make_save_output('layer1'))
        backbone.layer2.register_forward_hook(make_save_output('layer2'))
        backbone.layer3.register_forward_hook(make_save_output('layer3'))
        backbone.layer4.register_forward_hook(make_save_output('layer4'))

        # Remove head of backbone.
        self.backbone = nn.Sequential(*list(backbone.children())[0:-2])

        # Setup layers for top-down pathway.
        # Use test input to determine the number of channels in each layer.
        self.backbone(torch.rand((1, 3, 256, 256)))
        self.cross_conv1 = nn.Conv2d(
            self.backbone_out['layer1'].shape[1], out_channels, 1)
        self.out_conv1 = nn.Conv2d(
            out_channels, out_channels, 3, 1, 1)

        self.cross_conv2 = nn.Conv2d(
            self.backbone_out['layer2'].shape[1], out_channels, 1)

        self.cross_conv3 = nn.Conv2d(
            self.backbone_out['layer3'].shape[1], out_channels, 1)

        self.cross_conv4 = nn.Conv2d(
            self.backbone_out['layer4'].shape[1], out_channels, 1)

    def forward(self, input):
        """Computes output of FPN.

        Args:
            input: (tensor) batch of images with shape (batch_sz, 3, h, w)

        Returns:
            (list) output of each level in the pyramid ordered same as
                self.strides. Each output is tensor with shape
                (batch_sz, 256, h*, w*) where h* and w* are height and width
                for that level of the pyramid.
        """
        self.backbone_out = {}
        self.backbone(input)

        # c* is cross output, d* is downsampling output
        c4 = self.cross_conv4(self.backbone_out['layer4'])
        d4 = c4

        c3 = self.cross_conv3(self.backbone_out['layer3'])
        d3 = c3 + nn.functional.interpolate(d4, c3.shape[2:])

        c2 = self.cross_conv2(self.backbone_out['layer2'])
        d2 = c2 + nn.functional.interpolate(d3, c2.shape[2:])

        c1 = self.cross_conv1(self.backbone_out['layer1'])
        d1 = c1 + nn.functional.interpolate(d2, c1.shape[2:])

        return self.out_conv1(d1)

class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        return self.conv2(x)

class CenterNet(nn.Module):
    """CenterNet object detector

    See https://arxiv.org/abs/1904.07850
    """
    def __init__(self, backbone_arch, num_labels, pretrained=True,
                 loss_alpha=2.0, loss_beta=4.0):
        super().__init__()
        self.num_labels = num_labels
        self.stride = 4
        self.subloss_names = ['total_loss', 'keypoint_loss', 'reg_loss']
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta

        out_channels = 256
        self.fpn = MyFPN(backbone_arch, out_channels=out_channels, pretrained=pretrained)
        self.keypoint_head = Head(out_channels, num_labels)
        self.reg_head = Head(out_channels, 2)
        
        # Retinanet init
        torch.nn.init.constant_(self.keypoint_head.conv2.bias, prob2logit(0.01))

    def forward(self, input, targets=None, get_head_out=False):
        """Forward pass

        Args:
            input: tensor<n, 3, h, w> with batch of images
            targets: None or list<BoxList> of length n with boxes and labels

        Returns:
            if targets is None, returns list<BoxList> of length n, containing
            boxes, labels, and scores for boxes with score > 0.05. Further
            filtering based on score should be done before considering the
            prediction "final".

            if targets is a list, returns the losses as dict of form {
                'keypoint_loss': <tensor[1]>,
                'reg_loss': <tensor[1]>,
            }

            if get_head_out is True, also returns the raw output of the head
        """
        fpn_out = self.fpn(input)
        keypoint = torch.sigmoid(self.keypoint_head(fpn_out))
        reg = torch.exp(self.reg_head(fpn_out))
        head_out = keypoint, reg

        img_height, img_width = input.shape[2:]
        positions = get_positions(
            img_height, img_width, self.stride, keypoint.device)

        if targets is None:
            boxlists = decode(
                keypoint, reg, positions, self.stride)
            if get_head_out:
                return boxlists, head_out
            return boxlists

        encoded_targets = encode(
            targets, positions, self.stride, self.num_labels)

        loss_dict = loss(
            head_out, encoded_targets, loss_alpha=self.loss_alpha, loss_beta=self.loss_beta)
        if get_head_out:
            return loss_dict, head_out
        return loss_dict