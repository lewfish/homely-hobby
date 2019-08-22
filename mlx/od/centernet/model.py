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
    def __init__(self, backbone_arch, num_labels, img_size, pretrained=True):
        # TODO ideally we could get rid of dependence on img_size
        super().__init__()
        self.num_labels = num_labels
        self.stride = 1
        self.subloss_names = ['keypoint_loss', 'reg_loss']

        backbone_arch = getattr(models, backbone_arch)
        body = create_body(backbone_arch, pretrained)
        model = DynamicUnet(
            body, n_classes=1, img_size=img_size, blur=False,
            blur_final=False,
            self_attention=False, y_range=None)
        apply_init(model[2], nn.init.kaiming_normal_)
        self.unet = model

        # TODO add headless option to Unet
        # Setup hook to grab output right before head of unet.
        def hook(layer, input, output):
            self.unet_out = output
        self.unet[-2].register_forward_hook(hook)
        model(torch.empty((1, 3, img_size[0], img_size[1])))
        unet_out_channels = self.unet_out.shape[1]

        self.keypoint_head = Head(unet_out_channels, num_labels)
        self.reg_head = Head(unet_out_channels, 2)
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
        self.unet(input)
        keypoint = self.keypoint_head(self.unet_out)
        reg = torch.exp(self.reg_head(self.unet_out))
        head_out = (keypoint, reg)

        img_height, img_width = input.shape[2:]
        positions = get_positions(
            img_height, img_width, self.stride, keypoint.device)

        if targets is None:
            boxlists = decode(
                torch.sigmoid(keypoint), reg, positions, self.stride)
            if get_head_out:
                return boxlists, head_out
            return boxlists

        encoded_targets = encode(
            targets, positions, self.stride, self.num_labels)
        loss_dict = loss(head_out, encoded_targets)
        if get_head_out:
            return loss_dict, head_out
        return loss_dict