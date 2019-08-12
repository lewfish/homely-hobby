from collections import defaultdict
import math

import torch
import torch.nn as nn
from torchvision import models

from mlx.od.fcos.decoder import decode_batch_output
from mlx.od.fcos.loss import fcos_batch_loss

class FPN(nn.Module):
    """Feature Pyramid Network backbone.

    See https://arxiv.org/abs/1612.03144
    """
    def __init__(self, backbone_arch, out_channels=256, pretrained=True,
                 levels=None):
        # Assumes backbone_arch in is Resnet family.
        super().__init__()

        # Strides of cells in each level of the pyramid. Should be in
        # descending order.
        self.strides = [64, 32, 16, 8, 4]
        self.levels = levels
        if levels is not None:
            self.strides = [self.strides[l] for l in levels]

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
        self.out_conv2 = nn.Conv2d(
            out_channels, out_channels, 3, 1, 1)

        self.cross_conv3 = nn.Conv2d(
            self.backbone_out['layer3'].shape[1], out_channels, 1)
        self.out_conv3 = nn.Conv2d(
            out_channels, out_channels, 3, 1, 1)

        self.cross_conv4 = nn.Conv2d(
            self.backbone_out['layer4'].shape[1], out_channels, 1)
        self.out_conv4 = nn.Conv2d(
            out_channels, out_channels, 3, 1, 1)

        self.up_conv5 = nn.Conv2d(
            out_channels, out_channels, 3, 2, 1)

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
        d4 = self.out_conv4(c4)

        u5 = self.up_conv5(d4)

        c3 = self.cross_conv3(self.backbone_out['layer3'])
        d3 = self.out_conv3(c3 + nn.functional.interpolate(d4, c3.shape[2:]))

        c2 = self.cross_conv2(self.backbone_out['layer2'])
        d2 = self.out_conv2(c2 + nn.functional.interpolate(d3, c2.shape[2:]))

        c1 = self.cross_conv1(self.backbone_out['layer1'])
        d1 = self.out_conv1(c1 + nn.functional.interpolate(d2, c1.shape[2:]))

        out = [u5, d4, d3, d2, d1]
        if self.levels is not None:
            out = [out[l] for l in self.levels]
        return out

class ConvBlock(nn.Module):
    """Sequence of conv2d, group norm, and relu."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding)
        self.gn = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = nn.functional.relu(x)
        return x

class FCOSHead(nn.Module):
    """Head for FCOS model.

    Outputs reg_arr, label_arr, and center_arr for one level of the pyramid,
    which can be decoded into a BoxList.
    """
    def __init__(self, num_labels, in_channels=256):
        super().__init__()
        c = in_channels

        self.reg_branch = nn.Sequential(
            *[ConvBlock(c, c, 3, padding=1) for i in range(4)])
        self.reg_conv = nn.Conv2d(c, 4, 3, padding=1)

        self.label_branch = nn.Sequential(
            *[ConvBlock(c, c, 3, padding=1) for i in range(4)])
        self.label_conv = nn.Conv2d(c, num_labels, 3, padding=1)

        self.center_conv = nn.Conv2d(c, 1, 3, padding=1)

        # initialization adapted from retinanet
        # https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/rpn/retinanet/retinanet.py
        for modules in [self.reg_branch, self.reg_conv, self.label_branch, self.label_conv, self.center_conv]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        prob = 0.01
        logit = math.log(prob / (1 - prob))
        torch.nn.init.constant_(self.label_conv.bias, logit)

    def forward(self, x, scale_param):
        """Computes output of head.

        Args:
            x: (tensor) with shape (batch_sz, 256, h*, w*) which is assumed to
                be output of one level of the FPN.
            scale_param: (tensor) with single element used to scale the values
                in the reg_arr and varies across levels of the pyramid.

        Returns:
            tuple of form (reg_arr, label_arr, center_arr) where
                - reg_arr is tensor<n, 4, h, w>,
                - label_arr is tensor<n, num_labels, h, w>
                - center_arr is tensor<n, 1, h, w>
            where label_arr and center_arr contain logits
        """
        reg_arr = torch.exp(scale_param * self.reg_conv(self.reg_branch(x)))
        label_branch_arr = self.label_branch(x)
        label_arr = self.label_conv(label_branch_arr)
        center_arr = self.center_conv(label_branch_arr)
        return (reg_arr, label_arr, center_arr)

class FCOS(nn.Module):
    """Fully convolutional one stage object detector

    See https://arxiv.org/abs/1904.01355
    """
    def __init__(self, backbone_arch, num_labels, pretrained=True,
                 levels=None):
        super().__init__()

        out_channels = 256
        self.num_labels = num_labels
        self.levels = levels
        self.fpn = FPN(backbone_arch, out_channels=out_channels,
                       pretrained=pretrained, levels=levels)
        num_scales = len(self.fpn.strides)
        self.scale_params = nn.Parameter(torch.ones((num_scales,)))
        self.head = FCOSHead(num_labels, in_channels=out_channels)

    def forward(self, input, targets=None, get_head_out=False):
        """Compute output of FCOS.

        Args:
            input: tensor<n, 3, h, w> with batch of images
            targets: list<BoxList> of length n with boxes and labels set
        Returns:
            if targets is None, returns list<BoxList> of length n, containing
            boxes, labels, and scores for boxes with score > 0.05. Further
            filtering based on score should be done before considering the
            prediction "final".

            if target is a list, returns the losses as dict of form {
                'reg_loss': <tensor[1]>,
                'label_loss': <tensor[1]>,
                'center_loss': <tensor[1]>
            }

            if get_head_out is True, also return a list with the raw output
            from the heads
        """
        fpn_out = self.fpn(input)

        img_height, img_width = input.shape[2:]
        strides = self.fpn.strides
        hws = [level_out.shape[2:] for level_out in fpn_out]
        max_box_sides = [256, 128, 64, 32, 16]
        if self.levels is not None:
            max_box_sides = [max_box_sides[l] for l in self.levels]
        self.pyramid_shape = [
            (s, m, h, w) for s, m, (h, w) in zip(strides, max_box_sides, hws)]

        head_out = []
        for i, level_out in enumerate(fpn_out):
            head_out.append(self.head(level_out, self.scale_params[i]))

        if targets is None:
            out = decode_batch_output(head_out, self.pyramid_shape, img_height, img_width)
            if get_head_out:
                return out, head_out
            return out

        loss_dict = fcos_batch_loss(
            head_out, targets, self.pyramid_shape, self.num_labels)
        if get_head_out:
            return loss_dict, head_out
        return loss_dict