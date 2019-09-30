import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from mlx.od.centernet.decoder import decode
from mlx.od.centernet.encoder import encode
from mlx.od.centernet.loss import loss
from mlx.od.centernet.utils import get_positions, prob2logit

class FCN(nn.Module):
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
        self.cross_conv2 = nn.Conv2d(
            self.backbone_out['layer2'].shape[1], out_channels, 1)
        self.cross_conv3 = nn.Conv2d(
            self.backbone_out['layer3'].shape[1], out_channels, 1)
        self.cross_conv4 = nn.Conv2d(
            self.backbone_out['layer4'].shape[1], out_channels, 1)

    def forward(self, input):
        """Computes output of FCN.

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

        c4 = self.cross_conv4(self.backbone_out['layer4'])
        d4 = c4

        c3 = self.cross_conv3(self.backbone_out['layer3'])
        d3 = c3 + nn.functional.interpolate(d4, c3.shape[2:], mode='bilinear')

        c2 = self.cross_conv2(self.backbone_out['layer2'])
        d2 = c2 + nn.functional.interpolate(d3, c2.shape[2:], mode='bilinear')

        c1 = self.cross_conv1(self.backbone_out['layer1'])
        d1 = c1 + nn.functional.interpolate(d2, c1.shape[2:], mode='bilinear')

        return d1

# from https://github.com/xingyizhou/CenterNet/blob/819e0d0dde02f7b8cb0644987a8d3a370aa8206a/src/lib/models/networks/resnet_dcn.py
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class SimpleBaseline(nn.Module):
    def __init__(self, backbone_arch, pretrained=True):
        # Assumes backbone_arch in is Resnet family.
        super().__init__()

        # Setup bottom-up backbone and hooks to capture output of stages.
        # Assumes there is layer1, 2, 3, 4, which is true for Resnets.
        backbone = getattr(models, backbone_arch)(pretrained=pretrained)
        self.backbone_out = {}

        def make_save_output(layer_name):
            def save_output(layer, input, output):
                self.backbone_out[layer_name] = output
            return save_output

        backbone.layer4.register_forward_hook(make_save_output('layer4'))

        # Remove head of backbone.
        self.backbone = nn.Sequential(*list(backbone.children())[0:-2])

        # Setup layers for top-down pathway.
        # Use test input to determine the number of channels in each layer.
        self.backbone(torch.rand((1, 3, 256, 256)))

        self.up_conv1 = nn.ConvTranspose2d(
            self.backbone_out['layer4'].shape[1],
            256, 4, 2, 1)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.up_conv2 = nn.ConvTranspose2d(
            256, 128, 4, 2, 1)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.up_conv3 = nn.ConvTranspose2d(
            128, 64, 4, 2, 1)
        self.batch_norm3 = nn.BatchNorm2d(64)

        '''
        fill_up_weights(self.up_conv1)
        fill_up_weights(self.up_conv2)
        fill_up_weights(self.up_conv3)
        '''

    def forward(self, input):
        """Computes output of FCN.

        Args:
            input: (tensor) batch of images with shape (batch_sz, 3, h, w)

        Returns:
            (list) output of each level in the pyramid ordered same as
                self.strides. Each output is tensor with shape
                (batch_sz, 256, h*, w*) where h* and w* are height and width
                for that level of the pyramid.
        """
        # note: height and width must be divisible by 32
        self.backbone_out = {}
        self.backbone(input)
        x = self.backbone_out['layer4']

        x = nn.functional.relu(self.batch_norm1(self.up_conv1(x)))
        x = nn.functional.relu(self.batch_norm2(self.up_conv2(x)))
        x = nn.functional.relu(self.batch_norm3(self.up_conv3(x)))

        return x

class SimpleBaselineUpsample(nn.Module):
    def __init__(self, backbone_arch, pretrained=True):
        # Assumes backbone_arch in is Resnet family.
        super().__init__()

        # Setup bottom-up backbone and hooks to capture output of stages.
        # Assumes there is layer1, 2, 3, 4, which is true for Resnets.
        backbone = getattr(models, backbone_arch)(pretrained=pretrained)
        self.backbone_out = {}

        def make_save_output(layer_name):
            def save_output(layer, input, output):
                self.backbone_out[layer_name] = output
            return save_output

        backbone.layer4.register_forward_hook(make_save_output('layer4'))

        # Remove head of backbone.
        self.backbone = nn.Sequential(*list(backbone.children())[0:-2])

        # Setup layers for top-down pathway.
        # Use test input to determine the number of channels in each layer.
        self.backbone(torch.rand((1, 3, 256, 256)))

        self.up_conv1 = nn.Conv2d(
            self.backbone_out['layer4'].shape[1],
            256, 3, 1, 1)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.up_conv2 = nn.Conv2d(
            256, 128, 3, 1, 1)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.up_conv3 = nn.Conv2d(
            128, 64, 3, 1, 1)
        self.batch_norm3 = nn.BatchNorm2d(64)

    def forward(self, input):
        """Computes output of FCN.

        Args:
            input: (tensor) batch of images with shape (batch_sz, 3, h, w)

        Returns:
            (list) output of each level in the pyramid ordered same as
                self.strides. Each output is tensor with shape
                (batch_sz, 256, h*, w*) where h* and w* are height and width
                for that level of the pyramid.
        """
        # note: height and width must be divisible by 32
        self.backbone_out = {}
        self.backbone(input)
        x = self.backbone_out['layer4']

        x = nn.functional.relu(self.batch_norm1(self.up_conv1(F.interpolate(x, scale_factor=2, mode='bilinear'))))
        x = nn.functional.relu(self.batch_norm2(self.up_conv2(F.interpolate(x, scale_factor=2, mode='bilinear'))))
        x = nn.functional.relu(self.batch_norm3(self.up_conv3(F.interpolate(x, scale_factor=2, mode='bilinear'))))

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        return x

class DeepHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            *[ConvBlock(in_channels, in_channels) for _ in range(num_blocks)])
        self.final_conv = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.final_conv(x)
        return x

class Head(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        return self.conv2(x)

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class CenterNet(nn.Module):
    """CenterNet object detector

    See https://arxiv.org/abs/1904.07850
    """
    def __init__(self, backbone_arch, num_labels, cfg, pretrained=True):
        super().__init__()
        self.num_labels = num_labels
        self.stride = 4
        self.subloss_names = ['total_loss', 'keypoint_loss', 'reg_loss']
        self.cfg = cfg
        cn_cfg = cfg.model.centernet

        if cn_cfg.body_arch == 'fcn':
            out_channels = 256
            self.body = FCN(backbone_arch, out_channels=out_channels, pretrained=pretrained)
        elif cn_cfg.body_arch == 'simple_baseline':
            out_channels = 64
            self.body = SimpleBaseline(backbone_arch, pretrained=pretrained)
        elif cn_cfg.body_arch == 'simple_baseline_upsample':
            out_channels = 64
            self.body = SimpleBaselineUpsample(backbone_arch, pretrained=pretrained)

        if cn_cfg.head.mode == 'centernet':
            self.keypoint_head = Head(out_channels, num_labels)
            self.reg_head = Head(out_channels, 2)
            fill_fc_weights(self.keypoint_head)
            fill_fc_weights(self.reg_head)
            torch.nn.init.constant_(self.keypoint_head.conv2.bias, prob2logit(cn_cfg.head.keypoint_init))
        elif cn_cfg.head.mode == 'deep':
            num_blocks = cn_cfg.head.num_blocks
            self.keypoint_head = DeepHead(out_channels, num_labels, num_blocks)
            self.reg_head = DeepHead(out_channels, 2, num_blocks)
            fill_fc_weights(self.keypoint_head)
            fill_fc_weights(self.reg_head)
            torch.nn.init.constant_(self.keypoint_head.final_conv.bias, prob2logit(cn_cfg.head.keypoint_init))

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
        body_out = self.body(input)
        keypoint = torch.sigmoid(self.keypoint_head(body_out))
        reg = torch.exp(self.reg_head(body_out))
        head_out = keypoint, reg

        img_height, img_width = input.shape[2:]
        positions = get_positions(
            img_height, img_width, self.stride, keypoint.device)

        if targets is None:
            if self.cfg.data.test_aug.hflip:
                hflip_input = torch.flip(input, [3])
                hflip_body_out = self.body(hflip_input)
                hflip_keypoint = torch.sigmoid(self.keypoint_head(hflip_body_out))
                hflip_reg = torch.exp(self.reg_head(hflip_body_out))

                keypoint = (keypoint + torch.flip(hflip_keypoint, [3])) / 2.
                reg = (reg + torch.flip(hflip_reg, [3])) / 2.
                head_out = keypoint, reg

            boxlists = decode(
                keypoint, reg, positions, self.stride, self.cfg)
            if self.cfg.model.centernet.nms:
                boxlists = [bl.nms() for bl in boxlists]
            if get_head_out:
                return boxlists, head_out
            return boxlists

        encoded_targets = encode(
            targets, positions, self.stride, self.num_labels, self.cfg)
        loss_dict = loss(head_out, encoded_targets, self.cfg)
        if get_head_out:
            return loss_dict, head_out
        return loss_dict
