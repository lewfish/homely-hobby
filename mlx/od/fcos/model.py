from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import models

from mlx.od.fcos.decoder import decode_output
from mlx.od.fcos.encoder import encode_targets
from mlx.od.fcos.nms import compute_nms
from mlx.od.fcos.loss import focal_loss

class FPN(nn.Module):
    """Feature Pyramid Network backbone.

    See https://arxiv.org/abs/1612.03144
    """
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
        self.cross_conv1 = nn.Conv2d(64, out_channels, 1)
        self.cross_conv2 = nn.Conv2d(128, out_channels, 1)
        self.cross_conv3 = nn.Conv2d(256, out_channels, 1)
        self.cross_conv4 = nn.Conv2d(512, out_channels, 1)

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

        return [d4, d3, d2, d1]

class ConvBlock(nn.Module):
    """Module containing sequence of conv2d, relu, and batch norm."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        return self.bn(x)

class FCOSHead(nn.Module):
    """Head for FCOS model.

    Outputs reg_arr and label_arr for one level of the pyramid, which can
    be decoded into boxes and labels.
    """
    def __init__(self, num_labels, in_channels=256):
        super().__init__()
        c = in_channels

        self.reg_branch = nn.Sequential(
            *[ConvBlock(c, c, 3, padding=1) for i in range(4)])
        self.reg_conv = nn.Conv2d(c, 4, 1)

        self.label_branch = nn.Sequential(
            *[ConvBlock(c, c, 3, padding=1) for i in range(4)])
        self.label_conv = nn.Conv2d(c, num_labels, 1)

        # Initialize weights so outputs have more reasonable values, following
        # Retinanet paper.
        self.label_conv.weight.data.fill_(0.0)
        prior = torch.tensor(0.01)
        logit = torch.log(prior / (1 - prior))
        self.label_conv.bias.data.fill_(logit.item())

        self.reg_conv.weight.data.fill_(0.0)
        self.reg_conv.bias.data.fill_(1.0)

    def forward(self, x, scale_param):
        """Computes output of head.

        Args:
            x: (tensor) with shape (batch_sz, 256, h*, w*) which is assumed to
                be output of one level of the FPN.
            scale_param: (tensor) with single element used to scale the values
                in the reg_arr and varies across levels of the pyramid.

        Returns:
            (dict) of form {
                'reg_arr': <tensor with shape (batch_sz, 4, h*, w*)>,
                'label_arr': <tensor with shape (batch_sz, num_labels, h*, w*>
            }
        """
        reg_arr = torch.exp(scale_param * self.reg_conv(self.reg_branch(x)))
        label_arr = self.label_conv(self.label_branch(x))
        return {'reg_arr': reg_arr, 'label_arr': label_arr}

class FCOS(nn.Module):
    """Fully convolutional one stage object detector

    See https://arxiv.org/abs/1904.01355
    """
    def __init__(self, backbone_arch, num_labels, pretrained=True):
        super().__init__()

        out_channels = 256
        self.num_labels = num_labels
        self.fpn = FPN(backbone_arch, out_channels=out_channels,
                       pretrained=pretrained)
        num_scales = len(self.fpn.strides)
        self.scale_params = nn.Parameter(torch.ones((num_scales,)))
        self.head = FCOSHead(num_labels, in_channels=out_channels)

    def loss(self, out, targets):
        """Compute loss for a single image.

        Args:
            out: (dict) the output of the heads for the whole pyramid
            targets: (dict) the encoded targets for the whole pyramid

            the format for both is:
            (dict) of form {
                'reg_arr': <tensor with shape (4, h*, w*)>,
                'label_arr': <tensor with shape (num_labels, h*, w*>
            }

        Returns:
            (tensor) with one float element containing loss
        """
        lmbda = 1.0
        # should we normalize by total number of cells in pyramid?
        for i, s in enumerate(out.keys()):
            pos_indicator = targets[s]['label_arr'].sum(dim=0)
            ll = focal_loss(
                out[s]['label_arr'], targets[s]['label_arr'])
            rl = pos_indicator.unsqueeze(0) * nn.functional.l1_loss(
                out[s]['reg_arr'], targets[s]['reg_arr'], reduction='none')
            # Put lower bound on npos to avoid dividing by zero.
            npos = pos_indicator.reshape(-1).sum()
            min_npos = torch.ones_like(npos)
            npos = torch.max(min_npos, npos)
            rl = rl.reshape(-1).sum() / npos
            if i == 0:
                loss = ll + lmbda * rl
            else:
                loss += ll + lmbda * rl
        return loss

    def forward(self, input, targets=None):
        """Compute output of FCOS.

        Note: boxes are in (ymin, xmin, ymax, xmax) format with values between
            0 and h or w. labels are class ids starting at 0.

        Args:
            input: (tensor) batch of images with shape (batch_sz, 3, h, w)
            targets: (list) of length batch_sz with elements of form
                {'boxes': <tensor with shape (n, 4)>,
                 'labels': <tensor with shape (n,)>}
        Returns:
            if targets is None, returns list of length batch_sz with elements of
            form {'boxes': <tensor with shape (n, 4)>,
                  'labels': <tensor with shape (n,)>,
                  'scores': <tensor with shape (n,)>}

            if target is a list, returns the loss as a single element tensor
        """
        fpn_out = self.fpn(input)

        batch_sz = input.shape[0]
        h, w = input.shape[2:]
        strides = self.fpn.strides
        hws = [level_out.shape[2:] for level_out in fpn_out]
        max_box_sides = [256, 128, 64, 32]
        iou_thresh = 0.5
        pyramid_shape = [
            (s, m, h, w) for s, m, (h, w) in zip(strides, max_box_sides, hws)]

        head_out = {}
        for i, (stride, level_out) in enumerate(zip(strides, fpn_out)):
            head_out[stride] = self.head(level_out, self.scale_params[i])

        if targets is None:
            out = []
            for i in range(batch_sz):
                single_head_out = {}
                for k, v in head_out.items():
                    single_head_out[k] = {
                        'reg_arr': v['reg_arr'][i],
                        'label_arr': v['label_arr'][i]
                    }
                boxes, labels, scores = decode_output(single_head_out)
                good_inds = compute_nms(
                    boxes.detach().cpu().numpy(),
                    labels.detach().cpu().numpy(),
                    scores.detach().cpu().numpy(), iou_thresh=iou_thresh)
                boxes, labels, scores = \
                    boxes[good_inds, :], labels[good_inds], scores[good_inds]
                out.append({'boxes': boxes, 'labels': labels, 'scores': scores})
            return out

        for i, single_target in enumerate(targets):
            boxes = single_target['boxes']
            labels = single_target['labels']
            encoded_targets = encode_targets(
                boxes, labels, pyramid_shape, self.num_labels)

            single_head_out = {}
            for s in strides:
                single_head_out[s] = {
                    'reg_arr': head_out[s]['reg_arr'][i],
                    'label_arr': head_out[s]['label_arr'][i]
                }
            if i == 0:
                loss = self.loss(single_head_out, encoded_targets)
            else:
                loss += self.loss(single_head_out, encoded_targets)
        return loss / batch_sz