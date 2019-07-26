from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops.boxes import batched_nms

from mlx.od.fcos.decoder import decode_output
from mlx.od.fcos.encoder import encode_targets
from mlx.od.fcos.nms import compute_nms
from mlx.od.fcos.loss import focal_loss

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
        self.strides = [32, 16, 8, 4]
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

        out = [d4, d3, d2, d1]
        if self.levels is not None:
            out = [out[l] for l in self.levels]
        return out

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

        self.center_conv = nn.Conv2d(c, 1, 1)

        # Initialize weights so outputs have more reasonable values, following
        # Retinanet paper.
        self.reg_conv.weight.data.fill_(0.0)
        self.reg_conv.bias.data.fill_(1.0)

        self.label_conv.weight.data.fill_(0.0)
        prior = torch.tensor(0.01)
        logit = torch.log(prior / (1 - prior))
        self.label_conv.bias.data.fill_(logit.item())

        self.center_conv.weight.data.fill_(0.0)
        prior = torch.tensor(0.5)
        logit = torch.log(prior / (1 - prior))
        self.center_conv.bias.data.fill_(logit)

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
            label_arr contains logits
        """
        reg_arr = torch.exp(scale_param * self.reg_conv(self.reg_branch(x)))
        label_branch_arr = self.label_branch(x)
        label_arr = self.label_conv(label_branch_arr)
        center_arr = self.center_conv(label_branch_arr)
        return {'reg_arr': reg_arr, 'label_arr': label_arr, 'center_arr': center_arr}

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

    def loss(self, out, targets):
        """Compute loss for a single image.

        Note: the label_arr and center_arr for output is assumed to contain
        logits, and is assumed to contain probabilities for targets.

        Args:
            out: (dict) the output of the heads for the whole pyramid
            targets: (dict) the encoded targets for the whole pyramid

            the format for both is:
            (dict) of form {
                'reg_arr': <tensor with shape (4, h*, w*)>,
                'label_arr': <tensor with shape (num_labels, h*, w*)>,
                'center_arr': <tensor with shape (1, h*, w*)>
            }

        Returns:
            (tensor) with one float element containing loss
        """
        lmbda = 1.0
        # should we normalize by total number of cells in pyramid?
        for i, s in enumerate(out.keys()):
            pos_indicator = targets[s]['label_arr'].sum(dim=0)
            # Put lower bound on npos to avoid dividing by zero.
            npos = pos_indicator.reshape(-1).sum()
            min_npos = torch.ones_like(npos)
            npos = torch.max(min_npos, npos)

            ll = focal_loss(
                out[s]['label_arr'], targets[s]['label_arr'])
            rl = pos_indicator.unsqueeze(0) * nn.functional.l1_loss(
                out[s]['reg_arr'], targets[s]['reg_arr'], reduction='none')
            rl = rl.reshape(-1).sum()
            cl = pos_indicator.unsqueeze(0) * nn.functional.binary_cross_entropy_with_logits(
                out[s]['center_arr'], targets[s]['center_arr'], reduction='none')
            cl = cl.reshape(-1).sum()

            l = (ll / npos) + lmbda * (rl / npos) + (cl / npos)
            if i == 0:
                loss = l
            else:
                loss += l
        return loss

    def forward(self, input, targets=None, get_head_out=False):
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
            this returns boxes with score > 0.05. Further filtering based on
            score should be done before considering the prediction "final".

            if target is a list, returns the loss as a single element tensor
        """
        fpn_out = self.fpn(input)

        batch_sz = input.shape[0]
        h, w = input.shape[2:]
        strides = self.fpn.strides
        hws = [level_out.shape[2:] for level_out in fpn_out]
        max_box_sides = [256, 128, 64, 32]
        if self.levels is not None:
            max_box_sides = [max_box_sides[l] for l in self.levels]
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
                    # Convert logits in label_arr and center_arr
                    # to probabilities since decode expects probabilities.
                    single_head_out[k] = {
                        'reg_arr': v['reg_arr'][i],
                        'label_arr': torch.sigmoid(v['label_arr'][i]),
                        'center_arr': torch.sigmoid(v['center_arr'][i])
                    }
                boxes, labels, scores, centerness = decode_output(single_head_out)
                nms_scores = scores * centerness

                boxes = torch.stack([
                    torch.clamp(boxes[:, 0], 0, h),
                    torch.clamp(boxes[:, 1], 0, w),
                    torch.clamp(boxes[:, 2], 0, h),
                    torch.clamp(boxes[:, 3], 0, w)
                ], dim=1)
                good_inds = batched_nms(boxes, nms_scores, labels, iou_thresh)
                boxes, labels, scores = \
                    boxes[good_inds, :], labels[good_inds], scores[good_inds]
                out.append({'boxes': boxes, 'labels': labels, 'scores': scores})
            if get_head_out:
                return out, head_out
            return out

        for i, single_target in enumerate(targets):
            boxes = single_target['boxes']
            labels = single_target['labels']
            encoded_targets = encode_targets(
                boxes, labels, pyramid_shape, self.num_labels)

            single_head_out = {}
            for s in strides:
                # Don't convert logits to probabilities for output since
                # loss function expects logits for output
                # (and probabilities for targets)
                single_head_out[s] = {
                    'reg_arr': head_out[s]['reg_arr'][i],
                    'label_arr': head_out[s]['label_arr'][i],
                    'center_arr': head_out[s]['center_arr'][i]
                }
            if i == 0:
                loss = self.loss(single_head_out, encoded_targets)
            else:
                loss += self.loss(single_head_out, encoded_targets)
        loss = loss / batch_sz
        if get_head_out:
            return loss, head_out
        return loss