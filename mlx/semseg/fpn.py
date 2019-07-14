import torch
import torch.nn as nn
from torchvision import models


class FPN(nn.Module):
    """Feature Pyramid Network backbone.

    See https://arxiv.org/abs/1612.03144
    """
    def __init__(self, backbone_arch):
        super().__init__()

        # Setup bottom-up backbone and hooks to capture output of stages.
        # Assumes there is layer1, 2, 3, 4, which is true for Resnets.
        self.backbone = getattr(models, backbone_arch)(pretrained=True)
        self.backbone_out = {}

        def make_save_output(layer_name):
            def save_output(layer, input, output):
                self.backbone_out[layer_name] = output
            return save_output

        # TODO don't compute head of backbone.
        self.backbone.layer1.register_forward_hook(make_save_output('layer1'))
        self.backbone.layer2.register_forward_hook(make_save_output('layer2'))
        self.backbone.layer3.register_forward_hook(make_save_output('layer3'))
        self.backbone.layer4.register_forward_hook(make_save_output('layer4'))

        '''
        self.backbone(torch.rand((1, 3, 256, 256)))
        self.backbone_shapes = dict(
            [(ln, out.shape[1:]) for ln, out in self.backbone_out.items()])
        print(self.backbone_shapes)
        '''

        # Setup layers for top-down pathway.
        out_channels = 256
        self.cross_conv1 = nn.Conv2d(64, out_channels, 1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3)
        self.cross_conv2 = nn.Conv2d(128, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.cross_conv3 = nn.Conv2d(256, out_channels, 1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3)
        self.cross_conv4 = nn.Conv2d(512, out_channels, 1)

    def forward(self, input):
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

        return {
            'p4': d4,
            'p3': self.conv3(d3),
            'p2': self.conv2(d2),
            'p1': self.conv1(d1)
        }


class SegmentationFPN(nn.Module):
    """A semantic segmentation model using FPN."""
    def __init__(self, backbone_arch, num_classes):
        super().__init__()
        self.fpn = FPN(backbone_arch)
        self.conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, input):
        x = self.fpn(input)['p1']
        x = nn.functional.relu(x)
        x = nn.functional.interpolate(
            x, input.shape[2:], mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x

    @staticmethod
    def loss(output, target):
        return nn.functional.cross_entropy(output, target.squeeze(dim=1))

if __name__ == '__main__':
    num_classes = 10
    model = SegmentationFPN('resnet18', num_classes)
    x = torch.rand((1, 3, 256, 256))
    y = model(x)
    print(y.shape)
