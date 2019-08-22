import unittest

import torch

from mlx.od.fcos.model import FPN, FCOSHead, FCOS
from mlx.od.boxlist import BoxList

class TestFPN(unittest.TestCase):
    def test_fpn(self):
        model = FPN('resnet18', pretrained=False)
        h, w = 64, 64
        c = 256
        img = torch.empty((1, 3, h, w))
        out = model(img)

        self.assertEqual(len(out), 5)
        for i, s in enumerate(model.strides):
            self.assertListEqual(list(out[i].shape), [1, c, h//s, w//s])

class TestFCOSHead(unittest.TestCase):
    def test_fcos_head(self):
        h, w = 64, 64
        c = 256
        num_labels = 3
        scale_param = 2
        head = FCOSHead(num_labels, in_channels=c)
        x = torch.empty((1, c, h, w))
        reg_arr, label_arr, center_arr = head(x, scale_param)

        self.assertListEqual(
            list(reg_arr.shape), [1, 4, h, w])
        self.assertListEqual(
            list(label_arr.shape), [1, num_labels, h, w])
        self.assertListEqual(
            list(center_arr.shape), [1, 1, h, w])

class TestFCOS(unittest.TestCase):
    def test_fcos(self):
        h, w = 64, 64
        num_labels = 3
        x = torch.empty((1, 3, h, w))
        model = FCOS('resnet18', num_labels, pretrained=False)
        boxlists = model(x)

        self.assertEqual(len(boxlists), 1)
        boxlist = boxlists[0]
        num_boxes = boxlist.boxes.shape[0]
        self.assertListEqual(list(boxlist.boxes.shape), [num_boxes, 4])
        self.assertListEqual(list(boxlist.get_field('labels').shape), [num_boxes])
        self.assertListEqual(list(boxlist.get_field('scores').shape), [num_boxes])

    def test_fcos_with_targets(self):
        h, w = 64, 64
        num_labels = 3
        x = torch.empty((1, 3, h, w))
        model = FCOS('resnet18', num_labels, pretrained=False)

        boxes = torch.tensor([
            [0, 0, 16, 16],
            [8, 8, 12, 12]
        ])
        labels = torch.tensor([0, 1])
        targets = [BoxList(boxes, labels=labels)]

        loss_dict = model(x, targets)
        self.assertTrue('label_loss' in loss_dict)
        self.assertTrue('reg_loss' in loss_dict)
        self.assertTrue('center_loss' in loss_dict)

    def test_backwards(self):
        h, w = 64, 64
        num_labels = 3
        x = 2.0 * torch.rand((1, 3, h, w)) - 1.0
        model = FCOS('resnet18', num_labels, pretrained=False)

        boxes = torch.tensor([
            [0, 0, 16, 16],
            [16, 16, 32, 32]
        ])
        labels = torch.tensor([0, 1])
        targets = [BoxList(boxes, labels=labels)]

        model.train()
        model.zero_grad()
        loss_dict = model(x, targets)
        loss = sum(list(loss_dict.values()))
        loss.backward()

        for param in model.parameters():
            self.assertTrue(len(torch.nonzero(param.grad)) > 0)

if __name__ == '__main__':
    unittest.main()