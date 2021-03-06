import unittest
import math

import torch

from mlx.od.fcos.encoder import encode_box, encode_single_targets

class TestEncodeBox(unittest.TestCase):
    def test_encode_too_small(self):
        num_labels = 3
        reg_arr = torch.zeros((4, 3, 3))
        label_arr = torch.zeros((num_labels, 3, 3))
        center_arr = torch.zeros((1, 3, 3))
        label = 1
        stride = 4
        box = torch.Tensor([0, 0, 2, 2])
        encode_box(reg_arr, label_arr, center_arr, stride, box, label)

        exp_reg_arr = torch.zeros((4, 3, 3))
        self.assertTrue(reg_arr.equal(exp_reg_arr))

    def test_encode1(self):
        num_labels = 3
        reg_arr = torch.zeros((4, 3, 3))
        label_arr = torch.zeros((num_labels, 3, 3))
        center_arr = torch.zeros((1, 3, 3))
        label = 1
        stride = 4
        box = torch.Tensor([0, 0, 4, 12])
        encode_box(reg_arr, label_arr, center_arr, stride, box, label)

        exp_reg_arr = torch.zeros((4, 3, 3))
        exp_reg_arr[:, 0, 0] = torch.Tensor([2, 2, 2, 10])
        exp_reg_arr[:, 0, 1] = torch.Tensor([2, 6, 2, 6])
        exp_reg_arr[:, 0, 2] = torch.Tensor([2, 10, 2, 2])
        self.assertTrue(reg_arr.equal(exp_reg_arr))

        exp_label_arr = torch.zeros((num_labels, 3, 3))
        exp_label_arr[label, 0, :] = 1
        self.assertTrue(label_arr.equal(exp_label_arr))

        exp_center_arr = torch.zeros((1, 3, 3))
        exp_center_arr[0, 0, 0] = math.sqrt(1/5)
        exp_center_arr[0, 0, 1] = math.sqrt(1)
        exp_center_arr[0, 0, 2] = math.sqrt(1/5)
        self.assertTrue(center_arr.equal(exp_center_arr))

    def test_encode2(self):
        num_labels = 3
        reg_arr = torch.zeros((4, 3, 3))
        label_arr = torch.zeros((num_labels, 3, 3))
        center_arr = torch.zeros((1, 3, 3))
        label = 1
        stride = 4
        box = torch.Tensor([0, 8, 12, 12])
        encode_box(reg_arr, label_arr, center_arr, stride, box, label)

        exp_reg_arr = torch.zeros((4, 3, 3))
        exp_reg_arr[:, 0, 2] = torch.Tensor([2, 2, 10, 2])
        exp_reg_arr[:, 1, 2] = torch.Tensor([6, 2, 6, 2])
        exp_reg_arr[:, 2, 2] = torch.Tensor([10, 2, 2, 2])
        self.assertTrue(reg_arr.equal(exp_reg_arr))

        exp_label_arr = torch.zeros((num_labels, 3, 3))
        exp_label_arr[label, :, 2] = 1
        self.assertTrue(label_arr.equal(exp_label_arr))

class TestEncodeTargets(unittest.TestCase):
    def test_encode(self):
        pyramid_shape = [
            (32, 32, 2, 2),
            (16, 16, 4, 4),
            (8, 8, 8, 8)
        ]
        num_labels = 2
        boxes = torch.Tensor([
            [0, 0, 16, 16],
            [16, 16, 32, 32],
            [0, 0, 32, 32]
        ])
        labels = torch.Tensor([0, 0, 1])
        targets = encode_single_targets(boxes, labels, pyramid_shape, num_labels)

        # stride 8
        reg_arr, label_arr, center_arr = targets[2]
        self.assertTrue(torch.all(reg_arr == 0))
        self.assertTrue(torch.all(label_arr == 0))
        self.assertTrue(reg_arr.shape == (4, 8, 8))
        self.assertTrue(label_arr.shape == (2, 8, 8))

        # stride 16
        reg_arr, label_arr, center_arr = targets[1]
        exp_reg_arr = torch.zeros((4, 4, 4))
        exp_reg_arr[:, 0, 0] = torch.Tensor([8, 8, 8, 8])
        exp_reg_arr[:, 1, 1] = torch.Tensor([8, 8, 8, 8])
        self.assertTrue(reg_arr.equal(exp_reg_arr))
        exp_label_arr = torch.zeros((2, 4, 4))
        exp_label_arr[0, 0, 0] = 1
        exp_label_arr[0, 1, 1] = 1
        self.assertTrue(label_arr.equal(exp_label_arr))

        # stride 32
        reg_arr, label_arr, center_arr = targets[0]
        exp_reg_arr = torch.zeros((4, 2, 2))
        exp_reg_arr[:, 0, 0] = torch.Tensor([16, 16, 16, 16])
        self.assertTrue(reg_arr.equal(exp_reg_arr))
        exp_label_arr = torch.zeros((2, 2, 2))
        exp_label_arr[1, 0, 0] = 1
        self.assertTrue(label_arr.equal(exp_label_arr))

if __name__ == '__main__':
    unittest.main()