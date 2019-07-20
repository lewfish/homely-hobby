import unittest

import torch

from mlx.od.fcos.encoder import encode_box, encode_targets

class TestEncodeBox(unittest.TestCase):
    def test_encode_too_small(self):
        num_labels = 3
        label_arr = torch.zeros((num_labels, 3, 3))
        label = 1
        reg_arr = torch.zeros((4, 3, 3))
        stride = 4
        box = torch.Tensor([0, 0, 2, 2])
        encode_box(reg_arr, label_arr, stride, box, label)

        exp_reg_arr = torch.zeros((4, 3, 3))
        self.assertTrue(reg_arr.equal(exp_reg_arr))

    def test_encode1(self):
        num_labels = 3
        label_arr = torch.zeros((num_labels, 3, 3))
        label = 1
        reg_arr = torch.zeros((4, 3, 3))
        stride = 4
        box = torch.Tensor([3, 3, 9, 9])
        encode_box(reg_arr, label_arr, stride, box, label)

        exp_reg_arr = torch.zeros((4, 3, 3))
        exp_reg_arr[:, 1, 1] = torch.Tensor([3, 3, 3, 3])
        self.assertTrue(reg_arr.equal(exp_reg_arr))

    def test_encode2(self):
        num_labels = 3
        label_arr = torch.zeros((num_labels, 3, 3))
        label = 1
        reg_arr = torch.zeros((4, 3, 3))
        stride = 4
        box = torch.Tensor([0, 0, 4, 12])
        encode_box(reg_arr, label_arr, stride, box, label)

        exp_reg_arr = torch.zeros((4, 3, 3))
        exp_reg_arr[:, 0, 0] = torch.Tensor([2, 2, 2, 10])
        exp_reg_arr[:, 0, 1] = torch.Tensor([2, 6, 2, 6])
        exp_reg_arr[:, 0, 2] = torch.Tensor([2, 10, 2, 2])
        self.assertTrue(reg_arr.equal(exp_reg_arr))

        exp_label_arr = torch.zeros((num_labels, 3, 3))
        exp_label_arr[label, 0, :] = 1
        self.assertTrue(label_arr.equal(exp_label_arr))

    def test_encode3(self):
        num_labels = 3
        label_arr = torch.zeros((num_labels, 3, 3))
        label = 1
        reg_arr = torch.zeros((4, 3, 3))
        stride = 4
        box = torch.Tensor([0, 8, 12, 12])
        encode_box(reg_arr, label_arr, stride, box, label)

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
        targets = encode_targets(boxes, labels, pyramid_shape, num_labels)

        self.assertTrue(torch.all(targets[8]['reg_arr'] == 0))
        self.assertTrue(torch.all(targets[8]['label_arr'] == 0))
        self.assertTrue(targets[8]['reg_arr'].shape == (4, 8, 8))
        self.assertTrue(targets[8]['label_arr'].shape == (2, 8, 8))

        exp_reg_arr = torch.zeros((4, 4, 4))
        exp_reg_arr[:, 0, 0] = torch.Tensor([8, 8, 8, 8])
        exp_reg_arr[:, 1, 1] = torch.Tensor([8, 8, 8, 8])
        self.assertTrue(targets[16]['reg_arr'].equal(exp_reg_arr))
        exp_label_arr = torch.zeros((2, 4, 4))
        exp_label_arr[0, 0, 0] = 1
        exp_label_arr[0, 1, 1] = 1
        self.assertTrue(targets[16]['label_arr'].equal(exp_label_arr))

        exp_reg_arr = torch.zeros((4, 2, 2))
        exp_reg_arr[:, 0, 0] = torch.Tensor([16, 16, 16, 16])
        self.assertTrue(targets[32]['reg_arr'].equal(exp_reg_arr))
        exp_label_arr = torch.zeros((2, 2, 2))
        exp_label_arr[1, 0, 0] = 1
        self.assertTrue(targets[32]['label_arr'].equal(exp_label_arr))

if __name__ == '__main__':
    unittest.main()