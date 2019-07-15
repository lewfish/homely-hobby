import unittest

import torch

from mlx.od.fcos.encoder import encode_box

class TestEncoder(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()