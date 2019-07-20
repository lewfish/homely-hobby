import unittest

import torch

from mlx.od.fcos.decoder import decode_level_output, decode_output
from mlx.od.fcos.encoder import encode_box, encode_targets

class TestDecodeLevelOutput(unittest.TestCase):
    def encode_decode_level(self, stride, h, w, exp_boxes):
        score_thresh = 0.2
        num_labels = 2
        exp_labels = torch.tensor([0, 1])
        reg_arr = torch.zeros((4, h, w))
        label_arr = torch.zeros((num_labels, h, w))
        for box, label in zip(exp_boxes, exp_labels):
            encode_box(reg_arr, label_arr, stride, box, label.int().item())
        boxes, labels, scores = decode_level_output(
            reg_arr, label_arr, stride, score_thresh=score_thresh)

        def make_tuple_set(boxes, labels):
            return set([tuple(b.int().tolist()) + (l.int().item(),)
                        for b, l in zip(boxes, labels)])

        self.assertEqual(make_tuple_set(boxes.int(), labels.int()),
                         make_tuple_set(exp_boxes, exp_labels))
        self.assertTrue(torch.all(scores == 1))

    def test_decode1(self):
        stride = 4
        h, w = 2, 2
        exp_boxes = torch.tensor([
            [0, 0, 4, 4],
            [4, 4, 8, 8]
        ])
        self.encode_decode_level(stride, h, w, exp_boxes)

    def test_decode2(self):
        stride = 4
        h, w = 4, 4
        exp_boxes = torch.tensor([
            [1, 1, 9, 9],
            [7, 7, 15, 15]
        ])
        self.encode_decode_level(stride, h, w, exp_boxes)

    def test_decode3(self):
        stride = 4
        h, w = 4, 4
        exp_boxes = torch.tensor([
            [3, 3, 7, 7],
            [11, 11, 15, 15]
        ])
        self.encode_decode_level(stride, h, w, exp_boxes)

    def test_decode4(self):
        stride = 4
        h, w = 4, 4
        exp_boxes = torch.tensor([
            [3, 3, 7, 7],
            [11, 11, 15, 15]
        ])
        self.encode_decode_level(stride, h, w, exp_boxes)

    def test_decode5(self):
        stride = 4
        h, w = 4, 4
        exp_boxes = torch.tensor([
            [0, 0, 16, 16],
            [8, 8, 12, 12]
        ])
        self.encode_decode_level(stride, h, w, exp_boxes)

    def test_decode6(self):
        stride = 4
        h, w = 4, 4
        exp_boxes = torch.tensor([
            [0, 0, 16, 16],
            [8, 8, 12, 12]
        ])
        self.encode_decode_level(stride, h, w, exp_boxes)

    def test_decode7(self):
        stride = 4
        h, w = 7, 11
        exp_boxes = torch.tensor([
            [1, 3, 17, 23],
            [5, 7, 9, 11]
        ])
        self.encode_decode_level(stride, h, w, exp_boxes)

class TestDecodeOutput(unittest.TestCase):
    def encode_decode_output(self, pyramid_shape, exp_boxes, exp_labels):
        score_thresh = 0.2
        num_labels = 2

        targets = encode_targets(
            exp_boxes, exp_labels, pyramid_shape, num_labels)
        boxes, labels, scores = decode_output(
            targets, score_thresh=score_thresh)

        def make_tuple_set(boxes, labels):
            return set([tuple(b.int().tolist()) + (l.int().item(),)
                        for b, l in zip(boxes, labels)])

        self.assertEqual(make_tuple_set(boxes.int(), labels.int()),
                         make_tuple_set(exp_boxes, exp_labels))
        self.assertTrue(torch.all(scores == 1))

    def test_decode(self):
        pyramid_shape = [
            (32, 32, 2, 2),
            (16, 16, 4, 4),
            (8, 8, 8, 8)
        ]
        exp_boxes = torch.Tensor([
            [0, 0, 16, 16],
            [16, 16, 32, 32],
            [0, 0, 32, 32]
        ])
        exp_labels = torch.Tensor([0, 0, 1])
        self.encode_decode_output(pyramid_shape, exp_boxes, exp_labels)

if __name__ == '__main__':
    unittest.main()
