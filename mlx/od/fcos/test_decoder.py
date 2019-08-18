import unittest

import torch

from mlx.od.fcos.decoder import (
    decode_level_output, decode_single_output, decode_batch_output)
from mlx.od.fcos.encoder import encode_box, encode_single_targets

def make_tuple_set(boxes, labels):
    return set([tuple(b.int().tolist()) + (l.int().item(),)
                for b, l in zip(boxes, labels)])

class TestDecodeLevelOutput(unittest.TestCase):
    def encode_decode_level(self, stride, h, w, exp_boxes):
        score_thresh = 0.2
        num_labels = 2
        exp_labels = torch.tensor([0, 1])
        reg_arr = torch.zeros((4, h, w))
        label_arr = torch.zeros((num_labels, h, w))
        center_arr = torch.zeros((1, h, w))
        for box, label in zip(exp_boxes, exp_labels):
            encode_box(reg_arr, label_arr, center_arr, stride, box,
                       label.int().item())
        boxlist = decode_level_output(
            reg_arr, label_arr, center_arr, stride, score_thresh=score_thresh)

        self.assertEqual(make_tuple_set(boxlist.boxes.int(), boxlist.labels.int()),
                         make_tuple_set(exp_boxes, exp_labels))
        self.assertTrue(torch.all(boxlist.scores == 1))

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

    def test_decode_centerness(self):
        reg_arr = torch.zeros((4, 3, 3))
        label_arr = torch.zeros((1, 3, 3))
        center_arr = torch.zeros((1, 3, 3))
        stride = 4

        label_arr[0, 0, 0] = 0.5
        label_arr[0, 0, 1] = 0.5
        center_arr[0, 0, 0] = 0.1
        center_arr[0, 0, 1] = 0.2
        center_arr[0, 1, 0] = 0.3
        center_arr[0, 1, 1] = 0.4

        boxlist = decode_level_output(
            reg_arr, label_arr, center_arr, stride)

        exp_labels = torch.tensor([0, 0])
        exp_centerness = torch.tensor([0.1, 0.2])
        self.assertTrue(boxlist.labels.equal(exp_labels))
        self.assertTrue(boxlist.centerness.equal(exp_centerness))

class TestDecodeSingleOutput(unittest.TestCase):
    def encode_decode_output(self, pyramid_shape, exp_boxes, exp_labels):
        score_thresh = 0.2
        num_labels = 2

        targets = encode_single_targets(
            exp_boxes, exp_labels, pyramid_shape, num_labels)
        boxlist = decode_single_output(
            targets, pyramid_shape, score_thresh=score_thresh)

        self.assertEqual(make_tuple_set(boxlist.boxes.int(), boxlist.labels.int()),
                         make_tuple_set(exp_boxes, exp_labels))
        self.assertTrue(torch.all(boxlist.scores == 1))

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

class TestDecodeBatchOutput(unittest.TestCase):
    def test_decode(self):
        pyramid_shape = [
            (32, 32, 2, 2),
            (16, 16, 4, 4),
            (8, 8, 8, 8)
        ]
        num_labels = 2
        img_height, img_width = (64, 64)

        exp_boxes = [
            torch.Tensor([
                [0, 0, 16, 16],
                [16, 16, 32, 32],
                [0, 0, 32, 32]
            ]),
            torch.Tensor([
                [0, 0, 16, 16],
            ])
        ]

        exp_labels = [
            torch.Tensor([0, 0, 1]),
            torch.Tensor([1]),
        ]

        targets = []
        targets.append(encode_single_targets(
            exp_boxes[0], exp_labels[0], pyramid_shape, num_labels))
        targets.append(encode_single_targets(
            exp_boxes[1], exp_labels[1], pyramid_shape, num_labels))

        # Merge single targets into a batch.
        pyramid_sz = len(pyramid_shape)
        batch_sz = len(exp_boxes)
        batch_targets = []
        for level_ind in range(pyramid_sz):
            reg_arrs, label_arrs, center_arrs = [], [], []
            for batch_ind in range(batch_sz):
                reg_arr, label_arr, center_arr = targets[batch_ind][level_ind]
                def prob2logit(prob):
                    return torch.log(prob / (1 - prob))
                label_arr = prob2logit(label_arr)
                center_arr = prob2logit(center_arr)
                reg_arrs.append(reg_arr.unsqueeze(0))
                label_arrs.append(label_arr.unsqueeze(0))
                center_arrs.append(center_arr.unsqueeze(0))
            batch_targets.append((
                torch.cat(reg_arrs), torch.cat(label_arrs), torch.cat(center_arrs)))

        boxlists = decode_batch_output(
            batch_targets, pyramid_shape, img_height, img_width)
        for ind, boxlist in enumerate(boxlists):
            self.assertEqual(make_tuple_set(boxlist.boxes.int(), boxlist.labels.int()),
                             make_tuple_set(exp_boxes[ind], exp_labels[ind]))
            self.assertTrue(torch.all(boxlist.scores == 1))

if __name__ == '__main__':
    unittest.main()
