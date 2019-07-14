import unittest
import torch
from torch.nn.functional import binary_cross_entropy as bce, l1_loss

from mlx.od.ssd.utils import (
    ObjectDetectionGrid, BoxList, compute_intersection, compute_iou, F1)

class TestIOU(unittest.TestCase):
    def test_compute_intersection(self):
        a = torch.tensor([[0, 0, 2, 2],
                        [1, 1, 3, 3],
                        [2, 2, 4, 4]], dtype=torch.float)
        b = torch.tensor([[0, 0, 2, 2],
                         [1, 1, 3, 3]], dtype=torch.float)

        inter = compute_intersection(a, b)
        exp_inter = torch.tensor(
            [[4, 1],
            [1, 4],
            [0, 1]], dtype=torch.float)
        self.assertTrue(inter.equal(exp_inter))

    def test_compute_iou(self):
        a = torch.tensor([[0, 0, 2, 2],
                          [1, 1, 3, 3],
                          [2, 2, 4, 4]], dtype=torch.float)
        b = torch.tensor([[0, 0, 2, 2],
                          [1, 1, 3, 3]], dtype=torch.float)
        inter = compute_iou(a, b)
        exp_inter = torch.tensor(
            [[1, 1./7],
             [1./7, 1],
             [0, 1./7]], dtype=torch.float)
        self.assertTrue(inter.equal(exp_inter))

class TestBoxList(unittest.TestCase):
    def test_score_filter(self):
        boxes = torch.tensor([[0, 0, 2, 2],
                              [1, 1, 3, 3]], dtype=torch.float)
        labels = torch.tensor([0, 1])
        scores = torch.tensor([0.3, 0.7])
        bl = BoxList(boxes, labels, scores)
        filt_bl = bl.score_filter(0.5)

        exp_bl = BoxList(torch.tensor([[1, 1, 3, 3]], dtype=torch.float),
                         torch.tensor([1]),
                         torch.tensor([0.7]))
        self.assertTrue(filt_bl.equal(exp_bl))

    def test_nms(self):
        boxes = torch.tensor([[0, 0, 10, 10],
                              [1, 1, 11, 11],
                              [9, 9, 19, 19],
                              [0, 0, 10, 10],
                              [20, 20, 21, 21]], dtype=torch.float)
        labels = torch.tensor([0, 0, 0, 1, 1])
        scores = torch.tensor([0.5, 0.7, 0.5, 0.5, 0.5])
        bl = BoxList(boxes, labels, scores)
        bl = bl.nms(0.5)

        exp_boxes = torch.tensor([[1, 1, 11, 11],
                                  [9, 9, 19, 19],
                                  [0, 0, 10, 10],
                                  [20, 20, 21, 21]], dtype=torch.float)
        exp_labels = torch.tensor([0, 0, 1, 1])
        exp_scores = torch.tensor([0.7, 0.5, 0.5, 0.5])
        exp_bl = BoxList(exp_boxes, exp_labels, exp_scores)
        self.assertTrue(bl.equal(exp_bl))

class TestDetectorGrid(unittest.TestCase):
    def setUp(self):
        grid_sz = 2
        anc_sizes = torch.tensor([
            [2, 0.5],
            [0.5, 2]])
        num_classes = 2
        self.grid = ObjectDetectionGrid(grid_sz, anc_sizes, num_classes)

    def test_decode(self):
        batch_sz = 1
        out = torch.zeros(self.grid.get_out_shape(batch_sz), dtype=torch.float)
        # y_offset, x_offset, y_scale, x_scale, c0, c1
        out[0, 1, :, 0, 0] = torch.tensor([0.5, 0, 1, 1, 0.1, 0.7])

        exp_boxes = torch.tensor([-0.25, -1.5, 0.25, 0.5])
        exp_labels = torch.ones((1, 8), dtype=torch.long)
        exp_labels[0, 1] = torch.tensor(1)
        exp_scores = torch.zeros((1, 8))
        exp_scores[0, 1] = torch.tensor(0.7)

        boxes, labels, scores = self.grid.decode(out)
        self.assertTrue(boxes[0, 1, :].equal(exp_boxes))
        self.assertTrue(labels.equal(exp_labels))
        self.assertTrue(scores.equal(exp_scores))

    def test_encode(self):
        exp_out = torch.zeros(self.grid.get_out_shape(1), dtype=torch.float)
        # y_offset, x_offset, y_scale, x_scale, c0, c1
        exp_out[0, 1, :, 0, 1] = torch.tensor([0, 0, 1, 0.5, 0, 1])

        boxes = torch.tensor([[[-0.75, 0, -0.25, 1]]])
        labels = torch.tensor([[1]])
        out = self.grid.encode(boxes, labels)
        self.assertTrue(out.equal(exp_out))

    def test_get_preds(self):
        grid_sz = 2
        anc_sizes = torch.tensor([
            [1., 1],
            [2, 2]])
        num_classes = 2
        grid = ObjectDetectionGrid(grid_sz, anc_sizes, num_classes)

        boxes = torch.tensor([[[0, 0, 0.5, 0.5]]])
        labels = torch.tensor([[1]])
        output = grid.encode(boxes, labels)
        b, l, s = grid.get_preds(output)
        self.assertTrue(b.equal(boxes))
        self.assertTrue(l.equal(labels))

    def test_compute_losses(self):
        boxes = torch.tensor([[[-0.75, 0, -0.25, 1]]])
        labels = torch.tensor([[1]])
        gt = self.grid.encode(boxes, labels)

        boxes = torch.tensor([[[-1., 0, 0, 1]]])
        labels = torch.tensor([[0]])
        out = self.grid.encode(boxes, labels)

        bl, cl = self.grid.compute_losses(out, gt)
        bl, cl = bl.item(), cl.item()

        exp_bl = l1_loss(torch.tensor([0, 0, 1, 0.5]),
                         torch.tensor([0, 0, 2, 0.5])).item()
        self.assertEqual(bl, exp_bl)

        num_class_els = 16
        exp_cl = ((2 * bce(torch.tensor(1.), torch.tensor(0.))).item() /
                  num_class_els)
        self.assertEqual(cl, exp_cl)

class TestF1(unittest.TestCase):
    def setUp(self):
        grid_sz = 2
        anc_sizes = torch.tensor([
            [1., 1],
            [2, 2]])
        num_classes = 3
        self.grid = ObjectDetectionGrid(grid_sz, anc_sizes, num_classes)
        self.f1 = F1(self.grid, score_thresh=0.3, iou_thresh=0.5)
        self.f1.on_epoch_begin()

    def test1(self):
        # Two images in each batch. Each image has:
        # Two boxes, both match.
        boxes = torch.tensor([
            [[0, 0, 0.5, 0.5], [-1, -1, -0.5, -0.5]],
            [[0, 0, 0.5, 0.5], [-1, -1, -0.5, -0.5]]
        ])
        labels = torch.tensor([[1, 1], [1, 1]])
        output = self.grid.encode(boxes, labels)

        target_boxes = torch.tensor([
            [[0, 0, 0.5, 0.5], [-1, -1, -0.5, -0.5]],
            [[0, 0, 0.5, 0.5], [-1, -1, -0.5, -0.5]]
        ])
        target_labels = torch.tensor([[1, 1], [1, 1]])
        target = (target_boxes, target_labels)

        # Simulate two batches
        self.f1.on_batch_end(output, target)
        self.f1.on_batch_end(output, target)
        metrics = self.f1.on_epoch_end({})
        exp_f1 = self.f1.compute_f1(8, 0, 0)
        self.assertEqual(exp_f1, 1.0)
        self.assertEqual(exp_f1, metrics['last_metrics'][0])

    def test2(self):
        # Two boxes, one matches, the other doesn't overlap enough.
        boxes = torch.tensor([
            [[0, 0, 0.5, 0.5], [-1, -1, -0.5, -0.5]]
        ])
        labels = torch.tensor([[1, 1]])
        output = self.grid.encode(boxes, labels)

        target_boxes = torch.tensor([
            [[0, 0, 0.1, 0.1], [-1, -1, -0.5, -0.5]],
        ])
        target_labels = torch.tensor([[1, 1]])
        target = (target_boxes, target_labels)

        self.f1.on_batch_end(output, target)
        metrics = self.f1.on_epoch_end({})
        exp_f1 = self.f1.compute_f1(1, 1, 1)
        self.assertEqual(exp_f1, 0.5)
        self.assertEqual(exp_f1, metrics['last_metrics'][0])

    def test3(self):
        # Three boxes, one matches, one overlaps but has the wrong label,
        # and one doesn't match.
        boxes = torch.tensor([
            [[0, 0, 0.5, 0.5], [-1, -1, -0.5, -0.5], [-0.5, 0, 0, 0.5]]
        ])
        labels = torch.tensor([[1, 2, 1]])
        output = self.grid.encode(boxes, labels)

        target_boxes = torch.tensor([
            [[0, 0, 0.5, 0.5], [-1, -1, -0.5, -0.5], [-0.5, 0, -0.4, 0.1]]
        ])
        target_labels = torch.tensor([[1, 1, 1]])
        target = (target_boxes, target_labels)

        self.f1.on_batch_end(output, target)
        metrics = self.f1.on_epoch_end({})
        exp_f1 = self.f1.compute_f1(1, 2, 2)
        self.assertEqual(exp_f1, metrics['last_metrics'][0])


if __name__ == '__main__':
    unittest.main()
