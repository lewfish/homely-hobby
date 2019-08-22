import unittest

import torch

from mlx.od.metrics import compute_coco_eval

class TestCocoEval(unittest.TestCase):
    def test_coco_eval(self):
        outputs = [
            (
                torch.tensor([[0, 0, 100, 100]]),
                torch.tensor([0]),
                torch.tensor([0.9])
            )
        ]
        targets = [
            (
                torch.tensor([[0, 0, 100, 100]]),
                torch.tensor([0])
            )
        ]
        num_labels = 2
        ap = compute_coco_eval(outputs, targets, num_labels)
        self.assertAlmostEqual(ap, 1.0)

    def test_coco_eval2(self):
        outputs = [
            (
                torch.tensor([
                    [0, 0, 100, 100],
                    [100, 100, 200, 200],
                ]),
                torch.tensor([0, 0]),
                torch.tensor([0.9, 0.91])
            ),
            (
                torch.tensor([[100, 100, 200, 200]]),
                torch.tensor([0,]),
                torch.tensor([0.9,])
            )
        ]
        targets = [
            (
                torch.tensor([[0, 0, 100, 100]]),
                torch.tensor([0])
            ),
            (
                torch.tensor([]),
                torch.tensor([])
            ),
        ]
        num_labels = 2
        ap = compute_coco_eval(outputs, targets, num_labels)
        self.assertAlmostEqual(ap, 0.5)

    def test_coco_eval3(self):
        outputs = [
            (
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([])
            )
        ]
        targets = [
            (
                torch.tensor([[0, 0, 100, 100]]),
                torch.tensor([0])
            )
        ]
        num_labels = 2
        ap = compute_coco_eval(outputs, targets, num_labels)
        self.assertEqual(ap, -1)

if __name__ == '__main__':
    unittest.main()