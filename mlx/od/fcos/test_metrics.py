import unittest

import torch

from mlx.od.fcos.metrics import compute_coco_eval

class TestCocoEval(unittest.TestCase):
    def test_coco_eval(self):
        outputs = [
            {
                'boxes': torch.tensor([
                    [0, 0, 100, 100],
                ]),
                'labels': torch.tensor([0]),
                'scores': torch.tensor([0.9])
            }
        ]
        targets = [
            {
                'boxes': torch.tensor([
                    [0, 0, 100, 100],
                ]),
                'labels': torch.tensor([0])
            }
        ]
        num_labels = 2
        ap = compute_coco_eval(outputs, targets, num_labels)
        self.assertAlmostEqual(ap, 1.0)

    def test_coco_eval2(self):
        outputs = [
            {
                'boxes': torch.tensor([
                    [0, 0, 100, 100],
                    [100, 100, 200, 200],
                ]),
                'labels': torch.tensor([0, 0]),
                'scores': torch.tensor([0.9, 0.91])
            },
            {
                'boxes': torch.tensor([
                    [100, 100, 200, 200],
                ]),
                'labels': torch.tensor([0,]),
                'scores': torch.tensor([0.9,])
            }
        ]
        targets = [
            {
                'boxes': torch.tensor([
                    [0, 0, 100, 100],
                ]),
                'labels': torch.tensor([0])
            },
            {
                'boxes': torch.tensor([]),
                'labels': torch.tensor([])
            },
        ]
        num_labels = 2
        ap = compute_coco_eval(outputs, targets, num_labels)
        self.assertAlmostEqual(ap, 0.5)

    def test_coco_eval3(self):
        outputs = [
            {
                'boxes': torch.tensor([]),
                'labels': torch.tensor([]),
                'scores': torch.tensor([])
            }
        ]
        targets = [
            {
                'boxes': torch.tensor([
                    [0, 0, 100, 100],
                ]),
                'labels': torch.tensor([0])
            }
        ]
        num_labels = 2
        ap = compute_coco_eval(outputs, targets, num_labels)
        self.assertEqual(ap, -1)

if __name__ == '__main__':
    unittest.main()