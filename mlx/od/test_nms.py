import unittest

import shapely
import numpy as np

from mlx.od.nms import compute_nms, compute_iou

class TestNMS(unittest.TestCase):
    def test_iou(self):
        geom1 = shapely.geometry.box(0, 0, 4, 4)
        geom2 = shapely.geometry.box(2, 2, 6, 6)
        iou = compute_iou(geom1, geom2)
        self.assertEqual(iou, 4 / (32 - 4))

    def test_nms(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [5, 5, 10, 10],
            [1, 1, 11, 11]
        ])
        labels = np.array([1, 1, 1, 2])
        scores = np.array([0.8, 0.9, 0.9, 0.9])
        good_inds = compute_nms(boxes, labels, scores, iou_thresh=0.5)
        self.assertListEqual(good_inds, [1, 2, 3])

        scores = np.array([0.9, 0.8, 0.9, 0.9])
        good_inds = compute_nms(boxes, labels, scores, iou_thresh=0.5)
        self.assertListEqual(good_inds, [0, 2, 3])

        scores = np.array([0.9, 0.8, 0.9, 0.9])
        good_inds = compute_nms(boxes, labels, scores, iou_thresh=0.9)
        self.assertListEqual(good_inds, [0, 1, 2, 3])

if __name__ == '__main__':
    unittest.main()