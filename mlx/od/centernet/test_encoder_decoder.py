import unittest

import torch

from mlx.od.centernet.encoder import encode
from mlx.od.centernet.boxlist import BoxList
from mlx.od.centernet.decoder import decode
from mlx.od.centernet.utils import get_positions
from mlx.od.centernet.plot import plot_encoded

class TestEncoderDecoder(unittest.TestCase):
    def test_encode_decode(self):
        height, width = 256, 256
        stride = 4
        num_labels = 2
        boxes = torch.tensor([
            [0., 0., 128., 128.],
            [128., 128., 192., 192.],
            [64., 64., 128., 128.],
        ])
        # shift by two since only numbers that fall in the middle of the stride
        # can be represented exactly
        boxes += 2
        labels = torch.tensor([0, 0, 1])
        bl = BoxList(boxes, labels=labels)
        boxlists = [bl]
        positions = get_positions(height, width, stride, boxes.device)

        keypoint, reg = encode(boxlists, positions, stride, num_labels)
        path = '/opt/data/pascal2007/encoded.png'
        plot_encoded(boxlists[0], stride, keypoint[0], reg[0], path)

        decoded_boxlists = decode(
            keypoint, reg, positions, stride, prob_thresh=0.05)
        path = '/opt/data/pascal2007/decoded.png'
        plot_encoded(decoded_boxlists[0], stride, keypoint[0], reg[0], path)

        del decoded_boxlists[0].extras['scores']
        self.assertTrue(boxlists[0].equal(decoded_boxlists[0]))

if __name__ == '__main__':
    unittest.main()