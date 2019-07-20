import unittest

import torch

from mlx.od.fcos.loss import focal_loss

class TestFocalLoss(unittest.TestCase):
    def test_focal_loss(self):
        gamma = 2
        alpha = 0.25
        p = torch.tensor([0.2, 0.7])
        # do inverse of sigmoid
        output = torch.log(p / (1-p))
        target = torch.tensor([0.0, 1.0])

        pt = torch.tensor([0.8, 0.7])
        bce = -torch.log(pt)
        alphat = torch.tensor([0.75, 0.25])
        weights = torch.pow((1 - pt), gamma) * alphat
        loss_arr = bce * weights
        exp_loss = loss_arr.reshape(-1).mean()

        loss = focal_loss(output, target, gamma, alpha)
        self.assertEqual(exp_loss, loss)

if __name__ == '__main__':
    unittest.main()