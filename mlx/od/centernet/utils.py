import math

import torch

def get_positions(height, width, stride, device):
    ypos = torch.arange(0, height, stride, device=device, dtype=torch.float)
    xpos = torch.arange(0, width, stride, device=device, dtype=torch.float)
    ypos, xpos = torch.meshgrid(ypos, xpos)
    pos = torch.cat((ypos[None, :, :], xpos[None, :, :]))
    return pos + stride / 2

def prob2logit(prob):
    return math.log(prob / (1 - prob))