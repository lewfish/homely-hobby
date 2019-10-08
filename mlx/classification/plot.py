from os.path import join
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import torch

from mlx.filesystem.utils import make_dir

def plot_xyz(ax, x, y, label_names, z=None):
    x = x.permute(1, 2, 0)
    if x.shape[2] == 1:
        x = torch.cat([x for _ in range(3)], dim=2)
    ax.imshow(x)
    title = label_names[y]
    if z is not None:
        title += '\n' + label_names[z]
    ax.set_title(title)
    ax.axis('off')
