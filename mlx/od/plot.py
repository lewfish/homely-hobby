from abc import abstractmethod
from os.path import join
import math

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from mlx.od.model import fcos, centernet
from mlx.filesystem.utils import make_dir

def plot_xy(ax, x, y=None, label_names=None):
    ax.imshow(x.permute(1, 2, 0))

    if y is not None:
        scores = y.get_field('scores')
        for box_ind, (box, label) in enumerate(zip(y.boxes, y.get_field('labels'))):
            rect = patches.Rectangle(
                (box[1], box[0]), box[3]-box[1], box[2]-box[0],
                linewidth=1, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)

            label_name = label_names[label]
            if scores is not None:
                score = scores[box_ind]
                label_name += ' {.2f}'.format(score)
            label_width = len(label_name) * 7
            rect = patches.Rectangle(
                (box[1], box[0] - 11), label_width, 11, color='cyan')
            ax.add_patch(rect)
            ax.text(box[1] + 2, box[0] - 2, label_name, fontsize=7)
    ax.axis('off')

def plot_dataset(dataset, label_names, output_dir, num_imgs=5):
    for img_ind, (x, y) in enumerate(dataset):
        if img_ind == num_imgs:
            break

        fig, ax = plt.subplots(1)
        plot_xy(ax, x, y, dataset.label_names)
        make_dir(output_dir)
        plt.savefig(join(output_dir, '{}.png'.format(img_ind)),
                    figsize=(6, 6))
        plt.close()

def plot_dataloader(dataloader, label_names, output_path):
    x, y = next(iter(dataloader))
    batch_sz = x.shape[0]

    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    ncols = nrows = math.ceil(math.sqrt(batch_sz))
    grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    for i in range(batch_sz):
        ax = fig.add_subplot(grid[i])
        plot_xy(ax, x[i], y[i], label_names)

    make_dir(output_path, use_dirname=True)
    plt.savefig(output_path)
    plt.close()

class Plotter():
    @abstractmethod
    def make_debug_plots(self, dataset, model, classes, output_dir, max_plots=25, score_thresh=0.25):
        pass

    def plot_dataloaders(self, databunch, output_dir):
        databunch.valid_ds[0]
        if databunch.train_dl:
            plot_dataloader(databunch.train_dl, databunch.label_names, join(output_dir, 'train_dl.png'))
        if databunch.valid_dl:
            plot_dataloader(databunch.valid_dl, databunch.label_names, join(output_dir, 'valid_dl.png'))

    def get_pred(self, x, model, score_thresh):
        with torch.no_grad():
            device = list(model.parameters())[0].device
            x = x.unsqueeze(0).to(device=device)
            out, head_out = model(x, get_head_out=True)
            boxlist = out[0]
            boxlist = boxlist.score_filter(score_thresh).cpu()

            return boxlist, head_out

    def plot_image_preds(self, x, y, z, label_names):
        """Plot original, ground truth, and predictions on single figure."""
        fig = plt.figure(constrained_layout=True, figsize=(6, 3))
        grid = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

        ax = fig.add_subplot(grid[0])
        ax.set_title('image')
        plot_xy(ax, x)

        ax = fig.add_subplot(grid[1])
        ax.set_title('ground truth')
        plot_xy(ax, x, y, label_names)

        ax = fig.add_subplot(grid[2])
        ax.set_title('predictions')
        plot_xy(ax, x, z, label_names)

        return fig

    def plot_label_arr(self, label_probs, classes, stride):
        fig = plt.figure(constrained_layout=True, figsize=(12, 12))
        num_labels = len(classes)
        ncols = nrows = math.ceil(math.sqrt(num_labels))
        grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        for l in range(num_labels):
            ax = fig.add_subplot(grid[l])
            ax.set_title(classes[l])
            ax.imshow(label_probs[l], vmin=0., vmax=1.)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.suptitle('label_arr for stride={}'.format(stride), size=20)
        return fig

def build_plotter(cfg):
    if cfg.model.type == fcos:
        from mlx.od.fcos.plot import FCOSPlotter
        return FCOSPlotter()
    elif cfg.model.type == centernet:
        from mlx.od.centernet.plot import CenterNetPlotter
        return CenterNetPlotter()
