from abc import abstractmethod
from os.path import join
import shutil
import math

import torch
from fastai.vision import (
    denormalize, imagenet_stats, ImageBBox, normalize)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from mlx.od.model import fcos, centernet
from mlx.od.boxlist import to_box_pixel
from mlx.filesystem.utils import make_dir, zipdir

class Plotter():
    @abstractmethod
    def make_debug_plots(self, dataset, model, classes, output_dir, max_plots=25, score_thresh=0.25):
        pass

    def plot_data(self, data, output_dir, max_per_split=25):
        """Plot images and ground truth coming out the dataloader."""
        def _plot_data(split):
            debug_chips_dir = join(output_dir, '{}-debug-chips'.format(split))
            zip_path = join(output_dir, '{}-debug-chips.zip'.format(split))
            make_dir(debug_chips_dir, force_empty=True)

            dl = data.train_dl if split == 'train' else data.valid_dl

            img_ind = 0
            for batch in dl:
                N = batch[0].shape[0]
                for i in range(N):
                    if img_ind == max_per_split:
                        break

                    img = batch[0][i].cpu()
                    boxes = batch[1][0][i].cpu()
                    labels = batch[1][1][i].cpu()

                    boxes = to_box_pixel(boxes, img.shape[1], img.shape[2])
                    mean, std = imagenet_stats
                    img = denormalize(img, torch.tensor(mean), torch.tensor(std))

                    fig,ax = plt.subplots(1)
                    ax.imshow(img.permute(1, 2, 0))
                    for box, label in zip(boxes, labels):
                        if label != 0.0:
                            rect = patches.Rectangle(
                                (box[1], box[0]), box[3]-box[1], box[2]-box[0],
                                linewidth=1, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                            label_name = data.classes[label.item()]
                            txt = ax.text(box[1] + 2, box[0] + 7, label_name, fontsize=10)
                            import matplotlib.patheffects as PathEffects
                            txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

                    plt.axis('off')
                    plt.savefig(join(debug_chips_dir, '{}.png'.format(img_ind)),
                                figsize=(6, 6), dpi=200)
                    plt.close()

                    img_ind += 1
            zipdir(debug_chips_dir, zip_path)
            shutil.rmtree(debug_chips_dir)

        if len(data.train_ds):
            _plot_data('train')

        try:
            data.valid_ds[0]
            _plot_data('val')
        except:
            pass

    def get_pred(self, img, model, score_thresh):
        device = list(model.parameters())[0].device
        x = img.data.unsqueeze(0).to(device=device)
        mean, std = imagenet_stats
        x = normalize(x, torch.tensor(mean, device=device), torch.tensor(std, device=device))
        out, head_out = model(x, get_head_out=True)
        boxlist = out[0]
        boxlist = boxlist.score_filter(score_thresh).cpu()
        return boxlist, head_out

    def plot_image_preds(self, x, y, out_boxlist, classes):
        "Plot original, ground truth, and predictions on single figure."""
        h, w = x.size
        fig = plt.figure(constrained_layout=True, figsize=(6, 3))
        grid = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

        ax = fig.add_subplot(grid[0])
        ax.set_title('image')
        x.show(ax=ax)

        ax = fig.add_subplot(grid[1])
        ax.set_title('ground truth')
        if len(y.labels) > 0:
            x.show(y=y, ax=ax)
        else:
            x.show(ax=ax)

        ax = fig.add_subplot(grid[2])
        ax.set_title('predictions')
        boxes, labels, scores = (
            out_boxlist.boxes, out_boxlist.get_field('labels'),
            out_boxlist.get_field('scores'))
        if boxes.shape[0] > 0:
            z = ImageBBox.create(h, w, boxes, labels,
                                classes=classes, scale=True)
            x.show(y=z, ax=ax)
        else:
            x.show(ax=ax)

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
