import math
from os.path import join
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import torch

from mlx.od.plot import Plotter
from mlx.filesystem.utils import make_dir, zipdir
from mlx.od.centernet.encoder import encode
from mlx.od.centernet.utils import get_positions
from mlx.od.boxlist import to_box_pixel, BoxList

def plot_encoded(boxlist, stride, keypoint, reg, classes=None):
    fig = plt.figure(constrained_layout=True, figsize=(12, 12))
    num_labels = keypoint.shape[0]
    num_plots = num_labels + 2
    ncols = nrows = math.ceil(math.sqrt(num_plots))
    grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    for l in range(num_labels):
        ax = fig.add_subplot(grid[l])
        class_name = classes[l] if classes is not None else str(l)
        ax.set_title('keypoint[{}]'.format(class_name))
        ax.imshow(keypoint[l], vmin=0., vmax=1.)

        for box, label in zip(boxlist.boxes, boxlist.get_field('labels')):
            if label == l:
                box = box / stride
                rect = patches.Rectangle(
                    (box[1], box[0]), box[3]-box[1], box[2]-box[0],
                    linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for r in range(2):
        ax = fig.add_subplot(grid[num_labels + r])
        ax.set_title('reg[{}]'.format(r))
        ax.imshow(reg[r], vmin=0., vmax=reg[r].shape[0])

        for box, label in zip(boxlist.boxes, boxlist.get_field('labels')):
            box = box / stride
            rect = patches.Rectangle(
                (box[1], box[0]), box[3]-box[1], box[2]-box[0],
                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    return fig

class CenterNetPlotter(Plotter):
    def make_debug_plots(self, dataset, model, classes, output_dir,
                         max_plots=25, score_thresh=0.4):
        preds_dir = join(output_dir, 'preds')
        zip_path = join(output_dir, 'preds.zip')
        make_dir(preds_dir, force_empty=True)

        model.eval()
        for img_id, (x, y) in enumerate(dataset):
            if img_id == max_plots:
                break

            # Get predictions
            boxlist, head_out = self.get_pred(x, model, score_thresh)

            # Plot image, ground truth, and predictions
            fig = self.plot_image_preds(x, y, boxlist, classes)
            plt.savefig(join(preds_dir, '{}-images.png'.format(img_id)), dpi=200,
                        bbox_inches='tight')
            plt.close(fig)

            keypoint, reg = head_out
            keypoint, reg = torch.sigmoid(keypoint[0]), reg[0]
            stride = model.stride

            # detach, cpu
            fig = plot_encoded(boxlist, stride, keypoint, reg, classes=classes)
            plt.savefig(
                join(preds_dir, '{}-output.png'.format(img_id)), dpi=100,
                bbox_inches='tight')
            plt.close(fig)

            # Get encoding of ground truth targets.
            h, w = x.size
            boxes, labels = y.data
            boxes = to_box_pixel(boxes, h, w)
            boxlist = BoxList(boxes, labels=labels)
            positions = get_positions(h, w, stride, boxes.device)
            keypoint, reg = encode([boxlist], positions, stride, len(classes))
            fig = plot_encoded(
                boxlist, stride, keypoint[0], reg[0], classes=classes)
            plt.savefig(
                join(preds_dir, '{}-targets.png'.format(img_id)), dpi=100,
                bbox_inches='tight')
            plt.close(fig)

        zipdir(preds_dir, zip_path)
        shutil.rmtree(preds_dir)