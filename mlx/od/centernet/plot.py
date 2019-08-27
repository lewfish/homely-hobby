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
    def make_debug_plots(self, dataloader, model, classes, output_dir,
                         max_plots=25, score_thresh=0.4):
        preds_dir = join(output_dir, 'preds')
        zip_path = join(output_dir, 'preds.zip')
        make_dir(preds_dir, force_empty=True)

        model.eval()
        for batch_x, batch_y in dataloader:
            with torch.no_grad():
                device = list(model.parameters())[0].device
                batch_x = batch_x.to(device=device)
                batch_sz = batch_x.shape[0]
                batch_boxlist, batch_head_out = model(batch_x, get_head_out=True)
            
            for img_ind in range(batch_sz):
                x = batch_x[img_ind].cpu()
                y = batch_y[img_ind].cpu()
                boxlist = batch_boxlist[img_ind].score_filter(score_thresh).cpu()
                head_out = (batch_head_out[0][img_ind], batch_head_out[1][img_ind])

                # Plot image, ground truth, and predictions
                fig = self.plot_image_preds(x, y, boxlist, classes)
                plt.savefig(join(preds_dir, '{}-images.png'.format(img_ind)), dpi=200,
                            bbox_inches='tight')
                plt.close(fig)

                # Plot raw output of network.
                keypoint, reg = head_out
                keypoint, reg = keypoint.cpu(), reg.cpu()
                stride = model.stride

                fig = plot_encoded(boxlist, stride, keypoint, reg, classes=classes)
                plt.savefig(
                    join(preds_dir, '{}-output.png'.format(img_ind)), dpi=100,
                    bbox_inches='tight')
                plt.close(fig)

                # Plot encoding of ground truth targets.
                h, w = x.shape[1:]
                positions = get_positions(h, w, stride, y.boxes.device)
                keypoint, reg = encode([y], positions, stride, len(classes))
                fig = plot_encoded(
                    y, stride, keypoint[0], reg[0], classes=classes)
                plt.savefig(
                    join(preds_dir, '{}-targets.png'.format(img_ind)), dpi=100,
                    bbox_inches='tight')
                plt.close(fig)
            break

        zipdir(preds_dir, zip_path)
        shutil.rmtree(preds_dir)
