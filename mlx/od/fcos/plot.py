from os.path import join
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec

from mlx.filesystem.utils import make_dir, zipdir
from mlx.od.fcos.encoder import encode_single_targets
from mlx.od.plot import Plotter

def plot_reg_center_arr(reg_arr, center_probs, stride):
    fig = plt.figure(constrained_layout=True, figsize=(12, 3.5))
    grid = gridspec.GridSpec(ncols=5, nrows=1, figure=fig)
    directions = ['top', 'left', 'bottom', 'right']
    max_reg_val = reg_arr.reshape(-1).max()
    for di, d in enumerate(directions):
        ax = fig.add_subplot(grid[di])
        ax.imshow(reg_arr[di], vmin=0, vmax=max_reg_val)
        ax.set_title(d)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    ax = fig.add_subplot(grid[4])
    ax.set_title('centerness')
    ax.imshow(center_probs, vmin=0, vmax=1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.suptitle(
        'reg_arr and center_arr for stride={}'.format(stride), size=20)
    return fig

class FCOSPlotter(Plotter):
    def make_debug_plots(self, dataset, model, classes, output_dir, max_plots=25, score_thresh=0.25):
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
            plt.savefig(join(preds_dir, '{}.png'.format(img_id)), dpi=200,
                        bbox_inches='tight')
            plt.close(fig)

            # Plot raw output of network at each level.
            for level, level_out in enumerate(head_out):
                stride = model.fpn.strides[level]
                reg_arr, label_arr, center_arr = level_out

                # Plot label_arr
                label_arr = label_arr[0].detach().cpu()
                label_probs = torch.sigmoid(label_arr)
                fig = self.plot_label_arr(label_probs, classes, stride)
                plt.savefig(
                    join(preds_dir, '{}-{}-label-arr.png'.format(img_id, stride)),
                    dpi=100, bbox_inches='tight')
                plt.close(fig)

                # Plot top, left, bottom, right from reg_arr and center_arr.
                reg_arr = reg_arr[0].detach().cpu()
                center_arr = center_arr[0][0].detach().cpu()
                center_probs = torch.sigmoid(center_arr)
                fig = plot_reg_center_arr(reg_arr, center_probs, stride)
                plt.savefig(
                    join(preds_dir, '{}-{}-reg-center-arr.png'.format(img_id, stride)),
                    dpi=100, bbox_inches='tight')
                plt.close(fig)

            # Get encoding of ground truth targets.
            h, w = x.size
            boxes, labels = y.data
            labels = torch.tensor(labels)
            boxes = (boxes + 1.0) / 2.0
            boxes *= torch.tensor([[h, w, h, w]], device=boxes.device, dtype=torch.float)
            targets = encode_single_targets(boxes, labels, model.pyramid_shape, model.num_labels)

            # Plot encoding of ground truth at each level.
            for level, level_targets in enumerate(targets):
                stride = model.fpn.strides[level]
                reg_arr, label_arr, center_arr = level_targets

                # Plot label_arr
                label_probs = label_arr.detach().cpu()
                fig = self.plot_label_arr(label_probs, classes, stride)
                plt.savefig(
                    join(preds_dir, '{}-{}-label-arr-gt.png'.format(img_id, stride)),
                    dpi=100, bbox_inches='tight')
                plt.close(fig)

                # Plot top, left, bottom, right from reg_arr and center_arr.
                reg_arr = reg_arr.detach().cpu()
                center_arr = center_arr[0].detach().cpu()
                center_probs = center_arr
                fig = plot_reg_center_arr(reg_arr, center_probs, stride)
                plt.savefig(
                    join(preds_dir, '{}-{}-reg-center-arr-gt.png'.format(img_id, stride)),
                    dpi=100, bbox_inches='tight')
                plt.close(fig)

        zipdir(preds_dir, zip_path)
        shutil.rmtree(preds_dir)