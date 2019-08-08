from os.path import join
import shutil
import math

from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastai.vision import ImageBBox, normalize, imagenet_stats
import torch
import matplotlib.gridspec as gridspec

from mlx.filesystem.utils import make_dir, zipdir
from mlx.od.fcos.encoder import encode_targets

def plot_data(data, output_dir, max_per_split=25):
    def _plot_data(split):
        debug_chips_dir = join(output_dir, '{}-debug-chips'.format(split))
        zip_path = join(output_dir, '{}-debug-chips.zip'.format(split))
        make_dir(debug_chips_dir, force_empty=True)

        ds = data.train_ds if split == 'train' else data.valid_ds
        for i, (x, y) in enumerate(ds):
            if i == max_per_split:
                break
            x.show(y=y)
            plt.savefig(join(debug_chips_dir, '{}.png'.format(i)),
                        figsize=(6, 6))
            plt.close()
        zipdir(debug_chips_dir, zip_path)
        shutil.rmtree(debug_chips_dir)

    _plot_data('train')
    _plot_data('val')

def get_pred(img, model, score_thresh):
    device = list(model.parameters())[0].device
    x = img.data.unsqueeze(0).to(device=device)
    mean, std = imagenet_stats
    x = normalize(x, torch.tensor(mean, device=device), torch.tensor(std, device=device))
    out, head_out = model(x, get_head_out=True)
    out = out[0]
    # Filter boxes whose score is high enough
    boxes = out['boxes'].cpu()
    labels = out['labels'].cpu()
    scores = out['scores'].cpu()
    keep_inds = scores > score_thresh
    boxes, labels, scores = boxes[keep_inds, :], labels[keep_inds], scores[keep_inds]
    out = {'boxes': boxes, 'labels': labels, 'scores': scores}
    return out, head_out

def plot_image_preds(x, y, out, classes):
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
    if out['boxes'].shape[0] > 0:
        z = ImageBBox.create(h, w, out['boxes'], out['labels'],
                             classes=classes, scale=True)
        x.show(y=z, ax=ax)
    else:
        x.show(ax=ax)

    return fig

def plot_label_arr(label_probs, classes, stride):
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

def plot_preds(dataset, model, classes, output_dir, max_plots=25, score_thresh=0.25):
    preds_dir = join(output_dir, 'preds')
    zip_path = join(output_dir, 'preds.zip')
    make_dir(preds_dir, force_empty=True)

    model.eval()
    for img_id, (x, y) in enumerate(dataset):
        if img_id == max_plots:
            break

        # Get predictions
        out, head_out = get_pred(x, model, score_thresh)

        # Plot image, ground truth, and predictions
        fig = plot_image_preds(x, y, out, classes)
        plt.savefig(join(preds_dir, '{}.png'.format(img_id)), dpi=200,
                    bbox_inches='tight')
        plt.close(fig)

        # Plot raw output of network at each level.
        for stride, level_out in head_out.items():
            # Plot label_arr
            label_arr = level_out['label_arr'][0].detach().cpu()
            label_probs = torch.sigmoid(label_arr)
            fig = plot_label_arr(label_probs, classes, stride)
            plt.savefig(
                join(preds_dir, '{}-{}-label-arr.png'.format(img_id, stride)),
                dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Plot top, left, bottom, right from reg_arr and center_arr.
            reg_arr = level_out['reg_arr'][0].detach().cpu()
            center_arr = level_out['center_arr'][0][0].detach().cpu()
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
        targets = encode_targets(boxes, labels, model.pyramid_shape, model.num_labels)

        # Plot encoding of ground truth at each level.
        for stride, level_targets in targets.items():
            # Plot label_arr
            label_probs = level_targets['label_arr'].detach().cpu()
            fig = plot_label_arr(label_probs, classes, stride)
            plt.savefig(
                join(preds_dir, '{}-{}-label-arr-gt.png'.format(img_id, stride)),
                dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Plot top, left, bottom, right from reg_arr and center_arr.
            reg_arr = level_targets['reg_arr'].detach().cpu()
            center_arr = level_targets['center_arr'][0].detach().cpu()
            center_probs = center_arr
            fig = plot_reg_center_arr(reg_arr, center_probs, stride)
            plt.savefig(
                join(preds_dir, '{}-{}-reg-center-arr-gt.png'.format(img_id, stride)),
                dpi=100, bbox_inches='tight')
            plt.close(fig)

    zipdir(preds_dir, zip_path)
    shutil.rmtree(preds_dir)