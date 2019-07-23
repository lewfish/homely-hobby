from os.path import join
import shutil

from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastai.vision import ImageBBox
import torch

from mlx.filesystem.utils import make_dir, zipdir


def plot_preds(data, model, classes, output_dir, max_plots=50, score_thresh=0.4):
    preds_dir = join(output_dir, 'preds')
    zip_path = join(output_dir, 'preds.zip')
    make_dir(preds_dir, force_empty=True)

    device = list(model.parameters())[0].device
    model.eval()
    ds = data.valid_ds
    for img_id, (x, y) in enumerate(ds):
        if img_id == max_plots:
            break
        x_data = x.data.unsqueeze(0).to(device=device)
        z, head_out = model(x_data, get_head_out=True)

        # Plot predictions
        h, w = x.shape[1:]
        z = z[0]
        # Filter boxes whose score is high enough
        boxes = z['boxes'].cpu()
        labels = z['labels'].cpu()
        scores = z['scores'].cpu()
        keep_inds = scores > score_thresh
        boxes, labels, scores = boxes[keep_inds, :], labels[keep_inds], scores[keep_inds]

        if z['boxes'].shape[0] > 0:
            z = ImageBBox.create(h, w, boxes, labels, classes=classes,
                                 scale=True)
            x.show(y=z)
        else:
            x.show()
        plt.savefig(join(preds_dir, '{}.png'.format(img_id)), figsize=(3, 3))
        plt.close()

        # Plot original image
        img = Image.fromarray(
            (x.data.cpu().numpy() * 255).transpose(1, 2, 0).astype(np.uint8))
        img.save(join(preds_dir, '{}-orig.png'.format(img_id)))

        # Plot raw output of network
        for stride, level_out in head_out.items():
            # Plot probs for each label
            label_arr = level_out['label_arr'][0].detach().cpu()
            label_probs = torch.sigmoid(label_arr).numpy()
            plt.gca()
            num_labels = len(classes)
            for l in range(1, num_labels):
                plt.subplot(5, 5, l)
                plt.title(classes[l])
                a = label_probs[l]
                plt.imshow(a, vmin=0., vmax=1.)
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.gca().axes.get_yaxis().set_visible(False)

            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            plt.suptitle('label probs for stride={}'.format(stride))
            plt.savefig(
                join(preds_dir, '{}-{}-label-arr.png'.format(img_id, stride)))

            # Plot top, left, bottom, right from reg_arr and center_arr.
            reg_arr = level_out['reg_arr'][0].detach().cpu().numpy()
            center_arr = level_out['center_arr'][0][0].detach().cpu()
            center_probs = torch.sigmoid(center_arr).numpy()

            plt.gca()
            directions = ['top', 'left', 'bottom', 'right']
            max_reg_val = np.amax(reg_arr)
            for di, d in enumerate(directions):
                plt.subplot(1, 5, di + 1)
                plt.title(d)
                a = reg_arr[di]
                plt.imshow(a, vmin=0, vmax=max_reg_val)
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.gca().axes.get_yaxis().set_visible(False)

            plt.subplot(1, 5, 5)
            plt.title('centerness')
            plt.imshow(center_probs, vmin=0, vmax=1)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            plt.suptitle('reg_arr and center_arr for stride={}'.format(stride))
            plt.savefig(
                join(preds_dir, '{}-{}-reg-center-arr.png'.format(img_id, stride)),
                figsize=(10, 10))

    zipdir(preds_dir, zip_path)
    shutil.rmtree(preds_dir)