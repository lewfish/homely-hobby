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
                plt.imshow(a)

            plt.suptitle('label probs for stride={}'.format(stride))
            plt.savefig(
                join(preds_dir, '{}-{}-label-arr.png'.format(img_id, stride)),
                figsize=(3, 3))

            # Plot top, left, bottom, right arrays.
            reg_arr = level_out['reg_arr'][0].detach().cpu().numpy()
            plt.gca()
            directions = ['top', 'left', 'bottom', 'right']
            for di, d in enumerate(directions):
                plt.subplot(1, 4, di + 1)
                plt.title(d)
                a = reg_arr[di]
                plt.imshow(a)

            plt.suptitle('distance to box edge for stride={}'.format(stride))
            plt.savefig(
                join(preds_dir, '{}-{}-reg-arr.png'.format(img_id, stride)),
                figsize=(10, 10))

    zipdir(preds_dir, zip_path)
    shutil.rmtree(preds_dir)