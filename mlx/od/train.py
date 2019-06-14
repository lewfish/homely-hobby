from pathlib import Path
import json
import uuid
from os.path import join
import shutil

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from fastai.vision import (
    get_annotations, ObjectItemList, ObjectCategoryList, get_transforms,
    bb_pad_collate, URLs, untar_data)
from fastai.basic_train import Learner
from fastai.callbacks import CSVLogger

from mlx.od.utils import ObjectDetectionGrid, BoxList, F1
from mlx.od.model import ObjectDetectionModel
from mlx.batch_utils import submit_job
from mlx.filesystem.utils import make_dir, sync_to_dir, zipdir

output_dir = '/opt/data/pascal2007/output/'
output_uri = 's3://raster-vision-lf-dev/pascal2007/output/'

def run_on_batch(test, debug):
    job_name = 'mlx_train_obj_det-' + str(uuid.uuid4())
    job_def = 'lewfishPyTorchCustomGpuJobDefinition'
    job_queue = 'lewfishRasterVisionGpuJobQueue'
    cmd_list = ['python', '-m', 'mlx.od.train', '--s3-data']
    if debug:
        cmd_list = [
            'python', '-m', 'ptvsd', '--host', '0.0.0.0', '--port', '6006',
            '--wait', '-m', 'mlx.od.train', '--s3-data']
    if test:
        cmd_list.append('--test')
    submit_job(job_name, job_def, job_queue, cmd_list)
    exit()

def plot_data(data, output_dir, max_per_split=50):
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
                        figsize=(3, 3))
            plt.close()
        zipdir(debug_chips_dir, zip_path)
        shutil.rmtree(debug_chips_dir)

    _plot_data('train')
    _plot_data('val')

def plot_preds(data, learn, output_dir, max_plots=50):
    preds_dir = join(output_dir, 'preds')
    zip_path = join(output_dir, 'preds.zip')
    make_dir(preds_dir, force_empty=True)

    ds = data.valid_ds
    for i, (x, y) in enumerate(ds):
        if i == max_plots:
            break
        z = learn.predict(x)
        x.show(y=z[0])
        plt.savefig(join(preds_dir, '{}.png'.format(i)), figsize=(3, 3))
        plt.close()

    zipdir(preds_dir, zip_path)
    shutil.rmtree(preds_dir)

@click.command()
@click.option('--test', is_flag=True, help='Run small test experiment')
@click.option('--s3-data', is_flag=True, help='Use data and store results on S3')
@click.option('--batch', is_flag=True, help='Submit Batch job for full experiment')
@click.option('--debug', is_flag=True, help='Run via debugger')
def main(test, s3_data, batch, debug):
    if batch:
        run_on_batch(test, debug)

    # Setup options
    bs = 16
    size = 256
    num_workers = 4
    num_epochs = 100
    lr = 1e-4
    # for size 256
    # Subtract 2 because there's no padding on final convolution
    grid_sz = 8 - 2

    if test:
        bs = 8
        size = 128
        num_debug_images = 32
        num_workers = 0
        num_epochs = 1
        # for size 128
        grid_sz = 4 - 2

    # Setup data
    make_dir(output_dir)

    data_dir = untar_data(URLs.PASCAL_2007, dest='/opt/data/pascal2007/data')
    img_path = data_dir/'train/'
    trn_path = data_dir/'train.json'
    trn_images, trn_lbl_bbox = get_annotations(trn_path)
    val_path = data_dir/'valid.json'
    val_images, val_lbl_bbox = get_annotations(val_path)

    images, lbl_bbox = trn_images+val_images, trn_lbl_bbox+val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    with open(trn_path) as f:
        d = json.load(f)
        classes = sorted(d['categories'], key=lambda x: x['id'])
        classes = [x['name'] for x in classes]
        classes = ['background'] + classes
        num_classes = len(classes)

    anc_sizes = torch.tensor([
        [1, 1],
        [2, 2],
        [3, 3],
        [3, 1],
        [1, 3]], dtype=torch.float32)
    grid = ObjectDetectionGrid(grid_sz, anc_sizes, num_classes)
    score_thresh = 0.1
    iou_thresh = 0.8

    class MyObjectCategoryList(ObjectCategoryList):
        def analyze_pred(self, pred):
            boxes, labels, _ = grid.get_preds(
                pred.unsqueeze(0), score_thresh=score_thresh,
                iou_thresh=iou_thresh)
            return (boxes[0], labels[0])

    class MyObjectItemList(ObjectItemList):
        _label_cls = MyObjectCategoryList

    def get_data(bs, size, ):
        src = MyObjectItemList.from_folder(img_path)
        if test:
            src = src[0:num_debug_images]
        src = src.split_by_files(val_images)
        src = src.label_from_func(get_y_func, classes=classes)
        src = src.transform(get_transforms(), size=size, tfm_y=True)
        return src.databunch(path=data_dir, bs=bs, collate_fn=bb_pad_collate,
                             num_workers=num_workers)

    data = get_data(bs, size)
    print(data)
    plot_data(data, output_dir)

    # Setup model
    model = ObjectDetectionModel(grid)

    def loss(out, gt_boxes, gt_classes):
        gt = model.grid.encode(gt_boxes, gt_classes)
        box_loss, class_loss = model.grid.compute_losses(out, gt)
        return box_loss + class_loss

    metrics = [F1(grid, score_thresh=score_thresh, iou_thresh=iou_thresh)]
    learn = Learner(data, model, metrics=metrics, loss_func=loss,
                    path=output_dir)
    callbacks = [
        CSVLogger(learn, filename='log')
    ]
    # model.freeze_body()
    learn.fit_one_cycle(num_epochs, lr, callbacks=callbacks)

    plot_preds(data, learn, output_dir)

    if s3_data:
        sync_to_dir(output_dir, output_uri)

if __name__ == '__main__':
    main()