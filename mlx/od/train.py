from pathlib import Path
import json
import uuid
from os.path import join

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

from mlx.od.utils import ObjectDetectionGrid, BoxList
from mlx.od.model import ObjectDetectionModel
from mlx.batch_utils import submit_job
from mlx.filesystem.utils import make_dir, sync_to_dir

output_dir = '/opt/data/pascal2007/output/'
output_uri = 's3://raster-vision-lf-dev/pascal2007/output/'

def run_on_batch(test):
    job_name = 'mlx_train_obj_det-' + str(uuid.uuid4())
    job_def = 'lewfishPyTorchCustomGpuJobDefinition'
    job_queue = 'lewfishRasterVisionGpuJobQueue'
    cmd_list = ['python', '-m', 'mlx.od.train', '--s3-data']
    if test:
        cmd_list = [
            'python', '-m', 'ptvsd', '--host', '0.0.0.0', '--port', '6006',
            '--wait', '-m', 'mlx.od.train', '--s3-data', '--test']
    submit_job(job_name, job_def, job_queue, cmd_list)
    exit()

@click.command()
@click.option('--test', is_flag=True, help='Run small test experiment')
@click.option('--s3-data', is_flag=True, help='Use data and store results on S3')
@click.option('--batch', is_flag=True, help='Submit Batch job for full experiment')
def main(test, s3_data, batch):
    if batch:
        run_on_batch(test)

    # Setup options
    bs = 16
    size = 256
    num_workers = 4
    num_epochs = 10
    lr = 1e-4
    # for size 256
    grid_sz = 8 - 2

    if test:
        bs = 4
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
        num_classes = len(classes)

    # Subtract 2 because there's no padding on final convolution
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
            boxes, labels, scores = grid.decode(pred.unsqueeze(0))
            bl = BoxList(boxes[0], labels[0], scores[0])
            bl = bl.score_filter(score_thresh=score_thresh).nms(iou_thresh=iou_thresh)
            return [bl.boxes, bl.labels]

    class MyObjectItemList(ObjectItemList):
        _label_cls = MyObjectCategoryList

    def get_data(bs, size):
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
    data.show_batch()
    plt.savefig(join(output_dir, 'data.png'))

    # Setup model
    model = ObjectDetectionModel(grid)

    def loss(out, gt_boxes, gt_classes):
        gt = model.grid.encode(gt_boxes, gt_classes)
        box_loss, class_loss = model.grid.compute_losses(out, gt)
        return box_loss + class_loss

    learn = Learner(data, model, loss_func=loss, path=output_dir)
    callbacks = [
        CSVLogger(learn, filename='log')
    ]
    # model.freeze_body()
    learn.fit(num_epochs, lr, callbacks=callbacks)

    learn.show_results()
    plt.savefig(join(output_dir, 'preds.png'))

    if s3_data:
        sync_to_dir(output_dir, output_uri)

if __name__ == '__main__':
    main()