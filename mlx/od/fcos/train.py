import json
import uuid
from os.path import join, isdir, dirname
import shutil
import tempfile

import numpy as np
import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fastai
from fastai.vision import (
    get_annotations, ObjectItemList, get_transforms,
    bb_pad_collate, URLs, untar_data)
from fastai.basic_train import Learner
from fastai.callbacks import CSVLogger
import torch
from torch import nn, Tensor
from fastai.callback import CallbackHandler
from fastai.core import ifnone
from fastai.torch_core import OptLossFunc, OptOptimizer, Optional, Tuple, Union

from mlx.od.fcos.model import FCOS
from mlx.od.fcos.metrics import CocoMetric
from mlx.od.fcos.plot import plot_preds
from mlx.batch_utils import submit_job
from mlx.filesystem.utils import make_dir, sync_to_dir, zipdir, unzip, download_if_needed

def run_on_batch(dataset_name, test, debug):
    job_name = 'mlx_train_fcos-' + str(uuid.uuid4())
    job_def = 'lewfishPyTorchCustomGpuJobDefinition'
    job_queue = 'lewfishRasterVisionGpuJobQueue'
    cmd_list = ['python', '-m', 'mlx.od.fcos.train', dataset_name, '--s3-data']
    if debug:
        cmd_list = [
            'python', '-m', 'ptvsd', '--host', '0.0.0.0', '--port', '6006',
            '--wait', '-m', 'mlx.od.fcos.train', dataset_name, '--s3-data']
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
                        figsize=(6, 6))
            plt.close()
        zipdir(debug_chips_dir, zip_path)
        shutil.rmtree(debug_chips_dir)

    _plot_data('train')
    _plot_data('val')

# Modified from fastai to handle model which only computes loss when targets
# are passed in, and only computes output otherwise. This should run faster
# thanks to not having to run the decoder and NMS during training, and not
# computing the loss for the validation which is not a great metric anyway.
def loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None,
               opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
    device = xb.device
    # Translate from fastai box format to torchvision.
    batch_sz = len(xb)
    images = xb
    targets = []
    for i in range(batch_sz):
        boxes = yb[0][i]
        labels = yb[1][i]
        # convert from (ymin, xmin, ymax, xmax) in range [-1,1] to
        # (xmin, ymin, xmax, ymax) in range [0, h) or [0, w)
        h, w = images[i].shape[1:]
        boxes = ((boxes + 1.0) / 2.0) * torch.tensor([[h, w, h, w]]).to(
            device=device, dtype=torch.float)

        targets.append({
            'boxes': boxes,
            'labels': labels
        })

    out = None
    loss = torch.Tensor([0.0]).to(device=device)
    if model.training:
        loss = model(images, targets)
    else:
        out = model(images)

    out = cb_handler.on_loss_begin(out)

    if opt is not None:
        loss,skip_bwd = cb_handler.on_backward_begin(loss)
        if not skip_bwd:                     loss.backward()
        if not cb_handler.on_backward_end(): opt.step()
        if not cb_handler.on_step_end():     opt.zero_grad()

    return loss.detach().cpu()

def setup_data(dataset_name):
    if dataset_name == 'pascal2007':
        output_uri = 's3://raster-vision-lf-dev/pascal2007/output/'
        output_dir = '/opt/data/pascal2007/output/'
        make_dir(output_dir)
        data_dir = '/opt/data/pascal2007/data'
        untar_data(URLs.PASCAL_2007, dest=data_dir)
        data_dir = join(data_dir, 'pascal_2007')
        return output_dir, output_uri, data_dir
    elif dataset_name == 'boxes':
        output_uri = 's3://raster-vision-lf-dev/boxes/output'
        output_dir = '/opt/data/boxes/output/'
        make_dir(output_dir)
        data_uri = 's3://raster-vision-lf-dev/boxes/boxes.zip'
        data_dir = '/opt/data/boxes/data'
        with tempfile.TemporaryDirectory() as tmp_dir:
            if not isdir(data_dir):
                zip_path = download_if_needed(data_uri, tmp_dir)
                unzip(zip_path, dirname(data_dir))
        return output_dir, output_uri, data_dir
    else:
        raise ValueError('dataset_name {} is invalid'.format(dataset_name))

@click.command()
@click.argument('dataset_name')
@click.option('--test', is_flag=True, help='Run small test experiment')
@click.option('--s3-data', is_flag=True, help='Use data and store results on S3')
@click.option('--batch', is_flag=True, help='Submit Batch job for full experiment')
@click.option('--debug', is_flag=True, help='Run via debugger')
def main(dataset_name, test, s3_data, batch, debug):
    if batch:
        run_on_batch(dataset_name, test, debug)

    # Setup options
    backbone_arch = 'resnet18'
    levels = [0]
    bs = 8
    size = 300
    num_workers = 4
    num_epochs = 100
    lr = 1e-4
    if test:
        bs = 1
        size = 200
        num_debug_images = 32
        num_workers = 0
        num_epochs = 1

    # Setup data
    output_dir, output_uri, data_dir = setup_data(dataset_name)

    img_path = join(data_dir, 'train')
    trn_path = join(data_dir, 'train.json')
    trn_images, trn_lbl_bbox = get_annotations(trn_path)
    val_path = join(data_dir, 'valid.json')
    val_images, val_lbl_bbox = get_annotations(val_path)

    images, lbl_bbox = trn_images+val_images, trn_lbl_bbox+val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    with open(trn_path) as f:
        d = json.load(f)
        classes = sorted(d['categories'], key=lambda x: x['id'])
        classes = [x['name'] for x in classes]
        classes = ['background'] + classes
        num_labels = len(classes)

    def get_data(bs, size):
        src = ObjectItemList.from_folder(img_path)
        if test:
            rand_inds = np.random.choice(
                list(range(len(src))), (num_debug_images,), replace=False)
            src = src[rand_inds]

        if dataset_name == 'pascal2007':
            src = src.split_by_files(val_images[0:int(len(trn_images) * 0.2)])
        else:
            src = src.split_by_files(val_images)

        src = src.label_from_func(get_y_func, classes=classes)
        if dataset_name != 'boxes':
            src = src.transform(get_transforms(), size=size, tfm_y=True)
        return src.databunch(path=data_dir, bs=bs, collate_fn=bb_pad_collate,
                             num_workers=num_workers)

    data = get_data(bs, size)
    print(data)
    plot_data(data, output_dir)

    # Setup model
    model = FCOS(backbone_arch, num_labels, levels=levels)
    metrics = [CocoMetric(num_labels)]
    learn = Learner(data, model, path=output_dir, metrics=metrics)
    fastai.basic_train.loss_batch = loss_batch
    callbacks = [
        CSVLogger(learn, filename='log')
    ]
    learn.fit_one_cycle(num_epochs, lr, callbacks=callbacks)

    plot_preds(data, learn.model, classes, output_dir)

    if s3_data:
        sync_to_dir(output_dir, output_uri)

if __name__ == '__main__':
    main()