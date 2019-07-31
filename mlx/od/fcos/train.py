import json
import uuid
from os.path import join, isdir, dirname
import shutil
import tempfile
import os

import numpy as np
import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fastai
from fastai.vision import (
    get_annotations, ObjectItemList, get_transforms,
    bb_pad_collate, URLs, untar_data, imagenet_stats, flip_affine)
from fastai.basic_train import Learner
import torch
from torch import nn, Tensor
from fastai.callback import CallbackHandler
from fastai.callbacks import TrackEpochCallback
from fastai.core import ifnone
from fastai.torch_core import OptLossFunc, OptOptimizer, Optional, Tuple, Union

from mlx.od.fcos.model import FCOS
from mlx.od.fcos.metrics import CocoMetric
from mlx.od.fcos.plot import plot_preds, plot_data
from mlx.od.fcos.callbacks import (
    MyCSVLogger, SyncCallback, TensorboardLogger,
    SubLossMetric, MySaveModelCallback)
from mlx.od.fcos.data import get_databunch, setup_output
from mlx.batch_utils import submit_job
from mlx.filesystem.utils import (
    make_dir, sync_to_dir, zipdir, unzip, download_if_needed)

def run_on_batch(dataset_name, test, overfit, debug, profile):
    job_name = 'mlx_train_fcos-' + str(uuid.uuid4())
    job_def = 'lewfishPyTorchCustomGpuJobDefinition'
    job_queue = 'lewfishRasterVisionGpuJobQueue'
    cmd_list = ['python', '-m', 'mlx.od.fcos.train', dataset_name, '--s3-data']

    if debug:
        cmd_list = [
            'python', '-m', 'ptvsd', '--host', '0.0.0.0', '--port', '6006',
            '--wait', '-m', 'mlx.od.fcos.train', dataset_name, '--s3-data']

    if profile:
        cmd_list = ['kernprof', '-v', '-l', '/opt/src/mlx/od/fcos/train.py',
                    dataset_name, '--s3-data']

    if test:
        cmd_list.append('--test')
    if overfit:
        cmd_list.append('--overfit')
    submit_job(job_name, job_def, job_queue, cmd_list, attempts=5)
    exit()

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
        # range [0, h) or [0, w)
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
        loss_dict = model(images, targets)
        loss = loss_dict['label_loss'] + loss_dict['reg_loss'] + loss_dict['center_loss']
        cb_handler.state_dict['loss_dict'] = loss_dict
    else:
        out = model(images)

    out = cb_handler.on_loss_begin(out)

    if opt is not None:
        loss,skip_bwd = cb_handler.on_backward_begin(loss)
        if not skip_bwd:                     loss.backward()
        if not cb_handler.on_backward_end(): opt.step()
        if not cb_handler.on_step_end():     opt.zero_grad()

    return loss.detach().cpu()

@click.command()
@click.argument('dataset')
@click.option('--test', is_flag=True, help='Run small test experiment')
@click.option('--overfit', is_flag=True, help='Run overfitting experiment')
@click.option('--s3-data', is_flag=True, help='Use data and store results on S3')
@click.option('--batch', is_flag=True, help='Submit Batch job for full experiment')
@click.option('--debug', is_flag=True, help='Run via debugger')
@click.option('--profile', is_flag=True, help='Run via profiler')
def main(dataset, test, overfit, s3_data, batch, debug, profile):
    if batch:
        run_on_batch(dataset, test, overfit, debug, profile)

    # Setup options
    backbone_arch = 'resnet18'
    levels = [1]
    lr = 1e-4
    num_epochs = 30
    sync_interval = 2

    if overfit:
        num_epochs = 500
        sync_interval = 1000
        lr = 1e-4

    if test:
        num_epochs = 1

    # Setup data
    databunch = get_databunch(dataset, test, overfit)
    output_dir, output_uri = setup_output(dataset, s3_data)
    print(databunch)
    plot_data(databunch, output_dir)

    # Setup model
    num_labels = databunch.c
    model = FCOS(backbone_arch, num_labels, levels=levels)
    metrics = [CocoMetric(num_labels)]
    learn = Learner(databunch, model, path=output_dir, metrics=metrics)
    fastai.basic_train.loss_batch = loss_batch
    best_model_path = join(output_dir, 'best_model.pth')
    last_model_path = join(output_dir, 'last_model.pth')

    # Train model
    callbacks = [
        MyCSVLogger(learn, filename='log'),
        SubLossMetric(learn)
    ]

    if s3_data:
        callbacks.append(SyncCallback(output_dir, output_uri, sync_interval))

    if overfit:
        learn.fit(num_epochs, lr, callbacks=callbacks)
        learn.validate(databunch.train_dl, metrics=metrics)
        plot_dataset = databunch.train_ds
        torch.save(learn.model.state_dict(), last_model_path)
    else:
        tb_logger = TensorboardLogger(learn, 'run')
        tb_logger.set_extra_args(['label_loss', 'reg_loss', 'center_loss'], overfit)

        extra_callbacks = [
            MySaveModelCallback(
                learn, best_model_path, monitor='coco_metric', every='improvement'),
            MySaveModelCallback(learn, last_model_path, every='epoch'),
            TrackEpochCallback(learn),
            tb_logger
        ]
        callbacks.extend(extra_callbacks)
        learn.fit_one_cycle(num_epochs, lr, callbacks=callbacks)
        plot_dataset = databunch.valid_ds

    print('Plotting predictions...')
    plot_preds(plot_dataset, learn.model, databunch.classes, output_dir)
    if s3_data:
        sync_to_dir(output_dir, output_uri)

if __name__ == '__main__':
    main()