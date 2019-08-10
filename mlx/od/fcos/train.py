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
from mlx.od.fcos.data import setup_output_dir, build_databunch
from mlx.od.fcos.utils import to_box_pixel
from mlx.od.fcos.config import load_config
from mlx.filesystem.utils import sync_to_dir

# Modified from fastai to handle model which only computes loss when targets
# are passed in, and only computes output otherwise. This should run faster
# thanks to not having to run the decoder and NMS during training, and not
# computing the loss for the validation which is not a great metric anyway.
# This also converts the input format.
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
        boxes = to_box_pixel(boxes, *images[0].shape[1:3])
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
@click.argument('config_path')
@click.argument('opts', nargs=-1)
def train(config_path, opts):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name

    cfg = load_config(config_path, opts)
    print(cfg)

    # Setup options
    sync_interval = cfg.solver.sync_interval
    if cfg.overfit_mode:
        num_epochs = cfg.overfit_num_epochs
        sync_interval = cfg.overfit_sync_interval
    if cfg.test_mode:
        num_epochs = cfg.test_num_epochs

    # Setup data
    databunch, full_databunch = build_databunch(cfg, tmp_dir)
    output_dir = setup_output_dir(cfg, tmp_dir)
    print(full_databunch)
    plot_data(databunch, output_dir)

    # Setup model
    num_labels = databunch.c
    model = FCOS(cfg.model.backbone_arch, num_labels, levels=cfg.model.levels)
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

    if cfg.output_uri.startswith('s3://'):
        callbacks.append(
            SyncCallback(output_dir, cfg.output_uri, sync_interval))

    if cfg.overfit_mode:
        learn.fit_one_cycle(num_epochs, cfg.solver.lr, callbacks=callbacks)
        learn.validate(databunch.train_dl, metrics=metrics)
        plot_dataset = databunch.train_ds
        torch.save(learn.model.state_dict(), last_model_path)
        print('Validating on training set...')
        learn.validate(full_databunch.train_dl, metrics=metrics)
    else:
        tb_logger = TensorboardLogger(learn, 'run')
        tb_logger.set_extra_args(
            ['label_loss', 'reg_loss', 'center_loss'], cfg.overfit_mode)

        extra_callbacks = [
            MySaveModelCallback(
                learn, best_model_path, monitor='coco_metric', every='improvement'),
            MySaveModelCallback(learn, last_model_path, every='epoch'),
            TrackEpochCallback(learn),
            tb_logger
        ]
        callbacks.extend(extra_callbacks)
        learn.fit_one_cycle(num_epochs, cfg.solver.lr, callbacks=callbacks)
        plot_dataset = databunch.valid_ds
        print('Validating on full validation set...')
        learn.validate(full_databunch.valid_dl, metrics=metrics)

    print('Plotting predictions...')
    plot_preds(plot_dataset, learn.model, databunch.classes, output_dir)
    if cfg.output_uri.startswith('s3://'):
        sync_to_dir(output_dir, cfg.output_uri)

if __name__ == '__main__':
    train()