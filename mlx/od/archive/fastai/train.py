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

from mlx.od.metrics import CocoMetric
from mlx.od.callbacks import (
    MyCSVLogger, SyncCallback, TensorboardLogger,
    SubLossMetric, MySaveModelCallback)
from mlx.od.data import setup_output_dir, build_databunch
from mlx.od.config import load_config
from mlx.od.boxlist import BoxList, to_box_pixel
from mlx.od.model import build_model
from mlx.filesystem.utils import sync_to_dir
from mlx.od.fcos.model import FCOS
from mlx.od.plot import build_plotter

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
        targets.append(BoxList(boxes, labels=labels))

    out = None
    loss = torch.Tensor([0.0]).to(device=device)
    if model.training:
        loss_dict = model(images, targets)
        loss = loss_dict['total_loss']
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

    # Setup data
    databunch, full_databunch = build_databunch(cfg, tmp_dir)
    output_dir = setup_output_dir(cfg, tmp_dir)
    print(full_databunch)

    plotter = build_plotter(cfg)
    if not cfg.lr_find_mode and not cfg.predict_mode:
        plotter.plot_data(databunch, output_dir)

    # Setup model
    num_labels = databunch.c
    model = build_model(cfg, num_labels)
    metrics = [CocoMetric(num_labels)]
    learn = Learner(databunch, model, path=output_dir, metrics=metrics)
    fastai.basic_train.loss_batch = loss_batch
    best_model_path = join(output_dir, 'best_model.pth')
    last_model_path = join(output_dir, 'last_model.pth')

    # Train model
    callbacks = [
        MyCSVLogger(learn, filename='log'),
        SubLossMetric(learn, model.subloss_names)
    ]

    if cfg.output_uri.startswith('s3://'):
        callbacks.append(
            SyncCallback(output_dir, cfg.output_uri, cfg.solver.sync_interval))

    if cfg.model.init_weights:
        device = next(model.parameters()).device
        model.load_state_dict(
            torch.load(cfg.model.init_weights, map_location=device))

    if not cfg.predict_mode:
        if cfg.overfit_mode:
            learn.fit_one_cycle(cfg.solver.num_epochs, cfg.solver.lr, callbacks=callbacks)
            torch.save(learn.model.state_dict(), best_model_path)
            learn.model.eval()
            print('Validating on training set...')
            learn.validate(full_databunch.train_dl, metrics=metrics)
        else:
            tb_logger = TensorboardLogger(learn, 'run')
            tb_logger.set_extra_args(
                model.subloss_names, cfg.overfit_mode)

            extra_callbacks = [
                MySaveModelCallback(
                    learn, best_model_path, monitor='coco_metric', every='improvement'),
                MySaveModelCallback(learn, last_model_path, every='epoch'),
                TrackEpochCallback(learn),
            ]
            callbacks.extend(extra_callbacks)
            if cfg.lr_find_mode:
                learn.lr_find()
                learn.recorder.plot(suggestion=True, return_fig=True)
                lr = learn.recorder.min_grad_lr
                print('lr_find() found lr: {}'.format(lr))
                exit()

            learn.fit_one_cycle(cfg.solver.num_epochs, cfg.solver.lr, callbacks=callbacks)
            print('Validating on full validation set...')
            learn.validate(full_databunch.valid_dl, metrics=metrics)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(
            torch.load(join(output_dir, 'best_model.pth'), map_location=device))
        model.eval()
        plot_dataset = databunch.train_ds

    print('Plotting predictions...')
    plot_dataset = databunch.train_ds if cfg.overfit_mode else databunch.valid_ds
    plotter.make_debug_plots(plot_dataset, model, databunch.classes, output_dir)
    if cfg.output_uri.startswith('s3://'):
        sync_to_dir(output_dir, cfg.output_uri)

if __name__ == '__main__':
    train()