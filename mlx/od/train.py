from os.path import join, isfile
import shutil
import tempfile
from collections import defaultdict
import csv
import warnings
warnings.filterwarnings('ignore')
import time
import os
import datetime

import click
import torch

from mlx.filesystem.utils import (
    sync_to_dir, sync_from_dir, json_to_file, file_to_json)
from mlx.od.metrics import compute_coco_eval
from mlx.od.data import setup_output_dir, build_databunch
from mlx.od.config import load_config
from mlx.od.model import build_model
from mlx.od.plot import build_plotter
from mlx.od.optimizer import build_optimizer, build_scheduler

def train_epoch(cfg, model, device, dl, opt, step_scheduler=None, epoch_scheduler=None):
    model.train()
    train_loss = defaultdict(lambda: 0.0)
    num_samples = 0

    for batch_ind, (x, y) in enumerate(dl):
        # skip partial batch at end to avoid messing up batchnorm
        if x.shape[0] < cfg.solver.batch_sz:
            continue

        x = x.to(device)
        y = [_y.to(device) for _y in y]

        opt.zero_grad()
        loss_dict = model(x, y)
        loss_dict['total_loss'].backward()
        opt.step()
        if step_scheduler:
            step_scheduler.step()

        for k, v in loss_dict.items():
            train_loss[k] += v.item()
        num_samples += x.shape[0]

        # print(str(batch_ind), flush=True)        

    for k, v in train_loss.items():
        train_loss[k] = v / num_samples

    return dict(train_loss)

def validate_epoch(cfg, model, device, dl, num_labels):
    model.eval()

    ys = []
    outs = []
    with torch.no_grad():
        for batch_ind, (x, y) in enumerate(dl):
            x = x.to(device)
            out = model(x)

            ys.extend([_y.cpu() for _y in y])
            outs.extend([_out.cpu() for _out in out])
            # print(str(batch_ind), flush=True)

    coco_metrics = compute_coco_eval(outs, ys, num_labels)
    metrics = {'map50': coco_metrics[1]}
    return metrics

def overfit_loop(cfg, databunch, model, opt, device, output_dir):
    if not (cfg.solver.batch_sz == len(databunch.train_ds) == len(databunch.test_ds)):
        raise ValueError(
            'batch_sz and length of train_ds and test_ds '
            'must be the same in overfit_mode')
        
    for step in range(cfg.solver.overfit_num_steps):
        train_loss = train_epoch(
            cfg, model, device, databunch.train_dl, opt)
        if (step + 1) % 25 == 0:
            print('step: {}'.format(step))
            print('train loss: {}'.format(train_loss))
    
    last_model_path = join(output_dir, 'last_model.pth')
    torch.save(model.state_dict(), last_model_path)

def train_loop(cfg, databunch, model, opt, device, output_dir):
    best_model_path = join(output_dir, 'best_model.pth')
    last_model_path = join(output_dir, 'last_model.pth')
    num_labels = len(databunch.label_names)

    best_metric = -1.0
    start_epoch = 0
    train_state_path = join(output_dir, 'train_state.json')
    log_path = join(output_dir, 'log.csv')
    if isfile(train_state_path):
        print('Resuming from checkpoint: {}\n'.format(last_model_path))
        train_state = file_to_json(train_state_path)
        start_epoch = train_state['epoch'] + 1
        best_metric = train_state['best_metric']
        model.load_state_dict(
            torch.load(last_model_path, map_location=device))

    if not isfile(log_path):
        with open(log_path, 'w') as log_file:
            log_writer = csv.writer(log_file)
            row = ['epoch'] + ['map50', 'time'] + model.subloss_names
            log_writer.writerow(row)

    step_scheduler, epoch_scheduler = build_scheduler(cfg, databunch, opt, start_epoch)

    for epoch in range(start_epoch, cfg.solver.num_epochs):
        start = time.time()
        train_loss = train_epoch(
            cfg, model, device, databunch.train_dl, opt, step_scheduler, epoch_scheduler)
        end = time.time()
        epoch_time = datetime.timedelta(seconds=end - start)
        if epoch_scheduler:
            epoch_scheduler.step()

        print('----------------------------------------')
        print('epoch: {}'.format(epoch), flush=True)
        print('train loss: {}'.format(train_loss), flush=True)
        print('elapsed: {}'.format(epoch_time), flush=True)

        metrics = validate_epoch(
            cfg, model, device, databunch.valid_dl, num_labels)
        print('validation metrics: {}'.format(metrics), flush=True)

        '''
        if metrics['map50'] > best_metric:
            best_metric = metrics['map50']
            torch.save(model.state_dict(), best_model_path)
        '''
        torch.save(model.state_dict(), best_model_path)
        torch.save(model.state_dict(), last_model_path)

        train_state = {'epoch': epoch, 'best_metric': best_metric}
        json_to_file(train_state, train_state_path)

        with open(log_path, 'a') as log_file:
            log_writer = csv.writer(log_file)
            row = [epoch]
            row += [metrics['map50'], epoch_time]
            row += [train_loss[k] for k in model.subloss_names]
            log_writer.writerow(row)

        if (cfg.output_uri.startswith('s3://') and 
                ((epoch + 1) % cfg.solver.sync_interval == 0)):
            sync_to_dir(output_dir, cfg.output_uri)

@click.command()
@click.argument('config_path')
@click.argument('opts', nargs=-1)
def train(config_path, opts):
    torch_cache_dir = '/opt/data/torch-cache'
    os.environ['TORCH_HOME'] = torch_cache_dir

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name

    cfg = load_config(config_path, opts)
    print(cfg)
    print()

    # Setup data
    databunch = build_databunch(cfg, tmp_dir)
    output_dir = setup_output_dir(cfg, tmp_dir)
    shutil.copyfile(config_path, join(output_dir, 'config.yml'))
    print(databunch)
    print()

    plotter = build_plotter(cfg)
    if not cfg.predict_mode:
        plotter.plot_dataloaders(databunch, output_dir)

    # Setup model
    num_labels = len(databunch.label_names)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(cfg, num_labels)
    model.to(device)
    opt = build_optimizer(cfg, model)

    # TODO tensorboard, progress bar
    if cfg.model.init_weights:
        model.load_state_dict(
            torch.load(cfg.model.init_weights, map_location=device))

    if not cfg.predict_mode:
        if cfg.overfit_mode:
            overfit_loop(cfg, databunch, model, opt, device, output_dir)
        else:
            train_loop(cfg, databunch, model, opt, device, output_dir)

    print('\nEvaluating on test set...')
    metrics = validate_epoch(
        cfg, model, device, databunch.test_dl, num_labels)
    print('test metrics: {}'.format(metrics))
    json_to_file(metrics, join(output_dir, 'test_metrics.json'))

    print('\nPlotting predictions...')
    plotter.make_debug_plots(
        databunch.test_dl, model, databunch.label_names, output_dir)

    if cfg.output_uri.startswith('s3://'):
        sync_to_dir(output_dir, cfg.output_uri)

if __name__ == '__main__':
    train()