from os.path import join, isfile
import tempfile
from collections import defaultdict
import csv

import click
import torch

from mlx.filesystem.utils import sync_to_dir, json_to_file, file_to_json
from mlx.od.metrics import compute_coco_eval
from mlx.od.data import setup_output_dir, build_databunch
from mlx.od.config import load_config
from mlx.od.model import build_model
from mlx.od.plot import build_plotter
from mlx.od.optimizer import build_optimizer

def train_epoch(cfg, model, device, dl, opt, epoch):
    model.train()
    train_loss = defaultdict(lambda: 0.0)
    num_samples = 0

    for batch_idx, (x, y) in enumerate(dl):
        # skip partial batch at end to avoid messing up batchnorm
        if x.shape[0] < cfg.solver.batch_sz:
            continue

        x = x.to(device)
        y = [_y.to(device) for _y in y]

        opt.zero_grad()
        loss_dict = model(x, y)
        loss_dict['total_loss'].backward()
        opt.step()

        for k, v in loss_dict.items():
            train_loss[k] += v.item()

        num_samples += x.shape[0]

    for k, v in train_loss.items():
        train_loss[k] = v / num_samples

    return dict(train_loss)

def validate_epoch(cfg, model, device, dl, num_labels):
    model.eval()
    ys = []
    outs = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = [_y.to(device) for _y in y]
            out = model(x)

            ys.extend(y)
            outs.extend(out)

    coco_metrics = compute_coco_eval(outs, ys, num_labels)
    metrics = {'map': coco_metrics[0]}
    return metrics

@click.command()
@click.argument('config_path')
@click.argument('opts', nargs=-1)
def train(config_path, opts):
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name

    cfg = load_config(config_path, opts)
    print(cfg)
    print()

    # Setup data
    databunch = build_databunch(cfg, tmp_dir)
    output_dir = setup_output_dir(cfg, tmp_dir)
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
    best_model_path = join(output_dir, 'best_model.pth')
    last_model_path = join(output_dir, 'last_model.pth')

    # TODO one cycle, tensorboard, progress bar, save config file

    if cfg.output_uri.startswith('s3://'):
        sync_to_dir(cfg.output_uri, output_dir)

    if cfg.model.init_weights:
        device = next(model.parameters()).device
        model.load_state_dict(
            torch.load(cfg.model.init_weights, map_location=device))

    if cfg.predict_mode:
        # TODO make some of this more configurable
        model.load_state_dict(
            torch.load(join(output_dir, 'last_model.pth'), map_location=device))
    elif cfg.overfit_mode:
        if not (cfg.solver.batch_sz == len(databunch.train_ds) == len(databunch.test_ds)):
            raise ValueError(
                'batch_sz and length of train_ds and test_ds '
                'must be the same in overfit_mode')
        for epoch in range(cfg.solver.num_epochs):
            train_loss = train_epoch(
                cfg, model, device, databunch.train_dl, opt, epoch)
            if epoch % 50 == 0:
                print('epoch: {}'.format(epoch))
                print('train loss: {}'.format(train_loss))
        torch.save(model.state_dict(), last_model_path)
    else:
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
                row = ['epoch'] + model.subloss_names + ['map']
                log_writer.writerow(row)

        for epoch in range(start_epoch, cfg.solver.num_epochs):
            train_loss = train_epoch(
                cfg, model, device, databunch.train_dl, opt, epoch)
            print('epoch: {}'.format(epoch))
            print('train loss: {}'.format(train_loss))

            metrics = validate_epoch(
                cfg, model, device, databunch.valid_dl, num_labels)
            print('validation metrics: {}'.format(metrics))

            if metrics['map'] > best_metric:
                best_metric = metrics['map']
                torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), last_model_path)

            train_state = {'epoch': epoch, 'best_metric': best_metric}
            json_to_file(train_state, train_state_path)

            with open(log_path, 'a') as log_file:
                log_writer = csv.writer(log_file)
                row = [epoch]
                row += [train_loss[k] for k in model.subloss_names]
                row += [metrics['map']]
                log_writer.writerow(row)

            if cfg.output_uri.startswith('s3://'):
                sync_to_dir(output_dir, cfg.output_uri)


    print('\nEvaluating on test set...')
    metrics = validate_epoch(
        cfg, model, device, databunch.test_dl, num_labels)
    print('test metrics: {}'.format(metrics))

    print('\nPlotting predictions...')
    plotter.make_debug_plots(
        databunch.test_ds, model, databunch.label_names, output_dir)

    if cfg.output_uri.startswith('s3://'):
        sync_to_dir(output_dir, cfg.output_uri)

if __name__ == '__main__':
    train()