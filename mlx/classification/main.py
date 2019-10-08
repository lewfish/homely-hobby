from os.path import join, isfile
import shutil
import tempfile
import warnings
warnings.filterwarnings('ignore')
import os

import click
import torch

from mlx.filesystem.utils import (
    sync_to_dir, json_to_file, make_dir, sync_from_dir, get_local_path, file_to_json)
from mlx.classification.config import load_config
from mlx.classification.optimizer import build_optimizer, build_scheduler
from mlx.classification.model import build_model
from mlx.classification.data import build_databunch
from mlx.classification.learner import Learner

@click.command()
@click.argument('config_path')
@click.argument('opts', nargs=-1)
def main(config_path, opts):
    # Load config and setup output_dir.
    torch_cache_dir = '/opt/data/torch-cache'
    os.environ['TORCH_HOME'] = torch_cache_dir
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name

    cfg = load_config(config_path, opts)
    if cfg.output_uri.startswith('s3://'):
        output_dir = get_local_path(cfg.output_uri, tmp_dir)
        make_dir(output_dir, force_empty=True)
        if not cfg.overfit_mode:
            sync_from_dir(cfg.output_uri, output_dir)
    else:
        output_dir = cfg.output_uri
        make_dir(output_dir)
    shutil.copyfile(config_path, join(output_dir, 'config.yml'))

    print(cfg)
    print()

    # Setup databunch and plot.
    databunch = build_databunch(cfg)
    print(databunch)
    print()
    if not cfg.predict_mode:
        databunch.plot_dataloaders(output_dir)

    # Setup learner.
    num_labels = len(databunch.label_names)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(cfg)
    model.to(device)
    if cfg.model.init_weights:
        model.load_state_dict(
            torch.load(cfg.model.init_weights, map_location=device))

    opt = build_optimizer(cfg, model)
    loss_fn = torch.nn.CrossEntropyLoss()
    start_epoch = 0
    train_state_path = join(output_dir, 'train_state.json')
    if isfile(train_state_path):
        train_state = file_to_json(train_state_path)
        start_epoch = train_state['epoch'] + 1
    num_samples = len(databunch.train_ds)
    step_scheduler, epoch_scheduler = build_scheduler(
        cfg, num_samples, opt, start_epoch)
    learner = Learner(
        cfg, databunch, output_dir, model, loss_fn, opt, device,
        epoch_scheduler, step_scheduler)

    # Train
    if not cfg.predict_mode:
        if cfg.overfit_mode:
            learner.overfit()
        else:
            learner.train()

    # Evaluate on test set and plot.
    if cfg.eval_train:
        print('\nEvaluating on train set...')
        metrics = learner.validate_epoch(databunch.train_dl)
        print('train metrics: {}'.format(metrics))
        json_to_file(metrics, join(output_dir, 'train_metrics.json'))

        print('\nPlotting training set predictions...')
        learner.plot_preds(databunch.train_dl, join(output_dir, 'train_preds.png'))

    print('\nEvaluating on test set...')
    metrics = learner.validate(databunch.test_dl)
    print('test metrics: {}'.format(metrics))
    json_to_file(metrics, join(output_dir, 'test_metrics.json'))

    print('\nPlotting test set predictions...')
    learner.plot_preds(databunch.test_dl, join(output_dir, 'test_preds.png'))

    if cfg.output_uri.startswith('s3://'):
        sync_to_dir(output_dir, cfg.output_uri)

if __name__ == '__main__':
    main()
