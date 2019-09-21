from os.path import join, isfile
import csv
import warnings
warnings.filterwarnings('ignore')
import time
import datetime

import click
import torch

from mlx.filesystem.utils import (
    sync_to_dir, json_to_file, file_to_json)
from mlx.classification.metrics import compute_conf_mat_metrics

class Learner():
    def __init__(self, cfg, databunch, output_dir, model, loss_fn, opt, device,
                 epoch_scheduler=None, step_scheduler=None):
        self.cfg = cfg
        self.databunch = databunch
        self.output_dir = output_dir
        self.model = model
        self.loss_fn = loss_fn
        self.opt = opt
        self.device = device
        self.epoch_scheduler = epoch_scheduler
        self.step_scheduler = step_scheduler

        self.num_labels = len(self.databunch.label_names)

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        with click.progressbar(data_loader, label='Training') as bar:
            for batch_ind, (x, y) in enumerate(bar):
                x = x.to(self.device)
                y = y.to(self.device)

                self.opt.zero_grad()
                out = self.model(x)
                loss = self.loss_fn(out, y)
                loss.backward()
                total_loss += loss.item()
                self.opt.step()
                if self.step_scheduler:
                    self.step_scheduler.step()
                num_samples += x.shape[0]

        return total_loss / num_samples

    def validate(self, data_loader):
        self.model.eval()
        conf_mat = torch.zeros((self.num_labels, self.num_labels))
        with torch.no_grad():
            with click.progressbar(data_loader, label='Validating') as bar:
                for batch_ind, (x, y) in enumerate(bar):
                    x = x.to(self.device)
                    out = self.model(x)

                    out = out.argmax(-1).view(-1).cpu()
                    y = y.cpu()
                    if batch_ind == 0:
                        labels = torch.arange(0, self.num_labels)

                    conf_mat += ((out == labels[:, None]) &
                                (y == labels[:, None, None])).sum(
                                    dim=2, dtype=torch.float32)

        return compute_conf_mat_metrics(conf_mat)

    def overfit(self):
        if not (self.cfg.solver.batch_sz == len(self.databunch.train_ds) == len(self.databunch.test_ds)):
            raise ValueError(
                'batch_sz and length of train_ds and test_ds '
                'must be the same in overfit_mode')

        for step in range(self.cfg.solver.overfit_num_steps):
            train_loss = self.train_epoch(self.databunch.train_dl)
            if (step + 1) % 25 == 0:
                print('step: {}'.format(step))
                print('train loss: {}'.format(train_loss))

        last_model_path = join(self.output_dir, 'last_model.pth')
        torch.save(self.model.state_dict(), last_model_path)

    def train(self):
        last_model_path = join(self.output_dir, 'last_model.pth')
        start_epoch = 0

        log_path = join(self.output_dir, 'log.csv')
        train_state_path = join(self.output_dir, 'train_state.json')
        if isfile(train_state_path):
            print('Resuming from checkpoint: {}\n'.format(last_model_path))
            train_state = file_to_json(train_state_path)
            start_epoch = train_state['epoch'] + 1
            self.model.load_state_dict(
                torch.load(last_model_path, map_location=self.device))

        metric_names = ['precision', 'recall', 'f1', 'accuracy']
        if not isfile(log_path):
            with open(log_path, 'w') as log_file:
                log_writer = csv.writer(log_file)
                row = ['epoch', 'time', 'train_loss'] + metric_names
                log_writer.writerow(row)

        for epoch in range(start_epoch, self.cfg.solver.num_epochs):
            start = time.time()
            train_loss = self.train_epoch(self.databunch.train_dl)
            end = time.time()
            epoch_time = datetime.timedelta(seconds=end - start)
            if self.epoch_scheduler:
                self.epoch_scheduler.step()

            print('----------------------------------------')
            print('epoch: {}'.format(epoch), flush=True)
            print('train loss: {}'.format(train_loss), flush=True)
            print('elapsed: {}'.format(epoch_time), flush=True)

            metrics = self.validate(self.databunch.valid_dl)
        print('validation metrics: {}'.format(metrics), flush=True)

        torch.save(self.model.state_dict(), last_model_path)
        train_state = {'epoch': epoch}
        json_to_file(train_state, train_state_path)

        with open(log_path, 'a') as log_file:
            log_writer = csv.writer(log_file)
            row = [epoch, epoch_time, train_loss]
            row += [metrics[k] for k in metric_names]
            log_writer.writerow(row)

        if (self.cfg.output_uri.startswith('s3://') and
                ((epoch + 1) % self.cfg.solver.sync_interval == 0)):
            sync_to_dir(self.output_dir, self.cfg.output_uri)
