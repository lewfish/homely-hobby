import shutil
from typing import Any
from os.path import join

import torch
from fastai.callbacks import (
    CSVLogger, Callback, TrackerCallback, LearnerCallback, add_metrics)
from fastai.basic_train import Learner

from mlx.filesystem.utils import (sync_to_dir)

class SyncCallback(Callback):
    """A callback to sync from_dir to to_uri at the end of epochs."""
    def __init__(self, from_dir, to_uri, sync_interval=1):
        self.from_dir = from_dir
        self.to_uri = to_uri
        self.sync_interval = sync_interval

    def on_epoch_end(self, **kwargs):
        if (kwargs['epoch'] + 1) % self.sync_interval == 0:
            sync_to_dir(self.from_dir, self.to_uri, delete=True)

class ExportModelCallback(TrackerCallback):
    """Export the model when monitored quantity is best."""
    def __init__(self, learn:Learner, model_path:str, monitor:str='valid_loss', mode:str='auto'):
        self.model_path = model_path
        super().__init__(learn, monitor=monitor, mode=mode)

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        current = self.get_monitor_value()

        if (epoch == 0 or
                (current is not None and self.operator(current, self.best))):
            print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
            self.best = current
            print(f'Saving to {self.model_path}')
            torch.save(self.learn.model.state_dict(), self.model_path)

class MyCSVLogger(CSVLogger):
    """Logs metrics to a CSV file after each epoch.

    Modified from fastai version to:
    - flush after each epoch
    - append to log if already exists
    """
    def __init__(self, learn, filename='history'):
        super().__init__(learn, filename)

    def on_train_begin(self, **kwargs):
        if self.path.exists():
            self.file = self.path.open('a')
        else:
            super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        out = super().on_epoch_end(
            epoch, smooth_loss, last_metrics, **kwargs)
        self.file.flush()
        return out

class SubLossMetric(LearnerCallback):
    _order=-20 # Needs to run before the recorder
    def __init__(self, learn):
        super().__init__(learn)

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['label_loss', 'reg_loss', 'center_loss'])

    def on_batch_end(self, train, **kwargs):
        if train:
            loss_dict = kwargs['loss_dict']
            self.label_loss += loss_dict['label_loss'].detach().cpu().item()
            self.reg_loss += loss_dict['reg_loss'].detach().cpu().item()
            self.center_loss += loss_dict['center_loss'].detach().cpu().item()

    def on_epoch_begin(self, **kwargs):
        self.label_loss = 0.
        self.reg_loss = 0.
        self.center_loss = 0.

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(
            last_metrics, [self.label_loss, self.reg_loss, self.center_loss])

# This code was adapted from
# https://github.com/Pendar2/fastai-tensorboard-callback/blob/master/fastai_tensorboard_callback/tensorboard_cb.py
from tensorboardX import SummaryWriter
from fastai.basics import *

@dataclass
class TensorboardLogger(Callback):
    learn:Learner
    run_name:str
    histogram_freq:int=100
    path:str=None

    def set_extra_metrics(self, extra_metrics):
        self.extra_metrics = extra_metrics

    def __post_init__(self):
        self.path = self.path or join(self.learn.path, "logs")
        self.log_dir = join(self.path, self.run_name)

    def on_train_begin(self, **kwargs):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_epoch_end(self, **kwargs):
        iteration = kwargs["iteration"]
        metrics = kwargs["last_metrics"]

        metrics_names = ["valid_loss"] + [o.__name__ for o in self.learn.metrics]
        if self.extra_metrics is not None:
            metrics_names += self.extra_metrics

        for val, name in zip(metrics, metrics_names):
            self.writer.add_scalar(name, val, iteration)

        for name, emb in self.learn.model.named_children():
            if isinstance(emb, nn.Embedding):
                self.writer.add_embedding(list(emb.parameters())[0], global_step=iteration, tag=name)

    def on_batch_end(self, **kwargs):
        iteration = kwargs["iteration"]
        loss = kwargs["last_loss"]

        self.writer.add_scalar("learning_rate", self.learn.opt.lr, iteration)
        self.writer.add_scalar("momentum", self.learn.opt.mom, iteration)

        self.writer.add_scalar("loss", loss, iteration)
        if iteration % self.histogram_freq == 0:
            for name, param in self.learn.model.named_parameters():
                self.writer.add_histogram(name, param, iteration)

    def on_train_end(self, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dummy_input = next(iter(self.learn.data.train_dl))[0]
                self.writer.add_graph(self.learn.model, tuple(dummy_input))
        except Exception as e:
            print("Unable to create graph.")
            print(e)
        self.writer.close()