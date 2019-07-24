import shutil
from typing import Any

import torch
from fastai.callbacks import CSVLogger, Callback, TrackerCallback
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