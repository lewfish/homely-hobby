from os.path import join
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from mlx.filesystem.utils import make_dir
from mlx.classification.plot import plot_xyz

mnist = 'mnist'
datasets = [mnist]

def validate_dataset(dataset):
    if dataset not in datasets:
        raise ValueError('dataset {} is invalid'.format(dataset))

class DataBunch():
    def __init__(self, train_ds, train_dl, valid_ds, valid_dl, test_ds, test_dl, label_names):
        self.train_ds = train_ds
        self.train_dl = train_dl
        self.valid_ds = valid_ds
        self.valid_dl = valid_dl
        self.test_ds = test_ds
        self.test_dl = test_dl
        self.label_names = label_names

    def __repr__(self):
        rep = ''
        if self.train_ds:
            rep += 'train_ds: {} items\n'.format(len(self.train_ds))
        if self.valid_ds:
            rep += 'valid_ds: {} items\n'.format(len(self.valid_ds))
        if self.test_ds:
            rep += 'test_ds: {} items\n'.format(len(self.test_ds))
        rep += 'label_names: ' + ','.join(self.label_names)
        return rep

    def plot_dataloader(self, dataloader, output_path):
        x, y = next(iter(dataloader))
        batch_sz = x.shape[0]

        ncols = nrows = math.ceil(math.sqrt(batch_sz))
        fig = plt.figure(constrained_layout=True, figsize=(3 * ncols, 3 * nrows))
        grid = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        for i in range(batch_sz):
            ax = fig.add_subplot(grid[i])
            plot_xyz(ax, x[i], y[i], self.label_names)

        make_dir(output_path, use_dirname=True)
        plt.savefig(output_path)
        plt.close()

    def plot_dataloaders(self, output_dir):
        if self.train_dl:
            self.plot_dataloader(self.train_dl, join(output_dir, 'dataloaders/train.png'))
        if self.valid_dl:
            self.plot_dataloader(self.valid_dl, join(output_dir, 'dataloaders/valid.png'))
        if self.test_dl:
            self.plot_dataloader(self.test_dl, join(output_dir, 'dataloaders/test.png'))


def build_databunch(cfg):
    dataset = cfg.data.dataset
    validate_dataset(dataset)

    data_dir = '/opt/data/data-cache'
    batch_sz = cfg.solver.batch_sz
    num_workers = cfg.data.num_workers
    train_ds, valid_ds, test_ds = None, None, None
    transform = Compose([ToTensor()])

    if dataset == mnist:
        if cfg.overfit_mode:
            train_ds = MNIST(data_dir, train=True, transform=transform, download=True)
            label_names = train_ds.classes
            train_ds = Subset(train_ds, range(batch_sz))
            test_ds = train_ds
        elif cfg.test_mode:
            train_ds = MNIST(data_dir, train=True, transform=transform, download=True)
            label_names = train_ds.classes
            train_ds = Subset(train_ds, range(batch_sz))
            valid_ds = train_ds
            test_ds = train_ds
        else:
            train_ds = MNIST(data_dir, train=True, transform=transform, download=True)
            label_names = train_ds.classes
            test_ds = MNIST(data_dir, train=False, transform=transform, download=True)
            valid_ds = Subset(test_ds, range(len(test_ds) // 5))

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_sz, num_workers=num_workers, pin_memory=True) \
        if train_ds else None
    valid_dl = DataLoader(valid_ds, batch_size=batch_sz, num_workers=num_workers, pin_memory=True) \
        if valid_ds else None
    test_dl = DataLoader(test_ds, batch_size=batch_sz, num_workers=num_workers, pin_memory=True) \
        if test_ds else None
    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, test_ds, test_dl, label_names)
