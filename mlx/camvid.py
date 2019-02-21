from pathlib import Path
import tempfile
import zipfile
import os
import csv

import torch
import click
import numpy as np
from fastai.vision import (
    get_image_files, SegmentationItemList, open_image, get_transforms,
    imagenet_stats, models, unet_learner)
from fastai.callbacks import SaveModelCallback, CSVLogger, Callback

import mlx.s3_utils as s3_utils
import mlx.batch_utils as batch_utils
from mlx.utils import make_dir


class S3SyncCallback(Callback):
    def __init__(self, from_dir, to_uri, sync_interval=1):
        self.from_dir = from_dir
        self.to_uri = to_uri
        self.sync_interval = sync_interval

    def on_epoch_end(self, **kwargs):
        if (kwargs['epoch'] + 1) % self.sync_interval == 0:
            s3_utils.sync_from_dir(self.from_dir, self.to_uri)


class MyCSVLogger(CSVLogger):
    """Custom CSVLogger

    Modified to:
    - flush after each epoch
    - append to log if already exists
    - use start_epoch
    """
    def __init__(self, learn, filename='history', start_epoch=1):
        super().__init__(learn, filename)
        self.start_epoch = start_epoch

    def on_train_begin(self, **kwargs):
        if self.path.exists():
            self.file = self.path.open('a')
        else:
            super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        effective_epoch = self.start_epoch + epoch - 1
        out = super().on_epoch_end(
            effective_epoch, smooth_loss, last_metrics, **kwargs)
        self.file.flush()
        return out


def get_last_epoch(log_path):
    with open(log_path, 'r') as f:
        num_rows = 0
        for row in csv.reader(f):
            num_rows += 1
        if num_rows >= 2:
            return int(row[0])
        return 0


codes = np.array(['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree',
                  'Sign', 'Fence', 'Car', 'Pedestrian', 'Cyclist', 'Void'])
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']


def acc_camvid(input, target):
    # Note: custom metrics need to be at global level for learner to be saved.
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


def run_on_batch():
    job_name = 'camvid'
    job_def = 'lewfishPyTorchCustomGpuJobDefinition'
    job_queue = 'lewfishRasterVisionGpuJobQueue'
    cmd_list = ['python', '-m', 'mlx.camvid', '--s3-data']
    batch_utils.run_on_batch(job_name, job_def, job_queue, cmd_list)
    exit()


@click.command()
@click.option('--test', is_flag=True, help='Run small test experiment')
@click.option('--s3-data', is_flag=True, help='Use data and store results on S3')
@click.option('--batch', is_flag=True, help='Submit Batch job for full experiment')
def train(test, s3_data, batch):
    """Train a segmentation model using fastai and PyTorch on the Camvid dataset.

    This will write to a CSV log after each epoch, sync output to S3, and resume training
    from a checkpoint. Note: This is an adaptation of
    https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid-tiramisu.ipynb
    and uses the Camvid Tiramisu-subset dataset described in the fast.ai course at
    half-resolution. This takes about a minute to get to around 90% accuracy on a
    p3.2xlarge.
    """
    if batch:
        run_on_batch()

    # Setup hyperparams.
    bs = 8
    wd = 1e-2
    lr = 2e-3
    num_epochs = 10
    sample_pct = 1.0
    model_arch = models.resnet34
    fp16 = False
    sync_interval = 20 # Don't sync during training for such a small job.
    seed = 1234

    if test:
        bs = 1
        num_epochs = 2
        sample_pct = 0.01
        model_arch = models.resnet18

    # Setup paths.
    data_uri = Path('/opt/data/camvid/CamVid')
    train_uri = Path('/opt/data/camvid/train')
    data_dir = data_uri
    train_dir = train_uri
    if s3_data:
        temp_dir_obj = tempfile.TemporaryDirectory()
        data_uri = 's3://raster-vision-lf-dev/camvid/CamVid.zip'
        train_uri = 's3://raster-vision-lf-dev/camvid/train'
        train_dir = Path(temp_dir_obj.name)/'train'
        data_dir = Path(temp_dir_obj.name)/'data'
    make_dir(train_dir)
    make_dir(data_dir)

    # Retrieve data and remote training directory.
    if s3_data:
        print('Downloading data...')
        data_zip = Path(temp_dir_obj.name)/'CamVid.zip'
        s3_utils.copy_from(data_uri, str(data_zip))
        zip_ref = zipfile.ZipFile(data_zip, 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()
        data_dir = data_dir/'CamVid'

        if s3_utils.list_paths(train_uri):
            print('Syncing train dir...')
            s3_utils.sync_to_dir(train_uri, str(train_dir))

    # Setup data loader.
    def get_y_fn(x):
        return Path(str(x.parent)+'annot')/x.name

    fnames = get_image_files(data_dir/'val')
    img = open_image(fnames[0])

    src_size = np.array(img.data.shape[1:])
    size = src_size // 2

    data = (SegmentationItemList.from_folder(data_dir)
               .use_partial_data(sample_pct, seed)
               .split_by_folder(valid='val')
               .label_from_func(get_y_fn, classes=codes)
               .transform(get_transforms(), size=size, tfm_y=True)
               .databunch(bs=bs)
               .normalize(imagenet_stats))

    # Setup metrics, callbacks, and then train model.
    metrics = [acc_camvid]
    model_path = train_dir/'stage-1'
    log_path = train_dir/'log'
    learn = unet_learner(data, model_arch, metrics=metrics, wd=wd, bottle=True)
    learn.unfreeze()
    if fp16 and torch.cuda.is_available():
        learn = learn.to_fp16()

    start_epoch = 1
    if os.path.isfile(str(model_path) + '.pth'):
        print('Loading saved model...')
        start_epoch = get_last_epoch(str(log_path) + '.csv') + 1
        if start_epoch > num_epochs:
            print('Training already done. If you would like to re-train, delete '
                  'previous results of training in {}.'.format(train_dir))
            exit()

        learn.load(model_path)
        print('Resuming from epoch {}'.format(start_epoch))
        print('Note: fastai does not support a start_epoch, so epoch 1 below '
              'corresponds to {}'.format(start_epoch))

    callbacks = [
        SaveModelCallback(learn, name=model_path),
        MyCSVLogger(learn, filename=log_path, start_epoch=start_epoch)
    ]
    if s3_data:
        callbacks.append(S3SyncCallback(train_dir, train_uri, sync_interval))

    epochs_left = num_epochs - start_epoch + 1
    lrs = slice(lr/100, lr)
    learn.fit_one_cycle(epochs_left, lrs, pct_start=0.8, callbacks=callbacks)

    if s3_data:
        s3_utils.sync_from_dir(train_dir, train_uri)


if __name__ == '__main__':
    train()

'''
learn.load('stage-1')

learn.unfreeze()
lrs = slice(lr/100, lr)
learn.fit_one_cycle(num_epochs, lrs, pct_start=0.8)
learn.save('stage-2')
'''

'''
# Go big
learn = None
gc.collect()

size = src_size
bs = 8

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, bottle=True)
learn.load('stage-2')

lr = 1e-3
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)
learn.save('stage-1-big')
learn.load('stage-1-big')
learn.unfreeze()

lrs = slice(lr/1000, lr/10)
learn.fit_one_cycle(10, lrs)
learn.save('stage-2-big')
'''
