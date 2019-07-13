import json
import uuid
from os.path import join
import shutil
import tempfile

import numpy as np
import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastai.vision import (
    get_image_files, SegmentationItemList, open_image, get_transforms,
    imagenet_stats)
from fastai.basic_train import Learner
from fastai.callbacks import CSVLogger, SaveModelCallback

from mlx.batch_utils import submit_job
from mlx.filesystem.utils import (
    make_dir, sync_to_dir, zipdir, unzip, download_if_needed)
from mlx.semseg.fpn import SegmentationFPN

local_data_uri = '/opt/data/camvid/CamVid'
remote_data_uri = 's3://raster-vision-lf-dev/camvid/CamVid.zip'
local_output_uri = '/opt/data/camvid/output'
remote_output_uri = 's3://raster-vision-lf-dev/camvid/output'


codes = np.array(['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree',
                  'Sign', 'Fence', 'Car', 'Pedestrian', 'Cyclist', 'Void'])
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    # Note: custom metrics need to be at global level for learner to be saved.
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def run_on_batch(test, debug):
    job_name = 'mlx_train_fpn-' + str(uuid.uuid4())
    job_def = 'lewfishPyTorchCustomGpuJobDefinition'
    job_queue = 'lewfishRasterVisionGpuJobQueue'
    cmd_list = ['python', '-m', 'mlx.semseg.train_fpn', '--s3-data']
    if debug:
        cmd_list = [
            'python', '-m', 'ptvsd', '--host', '0.0.0.0', '--port', '6006',
            '--wait', '-m', 'mlx.od.train', '--s3-data']
    if test:
        cmd_list.append('--test')
    submit_job(job_name, job_def, job_queue, cmd_list)
    exit()

def plot_data(data, output_dir, max_per_split=50):
    def _plot_data(split):
        debug_chips_dir = join(output_dir, '{}-debug-chips'.format(split))
        zip_path = join(output_dir, '{}-debug-chips.zip'.format(split))
        make_dir(debug_chips_dir, force_empty=True)

        ds = data.train_ds if split == 'train' else data.valid_ds
        for i, (x, y) in enumerate(ds):
            if i == max_per_split:
                break
            x.show(y=y)
            plt.savefig(join(debug_chips_dir, '{}.png'.format(i)),
                        figsize=(3, 3))
            plt.close()
        zipdir(debug_chips_dir, zip_path)
        shutil.rmtree(debug_chips_dir)

    _plot_data('train')
    _plot_data('val')

def plot_preds(data, learn, output_dir, max_plots=50):
    preds_dir = join(output_dir, 'preds')
    zip_path = join(output_dir, 'preds.zip')
    make_dir(preds_dir, force_empty=True)

    ds = data.valid_ds
    for i, (x, y) in enumerate(ds):
        if i == max_plots:
            break
        z = learn.predict(x)
        x.show(y=z[0])
        plt.savefig(join(preds_dir, '{}.png'.format(i)), figsize=(3, 3))
        plt.close()

    zipdir(preds_dir, zip_path)
    shutil.rmtree(preds_dir)

def download_data(s3_data, tmp_dir):
    if not s3_data:
        return local_data_uri

    data_zip_uri = remote_data_uri
    data_zip_path = download_if_needed(data_zip_uri, tmp_dir)
    unzip(data_zip_path, tmp_dir)
    data_dir = join(tmp_dir, 'CamVid')
    return data_dir

def get_databunch(data_dir, batch_sz=8, num_workers=4,
                  sample_pct=1.0, seed=1234):
    def get_y_fn(x):
        return join(str(x.parent)+'annot', x.name)

    fnames = get_image_files(join(data_dir, 'test'))
    img = open_image(fnames[0])
    src_size = np.array(img.data.shape[1:])
    size = src_size // 2

    data = (SegmentationItemList.from_folder(data_dir)
               .use_partial_data(sample_pct, seed)
               .split_by_folder(valid='val')
               .label_from_func(get_y_fn, classes=codes)
               .transform(get_transforms(), size=size, tfm_y=True)
               .databunch(bs=batch_sz, num_workers=num_workers)
               .normalize(imagenet_stats))

    return data

@click.command()
@click.option('--test', is_flag=True, help='Run small test experiment')
@click.option('--s3-data', is_flag=True, help='Use data and store results on S3')
@click.option('--batch', is_flag=True, help='Submit Batch job for full experiment')
@click.option('--debug', is_flag=True, help='Run via debugger')
def main(test, s3_data, batch, debug):
    """Train a semantic segmentation FPN model on the CamVid-Tiramisu dataset."""
    if batch:
        run_on_batch(test, debug)

    # Setup options
    batch_sz = 8
    num_workers = 4
    num_epochs = 20
    lr = 1e-4
    backbone_arch = 'resnet18'
    sample_pct = 1.0

    if test:
        batch_sz = 1
        num_workers = 0
        num_epochs = 2
        sample_pct = 0.01

    # Setup data
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    output_dir = local_output_uri
    make_dir(output_dir)

    data_dir = download_data(s3_data, tmp_dir)
    data = get_databunch(
        data_dir, sample_pct=sample_pct, batch_sz=batch_sz, num_workers=num_workers)
    print(data)
    plot_data(data, output_dir)

    # Setup and train model
    num_classes = data.c
    model = SegmentationFPN(backbone_arch, num_classes)
    metrics = [acc_camvid]
    learn = Learner(
        data, model, metrics=metrics, loss_func=SegmentationFPN.loss,
        path=output_dir)
    learn.unfreeze()

    callbacks = [
        SaveModelCallback(learn, monitor='valid_loss'),
        CSVLogger(learn, filename='log'),
    ]

    learn.fit_one_cycle(num_epochs, lr, callbacks=callbacks)

    # Plot predictions and sync
    plot_preds(data, learn, output_dir)

    if s3_data:
        sync_to_dir(output_dir, remote_output_uri)

if __name__ == '__main__':
    main()