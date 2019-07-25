from os.path import join, isdir, dirname
import json
import tempfile

import numpy as np
from fastai.vision import (
   URLs, get_annotations, ObjectItemList, untar_data, imagenet_stats,
   flip_affine, bb_pad_collate)

from mlx.filesystem.utils import (
    make_dir, sync_to_dir, zipdir, unzip, download_if_needed)

def setup_data(dataset_name, test):
    if dataset_name == 'pascal2007':
        output_uri = 's3://raster-vision-lf-dev/pascal2007/output-norm-fliplr/'
        output_dir = '/opt/data/pascal2007/output/'
        make_dir(output_dir, force_empty=True)
        databunch = get_pascal_databunch(test)
        return output_dir, output_uri, databunch
    elif dataset_name == 'penn-fudan':
        output_uri = 's3://raster-vision-lf-dev/penn-fudan/output'
        output_dir = '/opt/data/penn-fudan/output/'
        make_dir(output_dir, force_empty=True)
        databunch = get_penn_fudan_databunch(test)
        return output_dir, output_uri, databunch
    else:
        raise ValueError('dataset_name {} is invalid'.format(dataset_name))

def get_pascal_databunch(test):
    img_sz = 416
    batch_sz = 8
    num_workers = 4

    if test:
        img_sz = 200
        batch_sz = 1
        num_workers = 0

    data_dir = '/opt/data/pascal2007/data'
    untar_data(URLs.PASCAL_2007, dest=data_dir)
    data_dir = join(data_dir, 'pascal_2007')

    img_path = join(data_dir, 'train')
    trn_path = join(data_dir, 'train.json')
    trn_images, trn_lbl_bbox = get_annotations(trn_path)
    val_path = join(data_dir, 'valid.json')
    val_images, val_lbl_bbox = get_annotations(val_path)

    images, lbl_bbox = trn_images+val_images, trn_lbl_bbox+val_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    with open(trn_path) as f:
        d = json.load(f)
        classes = sorted(d['categories'], key=lambda x: x['id'])
        classes = [x['name'] for x in classes]
        classes = ['background'] + classes

        src = ObjectItemList.from_folder(img_path)
        if test:
            src = src.split_by_idxs(np.arange(0, 2), np.arange(2, 4))
        else:
            src = src.split_by_files(val_images[0:250])
        src = src.label_from_func(get_y_func, classes=classes)
        train_transforms = [flip_affine(p=0.5)]
        val_transforms = []
        src = src.transform(
            tfms=[train_transforms, val_transforms], size=img_sz, tfm_y=True)
        data = src.databunch(path=data_dir, bs=batch_sz, collate_fn=bb_pad_collate,
                             num_workers=num_workers)
        data = data.normalize(imagenet_stats)
    data.classes = classes
    return data

def get_penn_fudan_databunch(test):
    img_sz = 416
    batch_sz = 8
    num_workers = 4

    if test:
        img_sz = 256
        batch_sz = 1
        num_workers = 0

    data_uri = 's3://raster-vision-lf-dev/penn-fudan/penn-fudan.zip'
    data_dir = '/opt/data/penn-fudan/data'
    if not isdir(data_dir):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = download_if_needed(data_uri, tmp_dir)
            unzip(zip_path, dirname(data_dir))

    img_path = join(data_dir, 'PNGImages')
    coco_path = join(data_dir, 'coco.json')
    images, lbl_bbox = get_annotations(coco_path)
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]
    sorted_images = sorted(images)

    with open(coco_path) as f:
        d = json.load(f)
        classes = sorted(d['categories'], key=lambda x: x['id'])
        classes = [x['name'] for x in classes]
        classes = ['background'] + classes

    src = ObjectItemList.from_folder(img_path)
    if test:
        src = src.split_by_idxs(np.arange(0, 2), np.arange(2, 4))
    else:
        src = src.split_by_files(sorted_images[0:40])
    src = src.label_from_func(get_y_func, classes=classes)
    train_transforms = [flip_affine(p=0.5)]
    val_transforms = []
    src = src.transform(
        tfms=[train_transforms, val_transforms], size=img_sz, tfm_y=True)
    data = src.databunch(path=data_dir, bs=batch_sz, collate_fn=bb_pad_collate,
                         num_workers=num_workers)
    data = data.normalize(imagenet_stats)
    data.classes = classes
    return data