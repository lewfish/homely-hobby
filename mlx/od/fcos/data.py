from os.path import join, isdir, dirname, basename
import json
import tempfile

import numpy as np
from fastai.vision import (
   URLs, get_annotations, ObjectItemList, untar_data, imagenet_stats,
   flip_affine, bb_pad_collate, ResizeMethod)

from mlx.filesystem.utils import (
    make_dir, sync_to_dir, sync_from_dir, zipdir, unzip, download_if_needed)

pascal2007 = 'pascal2007'
penn_fudan = 'penn-fudan'
datasets = [pascal2007, penn_fudan]

output_config = {
    pascal2007: {
        'output_uri': 's3://raster-vision-lf-dev/pascal2007/output-1-100-448',
        'output_dir': '/opt/data/pascal2007/output-test/'
    },
    penn_fudan: {
        'output_uri': 's3://raster-vision-lf-dev/penn-fudan/output-overfit',
        'output_dir': '/opt/data/penn-fudan/output/'
    }
}

def validate_dataset(dataset):
    if dataset not in datasets:
        raise ValueError('dataset {} is invalid'.format(dataset))

def setup_output(dataset, s3_data=False):
    validate_dataset(dataset)
    output_uri = output_config[dataset]['output_uri']
    output_dir = output_config[dataset]['output_dir']
    make_dir(output_dir)
    if s3_data:
        make_dir(output_dir, force_empty=True)
        sync_from_dir(output_uri, output_dir)
    return output_dir, output_uri

def get_databunch(dataset, test=False, overfit=False):
    validate_dataset(dataset)
    if dataset == pascal2007:
        return get_pascal_databunch(test, overfit)
    elif dataset == penn_fudan:
        return get_penn_fudan_databunch(test, overfit)

def get_pascal_databunch(test=False, overfit=False):
    img_sz = 448
    batch_sz = 8
    num_workers = 4

    if test:
        img_sz = 224
        batch_sz = 4
        num_workers = 0
    if overfit:
        img_sz = 224
        batch_sz = 4

    data_dir = '/opt/data/pascal2007/data'
    untar_data(URLs.PASCAL_2007, dest=data_dir)
    data_dir = join(data_dir, 'pascal_2007')

    trn_path = join(data_dir, 'train.json')
    trn_images, trn_lbl_bbox = get_annotations(trn_path)
    val_path = join(data_dir, 'valid.json')
    val_images, val_lbl_bbox = get_annotations(val_path)
    test_path = join(data_dir, 'test.json')
    test_images, test_lbl_bbox = get_annotations(test_path)

    images, lbl_bbox = trn_images+val_images+test_images, trn_lbl_bbox+val_lbl_bbox+test_lbl_bbox
    img2bbox = dict(zip(images, lbl_bbox))
    get_y_func = lambda o: img2bbox[o.name]

    with open(trn_path) as f:
        d = json.load(f)
        classes = sorted(d['categories'], key=lambda x: x['id'])
        classes = [x['name'] for x in classes]
        classes = ['background'] + classes

    def get_databunch(full=True):
        src = ObjectItemList.from_folder(data_dir, presort=True)
        if overfit:
            # Don't use any validation set so training will run faster.
            src = src.split_by_idxs(np.arange(0, 4), [])
        elif test:
            src = src.split_by_idxs(np.arange(0, 4), np.arange(0, 4))
        else:
            def file_filter(path):
                fn = basename(str(path))
                return fn in trn_images or fn in val_images or fn in test_images[0:500]
            if not full:
                src = src.filter_by_func(file_filter)
            src = src.split_by_files(test_images)
        src = src.label_from_func(get_y_func, classes=classes)
        train_transforms, val_transforms = [], []
        if not overfit:
            train_transforms = [flip_affine(p=0.5)]
        src = src.transform(
            tfms=[train_transforms, val_transforms], size=img_sz, tfm_y=True,
            resize_method=ResizeMethod.SQUISH)
        data = src.databunch(path=data_dir, bs=batch_sz, collate_fn=bb_pad_collate,
                             num_workers=num_workers)
        data.classes = classes
        return data

    return get_databunch(full=False), get_databunch(full=True)

def get_penn_fudan_databunch(test=False, overfit=False):
    img_sz = 224
    batch_sz = 32
    num_workers = 4

    if test:
        img_sz = 224
        batch_sz = 4
        num_workers = 0
    if overfit:
        img_sz = 224
        batch_sz = 4

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

    with open(coco_path) as f:
        d = json.load(f)
        classes = sorted(d['categories'], key=lambda x: x['id'])
        classes = [x['name'] for x in classes]
        classes = ['background'] + classes

    src = ObjectItemList.from_folder(img_path, presort=True)
    if overfit:
        # Don't use any validation set so training will run faster.
        src = src.split_by_idxs(np.arange(4, 8), [])
    elif test:
        src = src.split_by_idxs(
            np.arange(0, batch_sz), np.arange(batch_sz, batch_sz * 2))
    else:
        src = src.split_by_idxs(
            np.arange(30, len(images)), np.arange(0, 30))

    src = src.label_from_func(get_y_func, classes=classes)
    train_transforms = [flip_affine(p=0.5)]
    val_transforms = []
    src = src.transform(
        tfms=[train_transforms, val_transforms], size=img_sz, tfm_y=True,
        resize_method=ResizeMethod.SQUISH)
    data = src.databunch(path=data_dir, bs=batch_sz, collate_fn=bb_pad_collate,
                         num_workers=num_workers)
    data.classes = classes
    return data, data