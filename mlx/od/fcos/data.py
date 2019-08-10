from os.path import join, isdir, dirname, basename
import json
import tempfile

import numpy as np
from fastai.vision import (
   URLs, get_annotations, ObjectItemList, untar_data, imagenet_stats,
   flip_affine, bb_pad_collate, ResizeMethod, imagenet_stats)

from mlx.filesystem.utils import (
    get_local_path, make_dir, sync_to_dir, sync_from_dir, zipdir, unzip,
    download_if_needed)

pascal2007 = 'pascal2007'
penn_fudan = 'penn-fudan'
datasets = [pascal2007, penn_fudan]

def validate_dataset(dataset):
    if dataset not in datasets:
        raise ValueError('dataset {} is invalid'.format(dataset))

def setup_output_dir(cfg, tmp_dir):
    output_uri = cfg.output_uri
    if not output_uri.startswith('s3://'):
        return output_uri

    output_dir = get_local_path(output_uri, tmp_dir)
    make_dir(output_dir, force_empty=True)
    sync_from_dir(output_uri, output_dir)
    return output_dir

def build_databunch(cfg, tmp_dir):
    dataset = cfg.data.dataset
    validate_dataset(dataset)

    img_sz = cfg.data.img_sz
    batch_sz = cfg.solver.batch_sz
    num_workers = 4

    if cfg.test_mode or cfg.overfit_mode:
        img_sz = cfg.data.img_sz // 2
        batch_sz = 4

    if cfg.test_mode:
        num_workers = 0

    data_uri = cfg.data.data_uri
    if cfg.data.dataset == pascal2007:
        data_dir = data_uri
        if data_uri.startswith('s3://'):
            data_dir = join(tmp_dir, 'pascal2007-data')
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

        ann_path = trn_path
        img_dir = data_dir
    elif cfg.data.dataset == penn_fudan:
        data_dir = data_uri
        if data_uri.startswith('s3://'):
            data_dir = join(tmp_dir, 'penn-fudan/data')
            zip_path = download_if_needed(data_uri, tmp_dir)
            unzip(zip_path, data_dir)

        ann_path = join(data_dir, 'coco.json')
        images, lbl_bbox = get_annotations(ann_path)
        img2bbox = dict(zip(images, lbl_bbox))
        get_y_func = lambda o: img2bbox[o.name]

        img_dir = join(data_dir, 'PNGImages')

    with open(ann_path) as f:
        d = json.load(f)
        classes = sorted(d['categories'], key=lambda x: x['id'])
        classes = [x['name'] for x in classes]
        classes = ['background'] + classes

    def get_databunch(full=True):
        src = ObjectItemList.from_folder(img_dir, presort=True)

        if cfg.overfit_mode:
            # Don't use any validation set so training will run faster.
            src = src.split_by_idxs(np.arange(0, 4), [])
        elif cfg.test_mode:
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
        if not cfg.overfit_mode:
            train_transforms = [flip_affine(p=0.5)]
        src = src.transform(
            tfms=[train_transforms, val_transforms], size=img_sz, tfm_y=True,
            resize_method=ResizeMethod.SQUISH)
        data = src.databunch(path=data_dir, bs=batch_sz, collate_fn=bb_pad_collate,
                             num_workers=num_workers)
        data.normalize(imagenet_stats)
        data.classes = classes
        return data

    return get_databunch(full=False), get_databunch(full=True)