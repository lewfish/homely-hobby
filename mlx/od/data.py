from os.path import join
from collections import defaultdict
import random
import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import torchvision
import cv2
from albumentations import (
    BboxParams,
    HorizontalFlip,
    Resize,
    Compose,
    RGBShift,
    ShiftScaleRotate
)
from albumentations.pytorch import ToTensor

from mlx.filesystem.utils import (download_if_needed, get_local_path, make_dir,
                                  sync_from_dir, unzip, file_to_json)
from mlx.od.boxlist import BoxList

pascal2007 = 'pascal2007'
datasets = [pascal2007]

def validate_dataset(dataset):
    if dataset not in datasets:
        raise ValueError('dataset {} is invalid'.format(dataset))

def setup_output_dir(cfg, tmp_dir):
    if not cfg.output_uri.startswith('s3://'):
        make_dir(cfg.output_uri)
        return cfg.output_uri

    output_dir = get_local_path(cfg.output_uri, tmp_dir)
    make_dir(output_dir, force_empty=True)
    if not cfg.overfit_mode:
        sync_from_dir(cfg.output_uri, output_dir)
    return output_dir

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

def collate_fn(data):
    x = [d[0].unsqueeze(0) for d in data]
    y = [d[1] for d in data]
    return (torch.cat(x), y)

class CocoDataset(Dataset):
    def __init__(self, img_dir, annotation_uris, transforms=None):
        self.img_dir = img_dir
        self.annotation_uris = annotation_uris
        self.transforms = transforms

        self.imgs = []
        self.img2id = {}
        self.id2img = {}
        self.id2boxes = defaultdict(lambda: [])
        self.id2labels = defaultdict(lambda: [])
        self.label2name = {}
        for annotation_uri in annotation_uris:
            ann_json = file_to_json(annotation_uri)
            for img in ann_json['images']:
                self.imgs.append(img['file_name'])
                self.img2id[img['file_name']] = img['id']
                self.id2img[img['id']] = img['file_name']
            for ann in ann_json['annotations']:
                img_id = ann['image_id']
                box = ann['bbox']
                label = ann['category_id']
                self.id2boxes[img_id].append(box)
                self.id2labels[img_id].append(label)

        random.seed(1234)
        random.shuffle(self.imgs)
        self.id2boxes = dict([(id, boxes) for id, boxes in self.id2boxes.items()])
        self.id2labels = dict([(id, labels) for id, labels in self.id2labels.items()])

    def __getitem__(self, ind):
        img_fn = self.imgs[ind]
        img_id = self.img2id[img_fn]
        img = np.array(Image.open(join(self.img_dir, img_fn)))
        boxes, labels = self.id2boxes[img_id], self.id2labels[img_id]
        if self.transforms:
            out = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = out['image']
            boxes = torch.tensor(out['bboxes'])
            labels = torch.tensor(out['labels'])

        if len(boxes) > 0:
            x, y, w, h = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
            boxes = torch.cat([y, x, y+h, x+w], dim=1)
            boxlist = BoxList(boxes, labels=labels)
        else:
            boxlist = BoxList(torch.empty((0, 4)), labels=torch.empty((0,)))
        return (img, boxlist)

    def __len__(self):
        return len(self.imgs)

def get_label_names(coco_path):
    categories = file_to_json(coco_path)['categories']
    label2name = dict([(cat['id'], cat['name']) for cat in categories])
    labels = ['background'] + [label2name[i] for i in range(1, len(label2name) + 1)]
    return labels

def build_databunch(cfg, tmp_dir):
    dataset = cfg.data.dataset
    validate_dataset(dataset)

    img_sz = cfg.data.img_sz
    batch_sz = cfg.solver.batch_sz
    num_workers = cfg.data.num_workers

    data_cache_dir = '/opt/data/data-cache'
    if cfg.data.dataset == pascal2007:
        if cfg.base_uri.startswith('s3://'):
            data_dir = join(data_cache_dir, 'pascal2007-data')
            if not os.path.isdir(data_dir):
                print('Downloading pascal2007.zip...')
                zip_path = download_if_needed(join(cfg.base_uri, 'pascal2007.zip'), data_cache_dir)
                unzip(zip_path, data_dir)
        else:
            data_dir = join(cfg.base_uri, 'data', 'pascal_2007')
        train_dir = join(data_dir, 'train')
        train_anns = [join(data_dir, 'train.json'), join(data_dir, 'valid.json')]
        test_dir = join(data_dir, 'test')
        test_anns = [join(data_dir, 'test.json')]

    label_names = get_label_names(train_anns[0])

    aug_transforms = []
    if cfg.data.train_aug.hflip:
        aug_transforms.append(HorizontalFlip(p=0.5))
    if cfg.data.train_aug.rgb_shift:
        aug_transforms.append(RGBShift(20, 20, 20, p=0.5))
    if cfg.data.train_aug.shift_scale_rotate:
        aug_transforms.append(ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.3, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT))
    basic_transforms = [Resize(img_sz, img_sz), ToTensor()]
    aug_transforms.extend(basic_transforms)

    bbox_params = BboxParams(format='coco', min_area=0., min_visibility=0.2, label_fields=['labels'])
    aug_transforms = Compose(aug_transforms, bbox_params=bbox_params)
    transforms = Compose(basic_transforms, bbox_params=bbox_params)

    train_ds, valid_ds, test_ds = None, None, None
    if cfg.overfit_mode:
        train_ds = CocoDataset(train_dir, train_anns, transforms=transforms)
        train_ds = Subset(train_ds, range(batch_sz))
        test_ds = train_ds
    elif cfg.test_mode:
        train_ds = CocoDataset(train_dir, train_anns, transforms=aug_transforms)
        train_ds = Subset(train_ds, range(batch_sz))
        valid_ds = train_ds
        test_ds = train_ds
    else:
        train_ds = CocoDataset(train_dir, train_anns, transforms=aug_transforms)
        test_ds = CocoDataset(test_dir, test_anns, transforms=transforms)
        valid_ds = Subset(test_ds, range(len(test_ds.imgs) // 5))

    train_dl = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=batch_sz, num_workers=num_workers, pin_memory=True) \
        if train_ds else None
    valid_dl = DataLoader(valid_ds, collate_fn=collate_fn, batch_size=batch_sz, num_workers=num_workers, pin_memory=True) \
        if valid_ds else None
    test_dl = DataLoader(test_ds, collate_fn=collate_fn, batch_size=batch_sz, num_workers=num_workers, pin_memory=True) \
        if test_ds else None
    return DataBunch(train_ds, train_dl, valid_ds, valid_dl, test_ds, test_dl, label_names)
