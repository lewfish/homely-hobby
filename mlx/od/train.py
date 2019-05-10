from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from fastai.vision import (
    get_annotations, ObjectItemList, get_transforms, bb_pad_collate)
from fastai.basic_train import Learner

from mlx.od.utils import ObjectDetectionGrid
from mlx.od.model import ObjectDetectionModel

# Setup options
remote = False
bs = 4
size = 128
num_workers = 0
num_epochs = 1
lr = 1e-4

# Setup data
path = Path('/opt/data/pascal2007')
img_path = path/'VOCdevkit/VOC2007/JPEGImages'

trn_path = path/'PASCAL_VOC/pascal_train2007.json'
trn_images, trn_lbl_bbox = get_annotations(trn_path)
val_path = path/'PASCAL_VOC/pascal_val2007.json'
val_images, val_lbl_bbox = get_annotations(val_path)

images, lbl_bbox = trn_images+val_images, trn_lbl_bbox+val_lbl_bbox
img2bbox = dict(zip(images, lbl_bbox))
get_y_func = lambda o:img2bbox[o.name]


with open(trn_path) as f:
    d = json.load(f)
    classes = sorted(d['categories'], key=lambda x: x['id'])
    classes = [x['name'] for x in classes]

def get_data(bs, size):
    src = ObjectItemList.from_folder(img_path)
    src = src[0:32]
    src = src.split_by_files(val_images)
    src = src.label_from_func(get_y_func, classes=classes)
    src = src.transform(get_transforms(), size=size, tfm_y=True)
    return src.databunch(path=path, bs=bs, collate_fn=bb_pad_collate,
                         num_workers=num_workers)

data = get_data(bs, size)
print(data)
num_classes = len(data.classes)

# Setup model
# Subtract 2 because there's no padding on final convolution
grid_sz = 4 - 2
anc_sizes = torch.tensor([
    [2, 0.5],
    [0.5, 2],
    [1, 1]])
grid = ObjectDetectionGrid(grid_sz, anc_sizes, num_classes)
model = ObjectDetectionModel(grid)

def loss(out, gt_boxes, gt_classes):
    gt = model.grid.encode(gt_boxes, gt_classes)
    box_loss, class_loss = model.grid.compute_losses(out, gt)
    return box_loss + class_loss

learn = Learner(data, model, loss_func=loss)
model.freeze_body()
learn.fit(num_epochs, lr)