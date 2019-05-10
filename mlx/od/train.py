from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from fastai.vision import (
    get_annotations, ObjectItemList, ObjectCategoryList, get_transforms,
    bb_pad_collate)
from fastai.basic_train import Learner

from mlx.od.utils import ObjectDetectionGrid, BoxList
from mlx.od.model import ObjectDetectionModel
from mlx.utils import make_dir

# Setup options
debug = True

bs = 16
size = 256
num_workers = 4
num_epochs = 10
lr = 1e-4

if debug:
    bs = 4
    size = 128
    num_debug_images = 32
    num_workers = 0
    num_epochs = 1

# Setup data
path = Path('/opt/data/pascal2007')
out_dir = path/'local-output'
make_dir(out_dir)
img_path = path/'VOCdevkit/VOC2007/JPEGImages'

trn_path = path/'PASCAL_VOC/pascal_train2007.json'
trn_images, trn_lbl_bbox = get_annotations(trn_path)
val_path = path/'PASCAL_VOC/pascal_val2007.json'
val_images, val_lbl_bbox = get_annotations(val_path)

images, lbl_bbox = trn_images+val_images, trn_lbl_bbox+val_lbl_bbox
img2bbox = dict(zip(images, lbl_bbox))
get_y_func = lambda o: img2bbox[o.name]

with open(trn_path) as f:
    d = json.load(f)
    classes = sorted(d['categories'], key=lambda x: x['id'])
    classes = [x['name'] for x in classes]
    num_classes = len(classes)

# Subtract 2 because there's no padding on final convolution
grid_sz = 4 - 2
anc_sizes = torch.tensor([
    [2, 0.5],
    [0.5, 2],
    [1, 1]])
grid = ObjectDetectionGrid(grid_sz, anc_sizes, num_classes)
score_thresh = 0.1
iou_thresh = 0.5

class MyObjectCategoryList(ObjectCategoryList):
    def analyze_pred(self, pred):
        boxes, labels, scores = grid.decode(pred.unsqueeze(0))
        bl = BoxList(boxes[0], labels[0], scores[0])
        bl = bl.score_filter(score_thresh=score_thresh).nms(iou_thresh=iou_thresh)
        return [bl.boxes, bl.labels]

class MyObjectItemList(ObjectItemList):
    _label_cls = MyObjectCategoryList

def get_data(bs, size):
    src = MyObjectItemList.from_folder(img_path)
    if debug:
        src = src[0:num_debug_images]
    src = src.split_by_files(val_images)
    src = src.label_from_func(get_y_func, classes=classes)
    src = src.transform(get_transforms(), size=size, tfm_y=True)
    return src.databunch(path=path, bs=bs, collate_fn=bb_pad_collate,
                         num_workers=num_workers)

data = get_data(bs, size)
print(data)
data.show_batch()
plt.savefig(out_dir/'data.png')

# Setup model
model = ObjectDetectionModel(grid)

def loss(out, gt_boxes, gt_classes):
    gt = model.grid.encode(gt_boxes, gt_classes)
    box_loss, class_loss = model.grid.compute_losses(out, gt)
    return box_loss + class_loss

learn = Learner(data, model, loss_func=loss)
model.freeze_body()
learn.fit(num_epochs, lr)

learn.show_results()
plt.savefig(out_dir/'preds.png')