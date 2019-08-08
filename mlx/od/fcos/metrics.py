from os.path import join
import tempfile

import torch
from fastai.callback import Callback, add_metrics
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mlx.filesystem.utils import json_to_file
from mlx.od.fcos.utils import to_box_pixel

def get_coco_gt(targets, num_labels):
    images = []
    annotations = []
    ann_id = 1
    for img_id, target in enumerate(targets, 1):
        # Use fake height, width, and filename because they don't matter.
        images.append({
            'id': img_id,
            'height': 1000,
            'width': 1000,
            'file_name': '{}.png'.format(img_id)
        })
        for box, label in zip(target['boxes'], target['labels']):
            box = box.float().tolist()
            label = label.item()
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': label,
                'area': (box[2] - box[0]) * (box[3] - box[1]),
                'bbox': [box[1], box[0], box[3]-box[1], box[2]-box[0]],
                'iscrowd': 0
            })
            ann_id += 1

    categories = [{'id': label, 'name': str(label), 'supercategory': 'super'} for label in range(num_labels)]
    coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    return coco

def get_coco_preds(outputs):
    preds = []
    for img_id, output in enumerate(outputs, 1):
        for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
            box = box.float().tolist()
            label = label.item()
            score = score.item()
            preds.append({
                'image_id': img_id,
                'category_id': label,
                'bbox': [box[1], box[0], box[3]-box[1], box[2]-box[0]],
                'score': score
            })
    return preds

def compute_coco_eval(outputs, targets, num_labels):
    """Return mAP averaged over 0.5-0.95 using pycocotools eval.

    Note: boxes are in (ymin, xmin, ymax, xmax) format with values ranging
        from 0 to h or w.

    Args:
        outputs: (list) of length m containing dicts of form
            {'boxes': <tensor with shape (n, 4)>,
             'labels': <tensor with shape (n,)>,
             'scores': <tensor with shape (n,)>}
        targets: (list) of length m containing dicts of form
            {'boxes': <tensor with shape (n, 4)>,
             'labels': <tensor with shape (n,)>}
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        preds = get_coco_preds(outputs)
        # ap is undefined when there are no predicted boxes
        if len(preds) == 0:
            return -1

        gt = get_coco_gt(targets, num_labels)
        gt_path = join(tmp_dir, 'gt.json')
        json_to_file(gt, gt_path)
        coco_gt = COCO(gt_path)

        pycocotools.coco.unicode = None
        coco_preds = coco_gt.loadRes(preds)

        ann_type = 'bbox'
        coco_eval = COCOeval(coco_gt, coco_preds, ann_type)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats[0]

class CocoMetric(Callback):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.__name__ = 'mAP'

    def on_epoch_begin(self, **kwargs):
        self.outputs = []
        self.targets = []

    def on_batch_begin(self, last_input, last_target, **kwargs):
        self.h, self.w = last_input.shape[2:]

    def on_batch_end(self, last_output, last_target, **kwargs):
        self.outputs.extend(last_output)
        self.targets.append(last_target)

    def on_epoch_end(self, last_metrics, **kwargs):
        # Convert from fastai format
        my_targets = []
        for batch_boxes, batch_labels in self.targets:
            for boxes, labels in zip(batch_boxes, batch_labels):
                non_pad_inds = labels != 0
                boxes = to_box_pixel(boxes, self.h, self.w)
                my_targets.append({
                    'boxes': boxes[non_pad_inds, :],
                    'labels': labels[non_pad_inds]})
        metric = compute_coco_eval(self.outputs, my_targets, self.num_labels)
        return add_metrics(last_metrics, metric)