from os.path import join
import tempfile

import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mlx.filesystem.utils import json_to_file

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
        gt = get_coco_gt(targets, num_labels)
        gt_path = join(tmp_dir, 'gt.json')
        json_to_file(gt, gt_path)
        coco_gt = COCO(gt_path)

        pycocotools.coco.unicode = None
        preds = get_coco_preds(outputs)
        coco_preds = coco_gt.loadRes(preds)
        ann_type = 'bbox'
        coco_eval = COCOeval(coco_gt, coco_preds, ann_type)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats[0]
