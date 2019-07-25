from os.path import join
import os

import click
from PIL import Image
import numpy as np

from mlx.filesystem.utils import make_dir, json_to_file

@click.command()
@click.argument('data_dir')
@click.argument('coco_json_path')
def main(data_dir, coco_json_path):
    """Convert Penn-Fudan pedestrian dataset to COCO JSON format."""
    im_id = 0
    ann_id = 0
    img_fns = list(sorted(os.listdir(join(data_dir, 'PNGImages'))))
    mask_fns = list(sorted(os.listdir(join(data_dir, 'PedMasks'))))
    images = []
    annotations = []

    for img_fn, mask_fn in zip(img_fns, mask_fns):
        # Code adapted from https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=mTgWtixZTs3X
        mask_path = os.path.join(data_dir, 'PedMasks', mask_fn)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            box = [xmin, ymin, xmax, ymax]
            box = [int(x) for x in box]
            boxes.append(box)

        images.append({
            'id': im_id,
            'height': mask.shape[0],
            'width': mask.shape[1],
            'file_name': img_fn
        })
        for box in boxes:
            annotations.append({
                'id': ann_id,
                'image_id': im_id,
                'category_id': 1,
                'area': (box[2] - box[0]) * (box[3] - box[1]),
                'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                'iscrowd': 0
            })
            ann_id += 1
        im_id += 1

    categories = [{'id': 1, 'name': 'Human'}]
    coco_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories}

    json_to_file(coco_json, coco_json_path)

if __name__ == '__main__':
    main()