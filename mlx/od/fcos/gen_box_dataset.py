from os.path import join

import click
from PIL import Image
import numpy as np

from mlx.filesystem.utils import make_dir, json_to_file

def make_scene(img_size, max_boxes):
    img = np.zeros((3, img_size, img_size), dtype=np.uint8)
    num_boxes = np.random.randint(0, max_boxes+1)
    boxes = np.empty((num_boxes, 4))

    for i in range(num_boxes):
        yx = np.random.randint(0, img_size-img_size//10, (2,))
        hw = np.array([
            np.random.randint(img_size//10, img.shape[1] - yx[0]),
            np.random.randint(img_size//10, img.shape[2] - yx[1])])
        box = np.concatenate((yx, yx+hw))
        boxes[i, :] = box
        img[:, box[0], box[1]:box[3]] = 255
        img[:, box[2], box[1]:box[3]] = 255
        img[:, box[0]:box[2], box[1]] = 255
        img[:, box[0]:box[2], box[3]] = 255

    return img, boxes

@click.command()
@click.argument('output_dir')
@click.option('--img-size', default=300)
@click.option('--max-boxes', default=3)
@click.option('--train-size', default=10)
@click.option('--val-size', default=10)
def main(output_dir, img_size, max_boxes, train_size, val_size):
    """Generate synthetic box dataset in Coco-format for testing obj det."""
    im_id = 0
    ann_id = 0

    def make_split(split, split_size):
        nonlocal im_id
        nonlocal ann_id

        split_dir = join(output_dir, split)
        make_dir(split_dir)

        images = []
        annotations = []
        for _ in range(split_size):
            img, boxes = make_scene(img_size, max_boxes)
            img = np.transpose(img, (1, 2, 0))
            file_name = '{}.png'.format(im_id)
            Image.fromarray(img).save(
                join(split_dir, file_name))
            images.append({
                'id': im_id,
                'height': img_size,
                'width': img_size,
                'file_name': file_name
            })
            for box in boxes:
                annotations.append({
                    'id': ann_id,
                    'image_id': im_id,
                    'category_id': 1,
                    'area': (box[2] - box[0]) * (box[3] - box[1]),
                    'bbox': [box[1], box[0], box[3]-box[1], box[2]-box[0]]
                })
                ann_id += 1
            im_id += 1

        categories = [{'id': 1, 'name': 'rectangle'}]
        labels = {
            'images': images,
            'annotations': annotations,
            'categories': categories}
        json_to_file(labels, join(output_dir, '{}.json'.format(split)))

    make_split('train', train_size)
    make_split('val', val_size)

if __name__ == '__main__':
    main()