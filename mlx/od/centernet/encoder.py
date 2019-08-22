import torch

from mlx.od.centernet.utils import get_positions

def get_gaussian2d(positions, center, sigma):
    offsets = positions - center[:, None, None]
    return torch.exp(-(offsets ** 2).sum(0) / (2 * (sigma ** 2)))

def encode(boxlists, positions, stride, num_labels):
    N = len(boxlists)
    device = boxlists[0].boxes.device
    h, w = positions.shape[1:]
    keypoint = torch.zeros((N, num_labels, h, w), device=device)
    reg = torch.zeros((N, 2, h, w), device=device)

    for n, boxlist in enumerate(boxlists):
        # skip offset for now
        boxes = boxlist.boxes
        labels = boxlist.get_field('labels')
        sizes = boxes[:, 2:] - boxes[:, 0:2]
        centers = boxes[:, 0:2] + sizes / 2

        # TODO vectorize this loop
        for center, size, label in zip(centers, sizes, labels):
            sigma = min(size) / 6
            gaussian2d = get_gaussian2d(positions, center, sigma)
            keypoint[n, label, :, :] = torch.max(keypoint[n, label, :, :], gaussian2d)
            y, x = int(center[0] / stride), int(center[1] / stride)
            reg[n, :, y, x] = size

    return keypoint, reg
