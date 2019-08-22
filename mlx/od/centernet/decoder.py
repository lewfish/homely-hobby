import torch
import torch.nn.functional as F

from mlx.od.boxlist import BoxList

def decode(keypoint, reg, positions, stride, prob_thresh=0.05):
    N = keypoint.shape[0]
    boxlists = []
    flat_positions = positions.permute((1, 2, 0)).reshape((-1, 2))
    img_height = positions.shape[1] * stride
    img_width = positions.shape[2] * stride

    for n in range(N):
        per_keypoint = keypoint[n]
        per_reg = reg[n]
        num_labels = per_keypoint.shape[0]

        is_local_max = per_keypoint == F.max_pool2d(
            per_keypoint, kernel_size=3, stride=1, padding=1)
        is_over_thresh = per_keypoint > prob_thresh
        is_pos = is_local_max * is_over_thresh
        num_pos = is_pos.sum()

        if num_pos == 0:
            bl = BoxList(
                torch.empty((0, 4)), labels=torch.empty((0,)),
                scores=torch.empty((0,)))
        else:
            flat_is_pos, _ = is_pos.permute((1, 2, 0)).reshape((-1, num_labels)).max(1)
            flat_per_reg = per_reg.permute((1, 2, 0)).reshape((-1, 2))
            flat_per_reg = flat_per_reg[flat_is_pos, :]
            sizes = flat_per_reg
            centers = flat_positions[flat_is_pos]
            boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], dim=1)

            flat_per_keypoint = per_keypoint.permute((1, 2, 0)).reshape((-1, num_labels))
            flat_per_keypoint = flat_per_keypoint[flat_is_pos, :]
            scores, labels = flat_per_keypoint.max(1)

            bl = BoxList(boxes, labels=labels, scores=scores)
            bl.clamp(img_height, img_width)
        boxlists.append(bl)
    return boxlists
