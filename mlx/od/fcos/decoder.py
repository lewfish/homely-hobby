import torch
import math

from mlx.od.boxlist import BoxList

def decode_level_output(reg_arr, label_arr, center_arr, stride, score_thresh=0.05):
    """Decode output of head for one level of the pyramid for one image.

    Args:
        reg_arr: (tensor) with shape (4, h, w). The first dimension ranges over
            t, l, b, r (ie. top, left, bottom, right).
        label_arr: (tensor) with shape (num_labels, h, w) containing
            probabilities
        center_arr: (tensor) with shape (1, h, w) containing values between
            0 and 1
        stride: (int) the stride of the level of the pyramid
        score_thresh: (float) probability score threshold used to determine
            if a box is present at a cell

    Returns:
        BoxList
    """
    device = reg_arr.device
    h, w = reg_arr.shape[1:]
    pos_arr = torch.empty((2, h, w), device=device)
    pos_arr[0, :, :] = torch.arange(
        stride//2, stride * h, stride, device=device)[:, None]
    pos_arr[1, :, :] = torch.arange(
        stride//2, stride * w, stride, device=device)[None, :]

    boxes = torch.empty((4, h, w), device=device)
    boxes[0:2, :, :] = pos_arr - reg_arr[0:2, :, :]
    boxes[2:, :, :] = pos_arr + reg_arr[2:, :, :]

    scores, labels = torch.max(label_arr, dim=0)

    boxes = boxes.reshape(4, -1).transpose(1, 0)
    labels = labels.reshape(-1)
    scores = scores.reshape(-1)
    centerness = center_arr.reshape(-1)
    return BoxList(boxes, labels=labels, scores=scores, centerness=centerness).score_filter(score_thresh)

def decode_single_output(output, pyramid_shape, score_thresh=0.05):
    """Decode output of heads for all levels of pyramid for one image.

    Args:
        output: list of tuples where each tuple corresponds to a pyramid level
            tuple is of form (reg_arr, label_arr, center_arr) where
                - reg_arr is tensor<4, h, w>,
                - label_arr is tensor<num_labels, h, w>
                - center_arr is tensor<1, h, w>
            and label_arr and center_arr are probabilities
        score_thresh: (float) probability score threshold used to determine
            if a box is present at a cell

    Returns:
        BoxList
    """
    boxlists = []
    for level, level_out in enumerate(output):
        stride = pyramid_shape[level][0]
        boxlist = decode_level_output(
            *level_out, stride, score_thresh=score_thresh)
        boxlists.append(boxlist)
    return BoxList.cat(boxlists)

def decode_batch_output(output, pyramid_shape, img_height, img_width,
                        iou_thresh=0.5):
    """Decode output for batch of images.

    Args:
        output: list of tuples where each tuple corresponds to a pyramid level
            tuple is of form (reg_arr, label_arr, center_arr) where
                - reg_arr is tensor<n, 4, h, w>,
                - label_arr is tensor<n, num_labels, h, w>
                - center_arr is tensor<n, 1, h, w>
            and label_arr and center_arr are logits
        pyramid_shape:
        img_height:
        img_width:
        iou_thresh: (float) iou threshold passed to NMS

    Returns:
        list of n BoxLists
    """
    boxlists = []
    batch_sz = output[0][0].shape[0]
    for i in range(batch_sz):
        single_head_out = []
        for level, (reg_arr, label_arr, center_arr) in enumerate(output):
            # Convert logits in label_arr and center_arr to probabilities.
            single_head_out.append((
                reg_arr[i],
                torch.sigmoid(label_arr[i]),
                torch.sigmoid(center_arr[i])))
        boxlist = decode_single_output(single_head_out, pyramid_shape)
        boxlist = BoxList(
            boxlist.boxes, labels=boxlist.get_field('labels'),
            scores=boxlist.get_field('scores') * boxlist.get_field('centerness'),
            centerness=boxlist.get_field('centerness'))
        boxlist = boxlist.clamp(img_height, img_width)
        boxlist = boxlist.nms(iou_thresh=iou_thresh)
        boxlists.append(boxlist)
    return boxlists