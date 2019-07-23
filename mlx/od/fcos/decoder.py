import torch
import math

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
        (boxes, labels, scores)
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
    keep_inds = scores > score_thresh
    return (boxes[keep_inds, :], labels[keep_inds], scores[keep_inds],
            centerness[keep_inds])

def decode_output(output, score_thresh=0.05):
    """Decode output of heads for all levels of pyramid for one image.

    Args:
        output: (dict) where keys are strides and values are dicts of form
            {'reg_arr': <tensor of shape (4, h, w)>,
             'label_arr': <tensor of shape (num_labels, h, w)>,
             'center_arr': <tensor of shape (1, h, w)>}
            and label and center values are between 0 and 1
        score_thresh: (float) probability score threshold used to determine
            if a box is present at a cell

    Returns:
        (boxes, labels, scores)
    """
    all_boxes, all_labels, all_scores, all_centerness = [], [], [], []
    for stride, level_out in output.items():
        reg_arr = level_out['reg_arr']
        label_arr = level_out['label_arr']
        center_arr = level_out['center_arr']
        boxes, labels, scores, centerness = decode_level_output(
            reg_arr, label_arr, center_arr, stride, score_thresh=score_thresh)
        all_boxes.append(boxes)
        all_labels.append(labels)
        all_scores.append(scores)
        all_centerness.append(centerness)

    return (torch.cat(all_boxes), torch.cat(all_labels), torch.cat(all_scores),
            torch.cat(all_centerness))
