import torch

def decode_level_output(reg_arr, label_arr, stride, score_thresh=0.05):
    """Decode output of head for one level of the pyramid for one image.

    Args:
        reg_arr: (tensor) with shape (4, h, w). The first dimension ranges over
            t, l, b, r (ie. top, left, bottom, right).
        label_arr: (tensor) with shape (num_labels, h, w).
        stride: (int) the stride of the level of the pyramid
        score_thresh: (float) score threshold used to determine if a box is
            present at a cell

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
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    keep_inds = scores > score_thresh
    return boxes[keep_inds, :], labels[keep_inds], scores[keep_inds]

def decode_output(output, score_thresh=0.05):
    """Decode output of heads for all levels of pyramid for one image.

    Args:
        output: (dict) where keys are strides, values are dicts of form
            {'reg_arr': <tensor>, 'label_arr': <tensor>} where reg_arr.shape is
            (4, h, w) and label_arr.shape is (num_labels, h, w)
        score_thresh: (float) score threshold used to determine if a box is
            present at a cell

    Returns:
        (boxes, labels, scores)
    """
    all_boxes, all_labels, all_scores = [], [], []
    for stride, arrs in output.items():
        reg_arr = arrs['reg_arr']
        label_arr = arrs['label_arr']
        boxes, labels, scores = decode_level_output(
            reg_arr, label_arr, stride, score_thresh=score_thresh)
        all_boxes.append(boxes)
        all_labels.append(labels)
        all_scores.append(scores)

    return torch.cat(all_boxes), torch.cat(all_labels), torch.cat(all_scores)
