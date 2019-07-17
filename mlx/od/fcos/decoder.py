import torch

def decode(reg_arr, label_arr, stride, score_thresh=0.05):
    h, w = reg_arr.shape[1:]
    pos_arr = torch.empty((2, h, w))
    pos_arr[0, :, :] = torch.arange(stride//2, stride * h, stride)[:, None]
    pos_arr[1, :, :] = torch.arange(stride//2, stride * w, stride)[None, :]

    boxes = torch.empty((4, h, w))
    boxes[0:2, :, :] = pos_arr - reg_arr[0:2, :, :]
    boxes[2:, :, :] = pos_arr + reg_arr[2:, :, :]

    scores, labels = torch.max(label_arr, dim=0)

    boxes = boxes.reshape(4, -1).transpose(1, 0)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    keep_inds = scores > score_thresh
    return boxes[keep_inds, :], labels[keep_inds], scores[keep_inds]

def decode_targets(targets, score_thresh=0.05):
    all_boxes, all_labels, all_scores = [], [], []
    for stride, arrs in targets.items():
        reg_arr = arrs['reg_arr']
        label_arr = arrs['label_arr']
        boxes, labels, scores = decode(
            reg_arr, label_arr, stride, score_thresh=score_thresh)
        all_boxes.append(boxes)
        all_labels.append(labels)
        all_scores.append(scores)

    return torch.cat(all_boxes), torch.cat(all_labels), torch.cat(all_scores)
