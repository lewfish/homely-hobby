import torch
from torch import nn

from mlx.od.fcos.encoder import encode_targets

def focal_loss(output, target, gamma=2, alpha=0.25):
    """Compute focal loss for label arrays.

    See https://arxiv.org/abs/1708.02002

    Args:
        output: (tensor) with shape (num_labels, h, w). Each value is a logit
            representing a label (ie. class).
        target: (tensor) with same shape as target. Has one if label is present,
            zero otherwise.

    Returns: (tensor) with single float value
    """
    # Add epsilon to avoid overflow when p is 0.0
    epsilon = 0.00001
    p = torch.sigmoid(output) + epsilon
    pt = (1-target) * (1-p) + target * p
    alphat = (1-target) * (1-alpha) + target * alpha
    bce = -torch.log(pt)
    weights = alphat * (1 - pt).pow(gamma)
    loss_arr = weights * bce
    return loss_arr.sum()

# Adapted from following to handle different ordering
# https://github.com/tianzhi0549/FCOS/blob/master/maskrcnn_benchmark/layers/iou_loss.py
class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 1]
        pred_top = pred[:, 0]
        pred_right = pred[:, 3]
        pred_bottom = pred[:, 2]

        target_left = target[:, 1]
        target_top = target[:, 0]
        target_right = target[:, 3]
        target_bottom = target[:, 2]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

def fcos_loss(out, targets):
    """Compute loss for a single image.

    Note: the label_arr and center_arr for output is assumed to contain
    logits, and is assumed to contain probabilities for targets.

    Args:
        out: (dict) the output of the heads for the whole pyramid
        targets: (dict) the encoded targets for the whole pyramid

        the format for both is a dict with keys that are strides (int)
        and values that are (dict) of form {`
            'reg_arr': <tensor with shape (4, h*, w*)>,
            'label_arr': <tensor with shape (num_labels, h*, w*)>,
            'center_arr': <tensor with shape (1, h*, w*)>
        }

    Returns:
        dict of form {
            'reg_loss': <tensor[1]>,
            'label_loss': <tensor[1]>,
            'center_loss': <tensor[1]>
        }
    """
    iou_loss = IOULoss()

    targets_label_arrs = []
    out_label_arrs = []
    targets_reg_arrs = []
    out_reg_arrs = []
    targets_center_arrs = []
    out_center_arrs = []

    for i, s in enumerate(out.keys()):
        num_labels = targets[s]['label_arr'].shape[0]
        targets_label_arr = targets[s]['label_arr'].reshape(num_labels, -1).permute(1, 0)
        out_label_arr = out[s]['label_arr'].reshape(num_labels, -1).permute(1, 0)

        pos_indicator = targets_label_arr.sum(1) > 0.0
        targets_reg_arr = targets[s]['reg_arr'].reshape(4, -1).permute(1, 0)[pos_indicator, :]
        targets_center_arr = targets[s]['center_arr'].reshape(1, -1).permute(1, 0)[pos_indicator, :]
        out_reg_arr = out[s]['reg_arr'].reshape(4, -1).permute(1, 0)[pos_indicator, :]
        out_center_arr = out[s]['center_arr'].reshape(1, -1).permute(1, 0)[pos_indicator, :]

        targets_label_arrs.append(targets_label_arr)
        out_label_arrs.append(out_label_arr)
        targets_reg_arrs.append(targets_reg_arr)
        out_reg_arrs.append(out_reg_arr)
        targets_center_arrs.append(targets_center_arr)
        out_center_arrs.append(out_center_arr)

    targets_label_arr = torch.cat(targets_label_arrs)
    out_label_arr = torch.cat(out_label_arrs)
    targets_reg_arr = torch.cat(targets_reg_arrs)
    out_reg_arr = torch.cat(out_reg_arrs)
    targets_center_arr = torch.cat(targets_center_arrs)
    out_center_arr = torch.cat(out_center_arrs)

    npos = targets_reg_arr.shape[0] + 1
    label_loss = focal_loss(out_label_arr, targets_label_arr) / npos
    reg_loss = torch.tensor(0.0, device=label_loss.device)
    center_loss = torch.tensor(0.0, device=label_loss.device)
    if npos > 1:
        reg_loss = iou_loss(out_reg_arr, targets_reg_arr, targets_center_arr)
        center_loss = nn.functional.binary_cross_entropy_with_logits(
            out_center_arr, targets_center_arr, reduction='mean')

    loss_dict = {'label_loss': label_loss, 'reg_loss': reg_loss, 'center_loss': center_loss}
    return loss_dict

def get_batch_loss(head_out, targets, pyramid_shape, num_labels):
    batch_sz = len(targets)
    for i, single_target in enumerate(targets):
        boxes = single_target['boxes']
        labels = single_target['labels']
        encoded_targets = encode_targets(
            boxes, labels, pyramid_shape, num_labels)

        single_head_out = {}
        for stride in head_out.keys():
            # Don't convert logits to probabilities for output since
            # loss function expects logits for output
            # (and probabilities for targets)
            single_head_out[stride] = {
                'reg_arr': head_out[stride]['reg_arr'][i],
                'label_arr': head_out[stride]['label_arr'][i],
                'center_arr': head_out[stride]['center_arr'][i]
            }
        if i == 0:
            loss_dict = fcos_loss(single_head_out, encoded_targets)
        else:
            ld = fcos_loss(single_head_out, encoded_targets)
            loss_dict['label_loss'] += ld['label_loss']
            loss_dict['reg_loss'] += ld['reg_loss']
            loss_dict['center_loss'] += ld['center_loss']

        loss_dict['label_loss'] /= batch_sz
        loss_dict['reg_loss'] /= batch_sz
        loss_dict['center_loss'] /= batch_sz
    return loss_dict