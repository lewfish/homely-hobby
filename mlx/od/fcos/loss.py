import torch
from torch import nn

from mlx.od.fcos.encoder import encode_single_targets

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

def flatten_output(output):
    reg_arrs = []
    label_arrs = []
    center_arrs = []

    for reg_arr, label_arr, center_arr in output:
        batch_sz, num_labels = label_arr.shape[0:2]
        # (N, 4, H, W) -> (N, H, W, 4) -> (N*H*W, 4)
        reg_arrs.append(reg_arr.permute((0, 2, 3, 1)).reshape((-1, 4)))
        # (N, C, H, W) -> (N, H, W, C) -> (N*H*W, C)
        label_arrs.append(label_arr.permute((0, 2, 3, 1)).reshape((-1, num_labels)))
        # (N, 1, H, W) -> (N, H, W, 1) -> (N*H*W,)
        center_arrs.append(center_arr.permute((0, 2, 3, 1)).reshape((-1,)))

    return torch.cat(reg_arrs), torch.cat(label_arrs), torch.cat(center_arrs)

def fcos_batch_loss(out, targets, pyramid_shape, num_labels):
    """Compute loss for a single image.

    Note: the label_arr and center_arr for output is assumed to contain
    logits, and is assumed to contain probabilities for targets.

    Args:
        out: the output of the heads for the whole pyramid
        targets: list<BoxList> of length n

        the format of out is a list of tuples where each tuple corresponds to a
        pyramid level. tuple is of form (reg_arr, label_arr, center_arr) where
            - reg_arr is tensor<n, 4, h, w>,
            - label_arr is tensor<n, num_labels, h, w>
            - center_arr is tensor<n, 1, h, w>

        and label_arr and center_arr values are logits.

    Returns:
        dict of form {
            'reg_loss': tensor<1>,
            'label_loss': tensor<1>,
            'center_loss': tensor<1>
        }
    """
    iou_loss = IOULoss()
    batch_sz = len(targets)
    reg_arrs, label_arrs, center_arrs = [], [], []
    for single_targets in targets:
        single_targets = encode_single_targets(
            single_targets.boxes, single_targets.get_field('labels'), pyramid_shape,
            num_labels)
        for reg_arr, label_arr, center_arr in single_targets:
            # (4, H, W) -> (H, W, 4) -> (H*W, 4)
            reg_arrs.append(reg_arr.permute((1, 2, 0)).reshape((-1, 4)))
            # (C, H, W) -> (H, W, C) -> (H*W, C)
            label_arrs.append(label_arr.permute((1, 2, 0)).reshape((-1, num_labels)))
            # (1, H, W) -> (H, W, 1) -> (H*W,)
            center_arrs.append(center_arr.permute((1, 2, 0)).reshape((-1,)))

    targets_reg_arr, targets_label_arr, targets_center_arr = (
        torch.cat(reg_arrs), torch.cat(label_arrs),
        torch.cat(center_arrs))
    out_reg_arr, out_label_arr, out_center_arr = flatten_output(out)

    pos_indicator = targets_label_arr.sum(1) > 0.0
    out_reg_arr = out_reg_arr[pos_indicator, :]
    targets_reg_arr = targets_reg_arr[pos_indicator, :]
    out_center_arr = out_center_arr[pos_indicator]
    targets_center_arr = targets_center_arr[pos_indicator]

    npos = targets_reg_arr.shape[0] + 1
    label_loss = focal_loss(out_label_arr, targets_label_arr) / npos
    reg_loss = torch.tensor(0.0, device=label_loss.device)
    center_loss = torch.tensor(0.0, device=label_loss.device)
    if npos > 1:
        reg_loss = iou_loss(out_reg_arr, targets_reg_arr, targets_center_arr)
        center_loss = nn.functional.binary_cross_entropy_with_logits(
            out_center_arr, targets_center_arr, reduction='mean')

    total_loss = label_loss + reg_loss + center_loss
    loss_dict = {'total_loss': total_loss, 'label_loss': label_loss,
                 'reg_loss': reg_loss, 'center_loss': center_loss}
    return loss_dict