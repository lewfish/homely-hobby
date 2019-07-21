import torch

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
    p = torch.sigmoid(output)
    pt = (1-target) * (1-p) + target * p
    alphat = (1-target) * (1-alpha) + target * alpha
    bce = -torch.log(pt)
    weights = alphat * (1 - pt).pow(gamma)
    loss_arr = weights * bce
    return loss_arr.reshape(-1).sum()