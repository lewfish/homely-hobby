import torch

def encode_box(reg_arr, label_arr, center_arr, stride, box, label):
    """Encode single box/label into array format for single level of pyramid.

    The array format is fed into the loss function and is in the same format
    as the output as the head of the FCOS model. This writes values to reg_arr
    and label_arr and does not return anything.

    Note that the height and width of the box has to be at least stride,
    or else it might fall in between the center of two cells, and not be
    encodable.

    Args:
        reg_arr: (tensor) with shape (4, h, w) the first dimension ranges over
            t, l, b, r (ie. top, left, bottom, right)
        label_arr: (tensor) with shape (num_labels, h, w) containing
            probabilities
        center_arr: (tensor) with shape (1, h, w) containing values between
            0 and 1
        stride: (int) the stride of the level of the pyramid these arrays
            are responsible for encoding
        box: (tensor) with shape (4,) with format (ymin, xmin, ymax, xmax)
        label: (int) label of box (ie. class)
    """
    box = box.int()
    if torch.min(box[2:] - box[0:2]) < stride:
        return

    device = reg_arr.device
    h, w = torch.tensor(list(reg_arr.shape[1:]), device=device) * stride
    half_stride = torch.tensor(
        [stride//2, stride//2], device=device, dtype=torch.int)
    nw_ij = (box[0:2] + half_stride) // stride
    se_ij = (box[2:] - half_stride) // stride
    nw_yx = nw_ij * stride + half_stride
    nw_tlbr = torch.cat((nw_yx - box[0:2], box[2:] - nw_yx))

    box_arr_shape = (4,) + tuple((se_ij - nw_ij + 1).tolist())
    box_reg_arr = torch.empty(box_arr_shape, device=device)
    box_reg_arr[:, :, :] = nw_tlbr[:, None, None]

    row_incs = torch.arange(
        0, box_reg_arr.shape[1] * stride, stride, dtype=torch.float32,
        device=device)
    col_incs = torch.arange(
        0, box_reg_arr.shape[2] * stride, stride, dtype=torch.float32,
        device=device)

    box_reg_arr[0, :, :] += row_incs[:, None]
    box_reg_arr[1, :, :] += col_incs[None, :]
    box_reg_arr[2, :, :] -= row_incs[:, None]
    box_reg_arr[3, :, :] -= col_incs[None, :]

    nw_ij, se_ij = nw_ij.tolist(), se_ij.tolist()
    reg_arr[:, nw_ij[0]:se_ij[0]+1, nw_ij[1]:se_ij[1]+1] = box_reg_arr

    label_arr[:, nw_ij[0]:se_ij[0]+1, nw_ij[1]:se_ij[1]+1] = 0
    label_arr[label, nw_ij[0]:se_ij[0]+1, nw_ij[1]:se_ij[1]+1] = 1

    t = box_reg_arr[0]
    l = box_reg_arr[1]
    b = box_reg_arr[2]
    r = box_reg_arr[3]
    box_center_arr = torch.sqrt(
        (torch.min(l, r) / torch.max(l, r)) *
        (torch.min(t, b) / torch.max(t, b)))
    center_arr[0, nw_ij[0]:se_ij[0]+1, nw_ij[1]:se_ij[1]+1] = box_center_arr

def get_stride(box_size, pyramid_shape):
    """Get level of pyramid to use for a box.

    Args:
        box_size: (int) maximum length side of box
        pyramid_shape: list of (stride, max_box_side, h, w) sorted in
            descending order by stride

    Returns:
        stride of level in pyramid that handles boxes of box_size
    """
    for stride, max_box_size, _, _ in reversed(pyramid_shape):
        if box_size <= max_box_size:
            return stride
    return stride

def init_targets(pyramid_shape, num_labels, device='cpu'):
    """Initialize storage for encoded targets for one image.

    Args:
        pyramid_shape: list of (stride, max_box_side, h, w) sorted in
            descending order by stride
        num_labels: (int) number of labels (ie. classes)

    Returns:
        (dict) where keys are strides, values are dicts of form
        {'reg_arr': <tensor with shape (4, h, w)>,
         'label_arr': <tensor with shape (num_labels, h, w)>,
         'center_arr': <tensor with shape (1, h, w)>}}
        with all values set to 0
    """
    targets = {}
    for stride, _, h, w in pyramid_shape:
        reg_arr = torch.zeros((4, h, w), device=device)
        label_arr = torch.zeros((num_labels, h, w), device=device)
        center_arr = torch.zeros((1, h, w), device=device)
        targets[stride] = {
            'reg_arr': reg_arr, 'label_arr': label_arr, 'center_arr': center_arr}
    return targets

def sort_boxes(boxes, labels):
    """Sort boxes and labels by area in descending order."""
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    _, box_inds = torch.sort(box_areas, descending=True)
    boxes = boxes[box_inds, :]
    labels = labels[box_inds]
    return boxes, labels

def encode_targets(boxes, labels, pyramid_shape, num_labels):
    """Encode boxes and labels into a pyramid of arrays for one image.

    Encodes each box and label into the arrays representing a single level of
    the pyramid based on the size of the box.

    Args:
        boxes: (tensor) with shape (n, 4) with format (ymin, xmin, ymax, xmax)
        labels: (tensor) with shape (n,) with class ids
        pyramid_shape: list of (stride, max_box_side, h, w) sorted in
            descending order by stride
        num_labels: (int) number of labels (ie. class ids)

    Returns:
        (dict) where keys are strides, values are dicts of form
        {'reg_arr': <tensor with shape (4, h, w)>,
         'label_arr': <tensor with shape (num_labels, h, w)>,
         'center_arr': <tensor with shape (1, h, w)>}}
        with label and center values between 0 and 1
    """
    # sort boxes and labels by box area in descending order
    # this is so that we encode smaller boxes later, to give them precedence
    # in cases where there is ambiguity.
    device = boxes.device
    boxes, labels = sort_boxes(boxes, labels)
    targets = init_targets(pyramid_shape, num_labels, device=device)

    if boxes.shape[0] == 0:
        return targets

    # for each box, get arrays for matching stride and encode box
    box_sizes, _ = torch.max(boxes[:, 2:] - boxes[:, 0:2], dim=1)
    for box, label, box_size in zip(boxes, labels, box_sizes):
        stride = get_stride(box_size, pyramid_shape)
        reg_arr = targets[stride]['reg_arr']
        label_arr = targets[stride]['label_arr']
        center_arr = targets[stride]['center_arr']
        encode_box(
            reg_arr, label_arr, center_arr, stride, box, int(label.item()))

    return targets
