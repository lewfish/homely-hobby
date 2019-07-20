import torch

def get_stride(box_size, pyramid_shape):
    for stride, max_box_size, _, _ in reversed(pyramid_shape):
        if box_size <= max_box_size:
            return stride
    return stride

def make_targets(pyramid_shape, num_labels, device='cpu'):
    # setup empty reg and label arrays for each level in pyramid
    targets = {}
    for stride, _, h, w in pyramid_shape:
        reg_arr = torch.zeros((4, h, w), device=device)
        label_arr = torch.zeros((num_labels, h, w), device=device)
        targets[stride] = {'reg_arr': reg_arr, 'label_arr': label_arr}
    return targets

def sort_boxes(boxes, labels):
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    _, box_inds = torch.sort(box_areas, descending=True)
    boxes = boxes[box_inds, :]
    labels = labels[box_inds]
    return boxes, labels

def encode_targets(boxes, labels, pyramid_shape, num_labels):
    # pyramid_shape is list of (stride, max_box_side, h, w) sorted in descending
    # order by stride

    # sort boxes and labels by box area in descending order
    # this is so that we encode smaller boxes later, to give them precedence
    # in cases where there is ambiguity.
    device = boxes.device
    boxes, labels = sort_boxes(boxes, labels)
    targets = make_targets(pyramid_shape, num_labels, device=device)

    # for each box, get arrays for matching stride and encode box
    box_sizes, _ = torch.max(boxes[:, 2:] - boxes[:, 0:2], dim=1)
    for box, label, box_size in zip(boxes, labels, box_sizes):
        stride = get_stride(box_size, pyramid_shape)
        reg_arr = targets[stride]['reg_arr']
        label_arr = targets[stride]['label_arr']
        encode(reg_arr, label_arr, stride, box, int(label.item()))

    return targets

def encode(reg_arr, label_arr, stride, box, label):
    # reg_arr is (4, h, w) where first dimension ranges over t, l, b, r
    # ie. (top, left, bottom, right)
    # class_arr is (c, h, w)

    # height and width of box have to be at least stride, or else it might
    # fall in between the center of two positions, and not be encodable.
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
    box_arr = torch.empty(box_arr_shape, device=device)
    box_arr[:, :, :] = nw_tlbr[:, None, None]

    row_incs = torch.arange(
        0, box_arr.shape[1] * stride, stride, dtype=torch.float32,
        device=device)
    col_incs = torch.arange(
        0, box_arr.shape[2] * stride, stride, dtype=torch.float32,
        device=device)

    box_arr[0, :, :] += row_incs[:, None]
    box_arr[1, :, :] += col_incs[None, :]
    box_arr[2, :, :] -= row_incs[:, None]
    box_arr[3, :, :] -= col_incs[None, :]

    nw_ij, se_ij = nw_ij.tolist(), se_ij.tolist()
    reg_arr[:, nw_ij[0]:se_ij[0]+1, nw_ij[1]:se_ij[1]+1] = box_arr

    label_arr[:, nw_ij[0]:se_ij[0]+1, nw_ij[1]:se_ij[1]+1] = 0
    label_arr[label, nw_ij[0]:se_ij[0]+1, nw_ij[1]:se_ij[1]+1] = 1
