import torch

def encode_box(reg_arr, label_arr, stride, box, label):
    # reg_arr is (4, h, w) where first dimension ranges over t, l, b, r
    # ie. (top, left, bottom, right)
    # class_arr is (c, h, w)

    # height and width of box have to be at least stride, or else it might
    # fall in between the center of two positions, and not be encodable.
    box = box.int()
    if torch.min(box[2:] - box[0:2]) < stride:
        return

    h, w = torch.Tensor(list(reg_arr.shape[1:])) * stride
    half_stride = torch.Tensor([stride//2, stride//2]).int()
    nw_ij = (box[0:2] + half_stride) // stride
    se_ij = (box[2:] - half_stride) // stride
    nw_yx = nw_ij * stride + half_stride
    nw_tlbr = torch.cat((nw_yx - box[0:2], box[2:] - nw_yx))

    box_arr_shape = (4,) + tuple((se_ij - nw_ij + 1).tolist())
    box_arr = torch.empty(box_arr_shape)
    box_arr[:, :, :] = nw_tlbr[:, None, None]

    row_incs = torch.arange(0, box_arr.shape[1] * stride, stride, dtype=torch.float32)
    col_incs = torch.arange(0, box_arr.shape[2] * stride, stride, dtype=torch.float32)

    box_arr[0, :, :] += row_incs[:, None]
    box_arr[1, :, :] += col_incs[None, :]
    box_arr[2, :, :] -= row_incs[:, None]
    box_arr[3, :, :] -= col_incs[None, :]

    nw_ij, se_ij = nw_ij.tolist(), se_ij.tolist()
    reg_arr[:, nw_ij[0]:se_ij[0]+1, nw_ij[1]:se_ij[1]+1] = box_arr

    label_arr[label, nw_ij[0]:se_ij[0]+1, nw_ij[1]:se_ij[1]+1] = 1
