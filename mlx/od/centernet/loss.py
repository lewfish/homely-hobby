import torch.nn.functional as F
import torch

def loss(head_out, encoded_target):
    out_keypoint, out_reg = head_out
    targ_keypoint, targ_reg = encoded_target
    num_labels = out_keypoint.shape[1]

    num_reg = 2
    # (n, c, h, w) -> (-1, c)
    flat_out_reg = out_reg.permute((0, 2, 3, 1)).reshape((-1, num_reg))
    flat_targ_reg = targ_reg.permute((0, 2, 3, 1)).reshape((-1, num_reg))
    is_pos = flat_targ_reg.max(1)[0] != 0
    num_pos = is_pos.sum() + 1
    reg_loss = F.mse_loss(flat_out_reg[is_pos, :], flat_targ_reg[is_pos, :])

    flat_out_keypoint = out_keypoint.permute((0, 2, 3, 1)).reshape((-1, num_labels))
    flat_targ_keypoint = targ_keypoint.permute((0, 2, 3, 1)).reshape((-1, num_labels))
    keypoint_loss = F.mse_loss(flat_out_keypoint, flat_targ_keypoint, reduction='sum') / num_pos

    total_loss = keypoint_loss + reg_loss
    return {'total_loss': total_loss, 'keypoint_loss': keypoint_loss,
            'reg_loss': reg_loss}