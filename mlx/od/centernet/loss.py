import torch.nn.functional as F
import torch

def loss(head_out, encoded_target, loss_alpha=2.0, loss_beta=4.0):
    out_keypoint, out_reg = head_out
    targ_keypoint, targ_reg = encoded_target
    num_labels = out_keypoint.shape[1]

    num_reg = 2
    # (n, c, h, w) -> (-1, c)
    flat_out_reg = out_reg.permute((0, 2, 3, 1)).reshape((-1, num_reg))
    flat_targ_reg = targ_reg.permute((0, 2, 3, 1)).reshape((-1, num_reg))
    is_pos = flat_targ_reg.max(1)[0] != 0
    num_pos = is_pos.sum() + 1
    
    reg_loss = F.l1_loss(flat_out_reg[is_pos, :], flat_targ_reg[is_pos, :], reduction='mean')

    flat_out_keypoint = out_keypoint.permute((0, 2, 3, 1)).reshape((-1, num_labels))
    flat_targ_keypoint = targ_keypoint.permute((0, 2, 3, 1)).reshape((-1, num_labels))

    flat_out_keypoint_pos = flat_out_keypoint[is_pos, :]
    pos_loss = (((1. - flat_out_keypoint_pos) ** loss_alpha) * 
                torch.log(flat_out_keypoint_pos)).sum()

    flat_out_keypoint_neg = flat_out_keypoint[~is_pos, :]
    flat_targ_keypoint_neg = flat_targ_keypoint[~is_pos, :]
    neg_loss = (((1. - flat_targ_keypoint_neg) ** loss_beta) * 
                (flat_out_keypoint_neg ** loss_alpha) * 
                torch.log(1. - flat_out_keypoint_neg)).sum()
    keypoint_loss = -(pos_loss + neg_loss) / num_pos

    reg_scale = 0.1
    total_loss = keypoint_loss + reg_scale * reg_loss
    return {'total_loss': total_loss, 'keypoint_loss': keypoint_loss,
            'reg_loss': reg_loss}