import torch

def to_box_pixel(boxes, img_height, img_width):
    # convert from (ymin, xmin, ymax, xmax) in range [-1,1] to
    # range [0, h) or [0, w)
    boxes = ((boxes + 1.0) / 2.0) * torch.tensor([[img_height, img_width, img_height, img_width]]).to(
        device=boxes.device, dtype=torch.float)
    return boxes