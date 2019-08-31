import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from mlx.od.boxlist import BoxList

class MyFasterRCNN(nn.Module):
    def __init__(self, num_labels, img_sz, pretrained=True):
        super().__init__()
        
        self.model = fasterrcnn_resnet50_fpn(
            pretrained=False, progress=True, num_classes=num_labels, 
            pretrained_backbone=pretrained, min_size=img_sz, max_size=img_sz)
        self.subloss_names = [
            'total_loss', 'loss_box_reg', 'loss_classifier', 'loss_objectness', 'loss_rpn_box_reg']

    def forward(self, input, targets=None):
        """Forward pass

        Args:
            input: tensor<n, 3, h, w> with batch of images
            targets: None or list<BoxList> of length n with boxes and labels

        Returns:
            if targets is None, returns list<BoxList> of length n, containing
            boxes, labels, and scores for boxes with score > 0.05. Further
            filtering based on score should be done before considering the
            prediction "final".

            if targets is a list, returns the losses as dict of form {
            }
        """
        if targets:
            _targets = [bl.xyxy() for bl in targets]
            _targets = [{'boxes': bl.boxes, 'labels': bl.get_field('labels')} for bl in _targets]
            loss_dict = self.model(input, _targets)
            loss_dict['total_loss'] = sum(list(loss_dict.values())) 
            return loss_dict

        out = self.model(input)
        return [BoxList(_out['boxes'], labels=_out['labels'], scores=_out['scores']).yxyx() for _out in out]
