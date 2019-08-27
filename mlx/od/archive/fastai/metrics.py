import torch
from fastai.callback import Callback, add_metrics

class CocoMetric(Callback):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.__name__ = 'mAP'

    def on_epoch_begin(self, **kwargs):
        self.outputs = []
        self.targets = []

    def on_batch_begin(self, last_input, last_target, **kwargs):
        self.h, self.w = last_input.shape[2:]

    def on_batch_end(self, last_output, last_target, **kwargs):
        self.outputs.extend(last_output)
        self.targets.append(last_target)

    def on_epoch_end(self, last_metrics, **kwargs):
        # Convert from fastai format
        my_targets = []
        for batch_boxes, batch_labels in self.targets:
            for boxes, labels in zip(batch_boxes, batch_labels):
                non_pad_inds = labels != 0
                boxes = to_box_pixel(boxes, self.h, self.w)
                my_targets.append((boxes[non_pad_inds, :], labels[non_pad_inds]))

        my_outputs = [
            (boxlist.boxes, boxlist.get_field('labels'), boxlist.get_field('scores'))
            for boxlist in self.outputs]
        metric = compute_coco_eval(my_outputs, my_targets, self.num_labels)[0]
        return add_metrics(last_metrics, metric)