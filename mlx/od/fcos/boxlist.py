import torch

from torchvision.ops.boxes import batched_nms

def to_box_pixel(boxes, img_height, img_width):
    # convert from (ymin, xmin, ymax, xmax) in range [-1,1] to
    # range [0, h) or [0, w)
    boxes = ((boxes + 1.0) / 2.0) * torch.tensor([[img_height, img_width, img_height, img_width]]).to(
        device=boxes.device, dtype=torch.float)
    return boxes

class BoxList():
    def __init__(self, boxes, labels, scores=None, centerness=None):
        """Constructor.

        Args:
            boxes: tensor (n, 4)
            labels: tensor (n,)
            scores: tensor (n,)
            centerness: tensor (n,)
        """
        self.boxes = boxes
        self.labels = labels
        self.scores = scores if not None else torch.zeros_like(labels)
        self.centerness = centerness if not None else torch.zeros_like(centerness)

    def copy(self):
        return BoxList(self.boxes.copy(), self.labels.copy(),
                       self.scores.copy(), self.centerness.copy())

    def cpu(self):
        return BoxList(self.boxes.cpu(), self.labels.cpu(), self.scores.cpu(),
                       self.centerness.cpu())

    def __len__(self):
        return self.boxes.shape[0]

    def tuple(self):
        return self.boxes, self.labels, self.scores, self.centerness

    @staticmethod
    def make_empty():
        return BoxList(torch.zeros((0, 4)), torch.tensor([]))

    @staticmethod
    def cat(box_lists):
        if len(box_lists) == 0:
            return BoxList.make_empty()
        boxes = torch.cat([bl.boxes for bl in box_lists], dim=0)
        labels = torch.cat([bl.labels for bl in box_lists])
        scores = torch.cat([bl.scores for bl in box_lists])
        centerness = torch.cat([bl.centerness for bl in box_lists])
        return BoxList(boxes, labels, scores, centerness)

    def equal(self, other):
        return (self.boxes.equal(other.boxes) and
                self.labels.equal(other.labels) and
                self.scores.equal(other.scores) and
                self.centerness.equal(other.centerness))

    def ind_filter(self, inds):
        return BoxList(self.boxes[inds, :], self.labels[inds],
                       self.scores[inds], self.centerness[inds])

    def score_filter(self, score_thresh=0.25):
        return self.ind_filter(self.scores > score_thresh)

    def label_filter(self, label):
        return self.ind_filter(self.labels == label)

    def get_unique_labels(self):
        return sorted(set(self.labels.tolist()))

    def clamp(self, img_height, img_width):
        boxes = torch.stack([
            torch.clamp(self.boxes[:, 0], 0, img_height),
            torch.clamp(self.boxes[:, 1], 0, img_width),
            torch.clamp(self.boxes[:, 2], 0, img_height),
            torch.clamp(self.boxes[:, 3], 0, img_width)
        ], dim=1)
        return BoxList(boxes, self.labels, self.scores, self.centerness)

    def nms(self, iou_thresh=0.5):
        if len(self) == 0:
            return self

        good_inds = batched_nms(self.boxes, self.scores, self.labels, iou_thresh)
        return self.ind_filter(good_inds)