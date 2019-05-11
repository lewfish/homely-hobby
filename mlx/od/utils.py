import torch
from torch.nn.functional import binary_cross_entropy as bce, l1_loss


def compute_intersection(a, b):
    """Compute intersection between boxes.

    Args:
        a: tensor (n, 4)
        b: tensor (n, 4)

    Returns:
        tensor (n, n)
    """
    # (n, n, 2)
    lb = torch.max(a[:, 0:2].unsqueeze(0), b[:, 0:2].unsqueeze(1))
    ub = torch.min(a[:, 2:4].unsqueeze(0), b[:, 2:4].unsqueeze(1))
    # (n, n)
    inter = (ub - lb).clamp(min=0).prod(2)
    return inter

def compute_iou(a, b):
    """Compute IOU between boxes.

    Args:
        a: tensor (n, 4)
        b: tensor (n, 4)

    Returns:
        tensor (n, n)
    """
    # (n, n, 2)
    inter = compute_intersection(a, b)
    a_area = (a[:, 2:4] - a[:, 0:2]).prod(1)
    b_area = (b[:, 2:4] - b[:, 0:2]).prod(1)
    union = (a_area + b_area - inter)
    iou = inter / union
    return iou

class BoxList():
    def __init__(self, boxes, labels, scores=None):
        """Constructor.

        Args:
            boxes: tensor (n, 4)
            labels: tensor (n,)
            scores: tensor (n,) or None
        """
        self.boxes = boxes
        self.labels = labels
        self.scores = (scores if scores is not None
                       else torch.zeros_like(self.labels))

    def copy(self):
        return BoxList(self.boxes.copy(), self.labels.copy(),
                       self.scores.copy())

    def __len__(self):
        return self.boxes.shape[0]

    @staticmethod
    def make_empty():
        return BoxList(torch.zeros((0, 4)), torch.tensor([]))

    @staticmethod
    def merge(box_lists):
        if len(box_lists) == 0:
            return BoxList.make_empty()
        boxes = torch.cat([bl.boxes for bl in box_lists], dim=0)
        labels = torch.cat([bl.labels for bl in box_lists])
        scores = torch.cat([bl.scores for bl in box_lists])
        return BoxList(boxes, labels, scores)

    def equal(self, other):
        return (self.boxes.equal(other.boxes) and
                self.labels.equal(other.labels) and
                self.scores.equal(other.scores))

    def ind_filter(self, inds):
        return BoxList(self.boxes[inds, :], self.labels[inds], self.scores[inds])

    def score_filter(self, score_thresh=0.5):
        return self.ind_filter(self.scores > score_thresh)

    def label_filter(self, label):
        return self.ind_filter(self.labels == label)

    def get_unique_labels(self):
        return sorted(set(self.labels.tolist()))

    def _label_nms(self, iou_thresh=0.5):
        """Assumes all labels are the same."""
        # This line is very inefficient because there's no need to compute
        # the IOU between boxes that don't overlap. It might be faster to use
        # an r-tree if there are tons of boxes if the image is huge.
        ious = compute_iou(self.boxes, self.boxes)
        remove_inds = set()
        keep_inds = []
        score_inds = torch.argsort(self.scores, descending=True)
        all_inds = torch.arange(0, self.boxes.shape[0])
        for i in score_inds.tolist():
            if i not in remove_inds:
                _remove_inds = all_inds[
                    (self.scores < self.scores[i]) * (ious[i, :] > iou_thresh)]
                remove_inds.update(_remove_inds.tolist())
                keep_inds.append(i)
        return self.ind_filter(keep_inds)

    def nms(self, iou_thresh=0.5):
        if len(self) == 0:
            return self

        box_lists = []
        for l in self.get_unique_labels():
            bl = self.label_filter(l)._label_nms(iou_thresh)
            box_lists.append(bl)
        return BoxList.merge(box_lists)

    def __repr__(self):
        return 'boxes: {}\nlabels: {}\nscores: {}'.format(
            self.boxes, self.labels, self.scores)

class ObjectDetectionGrid():
    """Represents of grid of anchor boxes for object detection.

    Some shorthand:
        b: batch_sz
        a: num_ancs
        d: det_sz
        g: grid_sz
        c: num_classes

    Boxes are represented as (ymin, xmin, ymax, xmax).
    """
    def __init__(self, grid_sz, anc_sizes, num_classes):
        """Constructor.

        Args:
            grid_sz: number of rows and cols (assumed to be equal)
            anc_sizes: (a, 2) tensor where columns are (height, width)
                values that are multiplied with the cell_sz to get the height
                and width of anchor boxes.
            num_classes: number of classes to predict
        """
        self.grid_sz = grid_sz
        # (a, 2)
        self.anc_sizes = anc_sizes
        self.num_ancs = anc_sizes.shape[0]
        self.num_classes = num_classes

        self.det_sz = 4 + num_classes
        self.cell_sz = 2.0 / grid_sz
        # (g, g, 2)
        self.grid_inds = self.get_grid_inds()
        # (g, g, 2)
        self.cell_centers = self.get_cell_centers(self.grid_inds)
        # (g, g, a, 4)
        self.ancs = self.get_anchors(self.grid_inds)

    def get_out_shape(self, batch_sz):
        return (batch_sz, self.num_ancs, self.det_sz,
                self.grid_sz, self.grid_sz)

    def get_cell_centers(self, grid_inds):
        """Get centers of each cell in grid.

        Args:
            grid_inds: tensor (g, g, 2) where last dim is (row, col) within
                grid

        Returns:
            tensor (g, g, 2) where last dim is (y, x)
        """
        return (grid_inds * self.cell_sz + self.cell_sz / 2) - 1

    def get_anchors(self, grid_inds):
        """Get anchor boxes over grid.

        Args:
            grid_inds: tensor (g, g, 2) where last dim is (row, col) within
                grid

        Returns:
            tensor (g, g, a, 4) where last dim is box
        """
        # (a, 2)
        half_ancs = (self.cell_sz * self.anc_sizes / 2)
        # (g, g, 1, 2)
        cell_centers = self.cell_centers.unsqueeze(2)
        # (g, g, a, 2)
        anc_mins = cell_centers - half_ancs
        anc_maxs = cell_centers + half_ancs
        # (g, g, a, 4)
        # should we clamp to -1, 1 here?
        return torch.cat((anc_mins, anc_maxs), dim=3)

    def get_grid_inds(self):
        """Get indices of cells in grid.

        Returns:
            tensor (g, g, 2) where last dim is (row, col) within grid
        """
        grid_inds = torch.empty(
            (self.grid_sz, self.grid_sz, 2))
        for y in range(self.grid_sz):
            for x in range(self.grid_sz):
                grid_inds[y, x, :] = torch.tensor([y, x])
        # (g, g, 2)
        return grid_inds

    def decode(self, out):
        """Decode output of network into boxes, labels, and scores.

        Args:
            out: tensor (b, a, d, g, g) where the values for each anchor are
                (yoffset, xoffset, yscale, xscale, p0, ..., pn)

        Returns: (boxes, labels, scores) where
            boxes: tensor (b, agg, 4)
            labels: tensor (b, agg) where each element is a class index
            scores: tensor (b, agg) where each element is a probability
        """
        device = out.device
        batch_sz = out.shape[0]
        # (b, g, g, a, d)
        out = out.permute(0, 3, 4, 1, 2)

        # (b, agg, c)
        probs = out[:, :, :, :, 4:].reshape((batch_sz, -1, self.num_classes))

        # (b, g, g, a, 2)
        det_offsets = out[:, :, :, :, 0:2]
        det_scales = out[:, :, :, :, 2:4]
        box_centers = (self.cell_centers.to(device).unsqueeze(2) + self.cell_sz *
                       det_offsets)
        box_sizes = self.anc_sizes * self.cell_sz * det_scales
        box_mins = box_centers - box_sizes / 2
        box_maxs = box_centers + box_sizes / 2

        # (b, g, g, a, 4) -> (b, agg, 4)
        boxes = torch.cat((box_mins, box_maxs), dim=4) \
                     .reshape((batch_sz, -1, 4))
        # (b, agg)
        scores, labels = probs.max(2)
        return boxes, labels, scores

    def encode(self, boxes, labels):
        """Encode ground truth boxes and labels into output of network.

        There are some for loops in this implementation which I'd like to
        vectorize, but I don't see how to without having to do way more
        extra operations than is necessary.

        Args:
            boxes: tensor (b, n, 4) where n is an arbitrary number
            labels: tensor (b, n)

        Returns: tensor (b, a, d, g, g) where the values for each anchor are
            (yoffset, xoffset, yscale, xscale, c0, ..., cn)
        """
        device = boxes.device
        batch_sz, n = boxes.shape[0:2]
        out = torch.zeros((batch_sz, self.num_ancs, self.det_sz,
                           self.grid_sz, self.grid_sz), dtype=torch.float,
                           device=device)

        for batch_ind in range(batch_sz):
            # (n, 2)
            box_mins = boxes[batch_ind, :, :2]
            box_maxs = boxes[batch_ind, :, 2:]
            box_centers = box_mins + (box_maxs - box_mins) / 2
            match_grid_inds = ((box_centers + 1.0) / self.cell_sz).int()

            for n_ind in range(n):
                # (4)
                box = boxes[batch_ind, n_ind, :]
                # Ignore padding boxes which have all zeros.
                if torch.any(box != 0.):
                    # (2)
                    gi = match_grid_inds[n_ind, :].tolist()
                    # (a, 4)
                    ancs = self.ancs[gi[0], gi[1], :, :].to(device)
                    # (a, 1)
                    ious = compute_iou(ancs, box.unsqueeze(0))
                    best_anc_ind = ious.squeeze().argmax()
                    # (4)
                    anc = ancs[best_anc_ind, :]
                    # (2)
                    anc_center = self.cell_centers[gi[0], gi[1], :].to(device)
                    # TODO handle collisions

                    # (2)
                    offset = (anc_center - box_centers[n_ind, :]) / self.cell_sz
                    scales = (box[2:]-box[:2]) / (anc[2:]-anc[:2]) / self.cell_sz

                    out[batch_ind, best_anc_ind, 0:2, gi[0], gi[1]] = offset
                    out[batch_ind, best_anc_ind, 2:4, gi[0], gi[1]] = scales
                    out[batch_ind, best_anc_ind,
                        4 + labels[batch_ind, n_ind], gi[0], gi[1]] = 1

        return out

    def compute_losses(self, out, gt):
        """Compute losses given network output and encoded ground truth.

        Args:
            out: tensor (b, ad, g, g)
            gt: tensor (b, ad, g, g)

        Returns:
            box_loss, class_loss (both tensor (1))
        """
        batch_sz = out.shape[0]

        # (b, a, d, gg)
        out = out.reshape((batch_sz, self.num_ancs, self.det_sz, -1))
        gt = gt.reshape((batch_sz, self.num_ancs, self.det_sz, -1))

        # (b, a, 4, gg)
        out_anc_params = out[:, :, :4, :]
        gt_anc_params = gt[:, :, :4, :]

        # (b, a, c, gg)
        out_probs = out[:, :, 4:, :]
        gt_probs = gt[:, :, 4:, :]

        # TODO: switch to use logits version?
        class_loss = bce(out_probs, gt_probs)

        # [b, a, 1, gg]
        has_object = gt_probs.sum(2, keepdim=True) != 0

        # (-1)
        out_anc_params = out_anc_params.masked_select(has_object)
        gt_anc_params = gt_anc_params.masked_select(has_object)

        box_loss = l1_loss(out_anc_params, gt_anc_params)

        return box_loss, class_loss