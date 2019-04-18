import torch

def compute_intersection(a, b):
    # a and b: [n, 4]
    # [n, n, 2]
    lb = torch.max(a[:, 0:2].unsqueeze(0), b[:, 0:2].unsqueeze(1))
    ub = torch.min(a[:, 2:4].unsqueeze(0), b[:, 2:4].unsqueeze(1))
    # [n, n]
    inter = (ub - lb).clamp(min=0).prod(2)
    return inter

def compute_iou(a, b):
    # a and b are [n, 4]
    # [n, n, 2]
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
            boxes: tensor [n, 4]
            labels: tensor [n]
            scores: tensor [n]
        """
        self.boxes = boxes
        self.labels = labels
        self.scores = (scores if scores is not None
                       else torch.zeros_like(self.labels))

    def copy(self):
        return BoxList(self.boxes.copy(), self.labels.copy(),
                       self.scores.copy())

    @staticmethod
    def merge(box_lists):
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

    def label_nms(self, iou_thresh=0.5):
        """Assumes all labels are the same."""
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
        box_lists = []
        for l in self.get_unique_labels():
            bl = self.label_filter(l).label_nms(iou_thresh)
            box_lists.append(bl)
        return BoxList.merge(box_lists)

    def __repr__(self):
        return 'boxes: {}\nlabels: {}\nscores: {}'.format(
            self.boxes, self.labels, self.scores)

class DetectorGrid():
    def __init__(self, grid_sz, anc_sizes, num_classes):
        """Constructor.

        Args:
            anc_sizes: [a, 2] tensor
        """
        self.grid_sz = grid_sz
        # [a, 2]
        self.anc_sizes = anc_sizes
        self.num_ancs = anc_sizes.shape[0]
        self.num_classes = num_classes

        self.det_sz = 4 + num_classes
        self.cell_sz = 2.0 / grid_sz
        # [g, g, 2]
        self.grid_inds = self.get_grid_inds()
        # [g, g, 2]
        self.cell_centers = self.get_cell_centers(self.grid_inds)
        # [g, g, a, 4]
        self.ancs = self.get_anchors(self.grid_inds)

    def get_out_shape(self, batch_sz):
        return (batch_sz, self.num_ancs * self.det_sz,
                self.grid_sz, self.grid_sz)

    def get_cell_centers(self, grid_inds):
        return (grid_inds * self.cell_sz + self.cell_sz / 2) - 1

    def get_anchors(self, grid_inds):
        # [a, 2]
        half_ancs = (self.cell_sz * self.anc_sizes / 2)
        # [g, g, 1, 2]
        cell_centers = self.cell_centers.unsqueeze(2)
        # [g, g, a, 2]
        anc_mins = cell_centers - half_ancs
        anc_maxs = cell_centers + half_ancs
        # [g, g, a, 4]
        # should we clamp to -1, 1 here?
        return torch.cat((anc_mins, anc_maxs), dim=3)

    def get_grid_inds(self):
        grid_inds = torch.empty((self.grid_sz, self.grid_sz, 2))
        for y in range(self.grid_sz):
            for x in range(self.grid_sz):
                grid_inds[y, x, :] = torch.tensor([y, x])
        # [g, g, 2]
        return grid_inds

    def decode(self, out):
        """Convert output of network to boxes and probs.

        Args:
            out: tensor [b, a*d, g, g]

        Returns: (boxes, labels, scores) where
            boxes: tensor [b, -1, 4]
            probs: tensor [b, -1]
            scores: tensor [b, -1]
        """
        batch_sz = out.shape[0]
        # [b, g, g, a, d]
        out = out.permute(0, 2, 3, 1) \
                 .reshape((batch_sz, self.grid_sz, self.grid_sz,
                           self.num_ancs, self.det_sz))

        # [b, -1, c]
        probs = out[:, :, :, :, 4:].reshape((batch_sz, -1, self.num_classes))

        # [b, g, g, a, 2]
        det_offsets = out[:, :, :, :, 0:2]
        det_scales = out[:, :, :, :, 2:4]
        box_centers = self.cell_centers.unsqueeze(2) + self.cell_sz * det_offsets
        box_sizes = self.anc_sizes * self.cell_sz * det_scales
        box_mins = box_centers - box_sizes / 2
        box_maxs = box_centers + box_sizes / 2

        # [b, g, g, a, 4] -> [b, -1, 4]
        # box shape is  (ymin, xmin, ymax, xmax)
        boxes = torch.cat((box_mins, box_maxs), dim=4) \
                     .reshape((batch_sz, -1, 4))
        # [b, -1]
        scores, labels = probs.max(2)
        return boxes, labels, scores

    def encode(self, boxes, labels):
        """Convert boxes and labels to output of network.

        Args:
            boxes: tensor [b, n, 4]
            labels: tensor [b, n]

        Returns: tensor [b, a*d, g, g]
        """
        batch_sz = boxes.shape[0]
        n = boxes.shape[1]
        out = torch.zeros((batch_sz, self.num_ancs, self.det_sz,
                           self.grid_sz, self.grid_sz), dtype=torch.float)

        for batch_ind in range(batch_sz):
            # [n, 2]
            box_mins = boxes[batch_ind, :, :2]
            box_maxs = boxes[batch_ind, :, 2:]
            box_centers = box_mins + (box_maxs - box_mins) / 2
            match_grid_inds = ((box_centers + 1.0) / self.cell_sz).int()

            for n_ind in range(n):
                # [4]
                box = boxes[batch_ind, n_ind, :]
                # [2]
                grid_ind = match_grid_inds[n_ind, :].tolist()
                # [a, 4]
                ancs = self.ancs[grid_ind[0], grid_ind[1], :, :]
                # [a, 1]
                ious = compute_iou(ancs, box.unsqueeze(0))
                best_anc_ind = ious.squeeze().argmax()
                # [4]
                anc = ancs[best_anc_ind, :]
                # [2]
                anc_center = self.cell_centers[grid_ind[0], grid_ind[1], :]
                # TODO handle collisions

                # [2]
                offset = (anc_center - box_centers[n_ind, :]) / self.cell_sz
                scales = (box[2:]-box[:2]) / (anc[2:]-anc[:2]) / self.cell_sz

                out[batch_ind, best_anc_ind, 0:2, grid_ind[0], grid_ind[1]] = offset
                out[batch_ind, best_anc_ind, 2:4, grid_ind[0], grid_ind[1]] = scales
                out[batch_ind, best_anc_ind, 4 + labels[batch_ind, n_ind], grid_ind[0], grid_ind[1]] = 1
        return out.reshape(self.get_out_shape(batch_sz))

    '''
    def encode2(self, boxes, labels, num_classes, anc_sizes, grid_sz):
        """Convert boxes and labels to output of network.

        Args:
            boxes: tensor [b, n, 4]
            labels: tensor [b, n]
            anc_sizes: tensor [d, 2]
        """
        num_dets = len(anc_sizes)
        cell_sz = get_cell_sz(grid_sz)
        batch_sz = boxes.shape[0]
        det_sz = get_det_sz(num_classes)

        out = torch.zeros((batch_sz, num_dets, det_sz, grid_sz, grid_sz),
                        dtype=torch.float)

        # [b, n, 2]
        box_mins = boxes[:, :, 0:2]
        box_maxs = boxes[:, :, 2:4]
        box_centers = box_mins + (box_maxs - box_mins) / 2
        match_cells = ((box_centers + 1.0) / cell_sz).trunc()
        cell_centers = get_cell_center(match_cells, cell_sz)

        # [d, 2]
        half_ancs = (cell_sz * anc_sizes / 2)
        # [b, n, 1, 2]
        cell_centers = cell_centers.unsqueeze(2)
        # [b, n, d, 2]
        anc_mins = cell_centers - half_ancs
        anc_maxs = cell_centers + half_ancs
        # [b, n, d, 4]
        ancs = torch.cat((anc_mins, anc_maxs), dim=3).clamp(-1, 1)

        # IOU for each anchor
        # compute offset and scale for each box
    '''