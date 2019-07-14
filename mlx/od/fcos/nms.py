from shapely import geometry
from shapely.strtree import STRtree
import numpy as np

def compute_iou(geom1, geom2):
    return geom1.intersection(geom2).area / geom1.union(geom2).area

def compute_nms(boxes, labels, scores, iou_thresh=0.7):
    good_inds = []
    for label in set(labels):
        label_inds = np.nonzero(labels == label)[0]

        geoms = []
        for ind in label_inds:
            box = boxes[ind]
            geom = geometry.box(box[1], box[0], box[3], box[2])
            geom.ind = ind
            geom.good = True
            geoms.append(geom)
        tree = STRtree(geoms)

        for geom in geoms:
            geom_score = scores[geom.ind]
            overlaps = tree.query(geom)
            for other in overlaps:
                if other != geom and other.good:
                    iou = compute_iou(geom, other)
                    if iou > iou_thresh:
                        other_score = scores[other.ind]
                        if other_score <= geom_score:
                            other.good = False

        good_inds.extend([geom.ind for geom in geoms if geom.good])
    return good_inds