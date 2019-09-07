from mlx.od.fcos.model import FCOS
from mlx.od.centernet.model import CenterNet
from mlx.od.faster_rcnn.model import MyFasterRCNN

fcos = 'fcos'
centernet = 'centernet'
faster_rcnn = 'faster_rcnn'
model_types = [fcos, centernet, faster_rcnn]

def validate_model_type(model_type):
    if model_type not in model_types:
        raise ValueError('{} is not a valid model_type'.format(model_type))

def build_model(cfg, num_labels):
    model_type = cfg.model.type
    validate_model_type(model_type)
    if model_type == fcos:
        return FCOS(
            cfg.model.fcos.backbone_arch, num_labels,
            levels=cfg.model.fcos.levels)
    elif model_type == centernet:
        centernet_opts = cfg.model.centernet
        return CenterNet(
            centernet_opts.backbone_arch, num_labels, 
            loss_alpha=centernet_opts.loss_alpha, 
            loss_beta=centernet_opts.loss_beta)
    elif model_type == faster_rcnn:
        return MyFasterRCNN(
            num_labels, cfg.data.img_sz)

