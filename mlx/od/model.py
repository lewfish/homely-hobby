from mlx.od.fcos.model import FCOS
from mlx.od.centernet.model import CenterNet

fcos = 'fcos'
centernet = 'centernet'
model_types = [fcos, centernet]

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
        return CenterNet(
            cfg.model.centernet.backbone_arch, num_labels,
            (cfg.data.img_sz, cfg.data.img_sz))
