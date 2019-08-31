import os
from os.path import join

from yacs.config import CfgNode as CN

cfg = CN()

cfg.model = CN()
cfg.model.type = "fcos"
cfg.model.init_weights = ""

cfg.model.fcos = CN()
cfg.model.fcos.backbone_arch = "resnet18"
cfg.model.fcos.levels = [2]

cfg.model.centernet = CN()
cfg.model.centernet.backbone_arch = "resnet18"

cfg.model.faster_rcnn = CN()

cfg.solver = CN()
cfg.solver.lr = 1e-4
cfg.solver.num_epochs = 25
cfg.solver.test_num_epochs = 2
cfg.solver.overfit_num_steps = 500
cfg.solver.test_overfit_num_steps = 2
cfg.solver.sync_interval = 2
cfg.solver.batch_sz = 16
cfg.solver.one_cycle = False
cfg.solver.multi_stage = []
cfg.solver.test_multi_stage = []

cfg.data = CN()
cfg.data.dataset = "pascal2007"
cfg.data.img_sz = 448
cfg.data.num_workers = 0

cfg.predict_mode = False
cfg.test_mode = False
cfg.overfit_mode = False
cfg.lr_find_mode = False
cfg.base_uri = ''
cfg.output_dir = 'output'

def process_config(cfg):
    if cfg.base_uri == '':
        raise ValueError('Must set base_uri')
    cfg.output_uri = join(cfg.base_uri, cfg.output_dir)

    img_sz = cfg.data.img_sz
    if cfg.overfit_mode:
        cfg.data.img_sz = img_sz // 2
        cfg.solver.batch_sz = 4
        if cfg.test_mode:
            cfg.solver.overfit_num_steps = cfg.solver.test_overfit_num_steps

    if cfg.test_mode:
        cfg.solver.num_epochs = cfg.solver.test_num_epochs
        cfg.data.img_sz = img_sz // 2
        cfg.solver.batch_sz = 4
        cfg.data.num_workers = 0
        cfg.solver.multi_stage = cfg.solver.test_multi_stage

def load_config(config_path, opts):
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(opts)
    process_config(cfg)
    cfg.freeze()
    return cfg
