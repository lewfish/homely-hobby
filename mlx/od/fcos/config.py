import os
from os.path import join

from yacs.config import CfgNode as CN

cfg = CN()

cfg.model = CN()
cfg.model.backbone_arch = "resnet18"
cfg.model.levels = [2]

cfg.solver = CN()
cfg.solver.lr = 1e-4
cfg.solver.num_epochs = 25
cfg.solver.sync_interval = 2
cfg.solver.batch_sz = 16

cfg.data = CN()
cfg.data.dataset = "pascal2007"
cfg.data.data_uri = ''
cfg.data.img_sz = 448

cfg.test_mode = False
cfg.overfit_mode = False
cfg.output_uri = ''
cfg.base_uri = ''

cfg.overfit_num_epochs = 500
cfg.overfit_sync_interval = 1000
cfg.test_num_epochs = 1

def load_config(config_path, opts):
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(opts)
    cfg.output_uri = join(cfg.base_uri, cfg.output_uri) if cfg.base_uri else cfg.output_uri
    cfg.data.data_uri = join(cfg.base_uri, cfg.data.data_uri) if cfg.base_uri else cfg.data.data_uri
    cfg.freeze()
    return cfg
