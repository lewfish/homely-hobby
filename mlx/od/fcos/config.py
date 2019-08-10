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
cfg.solver.overfit_num_epochs = 500
cfg.solver.overfit_sync_interval = 1000
cfg.solver.test_num_epochs = 1

cfg.data = CN()
cfg.data.dataset = "pascal2007"
cfg.data.data_uri = ''
cfg.data.img_sz = 448
cfg.data.num_workers = 4

cfg.test_mode = False
cfg.overfit_mode = False
cfg.output_uri = ''
cfg.base_uri = ''

def load_config(config_path, opts):
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(opts)

    if cfg.base_uri:
        cfg.output_uri = join(cfg.base_uri, cfg.output_uri)
        cfg.data.data_uri = join(cfg.base_uri, cfg.data.data_uri)

    img_sz = cfg.data.img_sz
    if cfg.overfit_mode:
        cfg.solver.num_epochs = cfg.solver.overfit_num_epochs
        cfg.solver.sync_interval = cfg.solver.overfit_sync_interval
        cfg.data.img_sz = img_sz // 2
        cfg.solver.batch_sz = 4

    if cfg.test_mode:
        cfg.solver.num_epochs = cfg.solver.test_num_epochs
        cfg.data.img_sz = img_sz // 2
        cfg.solver.batch_sz = 4
        cfg.data.num_workers = 0

    cfg.freeze()
    return cfg
