import os
from os.path import join

from yacs.config import CfgNode as CN

cfg = CN()

cfg.model = CN()
cfg.model.type = "tiny"
cfg.model.init_weights = ""

cfg.solver = CN()
cfg.solver.lr = 2e-4
cfg.solver.num_epochs = 10
cfg.solver.test_num_epochs = 2
cfg.solver.overfit_num_steps = 100
cfg.solver.test_overfit_num_steps = 2
cfg.solver.sync_interval = 2
cfg.solver.batch_sz = 64
cfg.solver.one_cycle = False
cfg.solver.multi_stage = []

cfg.data = CN()
cfg.data.dataset = "mnist"
cfg.data.img_sz = 32
cfg.data.num_workers = 4

cfg.predict_mode = False
cfg.test_mode = False
cfg.overfit_mode = False
cfg.eval_train = False
cfg.base_uri = ''
cfg.output_dir = 'output'

def process_config(cfg):
    if cfg.base_uri == '':
        raise ValueError('Must set base_uri')
    cfg.output_uri = join(cfg.base_uri, cfg.output_dir)

    img_sz = cfg.data.img_sz
    if cfg.overfit_mode:
        cfg.data.img_sz = img_sz // 2
        cfg.solver.batch_sz = 8
        if cfg.test_mode:
            cfg.solver.overfit_num_steps = cfg.solver.test_overfit_num_steps

    if cfg.test_mode:
        cfg.solver.num_epochs = cfg.solver.test_num_epochs
        cfg.data.img_sz = img_sz // 2
        cfg.solver.batch_sz = 4
        cfg.data.num_workers = 0

def load_config(config_path, opts):
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(opts)
    process_config(cfg)
    cfg.freeze()
    return cfg
