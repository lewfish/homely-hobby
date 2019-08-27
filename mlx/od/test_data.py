from mlx.od.config import cfg, process_config
from mlx.od.data import build_databunch
from mlx.od.plot import plot_dataloader

cfg.base_uri = '/opt/data/pascal2007'
cfg.test_mode = True
process_config(cfg)

tmp_dir = '/opt/data/'
databunch = build_databunch(cfg, tmp_dir)

output_dir = '/opt/data/test/'
plot_dataloader(databunch.train_dl, databunch.label_names, output_dir)
