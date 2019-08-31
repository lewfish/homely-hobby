import math
from os.path import join
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import torch

from mlx.od.plot import Plotter
from mlx.filesystem.utils import make_dir, zipdir
from mlx.od.centernet.encoder import encode
from mlx.od.centernet.utils import get_positions
from mlx.od.boxlist import to_box_pixel, BoxList

class FasterRCNNPlotter(Plotter):
    def make_debug_plots(self, dataloader, model, classes, output_dir,
                         max_plots=25, score_thresh=0.4):
        preds_dir = join(output_dir, 'preds')
        zip_path = join(output_dir, 'preds.zip')
        make_dir(preds_dir, force_empty=True)

        model.eval()
        for batch_x, batch_y in dataloader:
            with torch.no_grad():
                device = list(model.parameters())[0].device
                batch_x = batch_x.to(device=device)
                batch_sz = batch_x.shape[0]
                batch_boxlist = model(batch_x)
            
            for img_ind in range(batch_sz):
                x = batch_x[img_ind].cpu()
                y = batch_y[img_ind].cpu()
                boxlist = batch_boxlist[img_ind].score_filter(score_thresh).cpu()

                # Plot image, ground truth, and predictions
                fig = self.plot_image_preds(x, y, boxlist, classes)
                plt.savefig(join(preds_dir, '{}-images.png'.format(img_ind)), dpi=200,
                            bbox_inches='tight')
                plt.close(fig)
            break

        zipdir(preds_dir, zip_path)
        shutil.rmtree(preds_dir)
