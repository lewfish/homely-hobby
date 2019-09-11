import torch.optim as optim
import torch
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR

def build_optimizer(cfg, model):
    cfg.solver.lr
    opt = optim.Adam(model.parameters(), lr=cfg.solver.lr)
    return opt

def build_scheduler(cfg, databunch, opt, start_epoch):
    step_scheduler, epoch_scheduler = None, None
    if cfg.solver.one_cycle and cfg.solver.num_epochs > 1:
        steps_per_epoch = len(databunch.train_ds) // cfg.solver.batch_sz
        total_steps = cfg.solver.num_epochs * steps_per_epoch
        step_size_up = (cfg.solver.num_epochs // 2) * steps_per_epoch
        step_size_down = total_steps - step_size_up
        step_scheduler = CyclicLR(
            opt, base_lr=cfg.solver.lr / 10, max_lr=cfg.solver.lr, 
            step_size_up=step_size_up, step_size_down=step_size_down,
            cycle_momentum=False)
        for _ in range(start_epoch * steps_per_epoch):
            step_scheduler.step()
    elif cfg.solver.multi_stage:
        epoch_scheduler = MultiStepLR(
            opt, milestones=cfg.solver.multi_stage, gamma=0.1)
        for _ in range(start_epoch):
            step_scheduler.step()

    return step_scheduler, epoch_scheduler