import torch.optim as optim

def build_optimizer(cfg, model):
    cfg.solver.lr
    opt = optim.Adam(model.parameters(), lr=cfg.solver.lr)
    return opt
