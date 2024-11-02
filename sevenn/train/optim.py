import torch.nn as nn
import torch.optim.lr_scheduler as scheduler
from torch.optim import adagrad, adam, adamw, radam, sgd

optim_dict = {
    'sgd': sgd.SGD,
    'adagrad': adagrad.Adagrad,
    'adam': adam.Adam,
    'adamw': adamw.AdamW,
    'radam': radam.RAdam,
}


scheduler_dict = {
    'steplr': scheduler.StepLR,
    'multisteplr': scheduler.MultiStepLR,
    'exponentiallr': scheduler.ExponentialLR,
    'cosineannealinglr': scheduler.CosineAnnealingLR,
    'reducelronplateau': scheduler.ReduceLROnPlateau,
    'linearlr': scheduler.LinearLR,
}

loss_dict = {'mse': nn.MSELoss, 'huber': nn.HuberLoss}
