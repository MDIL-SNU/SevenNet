import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

optim_dict = {
    'sgd': optim.SGD,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'radam': optim.RAdam,
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
