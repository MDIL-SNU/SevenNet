import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn as nn

optim_dict = {
    'sgd': optim.SGD,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'radam': optim.RAdam
}

optim_param_name_type_dict = {
    'universial': {
        'lr': float,
        'weight_decay': float
    },
    'sgd': {
        'momentum': float,
        'dampening': float,
        'nesterov': bool,
        'maximize': bool,
        'foreach': bool
    },
    'adagrad': {
        'lr_decay': float,
        'eps': float,
        'foreach': bool,
        'maximize': bool
    },
    'adam': {
        'betas': tuple,  # How to make it tuple[float, float]?
        'eps': float,
        'amsgrad': bool,
        'foreach': bool,
        'maximize': bool,
        'capturable': bool,
        'fused': bool
    },
    'adamw': {
        'betas': tuple,  # How to make it tuple[float, float]?
        'eps': float,
        'amsgrad': bool,
        'maximize': bool,
        'foreach': bool,
        'capturable': bool
    },
    'radam': {
        'betas': tuple,  # How to make it tuple[float, float]?
        'eps': float,
        'foreach': bool
    }

}

# Some scheduler use lambda function (e.g. LambdaLR) as input.
# This is not possible using simple yaml configuration.
# TODO: How to implement this?

scheduler_dict = {
    'steplr': scheduler.StepLR,
    'multisteplr': scheduler.MultiStepLR,
    'exponentiallr': scheduler.ExponentialLR,
    'cosineannealinglr': scheduler.CosineAnnealingLR,
    'reducelronplateau': scheduler.ReduceLROnPlateau
}

scheduler_param_name_type_dict = {
    'universial': {
        'last_epoch': int,
        'verbose': bool
    },
    'steplr': {
        'step_size': int,
        'gamma': float
    },
    'multisteplr': {
        'milestones': list,  # How to make it list[int]?
        'gamma': float
    },
    'exponentiallr': {
        'gamma': float
    },
    'cosineannealinglr': {
        'T_max': int,
        'eta_min': float
    },
    'reducelronplateau': {
        'mode': str,
        'factor': float,
        'patience': int,
        'threshold': float,
        'threshold_mode': str,
        'cooldown': int,
        'min_lr': float,
        'eps': float,
    }
}

loss_dict = {"mse": nn.MSELoss, "huber": nn.HuberLoss}
loss_param_name_type_dict = {
    'universial': {},
    "mse": {},
    "huber": {"delta": float}  # default = 1.0
}


