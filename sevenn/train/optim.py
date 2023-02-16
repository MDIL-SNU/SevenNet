import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

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
    'cosineannealinglr': scheduler.CosineAnnealingLR
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
    }
}
