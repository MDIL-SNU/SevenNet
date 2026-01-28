import torch
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler
from torch.optim import adagrad, adam, adamw, radam, sgd


class L2MAE(nn.Module):
    """
    L2 norm (Frobenius norm) based MAE loss.
    For stress, norm of 3*3 matrix is used for invariance
    """

    def __init__(
        self,
        prop: str = 'force',
        reduction: str = 'mean',
    ):
        super().__init__()
        self.prop = prop
        self.dim = 3 if prop == 'force' else 6
        self.reduction = reduction

    def forward(self, input, target):
        if self.prop == 'force':
            diff = input.view([-1, self.dim]) - target.view([-1, self.dim])
        else:
            diff = input.view([-1, self.dim]) - target.view([-1, self.dim])
            diff = torch.cat((diff, diff[:, -3:]), dim=1)
        norm = torch.norm(diff, p=2, dim=-1)
        if self.reduction == 'none':
            # make it (# component * atoms (or structures))
            # for consistency with MSE & MAE
            return torch.repeat_interleave(norm, self.dim)
        if self.reduction == 'mean':
            return torch.mean(norm)
        if self.reduction == 'sum':
            return torch.sum(norm)


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
    'onecyclelr': scheduler.OneCycleLR,
}

loss_dict = {
    'mse': nn.MSELoss,
    'huber': nn.HuberLoss,
    'mae': nn.L1Loss,
    'l2mae': L2MAE,
}
