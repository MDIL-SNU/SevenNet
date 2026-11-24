import math

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler
from torch.optim import adagrad, adam, adamw, radam, sgd
from torch.optim.lr_scheduler import _LRScheduler


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


# Adapted from the cosine_annealing_warmup package, MIT License,
# Copyright (c) 2022 Naoki Katsura.
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup and warm restarts.

    Args:
        optimizer: wrapped optimizer.
        first_cycle_steps: number of steps in the first cycle.
        cycle_mult: cycle length magnification applied at each restart.
        max_lr: maximum (post-warmup) learning rate of the first cycle.
        min_lr: minimum learning rate.
        warmup_steps: number of linear warmup steps.
        gamma: max_lr decay factor applied each cycle.
        last_epoch: index of the last epoch.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super().__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self) -> None:
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int(
                        (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult
                    )
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch
                                / self.first_cycle_steps
                                * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = (
                        self.first_cycle_steps * self.cycle_mult**n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


scheduler_dict = {
    'steplr': scheduler.StepLR,
    'multisteplr': scheduler.MultiStepLR,
    'exponentiallr': scheduler.ExponentialLR,
    'cosineannealinglr': scheduler.CosineAnnealingLR,
    'cosineannealingwarmuplr': CosineAnnealingWarmupRestarts,
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
