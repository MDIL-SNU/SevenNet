import math
from typing import Optional

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler
from torch.optim import adagrad, adam, adamw, radam, sgd
from torch.optim.lr_scheduler import _LRScheduler

from .muon import MuonWithAuxAdam


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
    'muon': MuonWithAuxAdam,
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


class WarmupStableDecayLR(_LRScheduler):
    """
    Warmup-stable-decay (WSD) learning-rate schedule, stepped every batch.

    Per optimizer group with peak ``max_lr``: linear warmup from
    ``max_lr / div_factor`` to ``max_lr`` over the first ``pct_warmup`` of
    ``total_steps``, constant ``max_lr`` until ``pct_stable_end``, then
    cosine decay to ``max_lr / (div_factor * final_div_factor)`` at the last
    step (held there afterwards).  Initial/peak/final learning rates and the
    phase-boundary convention (``end_step = pct * total_steps - 1``) follow
    torch ``OneCycleLR``, so swapping ``onecyclelr`` for ``wsd`` with the same
    ``max_lr``/``div_factor``/``final_div_factor`` only changes the shape.

    Sources (idea only, own implementation):
      - MiniCPM, Hu et al. 2024, arXiv:2404.06395 (Sec. 4, WSD scheduler).
      - DPA-4, arXiv:2606.02419 (Table S-16: warmup/stable/decay settings).

    Args:
        optimizer: wrapped optimizer.
        max_lr: peak learning rate, float or one value per param group.
        total_steps: total number of scheduler steps; or give ``epochs`` and
            ``steps_per_epoch`` (``total_steps`` wins when both are given).
        pct_warmup: fraction of steps spent in linear warmup.
        pct_stable_end: fraction of steps at which the cosine decay starts.
        div_factor: initial lr = max_lr / div_factor.
        final_div_factor: final lr = initial lr / final_div_factor.
        last_epoch: index of the last step (-1 for a fresh schedule).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr,
        total_steps: Optional[int] = None,
        epochs: Optional[float] = None,
        steps_per_epoch: Optional[int] = None,
        pct_warmup: float = 0.01,
        pct_stable_end: float = 0.35,
        div_factor: float = 25.0,
        final_div_factor: float = 4.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps is None:
            if epochs is None or steps_per_epoch is None:
                raise ValueError(
                    'wsd: give total_steps, or epochs and steps_per_epoch'
                )
            total_steps = math.ceil(epochs * steps_per_epoch)
        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError(f'wsd: total_steps must be > 0, got {total_steps}')
        if not 0.0 <= pct_warmup <= pct_stable_end <= 1.0:
            raise ValueError(
                'wsd: need 0 <= pct_warmup <= pct_stable_end <= 1, got '
                f'{pct_warmup}, {pct_stable_end}'
            )
        n_group = len(optimizer.param_groups)
        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != n_group:
                raise ValueError(
                    f'wsd: expected {n_group} max_lr values, got {len(max_lr)}'
                )
            max_lrs = [float(v) for v in max_lr]
        else:
            max_lrs = [float(max_lr)] * n_group

        self.total_steps = total_steps
        self.pct_warmup = float(pct_warmup)
        self.pct_stable_end = float(pct_stable_end)
        self.max_lrs = max_lrs
        self.initial_lrs = [v / div_factor for v in max_lrs]
        self.min_lrs = [v / final_div_factor for v in self.initial_lrs]
        # phase ends in step units, OneCycleLR convention
        self.warmup_end = self.pct_warmup * total_steps - 1
        self.stable_end = self.pct_stable_end * total_steps - 1
        self.decay_end = float(total_steps - 1)
        if last_epoch == -1:
            for group, lr in zip(optimizer.param_groups, self.initial_lrs):
                group['initial_lr'] = lr
        super().__init__(optimizer, last_epoch)

    def _lr_at(self, step: float, idx: int) -> float:
        lr0 = self.initial_lrs[idx]
        lr_max = self.max_lrs[idx]
        lr_min = self.min_lrs[idx]
        if self.warmup_end > 0 and step <= self.warmup_end:
            return lr0 + (lr_max - lr0) * step / self.warmup_end
        if step <= self.stable_end:
            return lr_max
        if step >= self.decay_end:
            return lr_min
        start = max(self.stable_end, self.warmup_end, 0.0)
        pct = (step - start) / (self.decay_end - start)
        return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * pct))

    def get_lr(self):
        step = self.last_epoch
        return [self._lr_at(step, i) for i in range(len(self.max_lrs))]


scheduler_dict = {
    'steplr': scheduler.StepLR,
    'multisteplr': scheduler.MultiStepLR,
    'exponentiallr': scheduler.ExponentialLR,
    'cosineannealinglr': scheduler.CosineAnnealingLR,
    'cosineannealingwarmuplr': CosineAnnealingWarmupRestarts,
    'reducelronplateau': scheduler.ReduceLROnPlateau,
    'linearlr': scheduler.LinearLR,
    'onecyclelr': scheduler.OneCycleLR,
    'wsd': WarmupStableDecayLR,
}

loss_dict = {
    'mse': nn.MSELoss,
    'huber': nn.HuberLoss,
    'mae': nn.L1Loss,
    'l2mae': L2MAE,
}
