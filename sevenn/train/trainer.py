from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sevenn._keys as KEY
from sevenn._const import LossType
from sevenn.error_recorder import ErrorRecorder
from sevenn.train.optim import loss_dict, optim_dict, scheduler_dict
from sevenn.util import postprocess_output


class Trainer:
    def __init__(self, model, config: dict):
        self.distributed = config[KEY.IS_DDP]

        if self.distributed:
            device = torch.device('cuda', config[KEY.LOCAL_RANK])
            dist.barrier()
            self.model = DDP(model.to(device), device_ids=[device])
            self.model.module.set_is_batch_data(True)
            self.rank = config[KEY.LOCAL_RANK]
        else:
            device = config[KEY.DEVICE]
            self.model = model.to(device)
            self.model.set_is_batch_data(True)
        self.device = device

        self.loss_weights = {LossType.ENERGY: 1.0}
        self.loss_weights[LossType.FORCE] = config[KEY.FORCE_WEIGHT]

        # where trace stress is used for?
        self.is_trace_stress = config[KEY.IS_TRACE_STRESS]
        self.is_train_stress = config[KEY.IS_TRAIN_STRESS]
        self.is_stress = self.is_trace_stress or self.is_train_stress
        if self.is_train_stress:
            self.loss_weights[LossType.STRESS] = config[KEY.STRESS_WEIGHT]

        # init loss_types
        self.loss_types = [LossType.ENERGY, LossType.FORCE]
        if self.is_stress:
            self.loss_types.append(LossType.STRESS)

        param = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim_dict[config[KEY.OPTIMIZER].lower()]
        optim_param = config[KEY.OPTIM_PARAM]
        self.optimizer = optimizer(param, **optim_param)

        scheduler = scheduler_dict[config[KEY.SCHEDULER].lower()]
        scheduler_param = config[KEY.SCHEDULER_PARAM]
        self.scheduler = scheduler(self.optimizer, **scheduler_param)

        loss = loss_dict[config[KEY.LOSS].lower()]
        # TODO: handle this kind of case in parse_input not here
        try:
            loss_param = config[KEY.LOSS_PARAM]
        except KeyError:
            loss_param = {}
        self.criterion = loss(**loss_param)

    def run_one_epoch(
        self, loader, is_train=False, error_recorder: ErrorRecorder = None
    ):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        for step, batch in enumerate(loader):
            if is_train:
                self.optimizer.zero_grad()
            batch = batch.to(self.device, non_blocking=True)
            output = self.model(batch)
            error_recorder.update(output)
            if is_train:
                total_loss = self.loss_calculator(output)
                total_loss.backward()
                self.optimizer.step()

        if self.distributed:
            self.recorder_all_reduce(error_recorder)

    def loss_calculator(self, output):
        unit_converted = postprocess_output(output, self.loss_types)
        total_loss = torch.tensor([0.0], device=self.device)
        for loss_type in self.loss_types:
            pred, ref, _ = unit_converted[loss_type]
            total_loss += (
                self.criterion(pred, ref) * self.loss_weights[loss_type]
            )
        return total_loss

    def scheduler_step(self, metric=None):
        if self.scheduler is None:
            return
        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def recorder_all_reduce(self, recorder: ErrorRecorder):
        for metric in recorder.metrics:
            # metric.value._ddp_reduce(self.device)
            metric.ddp_reduce(self.device)

    # Not used, ddp automatically averages gradients
    def average_gradient(self):
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

    def get_checkpoint_dict(self):
        if self.distributed:
            model_state_dct = self.model.module.state_dict()
        else:
            model_state_dct = self.model.state_dict()
        return {
            'model_state_dict': model_state_dct,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

    def load_state_dicts(
        self,
        model_state_dict,
        optimizer_state_dict,
        scheduler_state_dict,
        strict=True,
    ):
        if self.distributed:
            self.model.module.load_state_dict(model_state_dict, strict=strict)
        else:
            self.model.load_state_dict(model_state_dict, strict=strict)

        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        if scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)
