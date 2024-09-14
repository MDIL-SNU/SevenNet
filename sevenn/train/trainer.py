from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn
from torch.nn.parallel import DistributedDataParallel as DDP

import sevenn._keys as KEY
from sevenn.error_recorder import ErrorRecorder

from .loss import get_loss_functions_from_config
from .optim import optim_dict, scheduler_dict


class Trainer:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.distributed = config[KEY.IS_DDP]
        device = config[KEY.DEVICE]

        if self.distributed:
            dist.barrier()
            backend = config[KEY.DDP_BACKEND]
            if backend == "nccl":
                device = torch.device("cuda", config[KEY.LOCAL_RANK])
                self.model = DDP(model.to(device), device_ids=[device])
            elif backend == "mpi":
                self.model = DDP(model.to(device))
            self.model.module.set_is_batch_data(True)
            self.rank = config[KEY.LOCAL_RANK]
        else:
            self.model = model.to(device)
            self.model.set_is_batch_data(True)
        self.device = device

        param = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim_dict[config[KEY.OPTIMIZER].lower()]
        optim_param = config[KEY.OPTIM_PARAM]
        self.optimizer = optimizer(param, **optim_param)

        scheduler = scheduler_dict[config[KEY.SCHEDULER].lower()]
        scheduler_param = config[KEY.SCHEDULER_PARAM]
        self.scheduler = scheduler(self.optimizer, **scheduler_param)

        # This should be outside of the trainer(?)
        # list of tuples (loss_definition, weight)
        self.loss_functions = get_loss_functions_from_config(config)

    def run_one_epoch(
        self,
        loader: Iterable,
        is_train: bool = False,
        error_recorder: Optional[ErrorRecorder] = None,
    ):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        for _, batch in enumerate(loader):
            if is_train:
                self.optimizer.zero_grad()
            batch = batch.to(self.device, non_blocking=True)
            output = self.model(batch)
            if error_recorder is not None:
                error_recorder.update(output)
            if is_train:
                total_loss = torch.tensor([0.0], device=self.device)
                for loss_def, w in self.loss_functions:
                    total_loss += loss_def.get_loss(output, self.model) * w
                total_loss.backward()
                self.optimizer.step()

        if self.distributed and error_recorder is not None:
            self.recorder_all_reduce(error_recorder)

    def scheduler_step(self, metric: Optional[float] = None):
        if self.scheduler is None:
            return
        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            assert isinstance(metric, float)
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def recorder_all_reduce(self, recorder: ErrorRecorder):
        for metric in recorder.metrics:
            # metric.value._ddp_reduce(self.device)
            metric.ddp_reduce(self.device)

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
        model_state_dict: Dict,
        optimizer_state_dict: Dict,
        scheduler_state_dict: Dict,
        strict: bool = True,
    ):
        if self.distributed:
            self.model.module.load_state_dict(model_state_dict, strict=strict)
        else:
            self.model.load_state_dict(model_state_dict, strict=strict)

        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        if scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)
