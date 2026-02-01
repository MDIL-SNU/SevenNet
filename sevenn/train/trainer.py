import os
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import sevenn._keys as KEY
from sevenn.error_recorder import ErrorRecorder
from sevenn.train.loss import L2Regularization, LossDefinition

from .loss import get_loss_functions_from_config, get_regularization_from_config
from .optim import optim_dict, scheduler_dict


class Trainer:
    """
    Training routine specialized for this package. Depends on 'sevenn.train.loss'

    Args:
        model: model to train
        loss_functions: List of tuples of [LossDefinition, float]. 'float' is for
                        loss weight for each Loss function
        optimizer_cls: torch optimizer class to initialize
        optimizer_args: optimizer keyword argument except 'param'
        scheduler_cls: torch scheduler class to initialize, can be None
        optimizer_args: optimizer keyword argument except 'optimizer'
        device: device to train model, defaults to 'auto'
        distributed: whether this is distributed training
        distributed_backend: torch DDP backend. Should be one of 'nccl', 'mpi'
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_functions: List[Tuple[LossDefinition, float]],
        reg_functions: Optional[List[Tuple[L2Regularization, float]]] = None,
        optimizer_cls=None,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls=None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        grad_clip_norm_th: Optional[float] = None,
        device: Union[torch.device, str] = 'auto',
        distributed: bool = False,
        distributed_backend: str = 'nccl',
    ) -> None:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if distributed_backend == 'mpi':
                device = 'cpu'

        if distributed:
            local_rank = int(os.environ['LOCAL_RANK'])
            self.rank = local_rank
            if distributed_backend == 'nccl':
                device = torch.device('cuda', local_rank)
                self.model = DDP(model.to(device), device_ids=[device])
            elif distributed_backend == 'mpi':
                self.model = DDP(model.to(device))
            else:
                raise ValueError(f'Unknown DDP backend: {distributed_backend}')
            dist.barrier()
            self.model.module.set_is_batch_data(True)
        else:
            self.model = model.to(device)
            self.model.set_is_batch_data(True)
            self.rank = 0

        self.device = torch.device(device)
        self.distributed = distributed

        optimizer_args = optimizer_args or {}
        param = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer_cls(param, **optimizer_args)
        if scheduler_cls is not None:
            scheduler_args = scheduler_args or {}
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_args)
        else:
            self.scheduler = None
        self.loss_functions = loss_functions
        self.reg_functions = reg_functions or []
        self.grad_clip_norm_th = grad_clip_norm_th

    @staticmethod
    def from_config(model: torch.nn.Module, config: Dict[str, Any]) -> 'Trainer':
        reg_functions = get_regularization_from_config(
            config, list(model._modules.keys())
        )
        trainer = Trainer(
            model,
            loss_functions=get_loss_functions_from_config(config),
            reg_functions=reg_functions,
            optimizer_cls=optim_dict[config.get(KEY.OPTIMIZER, 'adam').lower()],
            optimizer_args=config.get(KEY.OPTIM_PARAM, {}),
            scheduler_cls=scheduler_dict[
                config.get(KEY.SCHEDULER, 'exponentiallr').lower()
            ],
            scheduler_args=config.get(KEY.SCHEDULER_PARAM, {}),
            grad_clip_norm_th=config.get(KEY.GRAD_CLIP, None),
            device=config.get(KEY.DEVICE, 'auto'),
            distributed=config.get(KEY.IS_DDP, False),
            distributed_backend=config.get(KEY.DDP_BACKEND, 'nccl'),
        )
        return trainer

    @staticmethod
    def args_from_checkpoint(checkpoint: str) -> Tuple[Dict, Dict, Dict]:
        """
        Usage:
            trainer_args, optim_stct, scheduler_stct = args_from_checkpoint('7net-0')
            # Do what you want to do here
            trainer = Trainer(**trainer_args)
            trainer.load_state_dict(
                optimizer_state_dict=optim_stct,
                scheduler_state_dict=scheduler_stct,
        """
        from sevenn.util import load_checkpoint

        cp = load_checkpoint(checkpoint)

        model = cp.build_model()
        config = cp.config
        optimizer_cls = optim_dict[config[KEY.OPTIMIZER].lower()]
        scheduler_cls = scheduler_dict[config[KEY.SCHEDULER].lower()]
        loss_functions = get_loss_functions_from_config(config)
        reg_functions = get_regularization_from_config(
            config, list(model._modules.keys())
        )

        return (
            {
                'model': model,
                'loss_functions': loss_functions,
                'reg_functions': reg_functions,
                'optimizer_cls': optimizer_cls,
                'optimizer_args': config[KEY.OPTIM_PARAM],
                'scheduler_cls': scheduler_cls,
                'scheduler_args': config[KEY.SCHEDULER_PARAM],
            },
            cp.optimizer_state_dict,
            cp.scheduler_state_dict,
        )

    def run_one_epoch(
        self,
        loader: Iterable,
        is_train: bool = False,
        error_recorder: Optional[ErrorRecorder] = None,
        wrap_tqdm: Union[bool, int] = False,
    ) -> None:
        """
        Run single epoch with given dataloader
        Args:
            loader: iterable yields AtomGraphData
            is_train: if true, do backward() and optimizer step
            error_recorder: ErrorRecorder instance to compute errors (RMSEm MAE, ..)
            wrap_tqdm: wrap given dataloader with tqdm for progress bar
        """
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        if wrap_tqdm:
            total_len = wrap_tqdm if isinstance(wrap_tqdm, int) else None
            loader = tqdm(loader, total=total_len)
        _model = self.model if not self.distributed else self.model.module
        for _, batch in enumerate(loader):
            if is_train:
                self.optimizer.zero_grad()
            batch = batch.to(self.device, non_blocking=True)
            output = self.model(batch)
            if error_recorder is not None:
                error_recorder.update(output, model=_model)
            if is_train:
                total_loss = torch.tensor([0.0], device=self.device)
                for loss_def, w in self.loss_functions:
                    indv_loss = loss_def.get_loss(output, _model)
                    if indv_loss is not None:
                        total_loss += (indv_loss * w)
                for reg_def, w in self.reg_functions:
                    reg_loss = reg_def.get_loss(output, _model)
                    total_loss += reg_loss * w / 2
                total_loss.backward()
                if self.grad_clip_norm_th is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_norm_th,
                    )
                self.optimizer.step()  # DDP syncs weight here

        if self.distributed and error_recorder is not None:
            self.recorder_all_reduce(error_recorder)

    def train_one_batch(
        self,
        batch,
        error_recorder: Optional[ErrorRecorder] = None,
    ) -> None:
        """
        Train on a single batch. Used for batch-level training loop.

        Args:
            batch: AtomGraphData batch
            error_recorder: ErrorRecorder instance to compute errors
        """
        self.model.train()
        self.optimizer.zero_grad()
        batch = batch.to(self.device, non_blocking=True)
        output = self.model(batch)
        _model = self.model if not self.distributed else self.model.module

        if error_recorder is not None:
            error_recorder.update(output, model=_model)

        total_loss = torch.tensor([0.0], device=self.device)
        for loss_def, w in self.loss_functions:
            total_loss += loss_def.get_loss(output, _model) * w
        for reg_def, w in self.reg_functions:
            reg_loss = reg_def.get_loss(output, _model)
            total_loss += reg_loss * w / 2

        total_loss.backward()

        # TODO: NaN sanitizer - replace NaN/Inf gradients with zero
        # for name, p in self.model.named_parameters():
        #     if p.grad is None:
        #         continue
        #     if not torch.isfinite(p.grad).all():
        #         if self.rank == 0:
        #             print(
        #                 f'[nan2zero] NaN/Inf gradient detected in {p.shape}, '
        #                 'resetting to 0',
        #                 flush=True,
        #             )
        #         p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        # Grad clipping
        if self.grad_clip_norm_th is not None:
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm_th,
            )
            if norm > self.grad_clip_norm_th and self.rank == 0:
                print(
                    f'[Clipping] Grad norm {norm:.2f} into '
                    f'{self.grad_clip_norm_th}',
                    flush=True,
                )
        self.optimizer.step()

    def scheduler_step(self, metric: Optional[float] = None) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            assert isinstance(metric, float)
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]['lr'])

    def recorder_all_reduce(self, recorder: ErrorRecorder) -> None:
        for metric in recorder.metrics:
            # metric.value._ddp_reduce(self.device)
            metric.ddp_reduce(self.device)

    def get_checkpoint_dict(self) -> Dict[str, Any]:
        if self.distributed:
            model_state_dct = self.model.module.state_dict()
        else:
            model_state_dct = self.model.state_dict()

        return {
            'model_state_dict': model_state_dct,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'hash': uuid.uuid4().hex,
        }

    def write_checkpoint(self, path: str, **extra) -> None:
        if self.distributed and self.rank != 0:
            return
        cp = self.get_checkpoint_dict()
        cp.update(**extra)
        torch.save(cp, path)

    def load_state_dicts(
        self,
        model_state_dict: Optional[Dict] = None,
        optimizer_state_dict: Optional[Dict] = None,
        scheduler_state_dict: Optional[Dict] = None,
        strict: bool = True,
    ) -> None:
        if model_state_dict is not None:
            if self.distributed:
                self.model.module.load_state_dict(model_state_dict, strict=strict)
            else:
                self.model.load_state_dict(model_state_dict, strict=strict)

        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        if scheduler_state_dict is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)
