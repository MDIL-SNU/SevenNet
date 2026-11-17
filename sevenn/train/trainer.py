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
from sevenn.train.loss import LossDefinition

from .loss import get_loss_functions_from_config
from .optim import optim_dict, scheduler_dict


def freeze_to_modals(
    model: torch.nn.Module,
    train_only_modals: List[str],
    rank: int = 0,
) -> None:
    """
    Experimental knob (config key 'train_only_modals'): freeze every parameter
    except the modality-specific slices that belong to the listed modalities.

    Modality-specific parameters are sub-slices of otherwise-shared tensors:
      * modal IrrepsLinear (nn/linear.py): the modal one-hot weights, i.e. the
        last weight-view of `linear.weight`, shape (num_modalities, out_scalars).
        A modality is one row of that view.
      * ModalWiseRescale (nn/scale.py): rows of `shift` / `scale`, but only when
        they are modal-wise and were trainable to begin with.

    Because requires_grad is per-tensor, the partially-trained tensors keep
    requires_grad=True and a backward hook masks their gradient to the selected
    rows only. This blocks (a) data-loss gradient from leaking into the shared
    columns of those layers and (b) L2_modal regularization (train/loss.py) from
    leaking onto the non-selected modalities. Everything else is frozen via
    requires_grad=False and is therefore excluded from the optimizer.
    """
    from sevenn.nn.linear import IrrepsLinear
    from sevenn.nn.scale import ModalWiseRescale

    modal_map = getattr(model, 'modal_map', None)
    if not modal_map:
        raise ValueError(
            'train_only_modals is set but the model has no modal_map; '
            'use_modality must be True with modalities defined'
        )
    missing = [m for m in train_only_modals if m not in modal_map]
    if missing:
        raise ValueError(
            f'train_only_modals {missing} not found in model modal_map '
            f'{sorted(modal_map)}'
        )
    keep_idx = sorted({modal_map[m] for m in train_only_modals})

    def _mask_hook(mask: torch.Tensor):
        cache = {'mask': mask}

        def hook(grad):
            m = cache['mask']
            if m.device != grad.device or m.dtype != grad.dtype:
                m = m.to(device=grad.device, dtype=grad.dtype)
                cache['mask'] = m
            return grad * m

        return hook

    # Snapshot trainability before freezing so train_shift/train_scale (and any
    # other intentionally-frozen params) are respected, not silently re-enabled.
    was_trainable = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)

    report = []
    for name, module in model.named_modules():
        if isinstance(module, IrrepsLinear) and module.num_modalities > 1:
            linear = module.linear
            weight = linear.weight
            mask = torch.zeros_like(weight)
            modal_view = list(linear.weight_views(mask))[-1]
            if modal_view.shape[0] != module.num_modalities:
                raise RuntimeError(
                    f'{name}: last weight-view has {modal_view.shape[0]} rows != '
                    f'num_modalities {module.num_modalities}; modal weight layout '
                    'assumption violated'
                )
            modal_view[keep_idx, :] = 1.0  # writes through into `mask`
            weight.requires_grad_(True)
            weight.register_hook(_mask_hook(mask))
            report.append(f'{name}.linear.weight (rows {keep_idx})')
        elif isinstance(module, ModalWiseRescale):
            for attr, use_mw in (
                ('shift', module.use_modal_wise_shift),
                ('scale', module.use_modal_wise_scale),
            ):
                param = getattr(module, attr)
                if use_mw and was_trainable.get(param, False):
                    mask = torch.zeros_like(param)
                    mask[keep_idx, :] = 1.0
                    param.requires_grad_(True)
                    param.register_hook(_mask_hook(mask))
                    report.append(f'{name}.{attr} (rows {keep_idx})')

    if not report:
        raise ValueError(
            'train_only_modals matched no modal parameters; check that modal '
            'modules (use_modal_*) and/or modal-wise shift/scale are enabled'
        )

    if rank == 0:
        n_tensors = sum(1 for p in model.parameters() if p.requires_grad)
        n_elems = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f'[train_only_modals] modalities {train_only_modals} -> indices '
            f'{keep_idx}; trainable tensors={n_tensors}, '
            f'masked-trainable elements={n_elems}\n  '
            + '\n  '.join(report),
            flush=True,
        )


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
        optimizer_cls=None,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls=None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        grad_clip_norm_th: Optional[float] = None,
        device: Union[torch.device, str] = 'auto',
        distributed: bool = False,
        distributed_backend: str = 'nccl',
        train_only_modals: Optional[List[str]] = None,
    ) -> None:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if distributed_backend == 'mpi':
                device = 'cpu'

        if distributed:
            local_rank = int(os.environ['LOCAL_RANK'])
            # global rank: gates checkpoint writing and rank-0 prints. Using
            # LOCAL_RANK here made local rank 0 of EVERY node write the same
            # checkpoint file concurrently on multi-node runs.
            self.rank = dist.get_rank() if dist.is_initialized() else local_rank
            self.local_rank = local_rank
            if distributed_backend == 'nccl':
                device = torch.device('cuda', local_rank)
            elif distributed_backend != 'mpi':
                raise ValueError(f'Unknown DDP backend: {distributed_backend}')
        else:
            self.rank = 0
            self.local_rank = 0

        model = model.to(device)
        # Experimental: restrict trainable params to selected modalities' slices.
        # Must run after .to(device) (gradient-mask buffers match the param
        # device) and before the DDP wrap (so the reducer sees the final
        # requires_grad) and the optimizer build (so frozen params are excluded).
        if train_only_modals is not None:
            freeze_to_modals(model, train_only_modals, rank=self.rank)

        if distributed:
            if distributed_backend == 'nccl':
                self.model = DDP(model, device_ids=[device])
            else:  # mpi
                self.model = DDP(model)
            dist.barrier()
            self.model.module.set_is_batch_data(True)
        else:
            self.model = model
            self.model.set_is_batch_data(True)

        self.device = torch.device(device)
        self.distributed = distributed

        optimizer_args = optimizer_args or {}
        if hasattr(optimizer_cls, 'from_model'):  # builds its own param groups
            unwrapped_model = self.model.module if distributed else self.model
            self.optimizer = optimizer_cls.from_model(
                unwrapped_model, **optimizer_args
            )
        else:
            param = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optimizer_cls(param, **optimizer_args)
        if scheduler_cls is not None:
            scheduler_args = scheduler_args or {}
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_args)
        else:
            self.scheduler = None
        self.loss_functions = loss_functions
        self.grad_clip_norm_th = grad_clip_norm_th

    @staticmethod
    def from_config(
        model: torch.nn.Module,
        config: Dict[str, Any],
    ) -> 'Trainer':
        trainer = Trainer(
            model,
            loss_functions=get_loss_functions_from_config(config, model),
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
            train_only_modals=config.get(KEY.TRAIN_ONLY_MODALS, None),
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
        loss_functions = get_loss_functions_from_config(config, model)

        return (
            {
                'model': model,
                'loss_functions': loss_functions,
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
        memory_error_recorder: Optional[ErrorRecorder] = None,
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
                        total_loss += indv_loss * w
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

        total_loss.backward()

        # Grad clipping
        if self.grad_clip_norm_th is not None:
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm_th,
            )
            if norm > self.grad_clip_norm_th and self.rank == 0:
                print(
                    f'[Clipping] Grad norm {norm:.2f} into {self.grad_clip_norm_th}',
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
