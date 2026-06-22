from typing import Any, Dict, Iterable, Optional, Union

import torch
from tqdm import tqdm

import sevenn._keys as KEY
from sevenn.error_recorder import ErrorRecorder
from sevenn.train.loss import get_loss_functions_from_config
from sevenn.train.optim import optim_dict, scheduler_dict
from sevenn.train.trainer import Trainer


class ReewcTrainer(Trainer):
    """
    Trainer with reEWC experience replay: one extra optimizer step on a memory
    (old-task) batch per training batch, to mitigate catastrophic forgetting.
    """

    def __init__(
        self, *args, memory_loader: Optional[Iterable] = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory_loader = memory_loader

    @staticmethod
    def from_config(
        model: torch.nn.Module,
        config: Dict[str, Any],
        memory_loader: Optional[Iterable] = None,
    ) -> 'ReewcTrainer':
        trainer = ReewcTrainer(
            model,
            loss_functions=get_loss_functions_from_config(config),
            optimizer_cls=optim_dict[config.get(KEY.OPTIMIZER, 'adam').lower()],
            optimizer_args=config.get(KEY.OPTIM_PARAM, {}),
            scheduler_cls=scheduler_dict[
                config.get(KEY.SCHEDULER, 'exponentiallr').lower()
            ],
            scheduler_args=config.get(KEY.SCHEDULER_PARAM, {}),
            device=config.get(KEY.DEVICE, 'auto'),
            distributed=config.get(KEY.IS_DDP, False),
            distributed_backend=config.get(KEY.DDP_BACKEND, 'nccl'),
            memory_loader=memory_loader,
        )
        return trainer

    def run_one_epoch(
        self,
        loader: Iterable,
        is_train: bool = False,
        error_recorder: Optional[ErrorRecorder] = None,
        memory_error_recorder: Optional[ErrorRecorder] = None,
        wrap_tqdm: Union[bool, int] = False,
    ) -> None:
        # Without a memory set to replay, behave exactly like the base Trainer.
        if not (is_train and self.memory_loader is not None):
            super().run_one_epoch(
                loader, is_train, error_recorder, wrap_tqdm=wrap_tqdm
            )
            return

        self.model.train()
        if wrap_tqdm:
            total_len = wrap_tqdm if isinstance(wrap_tqdm, int) else None
            loader = tqdm(loader, total=total_len)

        mem_iter = iter(self.memory_loader)
        for _, batch in enumerate(loader):
            self.optimizer.zero_grad()
            batch = batch.to(self.device, non_blocking=True)
            output = self.model(batch)
            if error_recorder is not None:
                error_recorder.update(output)
            total_loss = torch.tensor([0.0], device=self.device)
            for loss_def, w in self.loss_functions:
                indv_loss = loss_def.get_loss(output, self.model)
                if indv_loss is not None:
                    total_loss += (indv_loss * w)
            total_loss.backward()
            self.optimizer.step()

            # reEWC rehearsal: replay one memory batch with an independent
            # optimizer step.
            try:
                mem_batch = next(mem_iter)
            except StopIteration:
                mem_iter = iter(self.memory_loader)
                mem_batch = next(mem_iter)
            mem_batch = mem_batch.to(self.device, non_blocking=True)
            mem_output = self.model(mem_batch)
            if memory_error_recorder is not None:
                memory_error_recorder.update(mem_output)
            self.optimizer.zero_grad()
            mem_loss = torch.tensor([0.0], device=self.device)
            for loss_def, w in self.loss_functions:
                indv_loss = loss_def.get_loss(mem_output, self.model)
                if indv_loss is not None:
                    mem_loss += (indv_loss * w)
            mem_loss.backward()
            self.optimizer.step()

        if self.distributed and error_recorder is not None:
            self.recorder_all_reduce(error_recorder)
