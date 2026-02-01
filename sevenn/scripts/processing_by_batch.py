import os
import time
from copy import deepcopy
from typing import Optional

import numpy as np

import sevenn._keys as KEY
from sevenn.error_recorder import AverageNumber, ErrorRecorder
from sevenn.logger import Logger
from sevenn.train.trainer import Trainer
from sevenn.util import unique_filepath


def processing_by_batch(
    config: dict,
    trainer: Trainer,
    loaders: dict,
    data_progress: dict,
    start_epoch: int = 1,
    train_loader_key: str = 'trainset',
    error_recorder: Optional[ErrorRecorder] = None,
    total_epoch: Optional[int] = None,
    per_epoch: Optional[float] = None,
    best_metric_loader_key: str = 'validset',
    best_metric: Optional[str] = None,
    write_csv: bool = True,
    working_dir: Optional[str] = None,
):
    """
    Batch-level training loop for large-scale training.

    Unlike epoch-level training (processing_epoch_v2), this function:
    - Saves checkpoints at configurable intervals within an epoch
    - Supports resuming from exact data position
    - Uses OrderedSampler for deterministic DDP training

    Args:
        config: Training configuration dictionary
        trainer: Trainer instance
        loaders: Dict of dataloaders (must include 'trainset')
        data_progress: Dict tracking data iteration progress
        start_epoch: Starting epoch number
        train_loader_key: Key for training dataloader
        error_recorder: ErrorRecorder instance (created from config if None)
        total_epoch: Total epochs to train (from config if None)
        per_epoch: Checkpoint frequency as fraction of epoch (from config if None)
        best_metric_loader_key: Key for validation loader used for best metric
        best_metric: Metric name for tracking best model
        write_csv: Whether to write CSV log
        working_dir: Working directory for checkpoints
    """
    log = Logger()
    write_csv = write_csv and log.rank == 0
    working_dir = working_dir or os.getcwd()
    prefix = f'{os.path.abspath(working_dir)}/'

    total_epoch = total_epoch or config[KEY.EPOCH]
    per_epoch = per_epoch or config.get(KEY.PER_EPOCH, 0.1)
    best_metric = best_metric or config.get(KEY.BEST_METRIC, 'TotalLoss')
    recorder = error_recorder or ErrorRecorder.from_config(
        config, trainer.loss_functions, trainer.reg_functions
    )
    recorders = {k: deepcopy(recorder) for k in loaders}

    best_val = float('inf')
    best_key = None
    if best_metric_loader_key in recorders:
        best_key = recorders[best_metric_loader_key].get_key_str(best_metric)
    if best_key is None:
        log.writeline(
            f'Failed to get error recorder key: {best_metric} or '
            + f'{best_metric_loader_key} is missing. There will be no best '
            + 'checkpoint.'
        )

    csv_path = unique_filepath(f'{prefix}/lc.csv')
    if write_csv:
        head = ['epoch', 'lr']
        for k, rec in recorders.items():
            head.extend(list(rec.get_dct(prefix=k)))
        with open(csv_path, 'w') as f:
            f.write(','.join(head) + '\n')

    train_loader = loaders['trainset']

    if data_progress[KEY.TOTAL_DATA_NUM] < 0:  # fresh start or reset
        data_progress[KEY.TOTAL_DATA_NUM] = len(train_loader.sampler.sequence)
    else:  # continue
        train_loader.sampler.continue_from_data_progress(**data_progress)

    effective_batch_size = config[KEY.WORLD_SIZE] * config[KEY.BATCH_SIZE]
    start_batch = (data_progress[KEY.CURRENT_DATA_IDX]) // effective_batch_size
    total_step = data_progress[KEY.TOTAL_DATA_NUM] // effective_batch_size
    if data_progress[KEY.TOTAL_DATA_NUM] % effective_batch_size != 0:
        total_step += 1
    save_per_epoch = int(1 / per_epoch)
    save_batch_idx = np.linspace(1, total_step, save_per_epoch + 1)
    save_batch_idx = [int(idx) for idx in save_batch_idx[1:]]

    if data_progress[KEY.TOTAL_DATA_NUM] == data_progress[KEY.CURRENT_DATA_IDX]:
        start_epoch += 1  # continuing from end of epoch
        start_batch = 0
        train_loader.sampler.permutate_sequence()  # update rng state
        train_loader.sampler.refresh_sequence()  # make starting index to zero

    trainer.write_checkpoint(
        f'{prefix}/checkpoint_initial.pth',
        config=config,
        epoch=0,
        data_progress=data_progress,
    )

    scheduler_update_every_batch = (
            config.get(KEY.SCHEDULER_BATCH_MODE, False)
    )

    # TODO: too long, refactor more
    log.writeline('Entering training loop')
    for epoch in range(start_epoch, total_epoch + 1):  # one indexing
        data_progress[KEY.NUMPY_RNG_STATE] = train_loader.sampler.get_rng_state()
        log.timer_start('epoch')
        log.timer_start('batch')

        dl_timing = AverageNumber()
        dl_end = time.time()
        for idx, batch in enumerate(train_loader):
            dl_timing.update(time.time() - dl_end)
            current_batch_idx = idx + 1
            if epoch == start_epoch:
                current_batch_idx += start_batch  # continuing from middle of epoch
            save = current_batch_idx in save_batch_idx

            if save:
                lr = trainer.get_lr()
                log.bar()
                log.writeline(
                    f'Epoch {epoch}/{total_epoch}  '
                    + f'Batch {current_batch_idx}/{total_step}  lr: {lr:8f}'
                )
                log.bar()

            trainer.train_one_batch(batch, recorders[train_loader_key])
            if scheduler_update_every_batch:  # onecyclelr
                trainer.scheduler_step(best_val)

            if save:
                csv_dct = {
                    'epoch': str(epoch),
                    'batch': str(current_batch_idx),
                    'lr': f'{trainer.get_lr():8f}',
                }
                errors = {}
                for k, loader in loaders.items():
                    rec = recorders[k]
                    if k != train_loader_key:
                        print('valid', k, flush=True)
                        trainer.run_one_epoch(loader, False, rec)
                    if trainer.distributed:
                        trainer.recorder_all_reduce(rec)
                    csv_dct.update(rec.get_dct(prefix=k))
                    errors[k] = rec.epoch_forward()
                log.write_full_table(list(errors.values()), list(errors))

                batch_name = (
                    f'_{save_batch_idx.index(current_batch_idx) + 1}'
                    if current_batch_idx != save_batch_idx[-1]
                    else ''
                )
                data_progress[KEY.CURRENT_DATA_IDX] = min(
                    current_batch_idx * effective_batch_size,
                    data_progress[KEY.TOTAL_DATA_NUM],
                )
                trainer.write_checkpoint(
                    f'{prefix}/checkpoint_{epoch}{batch_name}.pth',
                    config=config,
                    epoch=epoch,
                    data_progress=data_progress,
                )

                if write_csv:
                    with open(csv_path, 'a') as f:
                        f.write(','.join(list(csv_dct.values())) + '\n')

                if best_key and errors[best_metric_loader_key][best_key] < best_val:
                    trainer.write_checkpoint(
                        f'{prefix}/checkpoint_best.pth', config=config, epoch=epoch
                    )
                    best_val = errors[best_metric_loader_key][best_key]
                    log.writeline('Best checkpoint written')

                log.timer_end('batch', message=f'Batch {current_batch_idx} elapsed')
                if config[KEY.IS_DDP]:
                    dl_timing._ddp_reduce(trainer.device)
                log.writeline(f'data loading, per (sec): {dl_timing.get():.4f}')
                log.writeline(f'data loading, sum *ALL* (sec): {dl_timing._sum:.4f}')
                dl_timing = AverageNumber()
                log.timer_start('batch')
            dl_end = time.time()
            # batch loop indent

        if not scheduler_update_every_batch:
            trainer.scheduler_step(best_val)
        log.timer_end('epoch', message=f'Epoch {epoch} elapsed')
    return trainer
