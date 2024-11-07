import os
from copy import deepcopy
from typing import Optional

import torch

import sevenn._keys as KEY
from sevenn.error_recorder import ErrorRecorder
from sevenn.sevenn_logger import Logger
from sevenn.train.trainer import Trainer


def processing_epoch_v2(
    config: dict,
    trainer: Trainer,
    loaders: dict,  # dict[str, Dataset]
    start_epoch: int = 1,
    train_loader_key: str = 'trainset',
    error_recorder: Optional[ErrorRecorder] = None,
    total_epoch: Optional[int] = None,
    per_epoch: Optional[int] = None,
    best_metric_loader_key: str = 'validset',
    best_metric: Optional[str] = None,
    write_csv: bool = True,
    working_dir: Optional[str] = None,
):
    from sevenn.util import unique_filepath

    log = Logger()
    write_csv = write_csv and log.rank == 0
    working_dir = working_dir or os.getcwd()
    prefix = f'{os.path.abspath(working_dir)}/'

    total_epoch = total_epoch or config[KEY.EPOCH]
    per_epoch = per_epoch or config.get(KEY.PER_EPOCH, 10)
    best_metric = best_metric or config.get(KEY.BEST_METRIC, 'TotalLoss')
    recorder = error_recorder or ErrorRecorder.from_config(config)
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

    path = f'{prefix}/checkpoint_0.pth'  # save first epoch
    trainer.write_checkpoint(path, config=config, epoch=0)

    for epoch in range(start_epoch, total_epoch + 1):  # one indexing
        log.timer_start('epoch')
        lr = trainer.get_lr()
        log.bar()
        log.write(f'Epoch {epoch}/{total_epoch}  lr: {lr:8f}\n')
        log.bar()

        csv_dct = {'epoch': str(epoch), 'lr': f'{lr:8f}'}
        errors = {}
        for k, loader in loaders.items():
            rec = recorders[k]
            trainer.run_one_epoch(loader, k == train_loader_key, rec)
            csv_dct.update(rec.get_dct(prefix=k))
            errors[k] = rec.epoch_forward()
        log.write_full_table(list(errors.values()), list(errors))
        trainer.scheduler_step(best_val)

        if write_csv:
            with open(csv_path, 'a') as f:
                f.write(','.join(list(csv_dct.values())) + '\n')

        if best_key and errors[best_metric_loader_key][best_key] < best_val:
            path = f'{prefix}/checkpoint_best.pth'
            trainer.write_checkpoint(path, config=config, epoch=epoch)
            best_val = errors[best_metric_loader_key][best_key]
            log.writeline('Best checkpoint written')

        if epoch % per_epoch == 0:
            path = f'{prefix}/checkpoint_{epoch}.pth'
            trainer.write_checkpoint(path, config=config, epoch=epoch)

        log.timer_end('epoch', message=f'Epoch {epoch} elapsed')
    return trainer


def processing_epoch(trainer, config, loaders, start_epoch, init_csv, working_dir):
    log = Logger()
    prefix = f'{os.path.abspath(working_dir)}/'
    train_loader, valid_loader = loaders

    is_distributed = config[KEY.IS_DDP]
    rank = config[KEY.RANK]
    total_epoch = config[KEY.EPOCH]
    per_epoch = config[KEY.PER_EPOCH]
    train_recorder = ErrorRecorder.from_config(config)
    valid_recorder = ErrorRecorder.from_config(config)
    best_metric = config[KEY.BEST_METRIC]
    csv_fname = f'{prefix}{config[KEY.CSV_LOG]}'
    current_best = float('inf')

    if init_csv:
        csv_header = ['Epoch', 'Learning_rate']
        # Assume train valid have the same metrics
        for metric in train_recorder.get_metric_dict().keys():
            csv_header.append(f'Train_{metric}')
            csv_header.append(f'Valid_{metric}')
        log.init_csv(csv_fname, csv_header)

    def write_checkpoint(epoch, is_best=False):
        if is_distributed and rank != 0:
            return
        suffix = '_best' if is_best else f'_{epoch}'
        checkpoint = trainer.get_checkpoint_dict()
        checkpoint.update({'config': config, 'epoch': epoch})
        torch.save(checkpoint, f'{prefix}/checkpoint{suffix}.pth')

    fin_epoch = total_epoch + start_epoch
    for epoch in range(start_epoch, fin_epoch):
        lr = trainer.get_lr()
        log.timer_start('epoch')
        log.bar()
        log.write(f'Epoch {epoch}/{fin_epoch - 1}  lr: {lr:8f}\n')
        log.bar()

        trainer.run_one_epoch(
            train_loader, is_train=True, error_recorder=train_recorder
        )
        train_err = train_recorder.epoch_forward()

        trainer.run_one_epoch(valid_loader, error_recorder=valid_recorder)
        valid_err = valid_recorder.epoch_forward()

        csv_values = [epoch, lr]
        for metric in train_err:
            csv_values.append(train_err[metric])
            csv_values.append(valid_err[metric])
        log.append_csv(csv_fname, csv_values)

        log.write_full_table([train_err, valid_err], ['Train', 'Valid'])

        val = None
        for metric in valid_err:
            # loose string comparison,
            # e.g. "Energy" in "TotalEnergy" or "Energy_Loss"
            if best_metric in metric:
                val = valid_err[metric]
                break
        assert val is not None, f'Metric {best_metric} not found in {valid_err}'
        trainer.scheduler_step(val)

        log.timer_end('epoch', message=f'Epoch {epoch} elapsed')

        if val < current_best:
            current_best = val
            write_checkpoint(epoch, is_best=True)
            log.writeline('Best checkpoint written')
        if epoch % per_epoch == 0:
            write_checkpoint(epoch)
