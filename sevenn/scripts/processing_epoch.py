import os
import math
from typing import Dict

import torch
import numpy as np

from sevenn.sevenn_logger import Logger
from sevenn.train.trainer import Trainer
from sevenn.error_recorder import ErrorRecorder
from sevenn._const import LossType
import sevenn._keys as KEY


def processing_epoch(trainer, config, loaders, start_epoch, init_csv, working_dir):
    prefix = f"{os.path.abspath(working_dir)}/"
    train_loader, valid_loader, test_loader = loaders

    is_distributed = config[KEY.IS_DDP]
    rank = config[KEY.RANK]
    total_epoch = config[KEY.EPOCH]
    per_epoch = config[KEY.PER_EPOCH]
    train_recorder = ErrorRecorder.from_config(config)
    valid_recorder = ErrorRecorder.from_config(config)
    best_metric = config[KEY.BEST_METRIC]
    csv_fname = config[KEY.CSV_LOG]
    current_best = 99999

    if init_csv:
        csv_header = ["Epoch", "Learning_rate"]
        # Assume train valid have the same metrics
        for metric in train_recorder.get_metric_dict().keys():
            csv_header.append(f"Train_{metric}")
            csv_header.append(f"Valid_{metric}")
        Logger().init_csv(csv_fname, csv_header)

    def write_checkpoint(epoch, is_best=False):
        if is_distributed and rank != 0:
            return
        suffix = "_best" if is_best else f"_{epoch}"
        checkpoint = trainer.get_checkpoint_dict()
        checkpoint.update({'config': config, 'epoch': epoch})
        torch.save(checkpoint, f"{prefix}/checkpoint{suffix}.pth")

    fin_epoch = total_epoch + start_epoch
    for epoch in range(start_epoch, fin_epoch):
        lr = trainer.get_lr()
        Logger().timer_start("epoch")
        Logger().bar()
        Logger().write(f"Epoch {epoch}/{fin_epoch - 1}  lr: {lr:8f}\n")
        Logger().bar()

        trainer.run_one_epoch(train_loader, is_train=True,
                              error_recorder=train_recorder)
        train_err = train_recorder.epoch_forward()

        trainer.run_one_epoch(valid_loader, error_recorder=valid_recorder)
        valid_err = valid_recorder.epoch_forward()

        csv_values = [epoch, lr]
        for metric in train_err:
            csv_values.append(train_err[metric])
            csv_values.append(valid_err[metric])
        Logger().append_csv(csv_fname, csv_values)

        Logger().write_full_table([train_err, valid_err], ["Train", "Valid"])

        val = None
        for metric in valid_err:
            # loose string comparison,
            # e.g. "Energy" in "TotalEnergy" or "Energy_Loss"
            if best_metric in metric:
                val = valid_err[metric]
                break
        assert val is not None, f"Metric {best_metric} not found in {valid_err}"
        trainer.scheduler_step(val)

        #Logger().epoch_write_loss(train_rmse, valid_rmse)
        Logger().timer_end("epoch", message=f"Epoch {epoch} elapsed")

        if val < current_best and epoch > 10:  # skip first 10 epochs
            current_best = val
            write_checkpoint(epoch, is_best=True)
            Logger().writeline("Best checkpoint written")
        if epoch % per_epoch == 0:
            write_checkpoint(epoch)

