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


def processing_epoch(trainer, config, loaders, working_dir):
    prefix = f"{os.path.abspath(working_dir)}/"
    train_loader, valid_loader, test_loader = loaders

    is_distributed = config[KEY.IS_DDP]
    rank = config[KEY.RANK]
    total_epoch = config[KEY.EPOCH]
    per_epoch = config[KEY.PER_EPOCH]
    train_recorder = ErrorRecorder.from_config(config)
    valid_recorder = ErrorRecorder.from_config(config)
    best_metric = config[KEY.BEST_METRIC]
    current_best = 9999999

    def write_checkpoint(epoch, is_best=False):
        if is_distributed and rank != 0:
            return
        suffix = "_best" if is_best else f"_{epoch}"
        checkpoint = trainer.get_checkpoint_dict()
        checkpoint.update({'config': config, 'epoch': epoch})
        torch.save(checkpoint, f"{prefix}/checkpoint{suffix}.pth")

    for epoch in range(1, total_epoch + 1):
        Logger().timer_start("epoch")
        Logger().bar()
        Logger().write(f"Epoch {epoch}/{total_epoch}  lr: {trainer.get_lr():8f}\n")
        Logger().bar()

        Logger().timer_start("train")
        trainer.run_one_epoch(train_loader, is_train=True,
                              error_recorder=train_recorder)
        train_err = train_recorder.epoch_forward()
        #Logger().timer_end("train", message="Train elapsed")

        Logger().timer_start("valid")
        trainer.run_one_epoch(valid_loader, error_recorder=valid_recorder)
        valid_err = valid_recorder.epoch_forward()
        #Logger().timer_end("valid", message="Valid elapsed")

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

        if val < current_best:
            current_best = val
            write_checkpoint(epoch, is_best=True)
        if epoch % per_epoch == 0:
            write_checkpoint(epoch)

