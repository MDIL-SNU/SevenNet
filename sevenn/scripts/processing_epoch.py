import os
import math
from typing import Dict

import torch
import numpy as np

from sevenn.sevenn_logger import Logger
from sevenn.train.trainer import Trainer, DataSetType, LossType
import sevenn._keys as KEY

import multiprocessing


def processing_epoch(trainer, config, loaders, working_dir):
    prefix = f"{os.path.abspath(working_dir)}/"
    train_loader, valid_loader, test_loader = loaders

    min_loss = 10000
    total_epoch = config[KEY.EPOCH]
    per_epoch = config[KEY.PER_EPOCH]

    def sqrt_dict(dct):
        for key in dct.keys():
            if isinstance(dct[key], dict):
                sqrt_dict(dct[key])
            else:
                dct[key] = math.sqrt(dct[key])
        return dct

    def write_checkpoint(is_best=False, epoch=None):
        suffix = "_best" if is_best else f"_{epoch}"
        checkpoint = trainer.get_checkpoint_dict()
        checkpoint.update({'config': config, 'epoch': epoch})
        torch.save(checkpoint, f"{prefix}/checkpoint{suffix}.pth")

    for epoch in range(1, total_epoch + 1):
        Logger().timer_start("epoch")
        Logger().bar()
        Logger().write(f"Epoch {epoch}/{total_epoch}  learning_rate: {trainer.get_lr():8f}\n")
        Logger().bar()

        Logger().timer_start("train")
        train_mse, train_specie_mse, loss_dct =\
            trainer.run_one_epoch(train_loader, DataSetType.TRAIN)
        Logger().timer_end("train", message=f"Train elapsed")
        Logger().timer_start("valid")
        valid_mse, valid_specie_mse, _ =\
            trainer.run_one_epoch(valid_loader, DataSetType.VALID)
        Logger().timer_end("valid", message=f"Valid elapsed")

        train_rmse = sqrt_dict(train_mse)
        valid_rmse = sqrt_dict(valid_mse)
        #train_specie_rmse = sqrt_dict(train_specie_mse)
        #valid_specie_rmse = sqrt_dict(valid_specie_mse)

        Logger().epoch_write_loss(train_rmse, valid_rmse)
        #Logger().epoch_write_specie_wise_loss(train_specie_rmse, valid_specie_rmse)
        Logger().epoch_write_train_loss(loss_dct)
        Logger().timer_end("epoch", message=f"Epoch {epoch} elapsed")

        loss_tot = sum(loss_dct.values())
        if loss_tot < min_loss:
            min_loss = loss_tot
            Logger().write(f"best train loss at epoch: {epoch}\n")
            write_checkpoint(is_best=True)
        if epoch % per_epoch == 0:
            write_checkpoint(epoch=epoch)
