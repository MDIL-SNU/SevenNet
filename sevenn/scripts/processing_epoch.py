import os
import math
from typing import Dict

import torch
import numpy as np

from sevenn.sevenn_logger import Logger
from sevenn.scripts.deploy import deploy
from sevenn.scripts.plot import draw_learning_curve, draw_every_parity
from sevenn.train.trainer import Trainer, DataSetType, LossType
import sevenn._keys as KEY

import multiprocessing


def processing_epoch(trainer, config, loaders, working_dir):
    prefix = f"{os.path.abspath(working_dir)}/"
    train_loader, valid_loader, test_loader = loaders

    min_loss = 100
    total_epoch = config[KEY.EPOCH]
    skip_output_until = config[KEY.SKIP_OUTPUT_UNTIL]

    output_per_epoch = config[KEY.OUTPUT_PER_EPOCH]
    per_epoch = output_per_epoch[KEY.PER_EPOCH]
    per_epoch = total_epoch if per_epoch is False else per_epoch
    draw_parity = output_per_epoch[KEY.DRAW_PARITY]
    is_save_data_pickle = output_per_epoch[KEY.SAVE_DATA_PICKLE]
    is_model_check_point = output_per_epoch[KEY.MODEL_CHECK_POINT]
    is_deploy_model = output_per_epoch[KEY.DEPLOY_MODEL]
    draw_lc = config[KEY.DRAW_LC]

    def postprocess_loss(train_loss, valid_loss,
                         train_specie_loss, valid_specie_loss,
                         scale, loss_history):
        rescale_loss(train_loss, scale)
        rescale_loss(valid_loss, scale)

        rescale_specie_wise_floss(train_specie_loss, scale)
        rescale_specie_wise_floss(valid_specie_loss, scale)
        loss_history[DataSetType.TRAIN][LossType.ENERGY].append(
            train_loss['total'][LossType.ENERGY])
        loss_history[DataSetType.TRAIN][LossType.FORCE].append(
            train_loss['total'][LossType.FORCE])
        loss_history[DataSetType.VALID][LossType.ENERGY].append(
            valid_loss['total'][LossType.ENERGY])
        loss_history[DataSetType.VALID][LossType.FORCE].append(
            valid_loss['total'][LossType.FORCE])

    def rescale_loss(loss_record: Dict[str, Dict[LossType, float]], scale: float):
        for label in loss_record.keys():
            loss_labeld = loss_record[label]
            for loss_type in loss_labeld.keys():
                loss_labeld[loss_type] = math.sqrt(loss_labeld[loss_type]) * scale

    def rescale_specie_wise_floss(f_loss_record: Dict[str, float], scale: float):
        for specie in f_loss_record.keys():
            f_loss_record[specie] = math.sqrt(f_loss_record[specie]) * scale

    # TODO: implement multiprocessing. Drawing graph is too expansive
    def output(is_best):
        """
        by default, make all outputs if best (it will overwrite previous one)
        """
        suffix = "_best" if is_best else f"_{epoch}"

        if draw_lc:
            draw_learning_curve(loss_history, f"{prefix}/learning_curve.png")
        if is_deploy_model or is_best:
            deploy(trainer.model.state_dict(), config,
                   f"{prefix}/deployed_model{suffix}.pt")
        if is_model_check_point or is_best:
            checkpoint = trainer.get_checkpoint_dict()
            checkpoint.update({'config': config, 'epoch': epoch})
            torch.save(checkpoint, f"{prefix}/checkpoint{suffix}.pth")
        if is_save_data_pickle or is_best:
            info_parity = {"train": train_parity_set, "valid": valid_parity_set}
            #torch.save(loss_hist_print, f"{prefix}/loss_hist{suffix}.pth")
            torch.save(info_parity, f"{prefix}/parity_at{suffix}.pth")
        if draw_parity or is_best:
            pass
            #dirty things inside plot.py
            #draw_every_parity(train_parity_set, valid_parity_set,
            #                  train_loss, valid_loss, f"{prefix}/parity_at{suffix}")

    loss_history = {
        DataSetType.TRAIN: {LossType.ENERGY: [], LossType.FORCE: []},
        DataSetType.VALID: {LossType.ENERGY: [], LossType.FORCE: []}
    }


    for epoch in range(1, total_epoch + 1):
        Logger().timer_start("epoch")
        Logger().bar()
        Logger().write(f"Epoch {epoch}/{total_epoch}\n")
        Logger().bar()

        train_parity_set, train_loss, train_specie_loss =\
            trainer.run_one_epoch(train_loader, DataSetType.TRAIN)
        valid_parity_set, valid_loss, valid_specie_loss =\
            trainer.run_one_epoch(valid_loader, DataSetType.VALID)

        loss_dct = {k: np.mean(v) for k, v in valid_loss['total'].items()}
        valid_total_loss = trainer.loss_function(loss_dct)

        # subroutine for loss (rescale, record loss, ..)
        postprocess_loss(train_loss, valid_loss,
                         train_specie_loss, valid_specie_loss,
                         1, loss_history)

        Logger().epoch_write_loss(train_loss, valid_loss)
        Logger().epoch_write_specie_wise_loss(train_specie_loss, valid_specie_loss)
        Logger().timer_end("epoch", message=f"Epoch {epoch} elapsed")

        if epoch < skip_output_until:
            continue
        Logger().timer_start("output_write")
        if valid_total_loss < min_loss:
            min_loss = valid_total_loss
            output(is_best=True)
            Logger().write(f"output written at epoch(best): {epoch}\n")
            # continue  # skip per epoch output if best is written
        if epoch % per_epoch == 0:
            output(is_best=False)
            Logger().write(f"output written at epoch: {epoch}\n")
        Logger().timer_end("output_write", message=f"Output write elapsed")
    # deploy(best_model, config, f'{prefix}/deployed_model.pt')
    # subroutine for loss (rescale, record loss, ..)
