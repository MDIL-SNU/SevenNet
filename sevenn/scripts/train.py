from typing import Dict
import sys
import math
import os
import random
import copy

import numpy as np
import torch
import torch.autograd
from torch_geometric.loader import DataLoader
import e3nn.util.jit

import sevenn
import sevenn.train
from sevenn.train.trainer import Trainer, DataSetType, LossType
from sevenn.model_build import build_E3_equivariant_model
from sevenn.scripts.deploy import deploy
from sevenn.scripts.plot import draw_learning_curve
from sevenn.sevenn_logger import Logger
from sevenn.scripts.init_dataset import init_dataset
import sevenn._keys as KEY


# TODO: E3_equivariant model assumed
def train(config: Dict, working_dir: str):
    """
    main flow
    """
    prefix = f"{os.path.abspath(working_dir)}/"
    Logger().timer_start("total")
    seed = config[KEY.RANDOM_SEED]
    random.seed(seed)
    torch.manual_seed(seed)
    device = config[KEY.DEVICE]
    is_stress = (config[KEY.IS_TRACE_STRESS] or config[KEY.IS_TRAIN_STRESS])

    # load data set
    Logger().write("\nInitializing dataset...\n")
    try:
        dataset = init_dataset(config, working_dir, is_stress)
        Logger().write("Dataset initialization was successful\n")
        natoms = dataset.get_natoms(config[KEY.TYPE_MAP])

        Logger().write("\nNumber of atoms in total dataset:\n")
        Logger().natoms_write(natoms)

        # calculate shift and scale from dataset
        ignore_test = not config[KEY.USE_TESTSET]
        train_set, valid_set, test_set = \
            dataset.divide_dataset(config[KEY.RATIO], ignore_test=ignore_test)
        Logger().write("The dataset divided into train, valid, test set\n")
        Logger().write(Logger.format_k_v("training_set size", train_set.len()))
        Logger().write(Logger.format_k_v("validation_set size", valid_set.len()))
        Logger().write(Logger.format_k_v("test_set size", test_set.len()
                                         if test_set is not None else 0))

        Logger().write("\nCalculating shift and scale from training set...\n")
        #shift, scale = train_set.shift_scale_dataset()
        #TODO TODO TODO
        shift = train_set.get_per_atom_energy_mean()
        scale = train_set.get_force_rmse()
        config.update({KEY.SHIFT: shift, KEY.SCALE: scale})
        #valid_set.shift_scale_dataset(shift=shift, scale=scale)
        Logger().write(f"calculated per_atom_energy mean shift is {shift:.6f} eV\n")
        Logger().write(f"calculated force rms scale is {scale:.6f}\n")

        avg_num_neigh = config[KEY.AVG_NUM_NEIGHBOR]
        if avg_num_neigh:
            Logger().write("Calculating average number of neighbor...\n")
            avg_num_neigh = train_set.get_avg_num_neigh()
            Logger().write(f"average number of neighbor is {avg_num_neigh:.6f}\n")
            config[KEY.AVG_NUM_NEIGHBOR] = avg_num_neigh
        else:
            config[KEY.AVG_NUM_NEIGHBOR] = 1

        batch_size = config[KEY.BATCH_SIZE]
        num_workers = config[KEY.NUM_WORKERS]
        # pin_memory = (device != torch.device("cpu"))
        train_loader = DataLoader(train_set.to_list(), batch_size,
                                  shuffle=True, num_workers=num_workers,)
        valid_loader = DataLoader(valid_set.to_list(), batch_size,
                                  num_workers=num_workers,)
        if test_set is not None:
            test_set.shift_scale_dataset(shift=shift, scale=scale)
            test_loader = DataLoader(test_set.to_list(), batch_size)
    except Exception as e:
        Logger().error(e)
        sys.exit(1)

    Logger().write("\nModel building...\n")
    # initialize model
    try:
        model = build_E3_equivariant_model(config)
    except Exception as e:
        Logger().error(e)
        sys.exit(1)
    Logger().write("Model building was successful\n")
    user_labels = dataset.user_labels

    optimizer_state_dict, scheduler_state_dict = None, None
    if config[KEY.CONTINUE] is not False:
        continue_dct = config[KEY.CONTINUE]
        Logger().write("\nContinue found, loading checkpoint\n")

        checkpoint = torch.load(continue_dct[KEY.CHECKPOINT])
        reset_optimizer = continue_dct[KEY.RESET_OPTIMIZER]
        reset_scheduler = continue_dct[KEY.RESET_SCHEDULER]

        from_epoch = checkpoint['epoch']
        model_state_dict_cp = checkpoint['model_state_dict']
        optimizer_state_dict = \
            None if reset_optimizer else checkpoint['optimizer_state_dict']
        scheduler_state_dict = \
            None if reset_scheduler else checkpoint['scheduler_state_dict']
        loss_cp = checkpoint['loss']
        config_cp = checkpoint['config']

        if(avg_num_neigh != config_cp[KEY.AVG_NUM_NEIGHBOR]
                or shift != config_cp[KEY.SHIFT]
                or scale != config_cp[KEY.SCALE]):
            Logger().write("\nWARNING: dataset is updated\n")
            Logger().write(f"avg_num_neigh: {config_cp[KEY.AVG_NUM_NEIGHBOR]:.4f}"
                           + f"-> {avg_num_neigh:.4f}\n")
            Logger().write(f"shift: {config_cp[KEY.SHIFT]:.4f} -> {shift:.4f}\n")
            Logger().write(f"scale: {config_cp[KEY.SCALE]:.4f} -> {scale:.4f}\n")
            Logger().write("The model(trained on previous dataset) "
                           + "gonna use updated shift, scale and avg_num_neigh\n")

        #TODO: hard coded
        IGNORE_WIEHGT_KEYS = ["rescale.shift", "rescale.scale"]
        for i in range(0, config[KEY.NUM_CONVOLUTION]):
            IGNORE_WIEHGT_KEYS.append(f"{i}_convolution.denumerator")
        model_state_dict_cp = {k: v for k, v in model_state_dict_cp.items()
                               if k not in IGNORE_WIEHGT_KEYS}

        # it will raise error if not compatible
        check_config_compatible(config, config_cp)
        model.load_state_dict(model_state_dict_cp, strict=False)

        Logger().write(f"checkpoint previous epoch was: {from_epoch}\n")
        Logger().write("checkpoint loading was successful\n")

    trainer = Trainer(
        model, user_labels, config,
        energy_key=KEY.PRED_PER_ATOM_ENERGY,
        ref_energy_key=KEY.PER_ATOM_ENERGY,
        force_key=KEY.PRED_FORCE,
        ref_force_key=KEY.FORCE,
        stress_key=KEY.PRED_STRESS,
        ref_stress_key=KEY.STRESS,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict
    )

    num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger().write(f"Total number of weight in model is {num_weights}\n")

    Logger().write("Trainer initialized. program is ready to training\n")
    Logger().bar()

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
            #TODO: implement
            pass

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
        if valid_total_loss < min_loss:
            min_loss = valid_total_loss
            output(is_best=True)
            Logger().write(f"output written at epoch(best): {epoch}\n")
            # continue  # skip per epoch output if best is written
        if epoch % per_epoch == 0:
            output(is_best=False)
            Logger().write(f"output written at epoch: {epoch}\n")
    # deploy(best_model, config, f'{prefix}/deployed_model.pt')
    Logger().timer_end("total", message="Total wall time")


# subroutine for loss (rescale, record loss, ..)
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


def inference_poscar(config: Dict, working_dir: str):  # This is only for debugging
    prefix = f"{os.path.abspath(working_dir)}/"
    is_stress = (config[KEY.IS_TRACE_STRESS] or config[KEY.IS_TRAIN_STRESS])
    device = config[KEY.DEVICE]

    dataset = init_dataset(config, working_dir, is_stress, True)

    loader = DataLoader(dataset.to_list(), 1, shuffle=False)

    checkpoint = torch.load('checkpoint_20.pth')
    old_config = checkpoint['config']

    config.update({KEY.SHIFT: old_config[KEY.SHIFT],
                   KEY.SCALE: old_config[KEY.SCALE],
                   KEY.AVG_NUM_NEIGHBOR: old_config[KEY.AVG_NUM_NEIGHBOR]})

    for key, value in old_config.items():
        if key not in config.keys():
            print(f'{key} does not exist')
        elif config[key] != value:
            print(f'{key} is updated')

    model = build_E3_equivariant_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    for idx, data in enumerate(loader):
        data.to(device)

        result = model(data)

        if idx == 0:
            print(result[KEY.PRED_TOTAL_ENERGY])
            print(result[KEY.PRED_FORCE])
            print(result[KEY.SCALED_STRESS])
            break


def check_config_compatible(cf1, cf2):
    SHOULD_BE_SAME = [
        KEY.TYPE_MAP,
        KEY.NODE_FEATURE_MULTIPLICITY,
        KEY.LMAX,
        KEY.IS_PARITY,
        KEY.RADIAL_BASIS,
        KEY.CUTOFF_FUNCTION,
        KEY.CUTOFF,
        KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS,
        KEY.NUM_CONVOLUTION,
        KEY.ACTIVATION_GATE,
        KEY.ACTIVATION_SCARLAR,
        KEY.DTYPE,
        KEY.OPTIMIZER,
        KEY.OPTIM_PARAM,
        KEY.SCHEDULER,
        KEY.SCHEDULER_PARAM,
    ]

    for sbs in SHOULD_BE_SAME:
        if cf1[sbs] == cf2[sbs]:
            continue
        raise ValueError(f"Value of {sbs} should be same. \
                {cf1[sbs]} != {cf2[sbs]}")

    try:
        cntdct = cf1[KEY.CONTINUE]
    except KeyError:
        return

    TRAINABLE_CONFIGS = [KEY.TRAIN_AVG_NUM_NEIGH, KEY.TRAIN_SHIFT_SCALE]
    if any((not cntdct[KEY.RESET_SCHEDULER], not cntdct[KEY.RESET_OPTIMIZER])) \
       and all(cf1[k] == cf2[k] for k in TRAINABLE_CONFIGS) is False:
        raise ValueError("trainable shift_scale or avg_num_neigh should match"
                         + " if one of reset optimizer or scheduler is false")

