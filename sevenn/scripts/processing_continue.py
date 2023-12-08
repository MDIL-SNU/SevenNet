import os

import torch

import sevenn._keys as KEY
from sevenn.sevenn_logger import Logger

from sevenn.train.trainer import Trainer


def check_config_compatible(config, config_cp):
    # TODO: check more
    SHOULD_BE_SAME = [
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
        KEY.USE_SPECIES_WISE_SHIFT_SCALE,
        KEY.USE_BIAS_IN_LINEAR,
        KEY.OPTIMIZE_BY_REDUCE,
    ]
    for sbs in SHOULD_BE_SAME:
        if config[sbs] == config_cp[sbs]:
            continue
        raise ValueError(f"Value of {sbs} should be same. \
                {config[sbs]} != {config_cp[sbs]}")

    try:
        cntdct = config[KEY.CONTINUE]
    except KeyError:
        return

    TRAINABLE_CONFIGS = [KEY.TRAIN_AVG_NUM_NEIGH, KEY.TRAIN_SHIFT_SCALE]
    if any((not cntdct[KEY.RESET_SCHEDULER], not cntdct[KEY.RESET_OPTIMIZER])) \
       and all(config[k] == config_cp[k] for k in TRAINABLE_CONFIGS) is False:
        raise ValueError("reset optimizer and scheduler if you want to change "
                         + "trainable configs")
    #TODO add conition for changed optim/scheduler but not reset


def processing_continue(config):
    continue_dct = config[KEY.CONTINUE]
    Logger().write("\nContinue found, loading checkpoint\n")

    try:
        checkpoint = torch.load(continue_dct[KEY.CHECKPOINT], map_location="cpu")
        config_cp = checkpoint['config']
    except FileNotFoundError:
        raise FileNotFoundError(
            f"checkpoint file {continue_dct[KEY.CHECKPOINT]} not found"
        )
    # it will raise error if not compatible
    check_config_compatible(config, config_cp)
    Logger().write("Checkpoint config is compatible\n")

    from_epoch = checkpoint['epoch']
    model_state_dict_cp = checkpoint['model_state_dict']
    optimizer_state_dict_cp = checkpoint['optimizer_state_dict'] \
        if not continue_dct[KEY.RESET_OPTIMIZER] else None
    scheduler_state_dict_cp = checkpoint['scheduler_state_dict'] \
        if not continue_dct[KEY.RESET_SCHEDULER] else None

    # Use checkpoints values if not specified in yaml
    shift = model_state_dict_cp['rescale atomic energy.shift'].numpy() \
            if type(config[KEY.SHIFT]) not in [list, float] \
            else config[KEY.SHIFT]
    scale = model_state_dict_cp['rescale atomic energy.scale'].numpy() \
            if type(config[KEY.SCALE]) not in [list, float] \
            else config[KEY.SCALE]
    avg_num_neigh = (model_state_dict_cp['1 convolution.denumerator']**2).item() \
            if type(config[KEY.AVG_NUM_NEIGHBOR]) is not float \
            else config[KEY.AVG_NUM_NEIGHBOR]

    # update config
    config.update({
        KEY.SHIFT: shift,
        KEY.SCALE: scale,
        KEY.AVG_NUM_NEIGHBOR: avg_num_neigh,
    })
    Logger().write("Use shift, scale, avg_num_neigh of (defined in yaml or in checkpoint)\n")
    Logger().format_k_v("shift", ', '.join([f"{x:.4f}" for x in shift]), write=True)
    Logger().format_k_v("scale", ', '.join([f"{x:.4f}" for x in scale]), write=True)
    Logger().format_k_v("avg_num_neigh", f"{avg_num_neigh:.4f}", write=True)
    Logger().write("If you don't want to continue with these values, "
                   + "specify values you want to input yaml\n")
    Logger().write("If the values were trainable, it would be changed\n")

    chem_speices_related = {
        KEY.TYPE_MAP: config_cp[KEY.TYPE_MAP],
        KEY.NUM_SPECIES: config_cp[KEY.NUM_SPECIES],
        KEY.CHEMICAL_SPECIES: config_cp[KEY.CHEMICAL_SPECIES],
        KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER: config_cp[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER],
    }
    config.update(chem_speices_related)

    Logger().write(f"checkpoint previous epoch was: {from_epoch}\n")

    # decide start epoch
    reset_epoch = continue_dct[KEY.RESET_EPOCH]
    if reset_epoch:
        start_epoch = 1
        Logger().write("epoch reset to 1\n")
    else:
        start_epoch = from_epoch + 1
        Logger().write(f"epoch start from {start_epoch}\n")

    # decide csv file to continue
    init_csv = True
    csv_fname = config[KEY.CSV_LOG]
    if os.path.isfile(csv_fname):
        # I hope python compare dict well
        if config_cp[KEY.ERROR_RECORD] == config[KEY.ERROR_RECORD]:
            Logger().writeline("Same metric, csv file will be appended")
            init_csv = False
        else:
            raise ValueError(
                "Continue found old csv file with different metric. "
                + "Please backup your csv file or restore old metric"
            )
    else:
        Logger().writeline(
            f"{csv_fname} file not found, new csv file will be created"
        )

    Logger().writeline("checkpoint loading was successful")
    return (model_state_dict_cp, optimizer_state_dict_cp, scheduler_state_dict_cp), \
            start_epoch, init_csv
