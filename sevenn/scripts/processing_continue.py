import os

import torch

import sevenn._keys as KEY
from sevenn.sevenn_logger import Logger

from sevenn.train.trainer import Trainer


def check_config_compatible(config, config_cp):
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
        #KEY.OPTIMIZER,
        #KEY.OPTIM_PARAM,
        #KEY.SCHEDULER,
        #KEY.SCHEDULER_PARAM,
        KEY.USE_SPECIES_WISE_SHIFT_SCALE,
        KEY.USE_BIAS_IN_LINEAR,
        KEY.OPTIMIZE_BY_REDUCE,
    ]
    for sbs in SHOULD_BE_SAME:
        if config[sbs] == config_cp[sbs]:
            continue
        raise ValueError(f"Value of {sbs} should be same. \
                {config[sbs]} != {config_cp[sbs]}")

    #TODO: for old checkpoint files, remove later
    if KEY.TRAIN_AVG_NUM_NEIGH not in config_cp.keys() \
            or KEY.TRAIN_SHIFT_SCALE not in config_cp.keys():
        config_cp[KEY.TRAIN_AVG_NUM_NEIGH] = False
        config_cp[KEY.TRAIN_SHIFT_SCALE] = False

    try:
        cntdct = config[KEY.CONTINUE]
    except KeyError:
        return

    TRAINABLE_CONFIGS = [KEY.TRAIN_AVG_NUM_NEIGH, KEY.TRAIN_SHIFT_SCALE]
    if any((not cntdct[KEY.RESET_SCHEDULER], not cntdct[KEY.RESET_OPTIMIZER])) \
       and all(config[k] == config_cp[k] for k in TRAINABLE_CONFIGS) is False:
        raise ValueError("trainable shift_scale or avg_num_neigh should match"
                         + " ,if one of reset optimizer or scheduler")


def processing_continue(model, config):
    # model is updated here, not returned

    avg_num_neigh = config[KEY.AVG_NUM_NEIGHBOR]
    shift = config[KEY.SHIFT]
    scale = config[KEY.SCALE]

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
    config_cp = checkpoint['config']

    if avg_num_neigh != config_cp[KEY.AVG_NUM_NEIGHBOR]:
        Logger().write("\nWARNING: dataset is updated (prev vs now)\n")
        Logger().write(f"avg_num_neigh: {config_cp[KEY.AVG_NUM_NEIGHBOR]:.4f}"
                       + f"!= {avg_num_neigh:.4f}\n")
        Logger().write("Below comments include shift, scale and avg_num_neigh\n")
        Logger().write(
            "If current config states are not trainable, use updated value\n"
        )
        Logger().write(
            "Else, we will ignore updated shfit, scale and avg_num_neigh\n"
        )
        Logger().write("The model keep using previous values\n")

    #TODO: Updating shift, scale, avg~~ to updated ones, make it optional?
    """
    IGNORE_WIEHGT_KEYS = ["rescale.shift", "rescale.scale"]
    for i in range(0, config[KEY.NUM_CONVOLUTION]):
        IGNORE_WIEHGT_KEYS.append(f"{i}_convolution.denumerator")
    model_state_dict_cp = {k: v for k, v in model_state_dict_cp.items()
                           if k not in IGNORE_WIEHGT_KEYS}
    """

    # it will raise error if not compatible
    check_config_compatible(config, config_cp)

    # If trainable (optional for shift, scale, avg_num_neigh), model_state_dict_cp
    # includes thoes value as parameters, which leads to overwritting updated
    # dataset's shift, scale, avg_num_neigh. So, we need to ignore those values
    model.load_state_dict(model_state_dict_cp, strict=False)

    trainer = Trainer(model, config)
    trainer.optimizer.load_state_dict(optimizer_state_dict)
    trainer.scheduler.load_state_dict(scheduler_state_dict)

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
    return trainer, start_epoch, init_csv
