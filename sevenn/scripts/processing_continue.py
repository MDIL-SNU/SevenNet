import torch

import sevenn._keys as KEY
from sevenn.sevenn_logger import Logger


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
        KEY.OPTIMIZER,
        KEY.OPTIM_PARAM,
        KEY.SCHEDULER,
        KEY.SCHEDULER_PARAM,
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
                         + " if one of reset optimizer or scheduler is false")


def processing_continue(config, model, statistic_values):

    avg_num_neigh, shift, scale = statistic_values
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
    return optimizer_state_dict, scheduler_state_dict


