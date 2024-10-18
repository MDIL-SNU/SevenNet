import os
import warnings

import torch

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.sevenn_logger import Logger


def processing_continue_v2(config):  # simpler
    """
    Replacement of processing_continue,
    Skips model compatibility
    """
    log = Logger()
    continue_dct = config[KEY.CONTINUE]
    log.write('\nContinue found, loading checkpoint\n')

    checkpoint = torch.load(
        continue_dct[KEY.CHECKPOINT], map_location='cpu', weights_only=False
    )
    model_cp, config_cp = util.model_from_checkpoint(checkpoint)
    model_state_dict_cp = model_cp.state_dict()

    optimizer_state_dict_cp = (
        checkpoint['optimizer_state_dict']
        if not continue_dct[KEY.RESET_OPTIMIZER]
        else None
    )
    scheduler_state_dict_cp = (
        checkpoint['scheduler_state_dict']
        if not continue_dct[KEY.RESET_SCHEDULER]
        else None
    )

    # use_statistic_value_of_checkpoint always True
    # Overwrite config from model state dict, so graph_dataset.from_config
    # will not put statistic values to shift, scale, and conv_denominator
    config[KEY.SHIFT] = model_state_dict_cp['rescale_atomic_energy.shift'].tolist()
    config[KEY.SCALE] = model_state_dict_cp['rescale_atomic_energy.scale'].tolist()
    conv_denom = []
    for i in range(config_cp[KEY.NUM_CONVOLUTION]):
        conv_denom.append(model_state_dict_cp[f'{i}_convolution.denominator'].item())
    config[KEY.CONV_DENOMINATOR] = conv_denom
    log.writeline(
        f'{KEY.SHIFT}, {KEY.SCALE}, and {KEY.CONV_DENOMINATOR} are '
        + 'overwritten by model_state_dict of checkpoint'
    )

    chem_keys = [
        KEY.TYPE_MAP,
        KEY.NUM_SPECIES,
        KEY.CHEMICAL_SPECIES,
        KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER,
    ]
    config.update({k: config_cp[k] for k in chem_keys})
    log.writeline(
        'chemical_species are overwritten by checkpoint. '
        + f'This model knows {config[KEY.NUM_SPECIES]} species'
    )

    from_epoch = checkpoint['epoch']
    log.writeline(f'Checkpoint previous epoch was: {from_epoch}')
    epoch = 1 if continue_dct[KEY.RESET_EPOCH] else from_epoch + 1
    log.writeline(f'epoch start from {epoch}')

    log.writeline('checkpoint loading successful')

    state_dicts = [
        model_state_dict_cp,
        optimizer_state_dict_cp,
        scheduler_state_dict_cp,
    ]
    return state_dicts, epoch


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
        KEY.USE_BIAS_IN_LINEAR,
        KEY.SELF_CONNECTION_TYPE,
    ]
    for sbs in SHOULD_BE_SAME:
        if config[sbs] == config_cp[sbs]:
            continue
        if sbs == KEY.SELF_CONNECTION_TYPE and config_cp[sbs] == 'MACE':
            warnings.warn(
                'We do not support this version of checkpoints to continue '
                "Please use self_connection_type='linear' in input.yaml "
                'and train from scratch',
                UserWarning,
            )
        raise ValueError(
            f'Value of {sbs} should be same. {config[sbs]} != {config_cp[sbs]}'
        )

    try:
        cntdct = config[KEY.CONTINUE]
    except KeyError:
        return

    TRAINABLE_CONFIGS = [KEY.TRAIN_DENOMINTAOR, KEY.TRAIN_SHIFT_SCALE]
    if (
        any((not cntdct[KEY.RESET_SCHEDULER], not cntdct[KEY.RESET_OPTIMIZER]))
        and all(config[k] == config_cp[k] for k in TRAINABLE_CONFIGS) is False
    ):
        raise ValueError(
            'reset optimizer and scheduler if you want to change '
            + 'trainable configs'
        )

    # TODO add conition for changed optim/scheduler but not reset


def processing_continue(config):
    log = Logger()
    continue_dct = config[KEY.CONTINUE]
    log.write('\nContinue found, loading checkpoint\n')

    checkpoint = torch.load(
        continue_dct[KEY.CHECKPOINT], map_location='cpu', weights_only=False
    )
    config_cp = checkpoint['config']

    model_cp, config_cp = util.model_from_checkpoint(checkpoint)
    model_state_dict_cp = model_cp.state_dict()

    # it will raise error if not compatible
    check_config_compatible(config, config_cp)
    log.write('Checkpoint config is compatible\n')

    # for backward compat.
    config.update({KEY._NORMALIZE_SPH: config_cp[KEY._NORMALIZE_SPH]})

    from_epoch = checkpoint['epoch']
    optimizer_state_dict_cp = (
        checkpoint['optimizer_state_dict']
        if not continue_dct[KEY.RESET_OPTIMIZER]
        else None
    )
    scheduler_state_dict_cp = (
        checkpoint['scheduler_state_dict']
        if not continue_dct[KEY.RESET_SCHEDULER]
        else None
    )

    # These could be changed based on given continue_input.yaml
    # ex) adapt to statistics of fine-tuning dataset
    shift_cp = model_state_dict_cp['rescale_atomic_energy.shift'].numpy()
    del model_state_dict_cp['rescale_atomic_energy.shift']
    scale_cp = model_state_dict_cp['rescale_atomic_energy.scale'].numpy()
    del model_state_dict_cp['rescale_atomic_energy.scale']
    conv_denominators = []
    for i in range(config_cp[KEY.NUM_CONVOLUTION]):
        conv_denominators.append(
            (model_state_dict_cp[f'{i}_convolution.denominator']).item()
        )
        del model_state_dict_cp[f'{i}_convolution.denominator']

    # Further handled by processing_dataset.py
    config.update({
        KEY.SHIFT + '_cp': shift_cp,
        KEY.SCALE + '_cp': scale_cp,
        KEY.CONV_DENOMINATOR + '_cp': conv_denominators,
    })

    chem_keys = [
        KEY.TYPE_MAP,
        KEY.NUM_SPECIES,
        KEY.CHEMICAL_SPECIES,
        KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER,
    ]
    config.update({k: config_cp[k] for k in chem_keys})

    log.write(f'checkpoint previous epoch was: {from_epoch}\n')

    # decide start epoch
    reset_epoch = continue_dct[KEY.RESET_EPOCH]
    if reset_epoch:
        start_epoch = 1
        log.write('epoch reset to 1\n')
    else:
        start_epoch = from_epoch + 1
        log.write(f'epoch start from {start_epoch}\n')

    # decide csv file to continue
    init_csv = True
    csv_fname = config_cp[KEY.CSV_LOG]
    if os.path.isfile(csv_fname):
        # I hope python compare dict well
        if config_cp[KEY.ERROR_RECORD] == config[KEY.ERROR_RECORD]:
            log.writeline('Same metric, csv file will be appended')
            init_csv = False
    else:
        log.writeline(f'{csv_fname} file not found, new csv file will be created')
    log.writeline('checkpoint loading was successful')

    state_dicts = [
        model_state_dict_cp,
        optimizer_state_dict_cp,
        scheduler_state_dict_cp,
    ]
    return state_dicts, start_epoch, init_csv
