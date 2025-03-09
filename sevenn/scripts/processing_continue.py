import os
import warnings

import torch

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.logger import Logger
from sevenn.scripts.convert_model_modality import (
    append_modality_to_model_dct,
    get_single_modal_model_dct,
)


def processing_continue_v2(config):  # simpler
    """
    Replacement of processing_continue,
    Skips model compatibility
    """
    log = Logger()
    continue_dct = config[KEY.CONTINUE]
    log.write('\nContinue found, loading checkpoint\n')

    checkpoint = util.load_checkpoint(continue_dct[KEY.CHECKPOINT])
    model_cp = checkpoint.build_model()
    config_cp = checkpoint.config
    model_state_dict_cp = model_cp.state_dict()

    optimizer_state_dict_cp = (
        checkpoint.optimizer_state_dict
        if not continue_dct[KEY.RESET_OPTIMIZER]
        else None
    )
    scheduler_state_dict_cp = (
        checkpoint.scheduler_state_dict
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

    if config_cp.get(KEY.USE_MODALITY, False) != config.get(KEY.USE_MODALITY):
        raise ValueError('use_modality is not same. Check sevenn_cp')

    modal_map = config_cp.get(KEY.MODAL_MAP, None)  # dict | None
    if modal_map and len(modal_map) > 0:
        modalities = list(modal_map.keys())
        log.writeline(f'Multimodal model found: {modalities}')
        log.writeline('use_modality: True')
        config[KEY.USE_MODALITY] = True

    from_epoch = checkpoint.epoch or 0
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

    if (
        KEY.USE_MODALITY in config_cp.keys() and config_cp[KEY.USE_MODALITY]
    ):  # checkpoint model is multimodal
        config.update({
            KEY.MODAL_MAP + '_cp': config_cp[KEY.MODAL_MAP],
            KEY.USE_MODALITY + '_cp': True,
            KEY.NUM_MODALITIES + '_cp': len(config_cp[KEY.MODAL_MAP]),
        })
    else:
        config.update({
            KEY.MODAL_MAP + '_cp': {},
            KEY.USE_MODALITY + '_cp': False,
            KEY.NUM_MODALITIES + '_cp': 0,
        })

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


def convert_modality_of_checkpoint_state_dct(config, state_dicts):
    # TODO: this requires updating model state dict after seeing dataset
    model_state_dict_cp, optimizer_state_dict_cp, scheduler_state_dict_cp = (
        state_dicts
    )

    if config[KEY.USE_MODALITY]:  # current model is multimodal
        num_modalities_cp = len(config[KEY.MODAL_MAP + '_cp'])
        append_modal_length = config[KEY.NUM_MODALITIES] - num_modalities_cp

        model_state_dict_cp = append_modality_to_model_dct(
            model_state_dict_cp, config, num_modalities_cp, append_modal_length
        )

    else:  # current model is single modal
        if config[KEY.USE_MODALITY + '_cp']:  # checkpoint model is multimodal
            # change model state dict to single modal, default = "common"
            model_state_dict_cp = get_single_modal_model_dct(
                model_state_dict_cp,
                config,
                config[KEY.DEFAULT_MODAL],
                from_processing_cp=True,
            )

    state_dicts = (
        model_state_dict_cp,
        optimizer_state_dict_cp,
        scheduler_state_dict_cp,
    )

    return state_dicts
