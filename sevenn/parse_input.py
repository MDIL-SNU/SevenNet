import glob
import os
import warnings
from typing import Any, Callable, Dict

import torch
import yaml

import sevenn._const as _const
import sevenn._keys as KEY
import sevenn.util as util


def config_initialize(
    key: str,
    config: Dict,
    default: Any,
    conditions: Dict,
):
    # default value exist & no user input -> return default
    if key not in config.keys():
        return default

    # No validation method exist => accept user input
    user_input = config[key]
    if key in conditions:
        condition = conditions[key]
    else:
        return user_input

    if type(default) is dict and isinstance(condition, dict):
        for i_key, val in default.items():
            user_input[i_key] = config_initialize(
                i_key, user_input, val, condition
            )
        return user_input
    elif isinstance(condition, type):
        if isinstance(user_input, condition):
            return user_input
        else:
            try:
                return condition(user_input)  # try type casting
            except ValueError:
                raise ValueError(
                    f"Expect '{user_input}' for '{key}' is {condition}"
                )
    elif isinstance(condition, Callable) and condition(user_input):
        return user_input
    else:
        raise ValueError(
            f"Given input '{user_input}' for '{key}' is not valid"
        )


def init_model_config(config: Dict):
    # defaults = _const.model_defaults(config)
    model_meta = {}

    # init complicated ones
    if KEY.CHEMICAL_SPECIES not in config.keys():
        raise ValueError('required key chemical_species not exist')
    input_chem = config[KEY.CHEMICAL_SPECIES]
    if isinstance(input_chem, str) and input_chem.lower() == 'auto':
        model_meta[KEY.CHEMICAL_SPECIES] = 'auto'
        model_meta[KEY.NUM_SPECIES] = 'auto'
        model_meta[KEY.TYPE_MAP] = 'auto'
    elif isinstance(input_chem, str) and 'univ' in input_chem.lower():
        model_meta.update(util.chemical_species_preprocess([], universal=True))
    else:
        if isinstance(input_chem, list) and all(
            isinstance(x, str) for x in input_chem
        ):
            pass
        elif isinstance(input_chem, str):
            input_chem = (
                input_chem.replace('-', ',').replace(' ', ',').split(',')
            )
            input_chem = [chem for chem in input_chem if len(chem) != 0]
        else:
            raise ValueError(f'given {KEY.CHEMICAL_SPECIES} input is strange')
        model_meta.update(util.chemical_species_preprocess(input_chem))

    # deprecation warnings
    if KEY.AVG_NUM_NEIGH in config:
        warnings.warn(
            "key 'avg_num_neigh' is deprecated. Please use 'conv_denominator'."
            ' We use the default, the average number of neighbors in the'
            ' dataset, if not provided.',
            UserWarning,
        )
        config.pop(KEY.AVG_NUM_NEIGH)
    if KEY.TRAIN_AVG_NUM_NEIGH in config:
        warnings.warn(
            "key 'train_avg_num_neigh' is deprecated. Please use"
            " 'train_denominator'. We overwrite train_denominator as given"
            ' train_avg_num_neigh',
            UserWarning,
        )
        config[KEY.TRAIN_DENOMINTAOR] = config[KEY.TRAIN_AVG_NUM_NEIGH]
        config.pop(KEY.TRAIN_AVG_NUM_NEIGH)
    if KEY.OPTIMIZE_BY_REDUCE in config:
        warnings.warn(
            "key 'optimize_by_reduce' is deprecated. Always true",
            UserWarning,
        )
        config.pop(KEY.OPTIMIZE_BY_REDUCE)

    # init simpler ones
    for key, default in _const.DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG.items():
        model_meta[key] = config_initialize(
            key, config, default, _const.MODEL_CONFIG_CONDITION
        )

    unknown_keys = [
        key for key in config.keys() if key not in model_meta.keys()
    ]
    if len(unknown_keys) != 0:
        warnings.warn(
            f'Unexpected model keys: {unknown_keys} will be ignored',
            UserWarning,
        )

    return model_meta


def init_train_config(config: Dict):
    train_meta = {}
    # defaults = _const.train_defaults(config)

    try:
        device_input = config[KEY.DEVICE]
        train_meta[KEY.DEVICE] = torch.device(device_input)
    except KeyError:
        train_meta[KEY.DEVICE] = (
            torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu')
        )
    train_meta[KEY.DEVICE] = str(train_meta[KEY.DEVICE])

    # init simpler ones
    for key, default in _const.DEFAULT_TRAINING_CONFIG.items():
        train_meta[key] = config_initialize(
            key, config, default, _const.TRAINING_CONFIG_CONDITION
        )

    if KEY.CONTINUE in config.keys():
        cnt_dct = config[KEY.CONTINUE]
        if KEY.CHECKPOINT not in cnt_dct.keys():
            raise ValueError('no checkpoint is given in continue')
        checkpoint = cnt_dct[KEY.CHECKPOINT]
        if os.path.isfile(checkpoint):
            checkpoint_file = checkpoint
        else:
            checkpoint_file = util.pretrained_name_to_path(checkpoint)
        train_meta[KEY.CONTINUE].update({KEY.CHECKPOINT: checkpoint_file})

    unknown_keys = [
        key for key in config.keys() if key not in train_meta.keys()
    ]
    if len(unknown_keys) != 0:
        warnings.warn(
            f'Unexpected train keys: {unknown_keys} will be ignored',
            UserWarning,
        )
    return train_meta


def init_data_config(config: Dict):
    data_meta = {}
    # defaults = _const.data_defaults(config)

    load_data_keys = []
    for k in config:
        if k.startswith('load_') and k.endswith('_path'):
            load_data_keys.append(k)

    for load_data_key in load_data_keys:
        if load_data_key in config.keys():
            inp = config[load_data_key]
            extended = []
            if type(inp) not in [str, list]:
                raise ValueError(f'unexpected input {inp} for sturcture_list')
            if type(inp) is str:
                extended = glob.glob(inp)
            elif type(inp) is list:
                for i in inp:
                    extended.extend(glob.glob(i))
            if len(extended) == 0:
                raise ValueError(
                    f'Cannot find {inp} for {load_data_key}'
                    + ' or path is not given'
                )
            data_meta[load_data_key] = extended
        else:
            data_meta[load_data_key] = False

    for key, default in _const.DEFAULT_DATA_CONFIG.items():
        data_meta[key] = config_initialize(
            key, config, default, _const.DATA_CONFIG_CONDITION
        )

    unknown_keys = [
        key for key in config.keys() if key not in data_meta.keys()
    ]
    if len(unknown_keys) != 0:
        warnings.warn(
            f'Unexpected data keys: {unknown_keys} will be ignored',
            UserWarning,
        )
    return data_meta


def read_config_yaml(filename: str, return_separately: bool = False):
    with open(filename, 'r') as fstream:
        inputs = yaml.safe_load(fstream)

    model_meta, train_meta, data_meta = {}, {}, {}
    for key, config in inputs.items():
        if key == 'model':
            model_meta = init_model_config(config)
        elif key == 'train':
            train_meta = init_train_config(config)
        elif key == 'data':
            data_meta = init_data_config(config)
        else:
            raise ValueError(f'Unexpected input {key} given')

    if return_separately:
        return model_meta, train_meta, data_meta
    else:
        model_meta.update(train_meta)
        model_meta.update(data_meta)
        return model_meta


def main():
    filename = './input.yaml'
    read_config_yaml(filename)


if __name__ == '__main__':
    main()
