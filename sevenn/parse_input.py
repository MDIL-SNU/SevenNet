import glob
import os
import warnings
from typing import Callable, Any

import torch
import yaml

import sevenn._const as _const
import sevenn._keys as KEY
from sevenn.train.optim import (
    loss_dict,
    loss_param_name_type_dict,
    optim_dict,
    optim_param_name_type_dict,
    scheduler_dict,
    scheduler_param_name_type_dict,
)
from sevenn.util import chemical_species_preprocess


def config_initialize(
    key: str, config: dict, default: Any, conditions,
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
            user_input[i_key] =\
                config_initialize(i_key, user_input, val, condition)
        return user_input
    elif isinstance(condition, type): 
        if isinstance(user_input, condition):
            return user_input
        else:
            try:
                return condition(user_input)  # try type casting
            except ValueError:
                raise ValueError(f"Expect '{user_input}' for '{key}' is {condition}")
    elif isinstance(condition, Callable) and condition(user_input):
        return user_input
    else:
        raise ValueError(f"Given input '{user_input}' for '{key}' is not valid")


def init_model_config(config: dict):
    defaults = _const.model_defaults(config)
    model_meta = {}

    # init complicated ones
    if KEY.CHEMICAL_SPECIES not in config.keys():
        raise ValueError('required key chemical_species not exist')
    input_chem = config[KEY.CHEMICAL_SPECIES]
    if type(input_chem) == str and input_chem.lower() == 'auto':
        model_meta[KEY.CHEMICAL_SPECIES] = 'auto'
        model_meta[KEY.NUM_SPECIES] = 'auto'
        model_meta[KEY.TYPE_MAP] = 'auto'
    else:
        if type(input_chem) == list and all(
            type(x) == str for x in input_chem
        ):
            pass
        elif type(input_chem) == str:
            input_chem = (
                input_chem.replace('-', ',').replace(' ', ',').split(',')
            )
            input_chem = [chem for chem in input_chem if len(chem) != 0]
        else:
            raise ValueError(f'given {KEY.CHEMICAL_SPECIES} input is strange')
        model_meta.update(chemical_species_preprocess(input_chem))

    ######## deprecation warnings #########
    if KEY.AVG_NUM_NEIGH in config:
        warnings.warn(
            "key 'avg_num_neigh' is deprecated. Please use 'conv_denominator'. "
            "We use the default, the average number of neighbors in the dataset, "
            "if not provided.",
            UserWarning,
        )
        config.pop(KEY.AVG_NUM_NEIGH)
    if KEY.TRAIN_AVG_NUM_NEIGH in config:
        warnings.warn(
            "key 'train_avg_num_neigh' is deprecated. Please use 'train_denominator'. "
            "We overwrite train_denominator as given train_avg_num_neigh",
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
    ######## deprecation warnings #########

    # init simpler ones
    for key, default in _const.DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG.items():
        model_meta[key] =\
            config_initialize(key, config, default, _const.MODEL_CONFIG_CONDITION)

    unknown_keys = [
        key for key in config.keys() if key not in model_meta.keys()
    ]
    if len(unknown_keys) != 0:
        raise ValueError(f'unknown keys : {unknown_keys} is given')

    return model_meta


def init_train_config(config: dict):
    train_meta = {}
    defaults = _const.train_defaults(config)

    try:
        device_input = config[KEY.DEVICE]
        # TODO: device input sanity?
        train_meta[KEY.DEVICE] = torch.device(device_input)
    except KeyError:
        train_meta[KEY.DEVICE] = (
            torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu')
        )

    name_dicts = [optim_dict, scheduler_dict, loss_dict]
    name_keys = [KEY.OPTIMIZER, KEY.SCHEDULER, KEY.LOSS]
    for idx, type_key in enumerate(name_keys):
        if type_key not in config.keys():
            train_meta[type_key] = defaults[type_key]
            continue
        user_input = config[type_key].lower()
        available_keys = name_dicts[idx].keys()
        if type(user_input) is not str:
            raise ValueError(f'{type_key} should be type: string.')
        if user_input not in available_keys:
            ava_key_to_str = ''
            for i, key in enumerate(available_keys):
                if i == 0:
                    ava_key_to_str += f'{key}'
                else:
                    ava_key_to_str += f', {key}'
            raise ValueError(f'{type_key} should be one of {ava_key_to_str}')
        train_meta[type_key] = user_input

    param_type_dicts = [
        optim_param_name_type_dict,
        scheduler_param_name_type_dict,
        loss_param_name_type_dict,
    ]
    for idx, param_key in enumerate(
        [KEY.OPTIM_PARAM, KEY.SCHEDULER_PARAM, KEY.LOSS_PARAM]
    ):
        if param_key not in config.keys():
            continue
        user_input = config[param_key]
        type_value = train_meta[name_keys[idx]]
        universal_keys = list(param_type_dicts[idx]['universial'].keys())
        available_keys = list(param_type_dicts[idx][type_value].keys())
        available_keys.extend(universal_keys)
        for key, value in user_input.items():
            # key = key.lower()  # case sensitive detect of param name
            if key not in available_keys:
                ava_key_to_str = ''
                for i, k in enumerate(available_keys):
                    if i == 0:
                        ava_key_to_str += f'{k}'
                    else:
                        ava_key_to_str += f', {k}'
                raise ValueError(
                    f'{param_key}: {key} should be one of {available_keys}'
                )
            if key in universal_keys:
                type_ = param_type_dicts[idx]['universial'][key]
            else:
                type_ = param_type_dicts[idx][type_value][key]

            if type(value) is not type_:
                raise ValueError(f'{param_key}: {key} should be type: {type_}')
        train_meta[param_key] = user_input

    if KEY.CONTINUE in config.keys():
        cnt_dct = config[KEY.CONTINUE]
        if KEY.CHECKPOINT not in cnt_dct.keys():
            raise ValueError('no checkpoint is given in continue')
        checkpoint = cnt_dct[KEY.CHECKPOINT]
        if type(checkpoint) != str or os.path.isfile(checkpoint) is False:
            raise ValueError(f'Checkpoint file:{checkpoint} is not found')
        train_meta[KEY.CONTINUE] = {}
        train_meta[KEY.CONTINUE][KEY.CHECKPOINT] = checkpoint

    # init simpler ones
    for key, default in _const.DEFAULT_TRAINING_CONFIG.items():
        train_meta[key] =\
            config_initialize(key, config, default, _const.TRAINING_CONFIG_CONDITION)

    unknown_keys = [
        key for key in config.keys() if key not in train_meta.keys()
    ]
    if len(unknown_keys) != 0:
        raise ValueError(f'unknown keys : {unknown_keys} is given')

    return train_meta


def init_data_config(config: dict):
    data_meta = {}
    defaults = _const.data_defaults(config)

    if KEY.LOAD_DATASET not in config.keys():
        raise ValueError('load_dataset_path is not given')

    for load_data_key in [KEY.LOAD_DATASET, KEY.LOAD_VALIDSET]:
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
                raise ValueError(f'Cannot find {inp} for {load_data_key}'
                    + ' or path is not given')
            data_meta[load_data_key] = extended
        else:
            data_meta[load_data_key] = False

    for key, default in _const.DEFAULT_DATA_CONFIG.items():
        data_meta[key] =\
            config_initialize(key, config, default, _const.DATA_CONFIG_CONDITION)

    unknown_keys = [
        key for key in config.keys() if key not in data_meta.keys()
    ]
    if len(unknown_keys) != 0:
        raise ValueError(f'unknown keys : {unknown_keys} is given')
    return data_meta


def read_config_yaml(filename):
    with open(filename, 'r') as fstream:
        inputs = yaml.safe_load(fstream)

    model_meta, train_meta, data_meta = None, None, None
    for key, config in inputs.items():
        if key == 'model':
            model_meta = init_model_config(config)
        elif key == 'train':
            train_meta = init_train_config(config)
        elif key == 'data':
            data_meta = init_data_config(config)
        else:
            raise ValueError(f'unexpected input {key} given')

    # how about model_config is None and 'continue_train' is True?
    if model_meta is None or train_meta is None or data_meta is None:
        raise ValueError('one of data, train, model is not provided')

    return model_meta, train_meta, data_meta


def main():
    filename = './input.yaml'
    read_config_yaml(filename)


if __name__ == '__main__':
    main()
