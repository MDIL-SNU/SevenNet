import os
import glob
from typing import List, Callable

import yaml
import torch

from sevenn.train.optim import (optim_dict, scheduler_dict,
                                optim_param_name_type_dict,
                                scheduler_param_name_type_dict,
                                loss_dict, loss_param_name_type_dict)

from sevenn.util import chemical_species_preprocess
import sevenn._keys as KEY
import sevenn._const as _const

#TODO: fix for ex)data key on train key and some auto spell check


def config_initialize(key: str, config: dict, default_dct: dict,
                      condition: Callable = None):
    """
    check require type by checking type of defaults
    cast user input to required type(type of default value)
    if casting was successful, check condition(user_input)
    if true, return processed input

    condition is callable of one input and boolean return
    if return of condition is False, raise error
    at condition stage, user input type is valid

    do dirty jobs for trivial configs for int, float, boolean
    """

    # no default value exist -> you should not use this function
    assert key in default_dct.keys()

    # default value exist & no user input -> return default
    default_val = default_dct[key]
    if key not in config.keys():
        return default_val

    # check user input from type of default value
    required_type = type(default_val)
    assert required_type in (float, int, bool, str, dict, list)

    user_input = config[key]
    if required_type in (int, float, str):
        try:
            user_input = required_type(user_input)
        except ValueError:
            raise ValueError(f"given {user_input} for {key} is strange")
        except TypeError:
            raise TypeError(f"given {user_input} for {key} is strange")
    elif required_type == bool:
        if type(user_input) == str:
            if user_input.lower() in ('yes', 'true', 't', '.t.', '1'):
                user_input = True
            elif user_input.lower() in ('no', 'false', 'f', '.f.', '0'):
                user_input = False
        elif type(user_input) == int:
            if user_input == 1:
                user_input = True
            elif user_input == 0 or user_input == -1:
                user_input = False
        elif type(user_input) == bool:
            pass
        else:
            raise ValueError(f"given {user_input} for {key} is not boolean")
    elif required_type == dict:
        # special case. recursively call this function for each dict key
        if type(user_input) != dict:
            raise ValueError(f"{key} should be dictionary")
        for inside_key, val in default_val.items():
            if val is None:
                continue  # ignore None default value of dict
            inside_condition = None
            if type(condition) == dict and inside_key in condition.keys():
                inside_condition = condition[inside_key]
            user_input[inside_key] =\
                config_initialize(inside_key, user_input,
                                  default_val, inside_condition)
    elif required_type == list:
        # assume all list memeber have same type
        if type(user_input) == str:
            user_input = user_input.replace('-', ',').replace(' ', ',').split(',')
        elif type(user_input) == list:  # yaml support list type input!
            pass
        else:
            raise ValueError(f"given {user_input} for {key} \
                    cannot be interpreted as list")

        list_member_type = type(default_val[0])
        try:
            user_input = [list_member_type(x) for x in user_input]
        except ValueError:
            raise ValueError(f"strange list input {user_input} for {key}")

    # dict is special case
    if required_type is dict or condition is None or condition(user_input):
        return user_input

    raise ValueError(f"unexpected input {user_input} for {key}")


def init_model_config(config: dict):
    defaults = _const.DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG
    model_meta = {}

    # init complicated ones
    if KEY.CHEMICAL_SPECIES not in config.keys():
        raise ValueError('required key chemical_species not exist')
    input_chem = config[KEY.CHEMICAL_SPECIES]
    if type(input_chem) == str and input_chem.lower() == 'auto':
        model_meta[KEY.CHEMICAL_SPECIES] = "auto"
        model_meta[KEY.NUM_SPECIES] = "auto"
        model_meta[KEY.TYPE_MAP] = "auto"
    else:
        if type(input_chem) == list and \
           all(type(x) == str for x in input_chem):
            pass
        elif type(input_chem) == str:
            input_chem = input_chem.replace('-', ',').replace(' ', ',').split(',')
            input_chem = [chem for chem in input_chem if len(chem) != 0]
        else:
            raise ValueError(f'given {KEY.CHEMICAL_SPECIES} input is strange')
        model_meta.update(chemical_species_preprocess(input_chem))
        """
        chemical_specie = sorted([x.strip() for x in input_chem])
        model_meta[KEY.CHEMICAL_SPECIES] = chemical_specie
        model_meta[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER] = \
            [atomic_numbers[x] for x in chemical_specie]
        model_meta[KEY.NUM_SPECIES] = len(chemical_specie)
        model_meta[KEY.TYPE_MAP] = get_type_mapper_from_specie(chemical_specie)
        """

    for act_key in [KEY.ACTIVATION_GATE, KEY.ACTIVATION_SCARLAR]:
        if act_key not in config.keys():
            model_meta[act_key] = defaults[act_key]
            continue
        user_input = config[act_key]
        if type(user_input) is not dict:
            raise ValueError(f"{act_key} should be dict of 'o' and 'e'")
        for key, val in user_input.items():
            if key == 'e' and val in _const.ACTIVATION_FOR_EVEN.keys():
                continue
            elif key == 'o' and val in _const.ACTIVATION_FOR_ODD.keys():
                continue
            else:
                raise ValueError(f"{act_key} should be dict of 'o' and 'e'")
        model_meta[act_key] = user_input

    # init simpler ones
    for key, cond in _const.MODEL_CONFIG_CONDITION.items():
        model_meta[key] = config_initialize(key, config, defaults, cond)

    unknown_keys = [key for key in config.keys() if key not in model_meta.keys()]
    if len(unknown_keys) != 0:
        raise ValueError(f"unknown keys : {unknown_keys} is given")

    return model_meta


def init_train_config(config: dict):
    train_meta = {}
    defaults = _const.DEFAULT_TRAINING_CONFIG

    try:
        device_input = config[KEY.DEVICE]
        # TODO: device input sanity?
        train_meta[KEY.DEVICE] = torch.device(device_input)
    except KeyError:
        train_meta[KEY.DEVICE] = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")

    name_dicts = [optim_dict, scheduler_dict, loss_dict]
    name_keys = [KEY.OPTIMIZER, KEY.SCHEDULER, KEY.LOSS]
    for idx, type_key in enumerate(name_keys):
        if type_key not in config.keys():
            train_meta[type_key] = defaults[type_key]
            continue

        user_input = config[type_key].lower()
        available_keys = name_dicts[idx].keys()

        if type(user_input) is not str:
            raise ValueError(f"{type_key} should be type: string.")

        if user_input not in available_keys:
            ava_key_to_str = ''
            for i, key in enumerate(available_keys):
                if i == 0:
                    ava_key_to_str += f'{key}'
                else:
                    ava_key_to_str += f', {key}'
            raise ValueError(f'{type_key} should be one of {ava_key_to_str}')
        train_meta[type_key] = user_input

    param_type_dicts = [optim_param_name_type_dict,
                        scheduler_param_name_type_dict,
                        loss_param_name_type_dict]
    for idx, param_key in enumerate([KEY.OPTIM_PARAM, KEY.SCHEDULER_PARAM, KEY.LOSS_PARAM]):
        if param_key not in config.keys():
            continue
        user_input = config[param_key]
        type_value = train_meta[name_keys[idx]]
        universal_keys = \
            list(param_type_dicts[idx]['universial'].keys())
        available_keys = \
            list(param_type_dicts[idx][type_value].keys())
        available_keys.extend(universal_keys)
        for key, value in user_input.items():
            #key = key.lower()  # case sensitive detect of param name
            if key not in available_keys:
                ava_key_to_str = ''
                for i, k in enumerate(available_keys):
                    if i == 0:
                        ava_key_to_str += f'{k}'
                    else:
                        ava_key_to_str += f', {k}'
                raise ValueError(
                    f'{param_key}: {key} should be one of {available_keys}')
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
            raise ValueError("no checkpoint is given in continue")
        checkpoint = cnt_dct[KEY.CHECKPOINT]
        if type(checkpoint) != str or os.path.isfile(checkpoint) is False:
            raise ValueError(f"Checkpoint file:{checkpoint} is not found")
        train_meta[KEY.CONTINUE] = {}
        train_meta[KEY.CONTINUE][KEY.CHECKPOINT] = checkpoint

    # init simpler ones
    for key, cond in _const.TRAINING_CONFIG_CONDITION.items():
        train_meta[key] = config_initialize(key, config, defaults, cond)

    if KEY.CHECKPOINT not in train_meta[KEY.CONTINUE].keys():
        train_meta[KEY.CONTINUE][KEY.CHECKPOINT] = False

    unknown_keys = [key for key in config.keys() if key not in train_meta.keys()]
    if len(unknown_keys) != 0:
        raise ValueError(f"unknown keys : {unknown_keys} is given")

    return train_meta


def init_data_config(config: dict):
    data_meta = {}
    defaults = _const.DEFAULT_DATA_CONFIG

    if KEY.LOAD_DATASET not in config.keys():
        raise ValueError("load_dataset_path is not given")

    for load_data_key in [KEY.LOAD_DATASET, KEY.LOAD_VALIDSET]:
        if load_data_key in config.keys():
            inp = config[load_data_key]
            extended = []
            if type(inp) not in [str, list]:
                raise ValueError(f"unexpected input {inp} for sturcture_list")
            if type(inp) is str:
                extended = glob.glob(inp)
            elif type(inp) is list:
                for i in inp:
                    extended.extend(glob.glob(i))
            data_meta[load_data_key] = extended
        else:
            data_meta[load_data_key] = False

    if KEY.SAVE_DATASET in config.keys():
        inp = config[KEY.SAVE_DATASET]
        if type(inp) is not str:
            raise ValueError("save_dataset_path is given but it is not pathlike")
        data_meta[KEY.SAVE_DATASET] = inp
    else:
        data_meta[KEY.SAVE_DATASET] = False

    for skey in [KEY.SHIFT, KEY.SCALE]:
        if skey not in config.keys():
            data_meta[skey] = defaults[skey]
            continue
        inp = config[skey]
        if type(inp) is int or type(inp) is float:
            data_meta[skey] = float(inp)
        elif type(inp) is list and all([type(i) is float for i in inp]):
            data_meta[skey] = inp
            config[KEY.USE_SPECIES_WISE_SHIFT_SCALE] = True
            data_meta[KEY.USE_SPECIES_WISE_SHIFT_SCALE] = True
        else:
            raise ValueError(f"shift/scale should be float or list of float")

    for key, cond in _const.DATA_CONFIG_CONDITION.items():
        data_meta[key] = config_initialize(key, config, defaults, cond)

    unknown_keys = [key for key in config.keys() if key not in data_meta.keys()]
    if len(unknown_keys) != 0:
        raise ValueError(f"unknow keys : {unknown_keys} is given")
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


if __name__ == "__main__":
    main()

