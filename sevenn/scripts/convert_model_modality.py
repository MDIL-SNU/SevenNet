import math
from typing import List

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear

import sevenn._keys as KEY
from sevenn.model_build import build_E3_equivariant_model

modal_module_dict = {
    KEY.USE_MODAL_NODE_EMBEDDING: 'onehot_to_feature_x',
    KEY.USE_MODAL_SELF_INTER_INTRO: 'self_interaction_1',
    KEY.USE_MODAL_SELF_INTER_OUTRO: 'self_interaction_2',
    KEY.USE_MODAL_OUTPUT_BLOCK: 'reduce_input_to_hidden',
}


def _get_scalar_index(irreps: Irreps):
    scalar_indices = []
    for idx, (_, (l, p)) in enumerate(irreps):  # noqa
        if (
            l == 0 and p == 1
        ):  # get index of parameter for scalar (0e), which is used for modality
            scalar_indices.append(idx)

    return scalar_indices


def _reshape_weight_of_linear(
    irreps_in: Irreps, irreps_out: Irreps, weight: torch.Tensor
) -> List[torch.Tensor]:
    linear = Linear(irreps_in, irreps_out)
    linear.weight = nn.Parameter(weight)
    return list(linear.weight_views())


def _erase_linear_modal_params(
    model_state_dct: dict,
    erase_modal_indices: List[int],
    key: str,
    irreps_in: Irreps,
    irreps_out: Irreps,
):
    orig_input_dim = irreps_in.count('0e')
    new_input_dim = orig_input_dim - len(erase_modal_indices)

    orig_weight = model_state_dct[key + '.linear.weight']
    scalar_idx = _get_scalar_index(irreps_in)
    linear_weight_list = _reshape_weight_of_linear(
        irreps_in, irreps_out, orig_weight
    )

    new_weight_list = []

    for idx, l_p_weight in enumerate(linear_weight_list[:-1]):
        new_weight = torch.reshape(l_p_weight, (1, -1)).squeeze()
        if idx in scalar_idx:
            new_weight = new_weight * math.sqrt(new_input_dim / orig_input_dim)

        new_weight_list.append(new_weight)

    """
    Following works for normalization = `path`, which is not used in SEVENNet
    for l_p_weight in linear_weight_list[:-1]:
        new_weight_list.append(torch.reshape(l_p_weight, (1, -1)).squeeze())
    """

    flattened_weight = torch.cat(new_weight_list)

    return flattened_weight


def _get_modal_weight_as_bias(
    model_state_dct: dict,
    key: str,
    ref_index: int,
    irreps_in: Irreps,
    irreps_out: Irreps,
):
    assert ref_index != -1
    input_dim = irreps_in.count('0e')
    output_dim = irreps_out.count('0e')
    orig_weight = model_state_dct[key + '.linear.weight']
    orig_bias = model_state_dct[key + '.linear.bias']
    if len(orig_bias) == 0:
        orig_bias = torch.zeros(output_dim, dtype=orig_weight.dtype)

    modal_weight = _reshape_weight_of_linear(
        irreps_in, irreps_out, orig_weight
    )[-1]

    new_bias = orig_bias + modal_weight[ref_index] / math.sqrt(input_dim)

    return new_bias


def _append_modal_weight(
    model_state_dct: dict,  # state dict to be targeted
    key: str,  # linear weight modune name
    irreps_in: Irreps,  # irreps_in before modality append
    irreps_out: Irreps,
    append_number: int,
):
    # This works for normalization = `element`, default in SEVENNet.
    # (normalization = `path` is curruently deprecated in SEVENNet.)
    input_dim = irreps_in.count('0e')
    output_dim = irreps_out.count('0e')
    new_input_dim = input_dim + append_number
    orig_weight = model_state_dct[key + '.linear.weight']
    scalar_idx = _get_scalar_index(irreps_in)
    linear_weight_list = _reshape_weight_of_linear(
        irreps_in, irreps_out, orig_weight
    )

    new_weight_list = []

    # TODO: combine following as function with _erase_linear_modal_params

    for idx, l_p_weight in enumerate(linear_weight_list):
        new_weight = torch.reshape(l_p_weight, (1, -1)).squeeze()
        if idx in scalar_idx:
            new_weight = new_weight * math.sqrt(new_input_dim / input_dim)

        new_weight_list.append(new_weight)

    flattened_weight_list = []
    for l_p_weight in new_weight_list:
        flattened_weight_list.append(
            torch.reshape(l_p_weight, (1, -1)).squeeze()
        )
    flattened_weight = torch.cat(flattened_weight_list)

    append_weight = torch.cat([
        flattened_weight,
        torch.zeros(append_number * output_dim, dtype=flattened_weight.dtype),
    ])  # zeros: starting from common model

    return append_weight


def get_single_modal_model_dct(
    model_state_dct: dict,
    config: dict,
    ref_modal: str,
    from_processing_cp: bool = False,
    is_deploy: bool = False,
):
    """
    Convert multimodal model state dictionary to single modal model.
    Modal is selected by `ref_modal`

    `model_state_dct`: model state dictionary from multimodal checkpoint file
    `config`: dictionary containing configuration of the checkpoint model
    `ref_modal`: modal that are going to be converted
    `from_processing_cp`: if True, use modal_map of the checkpoint file
    `is_deploy`: if True, model is build with single-modal shift and scale
    """
    if (
        not from_processing_cp and not config[KEY.USE_MODALITY]
    ):  # model is already single modal
        return model_state_dct

    config[KEY.USE_BIAS_IN_LINEAR] = True
    config['_deploy'] = is_deploy

    model = build_E3_equivariant_model(config)
    del config['_deploy']
    key_add = '_cp' if from_processing_cp else ''
    modal_type_dict = config[KEY.MODAL_MAP + key_add]
    erase_modal_indices = range(len(modal_type_dict.keys()))  # starts with 0

    if ref_modal != 'common':
        try:
            ref_modal_index = modal_type_dict[ref_modal]
        except:
            raise KeyError(
                f'{ref_modal} not in modal type. Use one of'
                f' {modal_type_dict.keys()}.'
            )

    for module_key in model._modules.keys():
        for (
            use_modal_module_key,
            modal_module_name,
        ) in modal_module_dict.items():
            irreps_out = Irreps(model.get_irreps_in(module_key, 'irreps_out'))
            # TODO: directly using "irreps_in" might not be compatible
            # when changing `nn/linear.py`
            output_dim = irreps_out.count('0e')
            if (
                config[use_modal_module_key]
                and modal_module_name in module_key
            ):  # this module is used for giving modality

                irreps_in = Irreps(
                    model.get_irreps_in(module_key, 'irreps_in')
                )

                new_bias = (
                    torch.zeros(output_dim)
                    if ref_modal == 'common'
                    else _get_modal_weight_as_bias(
                        model_state_dct,
                        module_key,
                        ref_modal_index,
                        irreps_in,  # type: ignore
                        irreps_out,  # type: ignore
                    )
                )
                erased_modal_weight = _erase_linear_modal_params(
                    model_state_dct,
                    erase_modal_indices,
                    module_key,
                    irreps_in,  # type: ignore
                    irreps_out,  # type: ignore
                )

                model_state_dct[module_key + '.linear.weight'] = (
                    erased_modal_weight
                )
                model_state_dct[module_key + '.linear.bias'] = new_bias
            elif modal_module_name in module_key:
                model_state_dct[module_key + '.linear.bias'] = torch.zeros(
                    output_dim,
                    dtype=model_state_dct[module_key + '.linear.weight'].dtype,
                )

    final_block_key = 'reduce_hidden_to_energy'
    model_state_dct[final_block_key + '.linear.bias'] = torch.tensor(
        [0], dtype=model_state_dct[final_block_key + '.linear.weight'].dtype
    )

    if config[KEY.USE_MODAL_WISE_SHIFT] or config[KEY.USE_MODAL_WISE_SHIFT]:
        rescaler_names = []
        if config[KEY.USE_MODAL_WISE_SHIFT]:
            rescaler_names.append('shift')
        if config[KEY.USE_MODAL_WISE_SCALE]:
            rescaler_names.append('scale')
        config[KEY.USE_MODAL_WISE_SHIFT] = False
        config[KEY.USE_MODAL_WISE_SCALE] = False
        for rescaler_name in rescaler_names:
            rescaler_key = 'rescale_atomic_energy.' + rescaler_name
            rescaler = model_state_dct[rescaler_key][ref_modal_index]
            model_state_dct.update({rescaler_key: rescaler})
            config.update({rescaler_name: rescaler})

    config[KEY.USE_MODALITY] = False

    return model_state_dct


def append_modality_to_model_dct(
    model_state_dct: dict,
    config: dict,
    orig_num_modal: int,
    append_modal_length: int,
):
    """
    Append modal-wise parameters to the original linear layers.
    This enables expanding modal to single/multi modal model checkpoint.

    `model_state_dct`: model state dictionary from multimodal checkpoint file
    `config`: dictionary containing configuration of the checkpoint model
            + modality appended
    `orig_num_modal`: Number of modality used in original checkpoint
    `append_modal_length`: Number of modality to be appended in new checkpoint.
    """
    config_num_modal = config[KEY.NUM_MODALITIES]
    config.update({KEY.NUM_MODALITIES: orig_num_modal, KEY.USE_MODALITY: True})

    model = build_E3_equivariant_model(config)

    for module_key in model._modules.keys():
        for (
            use_modal_module_key,
            modal_module_name,
        ) in modal_module_dict.items():
            if (
                config[use_modal_module_key]
                and modal_module_name in module_key
            ):  # this module is used for giving modality
                irreps_in = model.get_irreps_in(
                    module_key, 'irreps_in'
                )
                # TODO: directly using "irreps_in" might not be compatible
                # when changing `nn/linear.py`
                irreps_out = model.get_irreps_in(module_key, 'irreps_out')
                irreps_in, irreps_out = Irreps(irreps_in), Irreps(irreps_out)

                append_weight = _append_modal_weight(
                    model_state_dct,
                    module_key,
                    irreps_in,  # type: ignore
                    irreps_out,  # type: ignore
                    append_modal_length,
                )
                model_state_dct[module_key + '.linear.weight'] = append_weight
    config[KEY.NUM_MODALITIES] = config_num_modal

    return model_state_dct
