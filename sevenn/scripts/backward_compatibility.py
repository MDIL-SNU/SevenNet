"""
Debt
keep old pre-trained checkpoints unchanged.
"""

import copy

import torch

import sevenn._keys as KEY


def version_tuple(v1):
    v1 = tuple(map(int, v1.split('.')))
    return v1


def patch_old_config(config):
    version = config.get('version', None)
    if not version:
        raise ValueError('No version found in config')

    major, minor, _ = version.split('.')[:3]
    major, minor = int(major), int(minor)

    if major == 0 and minor <= 9:
        if config[KEY.CUTOFF_FUNCTION][KEY.CUTOFF_FUNCTION_NAME] == 'XPLOR':
            config[KEY.CUTOFF_FUNCTION].pop('poly_cut_p_value', None)
        if KEY.TRAIN_DENOMINTAOR not in config:
            config[KEY.TRAIN_DENOMINTAOR] = config.pop('train_avg_num_neigh', False)
        _opt = config.pop('optimize_by_reduce', None)
        if _opt is False:
            raise ValueError(
                'This checkpoint(optimize_by_reduce: False) is no longer supported'
            )
        if KEY.CONV_DENOMINATOR not in config:
            config[KEY.CONV_DENOMINATOR] = 0.0
        if KEY._NORMALIZE_SPH not in config:
            config[KEY._NORMALIZE_SPH] = False

    return config


def map_old_model(old_model_state_dict):
    """
    For compatibility with old namings (before 'correct' branch merged 2404XX)
    Map old model's module names to new model's module names
    """
    _old_module_name_mapping = {
        'EdgeEmbedding': 'edge_embedding',
        'reducing nn input to hidden': 'reduce_input_to_hidden',
        'reducing nn hidden to energy': 'reduce_hidden_to_energy',
        'rescale atomic energy': 'rescale_atomic_energy',
    }
    for i in range(10):
        _old_module_name_mapping[f'{i} self connection intro'] = (
            f'{i}_self_connection_intro'
        )
        _old_module_name_mapping[f'{i} convolution'] = f'{i}_convolution'
        _old_module_name_mapping[f'{i} self interaction 2'] = (
            f'{i}_self_interaction_2'
        )
        _old_module_name_mapping[f'{i} equivariant gate'] = f'{i}_equivariant_gate'

    new_model_state_dict = {}
    for k, v in old_model_state_dict.items():
        key_name = k.split('.')[0]
        follower = '.'.join(k.split('.')[1:])
        if 'denumerator' in follower:
            follower = follower.replace('denumerator', 'denominator')
        if key_name in _old_module_name_mapping:
            new_key_name = _old_module_name_mapping[key_name] + '.' + follower
            new_model_state_dict[new_key_name] = v
        else:
            new_model_state_dict[k] = v
    return new_model_state_dict


def sort_old_convolution(model_now, state_dict):
    from e3nn.o3 import wigner_3j

    """
    Reason1: we have to sort instructions of convolution to be compatible with
    cuEquivariance. (therefore, sort weight)
    Reason2: some of old convolution module's w3j coeff has flipped sign. This also
    has to be fixed to be compatible with cuEquivarinace.
    """

    def patch(stct):
        inst_old = copy.copy(conv._instructions_before_sort)
        inst_old = [(inst[0], inst[1], inst[2]) for inst in inst_old]
        del conv._instructions_before_sort

        conv_args = conv.convolution_kwargs
        irreps_in1 = conv_args['irreps_in1']
        irreps_in2 = conv_args['irreps_in2']
        irreps_out = conv_args.get('irreps_out', conv_args.get('filter_irreps_out'))

        inst_sorted = sorted(inst_old, key=lambda x: x[2])

        inst_sorted = [
            # in1, in2, out, weights
            (inst[0], inst[1], inst[2], irreps_in1[inst[0]].mul)
            for inst in inst_sorted
        ]

        n = len(weight_nn.hs) - 2
        ww_key = f'{conv_key}.weight_nn.layer{n}.weight'
        ww = stct[ww_key]
        ww_sorted = [None] * len(inst_old)

        _prev_idx = 0
        for ist_src in inst_old:
            for j, ist_dst in enumerate(inst_sorted):
                if not all(ist_src[ii] == ist_dst[ii] for ii in range(3)):
                    continue

                numel = ist_dst[3]  # weight num
                ww_src = ww[:, _prev_idx : _prev_idx + numel]
                l1, l2, l3 = (
                    irreps_in1[ist_src[0]].ir.l,
                    irreps_in2[ist_src[1]].ir.l,
                    irreps_out[ist_src[2]].ir.l,
                )
                if l1 > 0 and l2 > 0 and l3 > 0:
                    w3j_key = f'_w3j_{l1}_{l2}_{l3}'
                    conv_w3j_key = (
                        f'{conv_key}.convolution._compiled_main_left_right.{w3j_key}'
                    )
                    w3j_old = stct[conv_w3j_key]
                    w3j_now = wigner_3j(l1, l2, l3)
                    if not torch.allclose(w3j_old.to(w3j_now.device), w3j_now):
                        assert torch.allclose(
                            w3j_old.to(w3j_now.device), -1 * w3j_now
                        )
                        ww_src = -1 * ww_src
                        stct[conv_w3j_key] *= -1  # stct updated
                _prev_idx += numel
                ww_sorted[j] = ww_src
        ww_sorted = torch.cat(ww_sorted, dim=1)  # type: ignore
        stct[ww_key] = ww_sorted.clone()  # stct updated

    conv_dicts = {}
    for k, v in state_dict.items():
        key_name = k.split('.')[0]
        if key_name.split('_')[1] == 'convolution':
            if key_name not in conv_dicts:
                conv_dicts[key_name] = {}
            conv_dicts[key_name].update({k: v})

    new_state_dict = {}
    new_state_dict.update(state_dict)
    for conv_key, conv_state_dict in conv_dicts.items():
        conv = model_now._modules[conv_key]
        weight_nn = conv.weight_nn
        patch(conv_state_dict)
        new_state_dict.update(conv_state_dict)

    return new_state_dict


def patch_state_dict_if_old(state_dict, config_cp, now_model):
    version = config_cp.get('version', None)
    if not version:
        raise ValueError('No version found in config')
    vs = version.split('.')
    vsuffix = ''
    if len(vs) == 4:
        vsuffix = vs[-1]
        vs = version_tuple('.'.join(vs[:3]))
    else:
        vs = version_tuple('.'.join(vs))

    if vs < version_tuple('0.10.0'):
        state_dict = map_old_model(state_dict)

    # TODO: change version criteria before release!!!
    #       it causes problem if model is sorted but this function is called
    #       ... more robust way? idk
    if vs < version_tuple('0.11.0') or (
        vs == version_tuple('0.11.0') and vsuffix == 'dev0'
    ):
        state_dict = sort_old_convolution(now_model, state_dict)
    return state_dict
