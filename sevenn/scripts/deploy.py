from datetime import datetime

import e3nn.util.jit
import torch
from ase.data import chemical_symbols

import sevenn._const as _const
import sevenn._keys as KEY
from sevenn.model_build import build_E3_equivariant_model


# TODO: this is E3_equivariant specific
def deploy(model_state_dct, config, fname):
    config[KEY.IS_TRACE_STRESS] = True
    config[KEY.IS_TRAIN_STRESS] = True
    model = build_E3_equivariant_model(config)
    missing, not_used = model.load_state_dict(model_state_dct, strict=False)
    assert len(missing) == 0, f'missing keys: {missing}'
    assert len(not_used) == 0, f'not used keys: {not_used}'

    model.set_is_batch_data(False)
    model.eval()

    model = e3nn.util.jit.script(model)
    model = torch.jit.freeze(model)

    # make some config need for md
    md_configs = {}
    type_map = config[KEY.TYPE_MAP]
    chem_list = ''
    for Z in type_map.keys():
        chem_list += chemical_symbols[Z] + ' '
    chem_list.strip()
    md_configs.update({'chemical_symbols_to_index': chem_list})
    md_configs.update({'cutoff': str(config[KEY.CUTOFF])})
    md_configs.update({'num_species': str(config[KEY.NUM_SPECIES])})
    md_configs.update({'model_type': config[KEY.MODEL_TYPE]})
    md_configs.update({'version': _const.SEVENN_VERSION})
    md_configs.update({'dtype': config[KEY.DTYPE]})
    md_configs.update({'time': datetime.now().strftime('%Y-%m-%d')})

    if fname.endswith('.pt') is False:
        fname += '.pt'
    torch.jit.save(model, fname, _extra_files=md_configs)


# TODO: this is E3_equivariant specific
def deploy_parallel(model_state_dct, config, fname):
    # Additional layer for ghost atom (and copy parameters from original)
    GHOST_LAYERS_KEYS = ['onehot_to_feature_x', '0_self_interaction_1']

    # TODO: stress inference
    config[KEY.IS_TRACE_STRESS] = False
    config[KEY.IS_TRAIN_STRESS] = False
    model_list = build_E3_equivariant_model(config, parallel=True)
    dct_temp = {}
    copy_counter = {gk: 0 for gk in GHOST_LAYERS_KEYS}
    for ghost_layer_key in GHOST_LAYERS_KEYS:
        for key, val in model_state_dct.items():
            if not key.startswith(ghost_layer_key):
                continue
            dct_temp.update({f'ghost_{key}': val})
            copy_counter[ghost_layer_key] += 1
    # Ensure reference weights are copied from state dict
    assert all(x > 0 for x in copy_counter.values())

    model_state_dct.update(dct_temp)

    for model_part in model_list:
        missing, _ =\
            model_part.load_state_dict(model_state_dct, strict=False)
        # Ensure all values are inserted
        assert len(missing) == 0

    # prepare some extra information for MD
    md_configs = {}
    type_map = config[KEY.TYPE_MAP]

    chem_list = ''
    for Z in type_map.keys():
        chem_list += chemical_symbols[Z] + ' '
    chem_list.strip()

    # dim of irreps_in of last model convolution is (max)comm_size
    # except first one, first of every model is embedding followed by convolution
    # TODO: this code is error prone
    comm_size = (
        model_list[-1][1].convolution.irreps_in1.dim
        if len(model_list) > 1
        else 0
    )

    md_configs.update({'chemical_symbols_to_index': chem_list})
    md_configs.update({'cutoff': str(config[KEY.CUTOFF])})
    md_configs.update({'num_species': str(config[KEY.NUM_SPECIES])})
    md_configs.update({'comm_size': str(comm_size)})
    md_configs.update({'model_type': config[KEY.MODEL_TYPE]})
    md_configs.update({'version': _const.SEVENN_VERSION})
    md_configs.update({'dtype': config[KEY.DTYPE]})
    md_configs.update({'time': datetime.now().strftime('%Y-%m-%d')})

    for idx, model in enumerate(model_list):
        fname_full = f'{fname}_{idx}.pt'
        model.set_is_batch_data(False)
        model.eval()

        model = e3nn.util.jit.script(model)
        model = torch.jit.freeze(model)

        torch.jit.save(model, fname_full, _extra_files=md_configs)
