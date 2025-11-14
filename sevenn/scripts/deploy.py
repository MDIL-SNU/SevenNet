import os
import pathlib
from datetime import datetime
from typing import Optional, Union

import e3nn.util.jit
import torch
from ase.data import chemical_symbols

import sevenn._keys as KEY
from sevenn import __version__
from sevenn.model_build import build_E3_equivariant_model
from sevenn.util import load_checkpoint


def deploy(
    checkpoint: Union[pathlib.Path, str],
    fname='deployed_serial.pt',
    modal: Optional[str] = None,
    use_flash: bool = False,
) -> None:
    from sevenn.nn.edge_embedding import EdgePreprocess
    from sevenn.nn.force_output import ForceStressOutput

    cp = load_checkpoint(checkpoint)

    model, config = (
        cp.build_model(
            enable_cueq=False, enable_flash=use_flash, _flash_lammps=use_flash
        ),
        cp.config,
    )

    model.prepand_module('edge_preprocess', EdgePreprocess(True))
    grad_module = ForceStressOutput()
    model.replace_module('force_output', grad_module)
    new_grad_key = grad_module.get_grad_key()
    model.key_grad = new_grad_key
    if hasattr(model, 'eval_type_map'):
        setattr(model, 'eval_type_map', False)

    if modal:
        model.prepare_modal_deploy(modal)
    elif model.modal_map is not None and len(model.modal_map) >= 1:
        raise ValueError(
            f'Modal is not given. It has: {list(model.modal_map.keys())}'
        )

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
    md_configs.update({'flashTP': 'yes' if use_flash else 'no'})
    md_configs.update(
        {'model_type': config.pop(KEY.MODEL_TYPE, 'E3_equivariant_model')}
    )
    md_configs.update({'version': __version__})
    md_configs.update({'dtype': config.pop(KEY.DTYPE, 'single')})
    md_configs.update({'time': datetime.now().strftime('%Y-%m-%d')})

    if fname.endswith('.pt') is False:
        fname += '.pt'
    torch.jit.save(model, fname, _extra_files=md_configs)


# TODO: build model only once
def deploy_parallel(
    checkpoint: Union[pathlib.Path, str],
    fname='deployed_parallel',
    modal: Optional[str] = None,
    use_flash: bool = False,
) -> None:
    # Additional layer for ghost atom (and copy parameters from original)
    GHOST_LAYERS_KEYS = ['onehot_to_feature_x', '0_self_interaction_1']

    cp = load_checkpoint(checkpoint)
    model, config = (
        cp.build_model(enable_cueq=False, enable_flash=use_flash),
        cp.config,
    )
    config[KEY.CUEQUIVARIANCE_CONFIG] = {'use': False}
    config[KEY.USE_FLASH_TP] = use_flash
    model_state_dct = model.state_dict()

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
        missing, _ = model_part.load_state_dict(model_state_dct, strict=False)
        if hasattr(model_part, 'eval_type_map'):
            setattr(model_part, 'eval_type_map', False)
        # Ensure all values are inserted
        assert len(missing) == 0 or use_flash, missing

    if modal:
        model_list[0].prepare_modal_deploy(modal)
    elif model.modal_map is not None and len(model.modal_map) >= 1:
        raise ValueError(
            f'Modal is not given. It has: {list(model_list[0].modal_map.keys())}'
        )

    # prepare some extra information for MD
    md_configs = {}
    type_map = config[KEY.TYPE_MAP]

    chem_list = ''
    for Z in type_map.keys():
        chem_list += chemical_symbols[Z] + ' '
    chem_list.strip()

    comm_size = max(
        [
            seg._modules[f'{t}_convolution']._comm_size  # type: ignore
            for t, seg in enumerate(model_list)
        ]
    )

    md_configs.update({'chemical_symbols_to_index': chem_list})
    md_configs.update({'cutoff': str(config[KEY.CUTOFF])})
    md_configs.update({'num_species': str(config[KEY.NUM_SPECIES])})
    md_configs.update({'comm_size': str(comm_size)})
    md_configs.update({'flashTP': 'yes' if use_flash else 'no'})
    md_configs.update(
        {'model_type': config.pop(KEY.MODEL_TYPE, 'E3_equivariant_model')}
    )
    md_configs.update({'version': __version__})
    md_configs.update({'dtype': config.pop(KEY.DTYPE, 'single')})
    md_configs.update({'time': datetime.now().strftime('%Y-%m-%d')})

    os.makedirs(fname)
    for idx, model in enumerate(model_list):
        fname_full = f'{fname}/deployed_parallel_{idx}.pt'
        model.set_is_batch_data(False)
        model.eval()

        model = e3nn.util.jit.script(model)
        model = torch.jit.freeze(model)

        torch.jit.save(model, fname_full, _extra_files=md_configs)
