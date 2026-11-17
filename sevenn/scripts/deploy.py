import os
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import e3nn.util.jit
import torch
from ase.data import chemical_symbols

import sevenn._keys as KEY
from sevenn import __version__
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.util import load_checkpoint, warn_no_tp_accelerator


def deploy(
    checkpoint: Union[pathlib.Path, str],
    fname='deployed_serial.pt',
    modal: Optional[str] = None,
    use_flash: bool = False,
    use_oeq: bool = False,
) -> None:
    if not (use_flash or use_oeq):
        warn_no_tp_accelerator('LAMMPS TorchScript deployment')

    cp = load_checkpoint(checkpoint)
    model, config = (
        cp.build_model(
            enable_cueq=False,
            enable_flash=use_flash,
            enable_oeq=use_oeq,
            _flash_lammps=use_flash,
        ),
        cp.config,
    )

    if 'force_output' in model._modules:
        model.delete_module_by_key('force_output')
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
    md_configs.update({'oeq': 'yes' if use_oeq else 'no'})
    md_configs.update(
        {'model_type': config.pop(KEY.MODEL_TYPE, 'E3_equivariant_model')}
    )
    md_configs.update({'version': __version__})
    md_configs.update({'dtype': config.pop(KEY.DTYPE, 'single')})
    md_configs.update({'time': datetime.now().strftime('%Y-%m-%d')})

    if fname.endswith('.pt') is False:
        fname += '.pt'
    torch.jit.save(model, fname, _extra_files=md_configs)


def build_parallel_model_list(
    model_state_dct: Dict[str, Any],
    config: Dict[str, Any],
    modal: Optional[str] = None,
    use_flash: bool = False,
    use_oeq: bool = False,
) -> List[AtomGraphSequential]:
    """
    Segments of a LAMMPS e3gnn/parallel model (in memory, eager) from the
    state dict and config of a serial model, exactly as deploy_parallel saves
    them (before TorchScript). Every 'ghost_<key>' entry (layer-0 twins acting
    on the ghost atoms, model_build._to_parallel_model) gets the weights of
    <key>.
    """
    config = dict(config)
    config[KEY.CUEQUIVARIANCE_CONFIG] = {'use': False}
    config[KEY.USE_FLASH_TP] = use_flash
    config[KEY.USE_OEQ] = use_oeq
    config[KEY.USE_LUNATP] = False  # ML-IAP only
    config['_flash_lammps'] = use_flash

    model_list = build_E3_equivariant_model(config, parallel=True)
    model_state_dct = dict(model_state_dct)
    ghost_keys = {
        key for seg in model_list for key in seg.state_dict()
        if key.startswith('ghost_')
    }
    assert ghost_keys, 'no ghost layers found'
    for key in sorted(ghost_keys):
        src = key[len('ghost_'):]
        # Ensure reference weights are copied from state dict
        assert src in model_state_dct, f'{key}: {src} not in the state dict'
        model_state_dct[key] = model_state_dct[src]

    for model_part in model_list:
        missing, _ = model_part.load_state_dict(model_state_dct, strict=False)
        if hasattr(model_part, 'eval_type_map'):
            setattr(model_part, 'eval_type_map', False)
        # Ensure all values are inserted
        assert len(missing) == 0 or use_flash or use_oeq, missing

    modal_map = model_list[0].modal_map
    if modal:
        model_list[0].prepare_modal_deploy(modal)
    elif modal_map is not None and len(modal_map) >= 1:
        raise ValueError(f'Modal is not given. It has: {list(modal_map.keys())}')
    return model_list


def deploy_parallel(
    checkpoint: Union[pathlib.Path, str],
    fname='deployed_parallel',
    modal: Optional[str] = None,
    use_flash: bool = False,
    use_oeq: bool = False,
) -> None:
    if not (use_flash or use_oeq):
        warn_no_tp_accelerator(
            'LAMMPS parallel TorchScript deployment',
        )

    cp = load_checkpoint(checkpoint)

    model, config = (
        cp.build_model(
            enable_cueq=False,
            enable_flash=use_flash,
            enable_oeq=use_oeq,
            _flash_lammps=use_flash,
        ),
        cp.config,
    )
    model_list = build_parallel_model_list(
        model.state_dict(), config, modal, use_flash, use_oeq
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
    md_configs.update({'oeq': 'yes' if use_oeq else 'no'})
    md_configs.update(
        {'model_type': config.pop(KEY.MODEL_TYPE, 'E3_equivariant_model')}
    )
    md_configs.update({'version': __version__})
    md_configs.update({'dtype': config.pop(KEY.DTYPE, 'single')})
    md_configs.update({'time': datetime.now().strftime('%Y-%m-%d')})

    os.makedirs(fname, exist_ok=True)
    for idx, model in enumerate(model_list):
        fname_full = f'{fname}/deployed_parallel_{idx}.pt'
        model.set_is_batch_data(False)
        model.eval()

        model = e3nn.util.jit.script(model)
        model = torch.jit.freeze(model)

        torch.jit.save(model, fname_full, _extra_files=md_configs)
