import os
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn
from e3nn.o3 import FullTensorProduct, Irreps

import sevenn._keys as KEY


def to_atom_graph_list(atom_graph_batch):
    """
    torch_geometric batched data to separate list
    original to_data_list() by PyG is not enough since
    it doesn't handle inferred tensors
    """
    is_stress = KEY.PRED_STRESS in atom_graph_batch

    data_list = atom_graph_batch.to_data_list()

    indices = atom_graph_batch[KEY.NUM_ATOMS].tolist()

    atomic_energy_list = torch.split(atom_graph_batch[KEY.ATOMIC_ENERGY], indices)
    inferred_total_energy_list = torch.unbind(
        atom_graph_batch[KEY.PRED_TOTAL_ENERGY]
    )
    inferred_force_list = torch.split(atom_graph_batch[KEY.PRED_FORCE], indices)

    inferred_stress_list = None
    if is_stress:
        inferred_stress_list = torch.unbind(atom_graph_batch[KEY.PRED_STRESS])

    for i, data in enumerate(data_list):
        data[KEY.ATOMIC_ENERGY] = atomic_energy_list[i]
        data[KEY.PRED_TOTAL_ENERGY] = inferred_total_energy_list[i]
        data[KEY.PRED_FORCE] = inferred_force_list[i]
        # To fit with KEY.STRESS (ref) format
        if is_stress and inferred_stress_list is not None:
            data[KEY.PRED_STRESS] = torch.unsqueeze(inferred_stress_list[i], 0)
    return data_list


def error_recorder_from_loss_functions(loss_functions):
    from .error_recorder import ErrorRecorder, MAError, RMSError, get_err_type
    from .train.loss import ForceLoss, PerAtomEnergyLoss, StressLoss

    metrics = []
    for loss_function, _ in loss_functions:
        ref_key = loss_function.ref_key
        pred_key = loss_function.pred_key
        # unit = loss_function.unit
        criterion = loss_function.criterion
        name = loss_function.name
        base = None
        if type(loss_function) is PerAtomEnergyLoss:
            base = get_err_type('Energy')
        elif type(loss_function) is ForceLoss:
            base = get_err_type('Force')
        elif type(loss_function) is StressLoss:
            base = get_err_type('Stress')
        else:
            base = {}
        base['name'] = name
        base['ref_key'] = ref_key
        base['pred_key'] = pred_key
        if type(criterion) is torch.nn.MSELoss:
            base['name'] = base['name'] + '_RMSE'
            metrics.append(RMSError(**base))
        elif type(criterion) is torch.nn.L1Loss:
            metrics.append(MAError(**base))
    return ErrorRecorder(metrics)


def postprocess_output(output, loss_types):
    from ._const import LossType

    """
    Postprocess output from model to be used for loss calculation
    Flatten all the output & unit converting and store them as (pred, ref, vdim)
    Averaging them without care of vdim results in component-wise something
    Args:
        output (dict): output from model
        loss_types (list): list of loss types to be calculated
    Returns:
        results (dict): dictionary of loss type and its corresponding
    """
    TO_KB = 1602.1766208  # eV/A^3 to kbar
    results = {}
    for loss_type in loss_types:
        if loss_type is LossType.ENERGY:
            # dim: (num_batch)
            num_atoms = output[KEY.NUM_ATOMS]
            pred = output[KEY.PRED_TOTAL_ENERGY] / num_atoms
            ref = output[KEY.ENERGY] / num_atoms
            vdim = 1
        elif loss_type is LossType.FORCE:
            # dim: (total_number_of_atoms_over_batch, 3)
            pred = torch.reshape(output[KEY.PRED_FORCE], (-1,))
            ref = torch.reshape(output[KEY.FORCE], (-1,))
            vdim = 3
        elif loss_type is LossType.STRESS:
            # dim: (num_batch, 6)
            # calculate stress loss based on kB unit (was eV/A^3)
            pred = torch.reshape(output[KEY.PRED_STRESS] * TO_KB, (-1,))
            ref = torch.reshape(output[KEY.STRESS] * TO_KB, (-1,))
            vdim = 6
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')
        results[loss_type] = (pred, ref, vdim)
    return results


def onehot_to_chem(one_hot_indices: List[int], type_map: Dict[int, int]):
    from ase.data import chemical_symbols

    type_map_rev = {v: k for k, v in type_map.items()}
    return [chemical_symbols[type_map_rev[x]] for x in one_hot_indices]


def _patch_old_config(config: Dict[str, Any]):
    # Fixing my old mistakes
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
        # Warn this in the docs, not here for SevenNet-0 (22May2024)
    return config


def _map_old_model(old_model_state_dict):
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


def model_from_checkpoint(checkpoint) -> Tuple[torch.nn.Module, Dict]:
    from ._const import model_defaults
    from .model_build import build_E3_equivariant_model

    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
    elif isinstance(checkpoint, dict):
        pass
    else:
        raise ValueError('checkpoint must be either str or dict')

    model_state_dict = checkpoint['model_state_dict']
    config = checkpoint['config']
    defaults = {**model_defaults(config)}
    config = _patch_old_config(config)

    for k, v in defaults.items():
        if k not in config:
            warnings.warn(f'{k} not in config, using default value {v}', UserWarning)
            config[k] = v

    # expect only non-tensor values in config, if exists, move to cpu
    # This can be happen if config has torch tensor as value (shift, scale)
    # TODO: save only non-tensors at first place is better
    for k, v in config.items():
        if isinstance(v, torch.Tensor):
            config[k] = v.cpu()

    model = build_E3_equivariant_model(config)
    assert isinstance(model, torch.nn.Module)
    missing, _ = model.load_state_dict(model_state_dict, strict=False)
    if len(missing) > 0:
        updated = _map_old_model(model_state_dict)
        missing, not_used = model.load_state_dict(updated, strict=False)
        if len(not_used) > 0:
            warnings.warn(f'Some keys are not used: {not_used}', UserWarning)

    assert len(missing) == 0, f'Missing keys: {missing}'

    return model, config


def unlabeled_atoms_to_input(atoms, cutoff: float, grad_key: str = KEY.EDGE_VEC):
    from .atom_graph_data import AtomGraphData
    from .train.dataload import unlabeled_atoms_to_graph

    atom_graph = AtomGraphData.from_numpy_dict(
        unlabeled_atoms_to_graph(atoms, cutoff)
    )
    atom_graph[grad_key].requires_grad_(True)
    atom_graph[KEY.BATCH] = torch.zeros([0])
    return atom_graph


def chemical_species_preprocess(input_chem: List[str], universal: bool = False):
    from ase.data import atomic_numbers, chemical_symbols

    from .nn.node_embedding import get_type_mapper_from_specie

    config = {}
    if not universal:
        input_chem = list(set(input_chem))
        chemical_specie = sorted([x.strip() for x in input_chem])
        config[KEY.CHEMICAL_SPECIES] = chemical_specie
        config[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER] = [
            atomic_numbers[x] for x in chemical_specie
        ]
        config[KEY.NUM_SPECIES] = len(chemical_specie)
        config[KEY.TYPE_MAP] = get_type_mapper_from_specie(chemical_specie)
    else:
        config[KEY.CHEMICAL_SPECIES] = chemical_symbols
        len_univ = len(chemical_symbols)
        config[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER] = list(range(len_univ))
        config[KEY.NUM_SPECIES] = len_univ
        config[KEY.TYPE_MAP] = {z: z for z in range(len_univ)}
    return config


def dtype_correct(
    v: Union[np.ndarray, torch.Tensor, int, float],
    float_dtype: torch.dtype = torch.float32,
    int_dtype: torch.dtype = torch.int64,
):
    if isinstance(v, np.ndarray):
        if np.issubdtype(v.dtype, np.floating):
            return torch.from_numpy(v).to(float_dtype)
        elif np.issubdtype(v.dtype, np.integer):
            return torch.from_numpy(v).to(int_dtype)
    elif isinstance(v, torch.Tensor):
        if v.dtype.is_floating_point:
            return v.to(float_dtype)  # convert to specified float dtype
        else:  # assuming non-floating point tensors are integers
            return v.to(int_dtype)  # convert to specified int dtype
    else:  # scalar values
        if isinstance(v, int):
            return torch.tensor(v, dtype=int_dtype)
        elif isinstance(v, float):
            return torch.tensor(v, dtype=float_dtype)
        else:  # Not numeric
            return v


def infer_irreps_out(
    irreps_x: Irreps,
    irreps_operand: Irreps,
    drop_l: Union[bool, int] = False,
    parity_mode: str = 'full',
    fix_multiplicity: Union[bool, int] = False,
):
    assert parity_mode in ['full', 'even', 'sph']
    # (mul, (ir, p))
    irreps_out = FullTensorProduct(irreps_x, irreps_operand).irreps_out.simplify()
    new_irreps_elem = []
    for mul, (l, p) in irreps_out:
        elem = (mul, (l, p))
        if drop_l is not False and l > drop_l:
            continue
        if parity_mode == 'even' and p == -1:
            continue
        elif parity_mode == 'sph' and p != (-1) ** l:
            continue
        if fix_multiplicity:
            elem = (fix_multiplicity, (l, p))
        new_irreps_elem.append(elem)
    return Irreps(new_irreps_elem)


def pretrained_name_to_path(name: str) -> str:
    import sevenn._const as _const

    name = name.lower()
    heads = ['sevennet-0', '7net-0']
    checkpoint_path = None
    if name in [f'{n}_11july2024' for n in heads] or name in heads:
        checkpoint_path = _const.SEVENNET_0_11July2024
    elif name in [f'{n}_22may2024' for n in heads]:
        checkpoint_path = _const.SEVENNET_0_22May2024
    else:
        raise ValueError('Not a valid potential')

    return checkpoint_path


def unique_filepath(filepath: str) -> str:
    if not os.path.isfile(filepath):
        return filepath
    else:
        dirname = os.path.dirname(filepath)
        fname = os.path.basename(filepath)
        name, ext = os.path.splitext(fname)
        cnt = 0
        new_name = f'{name}{cnt}{ext}'
        new_path = os.path.join(dirname, new_name)
        while os.path.exists(new_path):
            cnt += 1
            new_name = f'{name}{cnt}{ext}'
            new_path = os.path.join(dirname, new_name)
        return new_path


def get_error_recorder(
    recorder_tuples: List[Tuple[str, str]] = [
        ('Energy', 'RMSE'),
        ('Force', 'RMSE'),
        ('Stress', 'RMSE'),
        ('Energy', 'MAE'),
        ('Force', 'MAE'),
        ('Stress', 'MAE'),
    ],
):
    # TODO add criterion argument and loss recorder selections
    import sevenn.error_recorder as error_recorder

    config = recorder_tuples
    err_metrics = []
    for err_type, metric_name in config:
        metric_kwargs = error_recorder.get_err_type(err_type).copy()
        metric_kwargs['name'] += f'_{metric_name}'
        metric_cls = error_recorder.ErrorRecorder.METRIC_DICT[metric_name]
        err_metrics.append(metric_cls(**metric_kwargs))
    return error_recorder.ErrorRecorder(err_metrics)
