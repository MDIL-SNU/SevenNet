import copy
import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn
from e3nn.o3 import FullTensorProduct, Irreps

import sevenn._keys as KEY
import sevenn.scripts.backward_compatibility as compat


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


def onehot_to_chem(one_hot_indices: List[int], type_map: Dict[int, int]):
    from ase.data import chemical_symbols

    type_map_rev = {v: k for k, v in type_map.items()}
    return [chemical_symbols[type_map_rev[x]] for x in one_hot_indices]


def _config_cp_routine(config):
    from ._const import model_defaults

    defaults = {**model_defaults(config)}
    config = compat.patch_old_config(config)  # type: ignore

    for k, v in defaults.items():
        if k in config:
            continue
        if os.getenv('SEVENN_DEBUG', False):
            warnings.warn(f'{k} not in config, use default value {v}', UserWarning)
        config[k] = v

    # expect only non-tensor values in config, if exists, move to cpu
    # This can be happen if config has torch tensor as value (shift, scale)
    # TODO: save only non-tensors at first place is better
    for k, v in config.items():
        if isinstance(v, torch.Tensor):
            config[k] = v.cpu()
    return config


def _e3nn_to_cue(stct_src, stct_dst, src_config):
    """
    manually check keys and assert if something unexpected happens
    """
    n_layer = src_config['num_convolution_layer']

    linear_module_names = [
        'onehot_to_feature_x',
        'reduce_input_to_hidden',
        'reduce_hidden_to_energy',
    ]
    convolution_module_names = []
    fc_tensor_product_module_names = []
    for i in range(n_layer):
        linear_module_names.append(f'{i}_self_interaction_1')
        linear_module_names.append(f'{i}_self_interaction_2')
        if src_config.get(KEY.SELF_CONNECTION_TYPE) == 'linear':
            linear_module_names.append(f'{i}_self_connection_intro')
        elif src_config.get(KEY.SELF_CONNECTION_TYPE) == 'nequip':
            fc_tensor_product_module_names.append(f'{i}_self_connection_intro')
        convolution_module_names.append(f'{i}_convolution')

    # Rule: those keys can be safely ignored before state dict load,
    #       except for linear.bias. This should be aborted in advance to
    #       this function. Others are not parameters but constants.
    cue_only_linear_followers = ['linear.f.tp.f_fx.module.c']
    e3nn_only_linear_followers = ['linear.bias', 'linear.output_mask']
    ignores_in_linear = cue_only_linear_followers + e3nn_only_linear_followers

    cue_only_conv_followers = [
        'convolution.f.tp.f_fx.module.c'
    ]
    e3nn_only_conv_followers = [
        'convolution._compiled_main_left_right._w3j',
        'convolution.weight',
        'convolution.output_mask',
    ]
    ignores_in_conv = cue_only_conv_followers + e3nn_only_conv_followers

    cue_only_fc_followers = [
        'fc_tensor_product.f.tp.f_fx.module.c'
    ]
    e3nn_only_fc_followers = [
        'fc_tensor_product.output_mask',
    ]
    ignores_in_fc = cue_only_fc_followers + e3nn_only_fc_followers

    updated_keys = []
    stct_updated = stct_dst.copy()
    for k, v in stct_src.items():
        module_name = k.split('.')[0]
        flag = False
        if module_name in linear_module_names:
            for ignore in ignores_in_linear:
                if '.'.join([module_name, ignore]) in k:
                    flag = True
                    break
            if not flag and k == '.'.join([module_name, 'linear.weight']):
                updated_keys.append(k)
                stct_updated[k] = v.clone()
                flag = True
            assert flag, f'Unexpected key from linear: {k}'
        elif module_name in convolution_module_names:
            for ignore in ignores_in_conv:
                if '.'.join([module_name, ignore]) in k:
                    flag = True
                    break
            if not flag and (
                k.startswith(f'{module_name}.weight_nn')
                or k == '.'.join([module_name, 'denominator'])
            ):
                updated_keys.append(k)
                stct_updated[k] = v.clone()
                flag = True
            assert flag, f'Unexpected key from linear: {k}'
        elif module_name in fc_tensor_product_module_names:
            for ignore in ignores_in_fc:
                if '.'.join([module_name, ignore]) in k:
                    flag = True
                    break
            if not flag and k == '.'.join([module_name, 'fc_tensor_product.weight']):
                updated_keys.append(k)
                stct_updated[k] = v.clone()
                flag = True
            assert flag, f'Unexpected key from fc tensor product: {k}'
        else:
            # assert k in stct_updated
            updated_keys.append(k)
            stct_updated[k] = v.clone()

    return stct_updated


def model_from_checkpoint(
    checkpoint: Union[str, dict],
) -> Tuple[torch.nn.Module, Dict]:
    from .model_build import build_E3_equivariant_model

    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert isinstance(checkpoint, dict)
    elif isinstance(checkpoint, dict):
        pass
    else:
        raise ValueError('checkpoint must be either str or dict')

    stct_cp = checkpoint['model_state_dict']  # type: ignore
    config = _config_cp_routine(checkpoint.get('config'))

    model = build_E3_equivariant_model(config)

    stct_cp = compat.patch_state_dict_if_old(stct_cp, config, model)
    missing, not_used = model.load_state_dict(stct_cp, strict=False)
    if len(not_used) > 0:
        warnings.warn(f'Some keys are not used: {not_used}', UserWarning)

    assert len(missing) == 0, f'Missing keys: {missing}'

    return model, config


def model_from_checkpoint_with_backend(
    checkpoint: Union[str, dict],
    backend: str = 'e3nn',
) -> Tuple[torch.nn.Module, Dict]:
    from .model_build import build_E3_equivariant_model

    use_cue = backend.lower() in ['cue', 'cueq', 'cuequivariance']

    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert isinstance(checkpoint, dict)
    elif isinstance(checkpoint, dict):
        pass
    else:
        raise ValueError('checkpoint must be either str or dict')
    config_cp = _config_cp_routine(checkpoint.get('config'))

    # early return
    cue_cfg = config_cp.get(KEY.CUEQUIVARIANCE_CONFIG, {'use': False})
    cp_already_use_cue = cue_cfg.get('use', False)
    if use_cue == cp_already_use_cue:
        return model_from_checkpoint(checkpoint)

    print(f'convert the model to {backend}')

    # build empty [e3nn | cue] model
    config = copy.deepcopy(config_cp)
    config[KEY.CUEQUIVARIANCE_CONFIG] = {'use': use_cue}
    model = build_E3_equivariant_model(config)

    # patch model checkpoint's state dict
    stct_src = checkpoint['model_state_dict'].copy()  # type: ignore
    stct_src = compat.patch_state_dict_if_old(stct_src, config_cp, model)

    # get empty [e3nn | cue] state dicts
    stct_dst = model.state_dict()
    # patch state dicts
    stct_new = (
        _e3nn_to_cue(stct_src, stct_dst, config_cp)
        if use_cue
        # TODO: for now, no special routine needed when src dst changed
        else _e3nn_to_cue(stct_src, stct_dst, config_cp)
    )

    missing, not_used = model.load_state_dict(stct_new, strict=False)
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
