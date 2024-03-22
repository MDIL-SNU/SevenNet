import numpy as np
import torch
from torch_geometric.loader import DataLoader

import sevenn.train.dataload
import sevenn._keys as KEY


class AverageNumber:
    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, values: torch.Tensor):
        self._sum += values.sum().item()
        self._count += values.numel()

    def _ddp_reduce(self, device):
        _sum = torch.tensor(self._sum, device=device)
        _count = torch.tensor(self._count, device=device)
        torch.distributed.all_reduce(_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(_count, op=torch.distributed.ReduceOp.SUM)
        self._sum = _sum.item()
        self._count = _count.item()

    def get(self):
        if self._count == 0:
            return np.nan
        return self._sum / self._count


def to_atom_graph_list(atom_graph_batch):
    """
    torch_geometric batched data to seperate list
    original to_data_list() by PyG is not enough since
    it doesn't handle inferred tensors
    """
    is_stress = KEY.PRED_STRESS in atom_graph_batch

    data_list = atom_graph_batch.to_data_list()

    indices = atom_graph_batch[KEY.NUM_ATOMS].tolist()

    atomic_energy_list = torch.split(
        atom_graph_batch[KEY.ATOMIC_ENERGY], indices
    )
    inferred_total_energy_list = torch.unbind(
        atom_graph_batch[KEY.PRED_TOTAL_ENERGY]
    )
    inferred_force_list = torch.split(
        atom_graph_batch[KEY.PRED_FORCE], indices
    )

    if is_stress:
        inferred_stress_list = torch.unbind(atom_graph_batch[KEY.PRED_STRESS])

    for i, data in enumerate(data_list):
        data[KEY.ATOMIC_ENERGY] = atomic_energy_list[i]
        data[KEY.PRED_TOTAL_ENERGY] = inferred_total_energy_list[i]
        data[KEY.PRED_FORCE] = inferred_force_list[i]
        # To fit with KEY.STRESS (ref) format
        if is_stress:
            data[KEY.PRED_STRESS] = torch.unsqueeze(inferred_stress_list[i], 0)
    return data_list


def postprocess_output_with_label(output, loss_types):
    from sevenn._const import LossType

    results = postprocess_output(output, loss_types)
    batched_label = output[KEY.USER_LABEL]
    label_set = set(batched_label)
    labeled = {k: {} for k in label_set}
    for loss_type, (pred, ref, vdim) in results.items():
        i_from = 0
        i_to = None
        # if loss_type in [LossType.ENERGY, LossType.STRESS]:
        #    i_to = vdim
        for idx, label in enumerate(batched_label):
            if loss_type is LossType.FORCE:
                i_to = i_from + vdim * results[KEY.NUM_ATOMS][idx].item()
            else:
                i_to = i_from + vdim
            labeled[label][loss_type] = (pred[i_from:i_to], ref[i_from:i_to])
            i_from = i_to
    return labeled


def postprocess_output(output, loss_types):
    from sevenn._const import LossType

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


def squared_error(pred, ref, vdim):
    MSE = torch.nn.MSELoss(reduction='none')
    return torch.reshape(MSE(pred, ref), (-1, vdim)).sum(dim=1)


def onehot_to_chem(one_hot_indicies, type_map):
    from ase.data import chemical_symbols

    type_map_rev = {v: k for k, v in type_map.items()}
    return [chemical_symbols[type_map_rev[x]] for x in one_hot_indicies]


def model_from_checkpoint(checkpoint):
    from sevenn._const import (
        DEFAULT_DATA_CONFIG,
        DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG,
        DEFAULT_TRAINING_CONFIG,
    )
    from sevenn.model_build import build_E3_equivariant_model

    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location='cpu')
    elif isinstance(checkpoint, dict):
        pass
    else:
        raise ValueError('checkpoint must be either str or dict')

    defaults = {
        **DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG,
        **DEFAULT_DATA_CONFIG,
        **DEFAULT_TRAINING_CONFIG,
    }

    model_state_dict = checkpoint['model_state_dict']
    config = checkpoint['config']

    for k, v in defaults.items():
        if k not in config:
            print(f'Warning: {k} not in config, using default value {v}')
            config[k] = v

    # expect only non-tensor values in config, if exists, move to cpu
    # This can be happen if config has torch tensor as value (shift, scale)
    # TODO: putting only non-tensors at first place is better
    for k, v in config.items():
        if isinstance(v, torch.Tensor):
            config[k] = v.cpu()

    model = build_E3_equivariant_model(config)
    model.load_state_dict(model_state_dict, strict=False)

    return model, config


def unlabeled_atoms_to_input(atoms, cutoff):
    from sevenn.atom_graph_data import AtomGraphData
    atom_graph = AtomGraphData.from_numpy_dict(
        sevenn.train.dataload.unlabeled_atoms_to_graph(atoms, cutoff)
    )
    atom_graph[KEY.POS].requires_grad_(True)
    atom_graph[KEY.BATCH] = torch.zeros([0])
    return atom_graph


def chemical_species_preprocess(input_chem):
    from ase.data import atomic_numbers

    from sevenn.nn.node_embedding import get_type_mapper_from_specie

    config = {}
    chemical_specie = sorted([x.strip() for x in input_chem])
    config[KEY.CHEMICAL_SPECIES] = chemical_specie
    config[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER] = [
        atomic_numbers[x] for x in chemical_specie
    ]
    config[KEY.NUM_SPECIES] = len(chemical_specie)
    config[KEY.TYPE_MAP] = get_type_mapper_from_specie(chemical_specie)
    # print(config[KEY.TYPE_MAP])
    # print(config[KEY.NUM_SPECIES])  # why
    # print(config[KEY.CHEMICAL_SPECIES])  # we need
    # print(config[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER])  # all of this?
    return config


def dtype_correct(v, float_dtype=torch.float32, int_dtype=torch.int64):
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
        else:
            # non-number
            return v


def load_model_from_checkpoint(checkpoint):
    """
    Deprecated
    """
    from sevenn._const import (
        DEFAULT_DATA_CONFIG,
        DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG,
        DEFAULT_TRAINING_CONFIG,
    )
    from sevenn.model_build import build_E3_equivariant_model

    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location='cpu')
    elif isinstance(checkpoint, dict):
        pass
    else:
        raise ValueError('checkpoint must be either str or dict')

    defaults = {
        **DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG,
        **DEFAULT_DATA_CONFIG,
        **DEFAULT_TRAINING_CONFIG,
    }

    model_state_dict = checkpoint['model_state_dict']
    config = checkpoint['config']

    for k, v in defaults.items():
        if k not in config:
            print(f'Warning: {k} not in config, using default value {v}')
            config[k] = v

    # expect only non-tensor values in config, if exists, move to cpu
    # This can be happen if config has torch tensor as value (shift, scale)
    # TODO: putting only non-tensors at first place is better
    for k, v in config.items():
        if isinstance(v, torch.Tensor):
            config[k] = v.cpu()

    model = build_E3_equivariant_model(config)

    model.load_state_dict(model_state_dict, strict=False)

    return model



