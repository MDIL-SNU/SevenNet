import numpy as np
import torch

import sevenn._keys as KEY
from sevenn._const import LossType


def postprocess_output(output, loss_type, criterion=None,
                       energy_weight=1.0, force_weight=1.0, stress_weight=1.0):
    """
    Do dirty things about output of model (like unit conversion, normalization)
    From the output of model (maybe batched), calculated NOT averaged mse
    and loss if criterion is given.

    Args:
        output: output of model (maybe batched) (AtomGraphData)
        loss_type: LossType
        criterion: loss function (from torch)
        energy_weight: weight of energy loss
        force_weight: weight of force loss
        stress_weight: weight of stress loss

    Returns:
        pred: predicted value, torch Tensor (maybe batched)
        ref: reference value, torch Tensro (maybe batched)
        mse: squared error (not averaged therefore, strictly, it is not mse)
        loss: weighted backwardable loss, None if criterion is not given
    """
    TO_KB = 1602.1766208  # eV/A^3 to kbar

    # from the output of model, calculate mse, loss
    # since they're all LossType wise
    MSE = torch.nn.MSELoss(reduction='none')
    def get_vector_component_and_mse(pred_V: torch.Tensor,
                                     ref_V: torch.Tensor, vdim: int):
        pred_V_component = torch.reshape(pred_V, (-1,))
        ref_V_component = torch.reshape(ref_V, (-1,))
        mse = MSE(pred_V_component, ref_V_component)
        mse = torch.reshape(mse, (-1, vdim))
        mse = mse.sum(dim=1)
        return pred_V_component, ref_V_component, mse

    loss_weight = 0
    if loss_type is LossType.ENERGY:
        num_atoms = output[KEY.NUM_ATOMS]
        pred = torch.squeeze(output[KEY.PRED_TOTAL_ENERGY], -1) / num_atoms
        ref = output[KEY.ENERGY] / num_atoms
        mse = MSE(pred, ref)
        loss_weight = energy_weight  # energy loss weight is 1 (it is reference)
    elif loss_type is LossType.FORCE:
        pred_raw = output[KEY.PRED_FORCE]
        ref_raw = output[KEY.FORCE]
        pred, ref, mse = \
            get_vector_component_and_mse(pred_raw, ref_raw, 3)
        loss_weight = force_weight / 3  # normalize by # of comp
    elif loss_type is LossType.STRESS:
        # calculate stress loss based on kB unit (was eV/A^3)
        pred_raw = output[KEY.PRED_STRESS] * TO_KB
        ref_raw = output[KEY.STRESS] * TO_KB
        pred, ref, mse = \
            get_vector_component_and_mse(pred_raw, ref_raw, 6)
        loss_weight = stress_weight / 6  # normalize by # of comp
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')

    loss = None
    if criterion is not None:
        if isinstance(criterion, torch.nn.MSELoss):
            # mse for L2 loss is already calculated
            loss = torch.mean(mse) * loss_weight
        else:
            loss = criterion(pred, ref) * loss_weight

    return pred, ref, mse, loss


def onehot_to_chem(one_hot_indicies, type_map):
    from ase.data import chemical_symbols
    type_map_rev = {v: k for k, v in type_map.items()}
    return [chemical_symbols[type_map_rev[x]] for x in one_hot_indicies]


def load_model_from_checkpoint(checkpoint):
    from sevenn.model_build import build_E3_equivariant_model
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint)
    elif isinstance(checkpoint, dict):
        pass
    else:
        raise ValueError("checkpoint must be either str or dict")

    #mse_hist = checkpoint["loss"]
    model_state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]

    model = build_E3_equivariant_model(config)
    model.load_state_dict(model_state_dict, strict=False)

    return model


def chemical_species_preprocess(input_chem):
    from ase.data import atomic_numbers
    from sevenn.nn.node_embedding import get_type_mapper_from_specie
    config = {}
    chemical_specie = sorted([x.strip() for x in input_chem])
    config[KEY.CHEMICAL_SPECIES] = chemical_specie
    config[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER] = \
        [atomic_numbers[x] for x in chemical_specie]
    config[KEY.NUM_SPECIES] = len(chemical_specie)
    config[KEY.TYPE_MAP] = get_type_mapper_from_specie(chemical_specie)
    #print(config[KEY.TYPE_MAP])
    #print(config[KEY.NUM_SPECIES])  # why
    #print(config[KEY.CHEMICAL_SPECIES])  # we need
    #print(config[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER])  # all of this?
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
