import numpy as np
import torch

import sevenn._keys as KEY
import sevenn._const


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
