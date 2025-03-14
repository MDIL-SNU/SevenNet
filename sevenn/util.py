import os
import pathlib
import shutil
import tempfile
import urllib.error
import urllib.request
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn
from e3nn.o3 import FullTensorProduct, Irreps

import sevenn._const as CONST
import sevenn._keys as KEY
from sevenn.checkpoint import SevenNetCheckpoint


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


def model_from_checkpoint(
    checkpoint: str,
) -> Tuple[torch.nn.Module, Dict]:
    cp = load_checkpoint(checkpoint)
    model = cp.build_model()

    return model, cp.config


def model_from_checkpoint_with_backend(
    checkpoint: str,
    backend: str = 'e3nn',
) -> Tuple[torch.nn.Module, Dict]:
    cp = load_checkpoint(checkpoint)
    model = cp.build_model(backend)

    return model, cp.config


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
    for mul, (l, p) in irreps_out:  # noqa
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
    name = name.lower()
    heads = ['sevennet', '7net']
    checkpoint_path = None
    if (  # TODO: regex
        name in [f'{n}-0_11july2024' for n in heads]
        or name in [f'{n}-0_11jul2024' for n in heads]
        or name in ['sevennet-0', '7net-0']
    ):
        checkpoint_path = CONST.SEVENNET_0_11Jul2024
    elif name in [f'{n}-0_22may2024' for n in heads]:
        checkpoint_path = CONST.SEVENNET_0_22May2024
    elif name in [f'{n}-l3i5' for n in heads]:
        checkpoint_path = CONST.SEVENNET_l3i5
    elif name in [f'{n}-mf-0' for n in heads]:
        checkpoint_path = CONST.SEVENNET_MF_0
    elif name in [f'{n}-mf-ompa' for n in heads]:
        checkpoint_path = CONST.SEVENNET_MF_OMPA
    elif name in [f'{n}-omat' for n in heads]:
        checkpoint_path = CONST.SEVENNET_OMAT
    else:
        raise ValueError('Not a valid potential')

    checkpoint_path = check_and_download_checkpoint(checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint: Union[pathlib.Path, str]):
    if os.path.isfile(checkpoint):
        checkpoint_path = checkpoint
    else:
        try:
            checkpoint_path = pretrained_name_to_path(str(checkpoint))
        except ValueError:
            raise ValueError(
                f'Given {checkpoint} is not exists and not a pre-trained name'
            )
    return SevenNetCheckpoint(checkpoint_path)


def check_and_download_checkpoint(checkpoint_path: str):
    # check if the file exists
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    model_name = os.path.basename(os.path.dirname(checkpoint_path))
    home_save_path = os.path.expanduser(f'~/.cache/{model_name}')
    checkpoint_path2 = os.path.join(
        home_save_path, os.path.basename(checkpoint_path)
    )
    if os.path.isfile(checkpoint_path2):
        return checkpoint_path2

    # download the file
    download_url = CONST.SEVENNET_DOWNLOAD_LINK.get(checkpoint_path)
    print(f'Downloading {model_name} checkpoint', flush=True)
    try:
        save_path = os.path.dirname(checkpoint_path)
        os.makedirs(save_path, exist_ok=True)
    except Exception:
        try:
            save_path = home_save_path
            os.makedirs(save_path, exist_ok=True)
            checkpoint_path = checkpoint_path2
        except ValueError:
            raise ValueError(
                f'Failed to create save path for {model_name} checkpoint'
            )
    print(f'Saving to {save_path}', flush=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=save_path) as temp_file:
        temp_path = temp_file.name
        try:
            _, http_msg = urllib.request.urlretrieve(download_url, temp_path)
            print(f'Download complete to {save_path}', flush=True)
            shutil.move(temp_path, checkpoint_path)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            OSError,
            shutil.Error,
            KeyboardInterrupt,
        ) as e:
            raise ValueError(f'Failed to download {model_name} checkpoint: {e}')
        finally:
            if os.path.isfile(temp_path):
                os.remove(temp_path)
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
