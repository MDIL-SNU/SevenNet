import copy
import os.path
from functools import partial
from itertools import chain, islice
from typing import Callable, List, Optional

import ase
import ase.io
import numpy as np
import torch.multiprocessing as mp
from ase.io.vasp_parsers.vasp_outcar_parsers import (
    Cell,
    DefaultParsersContainer,
    Energy,
    OutcarChunkParser,
    PositionsAndForces,
    Stress,
    outcarchunks,
)
from ase.neighborlist import primitive_neighbor_list
from ase.utils import string2index
from braceexpand import braceexpand
from tqdm import tqdm

import sevenn._keys as KEY
from sevenn._const import LossType
from sevenn.atom_graph_data import AtomGraphData

from .dataset import AtomGraphDataset


def _graph_build_matscipy(cutoff: float, pbc, cell, pos):
    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]

    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(pos)) + 1

    # Extend cell in non-periodic directions
    # For models with more than 5 layers,
    # the multiplicative constant needs to be increased.
    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]
    # it does not have self-interaction
    edge_src, edge_dst, edge_vec, shifts = neighbour_list(
        quantities='ijDS',
        pbc=pbc,
        cell=cell,
        positions=pos,
        cutoff=cutoff,
    )
    # dtype issue
    edge_src = edge_src.astype(np.int64)
    edge_dst = edge_dst.astype(np.int64)

    return edge_src, edge_dst, edge_vec, shifts


def _graph_build_ase(cutoff: float, pbc, cell, pos):
    # building neighbor list
    edge_src, edge_dst, edge_vec, shifts = primitive_neighbor_list(
        'ijDS', pbc, cell, pos, cutoff, self_interaction=True
    )

    is_zero_idx = np.all(edge_vec == 0, axis=1)
    is_self_idx = edge_src == edge_dst
    non_trivials = ~(is_zero_idx & is_self_idx)
    shifts = np.array(shifts[non_trivials])

    edge_vec = edge_vec[non_trivials]
    edge_src = edge_src[non_trivials]
    edge_dst = edge_dst[non_trivials]

    return edge_src, edge_dst, edge_vec, shifts


_graph_build_f = _graph_build_ase
try:
    from matscipy.neighbours import neighbour_list

    _graph_build_f = _graph_build_matscipy
except ImportError:
    pass


def _correct_scalar(v):
    if isinstance(v, np.ndarray):
        v = v.squeeze()
        assert v.ndim == 0, f'given {v} is not a scalar'
        return v
    elif isinstance(v, (int, float, np.integer, np.floating)):
        return np.array(v)
    else:
        assert False, f'{type(v)} is not expected'


def unlabeled_atoms_to_graph(atoms: ase.Atoms, cutoff: float):
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    pbc = atoms.get_pbc()

    edge_src, edge_dst, edge_vec, shifts = _graph_build_f(cutoff, pbc, cell, pos)

    edge_idx = np.array([edge_src, edge_dst])

    atomic_numbers = atoms.get_atomic_numbers()

    cell = np.array(cell)
    vol = _correct_scalar(atoms.cell.volume)
    if vol == 0:
        vol = np.array(np.finfo(float).eps)

    data = {
        KEY.NODE_FEATURE: atomic_numbers,
        KEY.ATOMIC_NUMBERS: atomic_numbers,
        KEY.POS: pos,
        KEY.EDGE_IDX: edge_idx,
        KEY.EDGE_VEC: edge_vec,
        KEY.CELL: cell,
        KEY.CELL_SHIFT: shifts,
        KEY.CELL_VOLUME: vol,
        KEY.NUM_ATOMS: _correct_scalar(len(atomic_numbers)),
    }
    data[KEY.INFO] = {}
    return data


def atoms_to_graph(
    atoms: ase.Atoms,
    cutoff: float,
    transfer_info: bool = True,
    y_from_calc: bool = False,
    allow_unlabeled: bool = False,
):
    """
    From ase atoms, return AtomGraphData as graph based on cutoff radius
    Except for energy, force and stress labels must be numpy array type
    as other cases are not tested.
    Returns 'np.nan' with consistent shape for unlabeled data
    (ex. stress of non-pbc system)

    Args:
        atoms (Atoms): ase atoms
        cutoff (float): cutoff radius
        transfer_info (bool): if True, transfer ".info" from atoms to graph,
                              defaults to True
        y_from_calc: if True, get ref values from calculator, defaults to False
    Returns:
        numpy dict that can be used to initialize AtomGraphData
        by AtomGraphData(**atoms_to_graph(atoms, cutoff))
        , for scalar, its shape is (), and types are np.ndarray
    Requires grad is handled by 'dataset' not here.
    """
    if not y_from_calc:
        y_energy = atoms.info['y_energy']
        y_force = atoms.arrays['y_force']
        y_stress = atoms.info.get('y_stress', np.full((6,), np.nan))
        if y_stress.shape == (3, 3):
            y_stress = np.array(
                [
                    y_stress[0][0],
                    y_stress[1][1],
                    y_stress[2][2],
                    y_stress[0][1],
                    y_stress[1][2],
                    y_stress[2][0],
                ]
            )
        else:
            y_stress = y_stress.squeeze()
    else:
        from_calc = _y_from_calc(atoms)
        y_energy = from_calc['energy']
        y_force = from_calc['force']
        y_stress = from_calc['stress']
    assert y_stress.shape == (6,), 'If you see this, please raise a issue'

    if not allow_unlabeled and (np.isnan(y_energy) or np.isnan(y_force).any()):
        raise ValueError('Unlabeled E or F found, set allow_unlabeled True')

    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    pbc = atoms.get_pbc()

    edge_src, edge_dst, edge_vec, shifts = _graph_build_f(cutoff, pbc, cell, pos)

    edge_idx = np.array([edge_src, edge_dst])
    atomic_numbers = atoms.get_atomic_numbers()

    cell = np.array(cell)
    vol = _correct_scalar(atoms.cell.volume)
    if vol == 0:
        vol = np.array(np.finfo(float).eps)

    data = {
        KEY.NODE_FEATURE: atomic_numbers,
        KEY.ATOMIC_NUMBERS: atomic_numbers,
        KEY.POS: pos,
        KEY.EDGE_IDX: edge_idx,
        KEY.EDGE_VEC: edge_vec,
        KEY.ENERGY: _correct_scalar(y_energy),
        KEY.FORCE: y_force,
        KEY.STRESS: y_stress.reshape(1, 6),  # to make batch have (n_node, 6)
        KEY.CELL: cell,
        KEY.CELL_SHIFT: shifts,
        KEY.CELL_VOLUME: vol,
        KEY.NUM_ATOMS: _correct_scalar(len(atomic_numbers)),
        KEY.PER_ATOM_ENERGY: _correct_scalar(y_energy / len(pos)),
    }

    if transfer_info and atoms.info is not None:
        info = copy.deepcopy(atoms.info)
        # save only metadata
        info.pop('y_energy', None)
        info.pop('y_force', None)
        info.pop('y_stress', None)
        data[KEY.INFO] = info
    else:
        data[KEY.INFO] = {}

    return data


def graph_build(
    atoms_list: List,
    cutoff: float,
    num_cores: int = 1,
    transfer_info: bool = True,
    y_from_calc: bool = False,
    allow_unlabeled: bool = False,
) -> List[AtomGraphData]:
    """
    parallel version of graph_build
    build graph from atoms_list and return list of AtomGraphData
    Args:
        atoms_list (List): list of ASE atoms
        cutoff (float): cutoff radius of graph
        num_cores (int): number of cores to use
        transfer_info (bool): if True, copy info from atoms to graph,
                              defaults to True
        y_from_calc (bool): Get reference y labels from calculator, defaults to False
    Returns:
        List[AtomGraphData]: list of AtomGraphData
    """
    serial = num_cores == 1
    inputs = [
        (atoms, cutoff, transfer_info, y_from_calc, allow_unlabeled)
        for atoms in atoms_list
    ]

    if not serial:
        pool = mp.Pool(num_cores)
        graph_list = pool.starmap(
            atoms_to_graph,
            tqdm(inputs, total=len(atoms_list), desc=f'graph_build ({num_cores})'),
        )
        pool.close()
        pool.join()
    else:
        graph_list = [
            atoms_to_graph(*input_)
            for input_ in tqdm(inputs, desc='graph_build (1)')
        ]

    graph_list = [AtomGraphData.from_numpy_dict(g) for g in graph_list]

    return graph_list


def _y_from_calc(atoms: ase.Atoms):
    ret = {
        'energy': np.nan,
        'force': np.full((len(atoms), 3), np.nan),
        'stress': np.full((6,), np.nan),
    }

    if atoms.calc is None:
        return ret

    try:
        ret['energy'] = atoms.get_potential_energy(force_consistent=True)
    except NotImplementedError:
        ret['energy'] = atoms.get_potential_energy()

    try:
        ret['force'] = atoms.get_forces(apply_constraint=False)
    except NotImplementedError:
        pass

    try:
        y_stress = -1 * atoms.get_stress()  # it ensures correct shape
        ret['stress'] = np.array(y_stress[[0, 1, 2, 5, 3, 4]])
    except RuntimeError:
        pass
    return ret


def _set_atoms_y(
    atoms_list: List[ase.Atoms],
    energy_key: Optional[str] = None,
    force_key: Optional[str] = None,
    stress_key: Optional[str] = None,
) -> List[ase.Atoms]:
    """
    Define how SevenNet reads ASE.atoms object for its y label
    If energy_key, force_key, or stress_key is given, the corresponding
    label is obtained from .info dict of Atoms object. These values should
    have eV, eV/Angstrom, and eV/Angstrom^3 for energy, force, and stress,
    respectively. (stress in Voigt notation)

    Args:
        atoms_list (list[ase.Atoms]): target atoms to set y_labels
        energy_key (str, optional): key to get energy. Defaults to None.
        force_key (str, optional): key to get force. Defaults to None.
        stress_key (str, optional): key to get stress. Defaults to None.

    Returns:
        list[ase.Atoms]: list of ase.Atoms

    Raises:
        RuntimeError: if ase atoms are somewhat imperfect

    Use free_energy: atoms.get_potential_energy(force_consistent=True)
    If it is not available, use atoms.get_potential_energy()
    If stress is available, initialize stress tensor
    Ignore constraints like selective dynamics
    """
    for atoms in atoms_list:
        from_calc = _y_from_calc(atoms)
        if energy_key is not None:
            atoms.info['y_energy'] = atoms.info.pop(energy_key)
        else:
            atoms.info['y_energy'] = from_calc['energy']

        if force_key is not None:
            atoms.arrays['y_force'] = atoms.arrays.pop(force_key)
        else:
            atoms.arrays['y_force'] = from_calc['force']

        if stress_key is not None:
            y_stress = -1 * atoms.info.pop(stress_key)
            atoms.info['y_stress'] = np.array(y_stress[[0, 1, 2, 5, 3, 4]])
        else:
            atoms.info['y_stress'] = from_calc['stress']

    return atoms_list


def ase_reader(
    filename: str,
    energy_key: Optional[str] = None,
    force_key: Optional[str] = None,
    stress_key: Optional[str] = None,
    index: str = ':',
    **kwargs,
) -> List[ase.Atoms]:
    """
    Wrapper of ase.io.read
    """
    atoms_list = ase.io.read(filename, index=index, **kwargs)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    return _set_atoms_y(atoms_list, energy_key, force_key, stress_key)


# Reader
def structure_list_reader(filename: str, format_outputs: Optional[str] = None):
    """
    Read from structure_list using braceexpand and ASE

    Args:
        fname : filename of structure_list

    Returns:
        dictionary of lists of ASE structures.
        key is title of training data (user-define)
    """
    parsers = DefaultParsersContainer(
        PositionsAndForces, Stress, Energy, Cell
    ).make_parsers()
    ocp = OutcarChunkParser(parsers=parsers)

    def parse_label(line):
        line = line.strip()
        if line.startswith('[') is False:
            return False
        elif line.endswith(']') is False:
            raise ValueError('wrong structure_list title format')
        return line[1:-1]

    def parse_fileline(line):
        line = line.strip().split()
        if len(line) == 1:
            line.append(':')
        elif len(line) != 2:
            raise ValueError('wrong structure_list format')
        return line[0], line[1]

    structure_list_file = open(filename, 'r')
    lines = structure_list_file.readlines()

    raw_str_dict = {}
    label = 'Default'
    for line in lines:
        if line.strip() == '':
            continue
        tmp_label = parse_label(line)
        if tmp_label:
            label = tmp_label
            raw_str_dict[label] = []
            continue
        elif label in raw_str_dict:
            files_expr, index_expr = parse_fileline(line)
            raw_str_dict[label].append((files_expr, index_expr))
        else:
            raise ValueError('wrong structure_list format')
    structure_list_file.close()

    structures_dict = {}
    info_dct = {'data_from': 'user_OUTCAR'}
    for title, file_lines in raw_str_dict.items():
        stct_lists = []
        for file_line in file_lines:
            files_expr, index_expr = file_line
            index = string2index(index_expr)
            for expanded_filename in list(braceexpand(files_expr)):
                f_stream = open(expanded_filename, 'r')
                # generator of all outcar ionic steps
                gen_all = outcarchunks(f_stream, ocp)
                try:  # TODO: index may not slice, it can be integer
                    it_atoms = islice(gen_all, index.start, index.stop, index.step)
                except ValueError:
                    # TODO: support
                    # negative index
                    raise ValueError('Negative index is not supported yet')

                info_dct_f = {
                    **info_dct,
                    'file': os.path.abspath(expanded_filename),
                }
                for idx, o in enumerate(it_atoms):
                    try:
                        it_atoms = islice(
                            gen_all, index.start, index.stop, index.step
                        )
                    except ValueError:
                        # TODO: support
                        # negative index
                        raise ValueError('Negative index is not supported yet')

                    info_dct_f = {
                        **info_dct,
                        'file': os.path.abspath(expanded_filename),
                    }
                    for idx, o in enumerate(it_atoms):
                        try:
                            istep = index.start + idx * index.step  # type: ignore
                            atoms = o.build()
                            atoms.info = {**info_dct_f, 'ionic_step': istep}.copy()
                        except TypeError:  # it is not slice of ionic steps
                            atoms = o.build()
                            atoms.info = info_dct_f.copy()
                        stct_lists.append(atoms)
                    f_stream.close()
                else:
                    stct_lists += ase.io.read(
                        expanded_filename,
                        index=index_expr,
                        parallel=False,
                    )
        structures_dict[title] = stct_lists
    return {k: _set_atoms_y(v) for k, v in structures_dict.items()}


def dict_reader(data_dict: dict):
    data_dict_cp = copy.deepcopy(data_dict)

    ret = []
    file_list = data_dict_cp.pop('file_list', None)
    if file_list is None:
        raise KeyError('file_list is not found')

    data_weight_default = {
        'energy': 1.0,
        'force': 1.0,
        'stress': 1.0,
    }
    data_weight = data_weight_default.copy()
    data_weight.update(data_dict_cp.pop(KEY.DATA_WEIGHT, {}))

    for file_dct in file_list:
        ftype = file_dct.pop('data_format', 'ase')
        files = list(braceexpand(file_dct.pop('file')))
        if ftype == 'ase':
            ret.extend(chain(*[ase_reader(f, **file_dct) for f in files]))
        elif ftype == 'graph':
            continue
        else:
            raise ValueError(f'{ftype} yet')

    for atoms in ret:
        atoms.info.update(data_dict_cp)
        atoms.info.update({KEY.DATA_WEIGHT: data_weight})
    return _set_atoms_y(ret)


def match_reader(reader_name: str, **kwargs):
    reader = None
    metadata = {}
    if reader_name == 'structure_list':
        reader = partial(structure_list_reader, **kwargs)
        metadata.update({'origin': 'structure_list'})
    else:
        reader = partial(ase_reader, **kwargs)
        metadata.update({'origin': 'ase_reader'})
    return reader, metadata


def file_to_dataset(
    file: str,
    cutoff: float,
    cores: int = 1,
    reader: Callable = ase_reader,
    label: Optional[str] = None,
    transfer_info: bool = True,
    use_weight: bool = False,
    use_modality: bool = False,
):
    """
    Deprecated
    Read file by reader > get list of atoms or dict of atoms
    """

    # expect label: atoms_list dct or atoms or list of atoms
    atoms = reader(file)

    if type(atoms) is list:
        if label is None:
            label = KEY.LABEL_NONE
        atoms_dct = {label: atoms}
    elif isinstance(atoms, ase.Atoms):
        if label is None:
            label = KEY.LABEL_NONE
        atoms_dct = {label: [atoms]}
    elif isinstance(atoms, dict):
        atoms_dct = atoms
    else:
        raise TypeError('The return of reader is not list or dict')

    graph_dct = {}
    for label, atoms_list in atoms_dct.items():
        graph_list = graph_build(
            atoms_list=atoms_list,
            cutoff=cutoff,
            num_cores=cores,
            transfer_info=transfer_info,
            y_from_calc=False,
        )

        label_info = label.split(':')
        for graph in graph_list:
            graph[KEY.USER_LABEL] = label_info[0].strip()
            if use_weight:
                find_weight = False
                for info in label_info[1:]:
                    if 'w=' in info.lower():
                        weights = info.split('=')[1]
                        try:
                            if ',' in weights:
                                weight_list = list(map(float, weights.split(',')))
                            else:
                                weight_list = [float(weights)] * 3
                            weight_dict = {}
                            for idx, loss_type in enumerate(LossType):
                                weight_dict[loss_type.value] = (
                                    weight_list[idx] if idx < len(weight_list) else 1
                                )
                            graph[KEY.DATA_WEIGHT] = weight_dict
                            find_weight = True
                            break
                        except:
                            raise ValueError(
                                'Weight must be a real number, but'
                                f' {weights} is given for {label}'
                            )
                if not find_weight:
                    weight_dict = {}
                    for loss_type in LossType:
                        weight_dict[loss_type.value] = 1
                    graph[KEY.DATA_WEIGHT] = weight_dict
            if use_modality:
                find_modality = False
                for info in label_info[1:]:
                    if 'm=' in info.lower():
                        graph[KEY.DATA_MODALITY] = (info.split('=')[1]).strip()
                        find_modality = True
                        break
                if not find_modality:
                    raise ValueError(f'Modality not given for {label}')

        graph_dct[label_info[0].strip()] = graph_list
    db = AtomGraphDataset(graph_dct, cutoff)
    return db
