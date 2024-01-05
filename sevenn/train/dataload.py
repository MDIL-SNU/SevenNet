import os.path
import pickle
from functools import partial
from itertools import islice
from typing import Dict, List, Optional

import ase
import ase.io
import numpy as np
import torch.multiprocessing as mp
import tqdm
from ase.io.utils import string2index
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
from braceexpand import braceexpand

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataset import AtomGraphDataset


def atoms_to_graph(
    atoms: ase.Atoms, cutoff: float, transfer_info: bool = True
):
    """
    From ase atoms, return AtomGraphData as graph based on cutoff radius
    Args:
        atoms (Atoms): ase atoms
        cutoff (float): cutoff radius
        transfer_info (bool): if True, transfer ".info" from atoms to graph
    Returns:
        numpy dict that can be used to initialize AtomGraphData
        by AtomGraphData(**atoms_to_graph(atoms, cutoff))
    Raises:
        RuntimeError: if ase atoms are somewhat imperfect

    Use free_energy by default (atoms.get_potential_energy(force_consistent=True))
    If it is not available, use energy (atoms.get_potential_energy())
    If stress is available, initialize stress tensor
    Ignore constraints like selective dynamics

    Requires grad is handled by 'dataset' not here.
    """

    try:
        y_energy = atoms.get_potential_energy(force_consistent=True)
    except NotImplementedError as e:
        y_energy = atoms.get_potential_energy()
    y_force = atoms.get_forces(apply_constraint=False)
    try:
        # xx yy zz xy yz zx order
        # We expect this is eV/A^3 unit
        # (ASE automatically converts vasp kB to eV/A^3)
        # So we restore it
        y_stress = -1 * atoms.get_stress()
        y_stress = np.array([y_stress[[0, 1, 2, 5, 3, 4]]])
    except RuntimeError as e:
        y_stress = None

    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())

    # building neighbor list
    edge_src, edge_dst, edge_vec, shifts = primitive_neighbor_list(
        'ijDS', atoms.get_pbc(), cell, pos, cutoff, self_interaction=True
    )

    # remove redundant edges (self interaction) but saves self interaction cross PBC
    is_zero_idx = np.all(edge_vec == 0, axis=1)
    is_self_idx = edge_src == edge_dst
    non_trivials = ~(is_zero_idx & is_self_idx)
    cell_shift = np.array(shifts[non_trivials])

    edge_vec = edge_vec[non_trivials]
    edge_src = edge_src[non_trivials]
    edge_dst = edge_dst[non_trivials]
    edge_idx = np.array([edge_src, edge_dst])

    atomic_numbers = atoms.get_atomic_numbers()

    cell = np.array(cell)

    data = {
        KEY.NODE_FEATURE: atomic_numbers,
        KEY.ATOMIC_NUMBERS: atomic_numbers,
        KEY.POS: pos,
        KEY.EDGE_IDX: edge_idx,
        KEY.EDGE_VEC: edge_vec,
        KEY.ENERGY: y_energy,
        KEY.FORCE: y_force,
        KEY.STRESS: y_stress,
        KEY.CELL: cell,
        KEY.CELL_SHIFT: cell_shift,
        KEY.CELL_VOLUME: np.einsum(
            'i,i', cell[0, :], np.cross(cell[1, :], cell[2, :])
        ),
        KEY.NUM_ATOMS: len(atomic_numbers),
        KEY.PER_ATOM_ENERGY: y_energy / len(pos),
    }
    # data.num_nodes = data[KEY.NUM_ATOMS]  # is it really necessary?
    if transfer_info and atoms.info is not None:
        data[KEY.INFO] = atoms.info
    else:
        data[KEY.INFO] = {}

    return data


def graph_build(
    atoms_list: List,
    cutoff: float,
    num_cores: int = 1,
    transfer_info: Optional[bool] = True,
) -> List[AtomGraphData]:
    """
    parallel version of graph_build
    build graph from atoms_list and return list of AtomGraphData
    Args:
        atoms_list (List): list of ASE atoms
        cutoff (float): cutoff radius of graph
        num_cores (int, Optional): number of cores to use
        transfer_info (bool, Optional): if True, copy info from atoms to graph
    Returns:
        List[AtomGraphData]: list of AtomGraphData
    """
    serial = num_cores == 1
    inputs = [(atoms, cutoff, transfer_info) for atoms in atoms_list]

    if not serial:
        pool = mp.Pool(num_cores)
        # this is not strictly correct because it updates for every input not output
        graph_list = pool.starmap(
            atoms_to_graph, tqdm.tqdm(inputs, total=len(atoms_list))
        )
        pool.close()
        pool.join()
    else:
        graph_list = [atoms_to_graph(*input_) for input_ in inputs]

    graph_list = [AtomGraphData.from_numpy_dict(g) for g in graph_list]

    return graph_list


def ase_reader(fname, **kwargs):
    index = kwargs.pop('index', None)
    if index is None:
        index = ':'  # new default for ours
    return ase.io.read(fname, index=index, **kwargs)


def pkl_atoms_reader(fname):
    """
    Assume the content is plane list of ase.Atoms
    """
    with open(fname, 'rb') as f:
        atoms_list = pickle.load(f)
    if type(atoms_list) != list:
        raise TypeError('The content of the pkl is not list')
    if type(atoms_list[0]) != ase.Atoms:
        raise TypeError('The content of the pkl is not list of ase.Atoms')
    return atoms_list


# Reader
def structure_list_reader(filename: str, format_outputs='vasp-out'):
    parsers = DefaultParsersContainer(
        PositionsAndForces, Stress, Energy, Cell
    ).make_parsers()
    ocp = OutcarChunkParser(parsers=parsers)
    """
    Read from structure_list using braceexpand and ASE

    Args:
        fname : filename of structure_list

    Returns:
        dictionary of lists of ASE structures.
        key is title of training data (user-define)
    """

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
    for i, line in enumerate(lines):
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
                """
                stct_lists += io.read(expanded_filename, index=index_expr,
                                      format=format_outputs, parallel=False)
                """
                # generator of all outcar ionic steps
                gen_all = outcarchunks(f_stream, ocp)
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
                        istep = index.start + idx * index.step
                        atoms = o.build()
                        atoms.info = {**info_dct_f, 'ionic_step': istep}
                    except TypeError:  # it is not slice of ionic steps
                        atoms = o.build()
                        atoms.info = info_dct_f
                    stct_lists.append(atoms)
                f_stream.close()
        structures_dict[title] = stct_lists
    return structures_dict


def match_reader(reader_name: str, **kwargs):
    reader = None
    metadata = {}
    if reader_name == 'pkl' or reader_name == 'pickle':
        reader = partial(pkl_atoms_reader, **kwargs)
        metadata.update({'origin': 'atoms_pkl'})
    elif reader_name == 'structure_list':
        reader = partial(structure_list_reader, **kwargs)
        metadata.update({'origin': 'structure_list'})
    else:
        reader = partial(ase_reader, **kwargs)
        metadata.update({'origin': f'ase_reader'})
    return reader, metadata


def file_to_dataset(
    file: str,
    cutoff: float,
    cores=1,
    reader=None,
    label: str = None,
    transfer_info: bool = True,
):
    """
    Read file by reader > get list of atoms or dict of atoms
    """

    # expect label: atoms_list dct or atoms or list of atoms
    atoms = reader(file)

    if type(atoms) == list:
        if label is None:
            label = KEY.LABEL_NONE
        atoms_dct = {label: atoms}
    elif type(atoms) == ase.Atoms:
        if label is None:
            label = KEY.LABEL_NONE
        atoms_dct = {label: [atoms]}
    elif type(atoms) == dict:
        atoms_dct = atoms
    else:
        raise TypeError('The return of reader is not list or dict')

    graph_dct = {}
    for label, atoms_list in atoms_dct.items():
        graph_list = graph_build(
            atoms_list, cutoff, cores, transfer_info=transfer_info
        )
        for graph in graph_list:
            graph[KEY.USER_LABEL] = label
        graph_dct[label] = graph_list
    db = AtomGraphDataset(graph_dct, cutoff)
    return db
