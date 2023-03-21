from typing import List, Optional
import pickle

import numpy as np
from ase import io, units, Atoms
from ase.neighborlist import primitive_neighbor_list
from braceexpand import braceexpand


def ASE_atoms_to_data(atoms, cutoff: float):
    """ very primitive function to extract properties from atoms
    Args:
        atoms : 'atoms' object from ASE
        cutoff : float
    Returns:
        data represents one 'graph'

        E : total energy of graph
        F : forces of each atom (n, 3)

        pos : (N(atoms), 3)
        edge_src : index of atoms for edge src (N(edge))
        edge_dst : index of atoms for edge dst (N(edge))
        edge_vec : vector representing edge (N(edge), 3)
        atomic_numbers : list of atomic number by index (n)
        chemical_symbol: list of chemical symbol by index (n)

        * this is full neighborlist
    """

    # 'y' of data
    E = atoms.get_potential_energy(force_consistent=True)
    F = atoms.get_forces()
    # xx yy zz xy yz zx order
    S = -1 * atoms.get_stress() / units.GPa * 10
    S = S[[0, 1, 2, 5, 3, 4]]

    cutoffs = np.full(len(atoms), cutoff)
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    edge_src, edge_dst, edge_vec, shifts = primitive_neighbor_list(
        "ijDS", atoms.get_pbc(), cell, pos, cutoff, self_interaction=True
    )

    # trivial : src == dst and not crossing pbc
    # nontirivial_self_interatction : src == dst but cross pbc
    # below is for eliminate trivial ones

    is_zero_idx = np.all(edge_vec == 0, axis=1)
    is_self_idx = edge_src == edge_dst
    non_trivials = ~(is_zero_idx & is_self_idx)

    # 'x' of data
    edge_src = edge_src[non_trivials]
    edge_dst = edge_dst[non_trivials]
    edge_vec = edge_vec[non_trivials]
    shift = shifts[non_trivials]
    atomic_numbers = atoms.get_atomic_numbers()
    chemical_symbol = atoms.get_chemical_symbols()
    edge_idx = np.array([edge_src, edge_dst])

    return atomic_numbers, chemical_symbol, edge_idx, edge_vec, shift, pos, cell, E, F


def parse_structure_list(filename: str, format_outputs='vasp-out'):
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
        if len(line) != 2:
            raise ValueError('wrong structure_list fileline format')
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
        else:
            files_expr, index_expr = parse_fileline(line)
            raw_str_dict[label].append((files_expr, index_expr))
    structure_list_file.close()

    structures_dict = {}
    for title, file_lines in raw_str_dict.items():
        stct_lists = []
        for file_line in file_lines:
            files_expr, index_expr = file_line
            for expanded_filename in list(braceexpand(files_expr)):
                stct_lists += io.read(expanded_filename, index=index_expr,
                                      format=format_outputs, parallel=False)
        structures_dict[title] = stct_lists
    return structures_dict
