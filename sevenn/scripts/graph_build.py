from typing import List, Optional
import multiprocessing as mp

import tqdm

import torch
from torch_geometric.loader import DataLoader

from sevenn.train.dataload import atoms_to_graph
import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataset import AtomGraphDataset


def graph_build(atoms_list: List,
                cutoff: float,
                num_cores: int = 1,
                transfer_info: Optional[bool] = True) -> List[AtomGraphData]:
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
    pool = mp.Pool(num_cores)
    inputs = [(atoms, cutoff, transfer_info) for atoms in atoms_list]
    # this is not strictly correct because it updates for every input not output
    graph_list = pool.starmap(atoms_to_graph,
                              tqdm.tqdm(inputs, total=len(atoms_list)))
    pool.close()
    pool.join()
    return graph_list


def label_atoms_dict_to_dataset(data_dict, cutoff, ncores, metadata=None):
    """
    Script that create AtomGraphDataset from structure_list dict
    each data is correctly labeled by the key of the dict

    Args:
        structure_list_dict (Dict[str, List[Atoms]):
            return of sevenn.train.dataload.parse_structure_list
            or something similar, key is label of the atoms list
        cutoff (float): cutoff radius of graph
    Returns:
        AtomGraphDataset: dataset of graphs
    """
    label_list = []
    unrolled_atoms_list = []

    # unroll dict for efficient parallel processing
    for label, atoms_list in data_dict.items():
        for atoms in atoms_list:
            label_list.append(label)
            unrolled_atoms_list.append(atoms)

    graph_list = graph_build(unrolled_atoms_list, cutoff, ncores)
    for graph in graph_list:
        graph[KEY.USER_LABEL] = label_list.pop(0)

    dataset = AtomGraphDataset(graph_list, cutoff, metadata=metadata)
    dataset.group_by_key(KEY.USER_LABEL)  # in place operation
    return dataset
