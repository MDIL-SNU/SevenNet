import os
import sys
import pickle
from typing import List, Optional
from collections import Counter

import tqdm
import torch
import torch.multiprocessing as mp
from torch_geometric.loader import DataLoader

from ase import Atoms
import ase.io

from sevenn.train.dataload import atoms_to_graph
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataset import AtomGraphDataset
from sevenn.train.dataload import parse_structure_list
from sevenn.sevenn_logger import Logger
import sevenn._keys as KEY


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
    serial = num_cores == 1
    inputs = [(atoms, cutoff, transfer_info) for atoms in atoms_list]

    if not serial:
        pool = mp.Pool(num_cores)
        # this is not strictly correct because it updates for every input not output
        graph_list = pool.starmap(atoms_to_graph,
                                  tqdm.tqdm(inputs, total=len(atoms_list)))
        pool.close()
        pool.join()
    else:
        graph_list = [atoms_to_graph(*input_) for input_ in inputs]

    graph_list = [AtomGraphData.from_numpy_dict(g) for g in graph_list]

    return graph_list


def dataset_finalize(dataset, labels, metadata, out,
                     save_by_label=False, verbose=True):
    """
    Common finalization of dataset include logging and saving
    """
    natoms = dataset.get_natoms()
    species = dataset.get_species()
    metadata = {**metadata,
                "labels": labels,
                "natoms": natoms,
                "species": species}
    dataset.meta = metadata

    if save_by_label:
        out = os.path.dirname(out)
    elif os.path.isdir(out) and save_by_label is False:
        out = os.path.join(out, "graph_built.sevenn_data")
    elif out.endswith(".sevenn_data") is False:
        out = out + ".sevenn_data"

    if verbose:
        Logger().writeline("The metadata of the dataset is...")
        for k, v in metadata.items():
            Logger().format_k_v(k, v, write=True)
    dataset.save(out, save_by_label)
    Logger().writeline(f"dataset is saved to {out}")

    return dataset


def pkl_atoms_reader(fname):
    """
    Assume the content is plane list of ase.Atoms
    """
    with open(fname, 'rb') as f:
        atoms_list = pickle.load(f)
    if type(atoms_list) != list:
        raise TypeError("The content of the pkl is not list")
    if type(atoms_list[0]) != Atoms:
        raise TypeError("The content of the pkl is not list of ase.Atoms")
    return atoms_list


def ase_read_wrapper(index=":", format=None):
    def reader(fname):
        return ase.io.read(fname, index=index, format=format)
    return reader

def file_to_dataset(file: str,
                    cutoff: float,
                    cores = 1,
                    reader=None,
                    label: str = None,
                    transfer_info: bool =True,
                    ):
    # Assume key: atoms_list if return is dict and ignore label
    atoms = reader(file)

    if type(atoms) == list:
        if label is None:
            label = "unknown"
        atoms_dct = {label: atoms}
    elif type(atoms) == dict:
        atoms_dct = atoms
    else:
        raise TypeError("The return of reader is not list or dict")

    graph_dct = {}
    for label, atoms_list in atoms_dct.items():
        graph_list = graph_build(atoms_list, cutoff, cores,
                                 transfer_info=transfer_info)
        for graph in graph_list:
            graph[KEY.USER_LABEL] = label
        graph_dct[label] = graph_list
    db = AtomGraphDataset(graph_dct, cutoff)
    return db

def build_script(source: str,
                 cutoff: float,
                 num_cores: int,
                 label_by: str,
                 out: str,
                 save_by_label: bool,
                 fmt: str,
                 suffix: str,
                 transfer_info: bool,
                 metadata: dict = None):

    if fmt == "pkl" or fmt == "pickle":
        reader = pkl_atoms_reader
        metadata.update({"origin": "pkl"})
    elif fmt == "structure_list":
        reader = parse_structure_list
        metadata.update({"origin": "structure_list"})
    else:
        reader = ase_read_wrapper(index=":", format=fmt)
        metadata.update({"origin": f"by ase format of {fmt}"})

    dataset = AtomGraphDataset({}, cutoff)

    if os.path.isdir(source):
        Logger().writeline(f"Look for source dir: {source}")
        if suffix is not None:
            Logger().writeline(f"Try to read files if it ends with {suffix}")
        for file in os.listdir(source):
            label = file.split('.')[0] if label_by == "auto" else label_by
            file = os.path.join(source, file)
            if suffix is not None and file.endswith(suffix) is False:
                continue
            Logger().writeline(f"Read from file: {file}")
            Logger().timer_start("graph_build")
            db = file_to_dataset(file, cutoff, num_cores, reader, label, transfer_info)
            dataset.augment(db)
            Logger().timer_end("graph_build", f"{label} graph build time")
    elif os.path.isfile(source):
        file = source
        label = file.split('.')[0] if label_by == "auto" else label_by
        Logger().writeline(f"Read from file: {file}")
        Logger().timer_start("graph_build")
        db = file_to_dataset(file, cutoff, num_cores, reader, label, transfer_info)
        dataset.augment(db)
        Logger().timer_end("graph_build", f"{label} graph build time")
    else:
        raise ValueError(f"source {source} is not a file or dir")

    dataset_finalize(dataset, label, metadata, out, save_by_label)
