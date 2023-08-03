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

from sevenn.train.dataload import atoms_to_graph
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataset import AtomGraphDataset
from sevenn.train.dataload import parse_structure_list
from sevenn.sevenn_logger import Logger
import sevenn._keys as KEY


def graph_build(atoms_list: List,
                cutoff: float,
                num_cores: int = 1,
                serial: bool = False,
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


def dataset_finalize(dataset, labels, metadata, out: str = None, verbose=True):
    """
    Common finalization of dataset include logging and saving
    if out is given
    """
    natoms = dataset.get_natoms()
    species = dataset.get_species()
    metadata = {**metadata,
                "labels": labels,
                "natoms": natoms,
                "species": species}
    dataset.meta = metadata

    if verbose:
        Logger().writeline("The metadata of the dataset is...")
        for k, v in metadata.items():
            Logger().format_k_v(k, v, write=True)
        if out is not None:
            Logger().writeline(f"dataset is saved to {out}")

    if out is not None:
        dataset.save(out)
    return dataset


def default_pkl_reader(fname):
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


# interface for reader??
def pkl_to_dataset(pkl_file: str,
                   cutoff: float,
                   label: str,
                   transfer_info=True,
                   reader=default_pkl_reader):
    atoms_list = reader(pkl_file)
    graph_list = graph_build(atoms_list, cutoff, serial=True,
                             transfer_info=transfer_info)
    for graph in graph_list:
        graph[KEY.USER_LABEL] = label
    db = AtomGraphDataset(graph_list, cutoff)
    db.group_by_key(KEY.USER_LABEL)
    return db


# method for parallel processing
def process_file(file, root, cutoff, label, transfer_info, metadata, path_to):
    src = os.path.join(root, file)
    save_to = os.path.join(path_to, f"{os.path.splitext(file)[0]}.sevenn_data")
    dataset = pkl_to_dataset(src, cutoff, label, transfer_info)
    metadata["source"] = src
    dataset_finalize(dataset, label, metadata, save_to, verbose=False)
    # the dataset necessariliy has only one label
    return Counter(dataset.get_natoms()[label]), dataset.get_species()


def from_root_dir(source: str,
                  cutoff: float,
                  num_cores: int,
                  label_by: str,
                  out: str,
                  suffix: str,
                  transfer_info: bool,
                  metadata: dict = None):

    Logger().writeline(f"Walk from root dir: {source}")
    Logger().writeline(f"Try to read files if it ends with {suffix}")
    Logger().writeline("The file should be python pkl containing list of ase.Atoms")

    if os.path.isdir(out) is False:
        raise ValueError(f"The output dir {out} does not exist")
    out = os.path.abspath(out)

    total_natoms = {"total": Counter()}
    total_species = {}
    Logger().timer_start("total_graph_build")
    abs_src_len = len(os.path.abspath(source))
    for root, dirs, files in os.walk(source):
        if len(files) == 0:
            continue
        label = root.strip().split('/')[-1] if label_by == "auto" else label_by
        if label not in total_natoms:
            total_natoms[label] = Counter()
            total_species[label] = set()

        path_to = os.path.join(out, os.path.abspath(root)[abs_src_len + 1:])
        if os.path.isdir(path_to) is False:
            os.makedirs(path_to, exist_ok=True)

        files_to_read = [file for file in files if file.endswith(suffix)]
        process_file_inps = [(file, root, cutoff, label,
                              transfer_info, metadata, path_to)
                             for file in files_to_read]
        Logger().writeline(f"{len(files_to_read)} files to read in {root}")
        Logger().writeline(f"Building graphs with label {label}")
        Logger().timer_start("graph_build")
        with mp.Pool(processes=num_cores) as pool:
            results = pool.starmap(process_file, process_file_inps)
        Logger().timer_end("graph_build", f"{label} graph build time")
        Logger().writeline(f"data is saved to {path_to}")
        Logger().writeline("")
        for natoms, species in results:
            total_natoms[label] += natoms
            total_species[label].update(species)
        Logger().writeline("current natoms:")
        Logger().dict_of_counter(total_natoms)
        Logger().bar()

    Logger().timer_end("total_graph_build", "Total graph build time")

    total_natoms["total"] = sum(total_natoms.values(), total_natoms["total"])
    Logger().dict_of_counter(total_natoms)
    Logger().bar()

    total_set = set()
    for k, species in total_species.items():
        total_set.update(species)
        Logger().format_k_v(k, species, write=True)
    Logger().format_k_v("total", total_set, write=True)


def label_atoms_dict_to_dataset(data_dict, cutoff, ncores):
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

    dataset = AtomGraphDataset(graph_list, cutoff)
    dataset.group_by_key(KEY.USER_LABEL)  # in place operation
    return dataset


def from_structure_list(structure_list: str,
                        cutoff: float,
                        num_cores: int,
                        label_by: str,
                        out: str,
                        metadata: dict = None):
    Logger().writeline("parsing structure list...")
    raw_dct = parse_structure_list(structure_list, format_outputs='vasp-out')
    Logger().writeline("structure_list is successfully parsed")

    Logger().writeline(f"building graph with cutoff={cutoff} and {num_cores} cores")
    Logger().timer_start("graph_build")
    dataset = label_atoms_dict_to_dataset(raw_dct, cutoff, num_cores)
    Logger().timer_end("graph_build", f"Graph build time with {num_cores}")
    Logger().writeline("graph build was succefully")

    metadata["source"] = structure_list
    labels = label_by if label_by != "auto" else dataset.user_labels
    if os.path.isdir(out):
        out = os.path.join(out, "structure_list.sevenn_data")
    elif out.endswith(".sevenn_data") is False:
        out = out + ".sevenn_data"
    dataset_finalize(dataset, labels, metadata.copy(), out)
