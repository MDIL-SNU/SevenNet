import os
import warnings
import random
from collections import Counter
from typing import List, Dict, Union, Any

import numpy as np
import torch
import torch.utils.data.dataset as torch_dataset
from ase.data import chemical_symbols

import sevenn.util as util
import sevenn._keys as KEY
import sevenn.train.dataload as dataload
from sevenn.atom_graph_data import AtomGraphData


"""
Replacement of sevenn/train/dataset.py
"""

def _tag_graphs(graph_list: List[AtomGraphData], tag: str):
    for g in graph_list:
        g[KEY.TAG] = tag
    return graph_list


class SevenNetGraphDataset(torch_dataset.Dataset):
    """
    Replacement of AtomGraphDataset.
    Holds list of AtomGraphData, python intatnce of .sevenn_data
    'tag' is replacement for 'label', and each datapoint has it as integer
    'tag' is usually parsed from if the structure_list of load_dataset
    Unnecessary attributed such as x_is_one_hot_idx or toggle grad are removed

    Args:
        dataset: List of AtomGraphData to save
        cutoff: edge cutoff of given AtomGraphData
        origin_files: auxilary arguments that explain where thy come from
    """

    def __init__(
        self,
        dataset: List[AtomGraphData],
        cutoff: float,
        origin_files: List[str] = [],
    ):
        self.cutoff = cutoff
        self.dataset = dataset
        self.origin_files = origin_files
        self.tag_map = {}
        self.statistics = {}
        self.finalized = False
        self.is_shuffled = False
        self._scanned = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __contains__(self, item):
        return item in self.dataset

    def __iter__(self):
        return iter(self.dataset)

    def __add__(self, _):
        raise NotImplementedError("Please use .augment instead")

    def augment(self, other):
        assert isinstance(other, SevenNetGraphDataset)
        self._scanned = False
        self.finalized = False
        if self.cutoff != other.cutoff:
            warnings.warn('Augmenting dataset with different cutoff', UserWarning)
        self.dataset.extend(other.dataset)
        self.origin_files.extend(other.origin_files)

    def split(self, ratio: float):
        assert ratio < 1.0
        random.shuffle(self.dataset)
        self.is_shuffled = True
        n_valid = int(len(self) * ratio)
        n_train = len(self) - n_valid
        train_list = self.dataset[0: n_train]
        valid_list = self.dataset[n_train:]
        train_set = SevenNetGraphDataset(
            train_list, self.cutoff, self.origin_files
        )
        valid_set = SevenNetGraphDataset(
            valid_list, self.cutoff, self.origin_files
        )
        return train_set, valid_set

    @property
    def species(self):
        if not self._scanned:
            self.run_stat()
        return [z for z in self.statistics['_natoms'].keys() if z != 'total']
    
    @property
    def natoms(self):
        if not self._scanned:
            self.run_stat()
        return self.statistics['_natoms']

    @property
    def per_atom_energy_mean(self):
        if not self._scanned:
            self.run_stat()
        return self.statistics[KEY.PER_ATOM_ENERGY]['mean']

    @property
    def elemwise_reference_energies(self):
        from sklearn.linear_model import Ridge
        if not self._scanned:
            self.run_stat()
        c = self.statistics['_composition'].numpy()
        y = self.statistics[KEY.ENERGY]['_array'].numpy()
        zero_indices = np.all(c == 0, axis=0)
        c_reduced = c[:, ~zero_indices]
        # will not 100% reproduce, as it is sorted by Z
        # train/dataset.py was sorted by alphabets of chemical species
        coef_reduced = (  
            Ridge(alpha=0.1, fit_intercept=False).fit(c_reduced, y).coef_
        )
        full_coeff = np.zeros(120)
        full_coeff[~zero_indices] = coef_reduced
        return full_coeff

    @property
    def force_rms(self):
        if not self._scanned:
            self.run_stat()
        mean = self.statistics[KEY.FORCE]['mean']
        std = self.statistics[KEY.FORCE]['std']
        return float((mean**2 + std**2)**(0.5))

    @property
    def per_atom_energy_std(self):
        if not self._scanned:
            self.run_stat()
        return self.statistics['per_atom_energy']['std']

    @property
    def avg_num_neigh(self):
        if not self._scanned:
            self.run_stat()
        return self.statistics['num_neighbor']['mean']

    @property
    def sqrt_avg_num_neigh(self):
        if not self._scanned:
            self.run_stat()
        return self.avg_num_neigh ** 0.5

    def run_stat(
        self, 
        y_keys: List[str] = [KEY.ENERGY, KEY.PER_ATOM_ENERGY, KEY.FORCE, KEY.STRESS]
    ):
        """
        Loop over dataset and init any statistics might need
        """
        n_neigh = []
        natoms_counter = Counter()
        composition = torch.zeros((len(self), 120))
        stats: Dict[str, Dict[str, Any]] = {y: {'_array': []} for y in y_keys}

        for i, graph in enumerate(self.dataset):
            natoms_counter.update(graph[KEY.ATOMIC_NUMBERS].tolist())
            composition[i] = torch.bincount(graph[KEY.ATOMIC_NUMBERS], minlength=120)
            n_neigh.append(
                torch.unique(graph[KEY.EDGE_IDX][0], return_counts=True)[1]
            )
            for y, dct in stats.items():
                dct['_array'].append(graph[y].reshape(-1,))

        stats.update({'num_neighbor': {'_array': n_neigh}})
        for y, dct in stats.items():
            array = torch.cat(dct['_array'])
            if array.dtype == torch.int64:  # because of n_neigh
                array = array.to(torch.float)
            dct.update({
                'mean': float(torch.mean(array)),
                'std': float(torch.std(array)),
                'median': float(torch.median(array)),
                'max': float(torch.max(array)),
                'min': float(torch.min(array)),
                '_array': array,
            })

        natoms = {chemical_symbols[int(z)]: cnt for z, cnt in natoms_counter.items()}
        natoms['total'] = sum(list(natoms.values()))
        self.statistics.update({
            '_composition': composition,
            '_natoms': natoms,
            **stats,
        })

        self._scanned = True

    @staticmethod
    def _from_sevenn_data(filename: str):
        from sevenn.train.dataset import AtomGraphDataset
        dataset = torch.load(filename, map_location='cpu', weights_only=False)
        if isinstance(dataset, AtomGraphDataset):
            # backward compatibility
            graph_list = []
            for _, graphs in dataset.dataset.items():
                # TODO: transfer label to tag (who gonna need this?)
                graph_list.extend(graphs)
            return SevenNetGraphDataset(
                dataset=graph_list, 
                cutoff=dataset.cutoff, 
                origin_files=[filename],
            )
        elif isinstance(dataset, SevenNetGraphDataset):
            return dataset
        else:
            raise ValueError(f'Not sevenn_data type: {type(dataset)}')

    @staticmethod
    def _from_structure_list(filename: str, cutoff: float, num_cores: int = 1):
        datadct = dataload.structure_list_reader(filename)
        graph_list = []
        for tag, atoms_list in datadct.items():
            tmp = dataload.graph_build(atoms_list, cutoff, num_cores)
            graph_list.extend(_tag_graphs(tmp, tag))
        return SevenNetGraphDataset(graph_list, cutoff, [filename])

    @staticmethod
    def _from_ase_readable(
        filename: str, 
        cutoff: float, 
        num_cores: int = 1, 
        tag: str = '', 
        **ase_kwargs
    ):
        atoms_list = dataload.ase_reader(filename, **ase_kwargs)
        graph_list = dataload.graph_build(atoms_list, cutoff, num_cores)
        if tag != '':
            graph_list = _tag_graphs(graph_list, tag)
        return SevenNetGraphDataset(graph_list, cutoff, [filename])

    @staticmethod
    def from_file(filename: str, cutoff: float, num_cores: int = 1, **ase_kwargs):
        if not os.path.isfile(filename):
            raise ValueError(f"No such file: {filename}")
        if filename.endswith('.sevenn_data'):
            return SevenNetGraphDataset._from_sevenn_data(filename)
        elif 'structure_list' in filename:
            return SevenNetGraphDataset._from_structure_list(
                filename, cutoff, num_cores
            )
        else:
            return SevenNetGraphDataset._from_ase_readable(
                filename, cutoff, num_cores, **ase_kwargs
            )

    @staticmethod
    def from_files(
        files: Union[str, List[str]], 
        cutoff: float, 
        num_cores: int = 1, 
        **kwargs
    ):
        if isinstance(files, str):
            files = [files]
        dataset = SevenNetGraphDataset([], cutoff)
        for file in files:
            dataset.augment(
                SevenNetGraphDataset.from_file(
                    filename=file, 
                    cutoff=cutoff, 
                    num_cores=num_cores, 
                    **kwargs
                )
            )
        ## post
        return dataset


# script, return dict of SevenNetGraphDataset
def from_config(config):
    from sevenn.sevenn_logger import Logger

    # initialize arguments for loading dataset
    load_dataset = config[KEY.LOAD_DATASET]
    common = {
        'cutoff': config[KEY.CUTOFF],
        'num_cores': config[KEY.PREPROCESS_NUM_CORES],
        **config[KEY.DATA_FORMAT_ARGS]
    }
    dataset_splits = {}

    # load train, valid, and test (if given)
    train_set = SevenNetGraphDataset.from_files(load_dataset, **common)
    if load_validset := config[KEY.LOAD_VALIDSET]:
        valid_set = SevenNetGraphDataset.from_files(load_validset, **common)
    else:
        train_set, valid_set = train_set.split(config[KEY.RATIO])
    dataset_splits.update({'train': train_set, 'valid': valid_set})

    if load_testset := config[KEY.LOAD_TESTSET]:
        test_set = SevenNetGraphDataset.from_files(load_testset, **common)
        dataset_splits.update({'test': test_set})

    if save_dataset := config[KEY.SAVE_DATASET]:
        dirname = os.path.dirname(save_dataset)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        for name, dataset in dataset_splits.items():
            fpath = os.path.join(dirname, f'{name}.sevenn_data')
            fpath = util.unique_filepath(fpath)
            torch.save(dataset, fpath)
            Logger().writeline(f'Saved: {fpath}')

    # print statistics of each dataset
    for name, dataset in dataset_splits.items():
        dataset.run_stat()
        Logger().bar()
        Logger().writeline(f'{name} statistics')
        Logger().statistic_write(dataset.statistics)
        Logger().format_k_v('# atoms', dataset.natoms, write=True)
        Logger().format_k_v('# structures', len(dataset), write=True)

    # retrieve shift, scale, conv_denominaotrs from user input (keyword)
    init_from_stats = [KEY.SHIFT, KEY.SCALE, KEY.CONV_DENOMINATOR]
    for k in init_from_stats:
        input = config[k]  # statistic key or numbers
        if isinstance(input, str) and hasattr(train_set, input):
            Logger().writeline(f'{k} is obtained from statistics')
            config.update({k: getattr(train_set, input)})
        # else, it should be float or list of float (with idx=Z)
        # either: continue training or manually given from yaml

    # initialize known species from dataset if 'auto'
    # sorted as aphabetical order (same as before)
    chem_keys = [KEY.CHEMICAL_SPECIES, KEY.NUM_SPECIES, KEY.TYPE_MAP]
    if all([ck == 'auto' for ck in chem_keys]):  # see parse_input.py
        Logger().writeline('Known species are derived from train set')
        chem_species = sorted(train_set.species)
        config.update(util.chemical_species_preprocess(chem_species))

    return dataset_splits
