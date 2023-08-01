import random
import itertools
from collections import Counter
from typing import List, Dict, Callable, Optional, Union

import torch
from ase.data import chemical_symbols
import numpy as np

import sevenn._keys as KEY


# TODO: inherit torch_geometry dataset?
# TODO: url things?
# TODO: label specific or atom specie wise statistic?
class AtomGraphDataset:
    """
    class representing dataset containing fully preprocessed data ready
    for model input
    if given data is List, it stores data as {KEY_DEFAULT: data}

    Args:
        data (Union[Dict[str, List], List]: dataset as dict or pure list
        preprocessor (Callable, Optional): preprocess function for each data
        metadata (Dict, Optional): metadata of data used for whether augment is valid

    for now, metadata 'might' have following keys:
        KEY.CUTFF (float), KEY.CHEMICAL_SPECIES (Dict)
    """
    DATA_KEY_ENERGY = KEY.ENERGY
    DATA_KEY_FORCE = KEY.FORCE
    KEY_DEFAULT = 'No_label'

    # TODO: mpi thing? logging?
    def __init__(self,
                 data: Union[Dict[str, List], List],
                 preprocessor: Optional[Callable] = None,
                 metadata: Optional[Dict] = None,
                 is_modified: bool = False):

        self.is_modified = is_modified
        self.meta = metadata
        if type(data) is list:
            data = {self.KEY_DEFAULT: data}

        self.dataset = {}
        for user_label, data_list in data.items():
            self.dataset[user_label] = []
            for datum in data_list:
                if preprocessor is not None:
                    datum = preprocessor(datum)
                if KEY.USER_LABEL not in datum:  # init label only when data is fresh
                    datum[KEY.USER_LABEL] = user_label
                self.dataset[user_label].append(datum)
        self.user_labels = list(data.keys())

    def len(self):
        if len(self.dataset.keys()) == 1 and \
           list(self.dataset.keys())[0] == AtomGraphDataset.KEY_DEFAULT:
            return len(self.dataset[AtomGraphDataset.KEY_DEFAULT])
        else:
            return {k: len(v) for k, v in self.dataset.items()}

    def get(self, idx, key=None):
        if key is None:
            key = self.KEY_DEFAULT
        return self.dataset[key][idx]

    def items(self):
        return self.dataset.items()

    def to_dct(self):
        dct_dataset = {}
        for label, data_list in self.dataset.items():
            dct_dataset[label] = [datum.to_dict() for datum in data_list]
        self.dataset = dct_dataset
        return self

    def toggle_requires_grad_of_data(self, key: str, requires_grad_value: bool):
        """
        set requires_grad of specific key of data(pos, edge_vec, ...)
        """
        for data_list in self.dataset.values():
            for datum in data_list:
                datum[key].requires_grad_(requires_grad_value)

    def divide_dataset(self, ratio: float, constant_ratio_btw_labels=True,
                       ignore_test=True):
        """
        divide dataset into 1-2*ratio : ratio : ratio
        return divided AtomGraphDataset
        returned value lost its dict key and became {KEY_DEFAULT: datalist}
        but KEY.USER_LABEL of each data is preserved
        """
        def divide(ratio: float, data_list: List, ignore_test=True):
            # Get list as input and return list divided by 1-2*ratio : ratio : ratio
            if ratio > 0.5:
                raise ValueError('Ratio must not exceed 0.5')
            data_len = len(data_list)
            random.shuffle(data_list)
            n_validation = int(data_len * ratio)
            if n_validation == 0:
                raise ValueError('# of validation set is 0, increase your dataset')

            if ignore_test:
                test_list = []
                n_train = data_len - n_validation
                train_list = data_list[0:n_train]
                valid_list = data_list[n_train:]
            else:
                n_train = data_len - 2 * n_validation
                train_list = data_list[0:n_train]
                valid_list = data_list[n_train:n_train + n_validation]
                test_list = data_list[n_train + n_validation:data_len]
            return train_list, valid_list, test_list

        lists = ([], [], [])  # train, valid, test
        if constant_ratio_btw_labels:
            for data_list in self.dataset.values():
                for store, divided in zip(lists, divide(ratio, data_list)):
                    store.extend(divided)
        else:
            lists = divide(ratio, self.to_list())

        return tuple(AtomGraphDataset(data, metadata=self.meta) for data in lists)

    def to_list(self):
        return list(itertools.chain(*self.dataset.values()))

    def get_natoms(self, type_map):
        """
        type_map: Z->one_hot_index(node_feature)
        return Dict{label: {symbol, natom}]}
        """
        KEY_TO_LOOK = KEY.NODE_FEATURE
        natoms = {}
        species = [chemical_symbols[Z] for Z in type_map.keys()]
        type_map_rev = {v: k for k, v in type_map.items()}
        for label in self.user_labels:
            data = self.dataset[label]
            natoms[label] = {sym: 0 for sym in species}
            for datum in data:
                cnt = Counter(torch.argmax(datum[KEY_TO_LOOK], dim=1))
                for k, v in cnt.items():
                    atomic_num = type_map_rev[k.item()]
                    natoms[label][chemical_symbols[atomic_num]] += v
        return natoms

    def get_per_atom_mean(self, key, key_num_atoms=KEY.NUM_ATOMS):
        """
        return per_atom mean of given data key
        """
        eng_list = torch.Tensor([x[key] / x[key_num_atoms] for x in self.to_list()])
        return float(torch.mean(eng_list))

    def get_per_atom_energy_mean(self):
        """
        alias for get_per_atom_mean(KEY.ENERGY)
        """
        return self.get_per_atom_mean(KEY.ENERGY)

    def get_force_rmse(self, force_key=KEY.FORCE):
        force_list = []
        for x in self.to_list():
            force_list.extend(x[force_key].reshape(-1,).tolist())
        force_list = torch.Tensor(force_list)
        return float(torch.sqrt(torch.mean(torch.pow(force_list, 2))))

    def get_avg_num_neigh(self):
        n_neigh = []
        for _, data_list in self.dataset.items():
            for data in data_list:
                n_neigh.extend(
                    np.unique(data[KEY.EDGE_IDX][0], return_counts=True)[1]
                )

        avg_num_neigh = np.average(n_neigh)
        return avg_num_neigh

    def augment(self, dataset, validator: Optional[Callable] = None):
        """check meta compatiblity here
        dataset(AtomGraphDataset): data to augment
        validator(Callable, Optional): function(self, dataset) -> bool

        if validator is None, by default it checks
        whether cutoff & chemical_species are same before augment
        """
        assert type(dataset) == AtomGraphDataset

        def default_validator(dataset1, dataset2):
            meta1 = dataset1.meta
            meta2 = dataset2.meta
            cutoff1 = dataset1.meta[KEY.CUTOFF]
            cutoff2 = dataset2.meta[KEY.CUTOFF]
            # compare unordered lists
            chem_1 = Counter(meta1[KEY.CHEMICAL_SPECIES])
            chem_2 = Counter(meta2[KEY.CHEMICAL_SPECIES])

            # info for print error
            info = f"cutoff1: {cutoff1}, cutoff2: {cutoff2},\n"\
                + f"chem_1: {chem_1}, chem_2: {chem_2}"
            return (cutoff1 == cutoff2 and chem_1 == chem_2, info)
        if validator is None:
            validator = default_validator
        is_valid, info = validator(self, dataset)
        if is_valid:
            for key, val in dataset.items():
                if key in self.dataset:
                    self.dataset[key].extend(val)
                else:
                    self.dataset.update({key: val})
        else:
            raise ValueError(f'given datasets are not compatible {info}')
        self.user_labels = list(self.dataset.keys())

    def save(self, path, by_label=False):
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        """
        if by_label:
            for label, data in self.dataset.items():
                to = f"{path}/{label}.sevenn_data"
                torch.save(AtomGraphDataset({label: data}, metadata=self.meta), to)
        else:
            torch.save(self, path)
