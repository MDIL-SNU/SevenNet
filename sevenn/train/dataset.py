import warnings
import random
import itertools
from collections import Counter
from typing import List, Dict, Callable, Optional, Union

import torch
from ase.data import chemical_symbols
import numpy as np

import sevenn.util
import sevenn._keys as KEY


# TODO: inherit torch_geometry dataset?
# TODO: url things?
# TODO: label specific or atom specie wise statistic?
class AtomGraphDataset:
    """
    class representing dataset of AtomGraphData
    the dataset is handled as dict, {label: data}
    if given data is List, it stores data as {KEY_DEFAULT: data}

    Every data expected to have one unique cutoff
    No validity or check of the condition is done inside the object

    attribute:
        dataset (Dict[str, List]): key is data label(str), value is list of data
        user_labels (List[str]): list of user labels same as dataset.keys()
        meta (Dict, Optional): metadata of dataset
    for now, metadata 'might' have following keys:
        KEY.CUTFF (float), KEY.CHEMICAL_SPECIES (Dict)
    """
    DATA_KEY_X = KEY.NODE_FEATURE
    DATA_KEY_ENERGY = KEY.ENERGY
    DATA_KEY_FORCE = KEY.FORCE
    KEY_DEFAULT = 'No_label'

    def __init__(self,
                 dataset: Union[Dict[str, List], List],
                 cutoff: float,
                 metadata: Optional[Dict] = None,
                 x_is_one_hot_idx: Optional[bool] = False):
        """
        Default constructor of AtomGraphDataset
        Args:
            dataset (Union[Dict[str, List], List]: dataset as dict or pure list
            metadata (Dict, Optional): metadata of data
            cutoff (float): cutoff radius of graphs inside the dataset
            x_is_one_hot_idx (bool, Optional): if True, x is one_hot_idx, eles 'Z'

        'x' (node feature) of dataset can have 3 states, atomic_numbers,
        one_hot_idx, or one_hot_vector.

        atomic_numbers is the most general one but cannot directly used for input
        one_hot_idx is can be input of the model but requires 'type_map'
        one_hot_idx to one_hot_vector is done by model, so it should not be used here
        """
        self.cutoff = cutoff
        self.x_is_one_hot_idx = x_is_one_hot_idx
        self.meta = metadata
        if type(dataset) is list:
            self.dataset = {self.KEY_DEFAULT: dataset}
        else:
            self.dataset = dataset
        self.user_labels = list(self.dataset.keys())

    def group_by_key(self, data_key=KEY.USER_LABEL):
        """
        group dataset list by given key and save it as dict
        and change in-place
        Args:
            data_key (str): data key to group by

        original use is USER_LABEL, but it can be used for other keys
        if someone established it from data[KEY.INFO]
        """
        data_list = self.to_list()
        self.dataset = {}
        for datum in data_list:
            key = datum[data_key]
            if key not in self.dataset:
                self.dataset[key] = []
            self.dataset[key].append(datum)
        self.user_labels = list(self.dataset.keys())

    def seperate_info(self, data_key=KEY.INFO):
        """
        seperate info from data and save it as list of dict
        to make it compatible with torch_geometric
        """
        data_list = self.to_list()
        info_list = []
        for datum in data_list:
            info_list.append(datum[data_key])
            del datum[data_key]  # It really changes the self.dataset
            datum[data_key] = len(info_list) - 1
        self.info_list = info_list

        return data_list, info_list

    def get_species(self):
        """
        You can also use get_natoms and extract keys from there istead of this
        (And it is more efficient)
        get chemical species of dataset
        return list of chemical species (as str)
        """
        if hasattr(self, "type_map"):
            natoms = self.get_natoms(self.type_map)
        else:
            natoms = self.get_natoms()
        species = set()
        for natom_dct in natoms.values():
            species.update(natom_dct.keys())
        species = sorted(list(species))
        return species

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

    def x_to_one_hot_idx(self, type_map: Dict[int, int]):
        """
        type_map is dict of {atomic_number: one_hot_idx}
        after this process, the dataset has dependency on type_map
        or chemical species user want to consider
        """
        assert self.x_is_one_hot_idx is False
        for data_list in self.dataset.values():
            for datum in data_list:
                datum[self.DATA_KEY_X] = \
                    torch.LongTensor([type_map[z.item()]
                                      for z in datum[self.DATA_KEY_X]])
        self.type_map = type_map
        self.x_is_one_hot_idx = True

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

        return tuple(AtomGraphDataset(data, self.cutoff,
                                      metadata=self.meta) for data in lists)

    def to_list(self):
        return list(itertools.chain(*self.dataset.values()))

    def get_natoms(self, type_map=None):
        """
        if x_is_one_hot_idx, type_map is required
        type_map: Z->one_hot_index(node_feature)
        return Dict{label: {symbol, natom}]}
        """
        assert not(self.x_is_one_hot_idx is True and type_map is None)
        natoms = {}
        if type_map is not None:
            type_map_rev = {v: k for k, v in type_map.items()}
        for label, data in self.dataset.items():
            natoms[label] = Counter()
            for datum in data:
                # list of atomic number
                Zs = [chemical_symbols[z] for z in datum[self.DATA_KEY_X].tolist()]
                if self.x_is_one_hot_idx:
                    Zs = [type_map_rev[z] for z in Zs]
                cnt = Counter(Zs)
                natoms[label] += cnt
            natoms[label] = dict(natoms[label])
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

        check consistent data type, float, double, long integer etc
        """
        assert type(dataset) == AtomGraphDataset

        def default_validator(dataset1, dataset2):
            meta1 = dataset1.meta
            meta2 = dataset2.meta
            cutoff1 = dataset1.cutoff
            cutoff2 = dataset2.cutoff
            cut_consis = cutoff1 == cutoff2
            # compare unordered lists
            try:
                chem_1 = Counter(meta1[KEY.CHEMICAL_SPECIES])
                chem_2 = Counter(meta2[KEY.CHEMICAL_SPECIES])
                chem_consis = chem_1 == chem_2
            except KeyError:
                chem_consis = dataset1.x_is_one_hot_idx is False \
                    and dataset2.x_is_one_hot_idx is False

            return cut_consis and chem_consis
        if validator is None:
            validator = default_validator
        is_valid = validator(self, dataset)
        if not is_valid:
            raise ValueError('given datasets are not compatible')
        for key, val in dataset.items():
            if key in self.dataset:
                self.dataset[key].extend(val)
            else:
                self.dataset.update({key: val})
        self.user_labels = list(self.dataset.keys())

    def unify_dtypes(self, float_dtype=torch.float32, int_dtype=torch.int64):
        data_list = self.to_list()
        for datum in data_list:
            for k, v in list(datum.items()):
                datum[k] = sevenn.util.dtype_correct(v, float_dtype, int_dtype)

    def delete_data_key(self, key):
        for data in self.to_list():
            del data[key]

    def save(self, path, by_label=False):
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        """
        if by_label:
            for label, data in self.dataset.items():
                to = f"{path}/{label}.sevenn_data"
                #torch.save(AtomGraphDataset({label: data}, metadata=self.meta), to)
                torch.save(AtomGraphDataset({label: data}, self.cutoff, metadata=self.meta), to)
        else:
            torch.save(self, path)
