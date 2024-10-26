import os
import random
import warnings
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch.utils.data
from ase.atoms import Atoms
from ase.data import chemical_symbols
from ase.io import write
from tqdm import tqdm

import sevenn._keys as KEY
import sevenn.train.dataload as dataload
import sevenn.util as util
from sevenn._const import NUM_UNIV_ELEMENT
from sevenn.atom_graph_data import AtomGraphData


class SevenNetAtomsDataset(torch.utils.data.Dataset):
    """ """

    def __init__(
        self,
        cutoff: float,
        files: Union[str, List[str]],
        atoms_filter: Optional[Callable] = None,
        atoms_transform: Optional[Callable] = None,
        graph_transform: Optional[Callable] = None,
        **process_kwargs,
    ):
        self.cutoff = cutoff
        if isinstance(files, str):
            files = [files]  # user convenience
        files = [os.path.abspath(file) for file in files]
        self._files = files
        self.atoms_filter = atoms_filter
        self.atoms_trasform = atoms_transform
        self.graph_trasform = graph_transform
        self._scanned = False
        self._avg_num_neigh_approx = None
        self.statistics = {}

        atoms_list = []
        for file in files:
            atoms_list.extend(
                SevenNetAtomsDataset.file_to_atoms_list(file, **process_kwargs)
            )
        self._atoms_list = atoms_list

        super().__init__()

    @staticmethod
    def file_to_atoms_list(filename: str, **kwargs) -> List[Atoms]:
        if 'structure_list' in filename:
            atoms_dct = dataload.structure_list_reader(filename)
            atoms_list = []
            for lst in atoms_dct.values():
                atoms_list.extend(lst)
        else:
            atoms_list = dataload.ase_reader(filename, **kwargs)
        return atoms_list

    def save(self, path):
        # Save atoms list as extxyz
        write(path, self._atoms_list, format='extxyz')

    def _graph_build(self, atoms):
        return dataload.atoms_to_graph(
            atoms, self.cutoff, transfer_info=False, y_from_calc=False
        )

    def __len__(self):
        return len(self._atoms_list)

    def __getitem__(self, index):
        atoms = self._atoms_list[index]
        if self.atoms_trasform is not None:
            atoms = self.atoms_trasform(atoms)

        graph = self._graph_build(atoms)
        if self.graph_trasform is not None:
            graph = self.graph_trasform(graph)

        return AtomGraphData.from_numpy_dict(graph)

    @property
    def species(self):
        self.run_stat()
        return [z for z in self.statistics['_natoms'].keys() if z != 'total']

    @property
    def natoms(self):
        self.run_stat()
        return self.statistics['_natoms']

    @property
    def per_atom_energy_mean(self):
        self.run_stat()
        return self.statistics[KEY.PER_ATOM_ENERGY]['mean']

    @property
    def elemwise_reference_energies(self):
        from sklearn.linear_model import Ridge

        c = self.statistics['_composition']
        y = self.statistics[KEY.ENERGY]['_array']
        zero_indices = np.all(c == 0, axis=0)
        c_reduced = c[:, ~zero_indices]
        # will not 100% reproduce, as it is sorted by Z
        # train/dataset.py was sorted by alphabets of chemical species
        coef_reduced = Ridge(alpha=0.1, fit_intercept=False).fit(c_reduced, y).coef_
        full_coeff = np.zeros(NUM_UNIV_ELEMENT)
        full_coeff[~zero_indices] = coef_reduced
        return full_coeff.tolist()  # ex: full_coeff[1] = H_reference_energy

    @property
    def force_rms(self):
        self.run_stat()
        mean = self.statistics[KEY.FORCE]['mean']
        std = self.statistics[KEY.FORCE]['std']
        return float((mean**2 + std**2) ** (0.5))

    @property
    def per_atom_energy_std(self):
        self.run_stat()
        return self.statistics['per_atom_energy']['std']

    @property
    def avg_num_neigh(self, n_sample=10000):
        if self._avg_num_neigh_approx is None:
            if len(self) > n_sample:
                warnings.warn(
                    """SevenNetAtomsDataset does not provide correct avg_num_neigh
                    as it does not build graph. We will compute only random 10000
                    structures graph to approximate this value. If you want more
                    precise avg_num_neigh, use SevenNetGraphDataset. If it is not
                    viable due to memory limit, you need online algorithm to do this
                    , which is not yet implemented in the SevenNet"""
                )
            n_sample = min(len(self), n_sample)
            indices = random.sample(range(len(self)), n_sample)
            n_neigh = []
            for i in indices:
                graph = self[i]
                _, nn = np.unique(graph[KEY.EDGE_IDX][0], return_counts=True)
                n_neigh.append(nn)
            n_neigh = np.concatenate(n_neigh)
            self._avg_num_neigh_approx = np.mean(n_neigh)
        return self._avg_num_neigh_approx

    @property
    def sqrt_avg_num_neigh(self):
        self.run_stat()
        return self.avg_num_neigh**0.5

    def run_stat(self):
        """
        Loop over dataset and init any statistics might need
        Unlink SevenNetGraphDataset, neighbors count is not computed as
        it requires to build graph
        """
        if self._scanned is True:
            return  # statistics already computed
        y_keys: List[str] = [KEY.ENERGY, KEY.PER_ATOM_ENERGY, KEY.FORCE, KEY.STRESS]
        natoms_counter = Counter()
        composition = np.zeros((len(self), NUM_UNIV_ELEMENT))
        stats: Dict[str, Dict[str, Any]] = {y: {'_array': []} for y in y_keys}

        for i, atoms in tqdm(
            enumerate(self._atoms_list), desc='run_stat', total=len(self)
        ):
            z = atoms.get_atomic_numbers()
            natoms_counter.update(z.tolist())
            composition[i] = np.bincount(z, minlength=NUM_UNIV_ELEMENT)
            for y, dct in stats.items():
                if y == KEY.ENERGY:
                    dct['_array'].append(atoms.info['y_energy'])
                elif y == KEY.PER_ATOM_ENERGY:
                    dct['_array'].append(atoms.info['y_energy'] / len(atoms))
                elif y == KEY.FORCE:
                    dct['_array'].append(atoms.arrays['y_force'].reshape(-1))
                elif y == KEY.STRESS:
                    dct['_array'].append(atoms.info['y_stress'].reshape(-1))

        for y, dct in stats.items():
            if y == KEY.FORCE:
                array = np.concatenate(dct['_array'])
            else:
                array = np.array(dct['_array']).reshape(-1)
            dct.update(
                {
                    'mean': float(np.mean(array)),
                    'std': float(np.std(array)),
                    'median': float(np.quantile(array, q=0.5)),
                    'max': float(np.max(array)),
                    'min': float(np.min(array)),
                    '_array': array,
                }
            )

        natoms = {chemical_symbols[int(z)]: cnt for z, cnt in natoms_counter.items()}
        natoms['total'] = sum(list(natoms.values()))
        self.statistics.update(
            {
                '_composition': composition,
                '_natoms': natoms,
                **stats,
            }
        )
        self._scanned = True


# script, return dict of SevenNetAtomsDataset
def from_config(
    config: dict[str, Any],
    working_dir: str = os.getcwd(),
    dataset_keys: Optional[list[str]] = None,
):
    from sevenn.sevenn_logger import Logger

    log = Logger()
    if dataset_keys is None:
        dataset_keys = []
        for k in config:
            if k.startswith('load_') and k.endswith('_path'):
                dataset_keys.append(k)

    if KEY.LOAD_TRAINSET not in dataset_keys:
        raise ValueError(f'{KEY.LOAD_TRAINSET} must be present in config')

    # initialize arguments for loading dataset
    dataset_args = {
        'cutoff': config[KEY.CUTOFF],
        **config[KEY.DATA_FORMAT_ARGS],
    }

    datasets = {}
    for dk in dataset_keys:
        if not (paths := config[dk]):
            continue
        if isinstance(paths, str):
            paths = [paths]
        name = dk.split('_')[1].strip()
        dataset_args.update({'files': paths})
        datasets[name] = SevenNetAtomsDataset(**dataset_args)

    if not config[KEY.COMPUTE_STATISTICS]:
        log.writeline(
            """
            Computing statistics is skipped, note that if any of other
            configurations requires statistics (shift, scale, avg_num_neigh,
            chemical_species as auto), SevenNet eventually raise an error!
            """
        )
        return datasets

    train_set = datasets['trainset']

    chem_species = set(train_set.species)
    # print statistics of each dataset
    for name, dataset in datasets.items():
        dataset.run_stat()
        log.bar()
        log.writeline(f'{name} distribution:')
        log.statistic_write(dataset.statistics)
        log.format_k_v('# atoms (node)', dataset.natoms, write=True)
        log.format_k_v('# structures (graph)', len(dataset), write=True)

        chem_species.update(dataset.species)
    log.bar()

    # initialize known species from dataset if 'auto'
    # sorted to alphabetical order (which is same as before)
    chem_keys = [KEY.CHEMICAL_SPECIES, KEY.NUM_SPECIES, KEY.TYPE_MAP]
    if all([config[ck] == 'auto' for ck in chem_keys]):  # see parse_input.py
        log.writeline('Known species are obtained from the dataset')
        config.update(util.chemical_species_preprocess(sorted(list(chem_species))))

    # retrieve shift, scale, conv_denominaotrs from user input (keyword)
    init_from_stats = [KEY.SHIFT, KEY.SCALE, KEY.CONV_DENOMINATOR]
    for k in init_from_stats:
        input = config[k]  # statistic key or numbers
        # If it is not 'str', 1: It is 'continue' training
        #                     2: User manually inserted numbers
        if isinstance(input, str) and hasattr(train_set, input):
            var = getattr(train_set, input)
            config.update({k: var})
            log.writeline(f'{k} is obtained from statistics')
        elif isinstance(input, str) and not hasattr(train_set, input):
            raise NotImplementedError(input)

    return datasets
