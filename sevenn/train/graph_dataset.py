import os
import warnings
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from ase.data import chemical_symbols
# from torch.utils.data import random_split
from torch_geometric.data.in_memory_dataset import InMemoryDataset

import sevenn._keys as KEY
import sevenn.train.dataload as dataload
import sevenn.util as util
from sevenn._const import NUM_UNIV_ELEMENT
from sevenn.atom_graph_data import AtomGraphData

# warning from PyG, for later torch versions
warnings.filterwarnings(
    'ignore',
    message='You are using `torch.load` with `weights_only=False`',
)


def _tag_graphs(graph_list: List[AtomGraphData], tag: str):
    """
    WIP: To be used
    """
    for g in graph_list:
        g[KEY.TAG] = tag
    return graph_list


def filename2args(pt_filename: str):
    """
    Return arg dict of root and processed_name from path to .pt
    Usage:
        dataset = SevenNetGraphDataset(
            **filename2args({path}/processed_7net/dataset.pt)
        )
    """
    processed_dir, basename = os.path.split(pt_filename)
    return {
        'root': os.path.dirname(processed_dir),
        'processed_name': os.path.basename(basename)
    }


class SevenNetGraphDataset(InMemoryDataset):
    """
    Replacement of AtomGraphDataset.
    Holds list of AtomGraphData, python intatnce of .sevenn_data
    'tag' is replacement for 'label', and each datapoint has it as integer
    'tag' is usually parsed from if the structure_list of load_dataset
    Unnecessary attributed such as x_is_one_hot_idx or toggle grad are removed

    Args:
        root: path to save/load processed PyG dataset
        cutoff: edge cutoff of given AtomGraphData
        files: list of filenames, extxyz, structure_list, .sevenn_data
        process_num_cores: # of cpu cores to build graph
        processed_name: name of .pt file to be saved in {root}/processed_7net
        ... some InMemoryDataset callables
        force_reload: if True, reload dataset from files even if there exist
                      {root}/processed_7net/{processed_name}
        **process_kwargs: keyword arguments that will be passed into ase.io.read
    """

    def __init__(
        self,
        cutoff: float,
        root: Optional[str] = None,
        files: Optional[Union[str, List[str]]] = None,
        process_num_cores: int = 1,
        processed_name: str = 'graph.pt',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
        **process_kwargs,
    ):
        self.cutoff = cutoff
        cutoff_given = cutoff
        if files is None:
            files = []
        elif isinstance(files, str):
            files = [files]  # user convenience
        self._files = files
        if not processed_name.endswith('.pt'):
            processed_name += '.pt'
        self._processed_name = processed_name
        self.process_num_cores = process_num_cores
        self.process_kwargs = process_kwargs

        super().__init__(
            root, transform, pre_transform, pre_filter,
            log=log, force_reload=force_reload
        )  # file saved at this moment
        self.load(self.processed_paths[0])

        if self.cutoff != cutoff_given:
            warnings.warn(
                f'!!!This dataset has built with different cutoff: {self.cutoff}!!!',
                UserWarning
            )

        self.tag_map = {}
        self.statistics = {}
        self.finalized = False
        self.is_shuffled = False
        self._scanned = False

    @property
    def raw_file_names(self) -> List[str]:
        return self._files

    @property
    def processed_file_names(self) -> List[str]:
        return [self._processed_name]

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed_7net')

    def process(self):
        graph_list: list[AtomGraphData] = []
        for file in self.raw_file_names:
            graph_list.extend(
                SevenNetGraphDataset._file_to_graph_list(
                    filename=file,
                    cutoff=self.cutoff,
                    num_cores=self.process_num_cores,
                    **self.process_kwargs,
                )
            )
        for data in graph_list:
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

        self.save(graph_list, self.processed_paths[0])

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
        full_coeff = np.zeros(NUM_UNIV_ELEMENT)
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
        composition = torch.zeros((len(self), NUM_UNIV_ELEMENT))
        stats: Dict[str, Dict[str, Any]] = {y: {'_array': []} for y in y_keys}

        for i, graph in enumerate(self):
            z_tensor = graph[KEY.ATOMIC_NUMBERS]
            natoms_counter.update(z_tensor.tolist())
            composition[i] = torch.bincount(z_tensor, minlength=NUM_UNIV_ELEMENT)
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
    def _read_sevenn_data(filename: str) -> tuple[list[AtomGraphData], float]:
        # backward compatibility
        from sevenn.train.dataset import AtomGraphDataset
        dataset = torch.load(filename, map_location='cpu', weights_only=False)
        if isinstance(dataset, AtomGraphDataset):
            graph_list = []
            for _, graphs in dataset.dataset.items():  # type: ignore
                # TODO: transfer label to tag (who gonna need this?)
                graph_list.extend(graphs)
            return graph_list, dataset.cutoff
        else:
            raise ValueError(f'Not sevenn_data type: {type(dataset)}')

    @staticmethod
    def _read_structure_list(
        filename: str,
        cutoff: float,
        num_cores: int = 1
    ) -> list[AtomGraphData]:
        datadct = dataload.structure_list_reader(filename)
        graph_list = []
        for tag, atoms_list in datadct.items():
            tmp = dataload.graph_build(atoms_list, cutoff, num_cores)
            graph_list.extend(_tag_graphs(tmp, tag))
        return graph_list

    @staticmethod
    def _read_ase_readable(
        filename: str,
        cutoff: float,
        num_cores: int = 1,
        tag: str = '',
        **ase_kwargs
    ) -> list[AtomGraphData]:
        atoms_list = dataload.ase_reader(filename, **ase_kwargs)
        graph_list = dataload.graph_build(atoms_list, cutoff, num_cores)
        if tag != '':
            graph_list = _tag_graphs(graph_list, tag)
        return graph_list

    @staticmethod
    def _file_to_graph_list(
        filename: str,
        cutoff: float,
        num_cores: int = 1,
        **kwargs
    ):
        """
        kwargs: if file is ase readable, passed to ase.io.read
        """
        if not os.path.isfile(filename):
            raise ValueError(f'No such file: {filename}')
        graph_list: list[AtomGraphData]
        if filename.endswith('.sevenn_data'):
            graph_list, cutoff_other =\
                SevenNetGraphDataset._read_sevenn_data(filename)
            if cutoff_other != cutoff:
                warnings.warn(
                    f'Given {filename} has different {cutoff_other}!', UserWarning
                )
            cutoff = cutoff_other
        elif 'structure_list' in filename:
            graph_list = SevenNetGraphDataset._read_structure_list(
                filename, cutoff, num_cores
            )
        else:
            graph_list = SevenNetGraphDataset._read_ase_readable(
                filename, cutoff, num_cores, **kwargs
            )
        return graph_list


# script, return dict of SevenNetGraphDataset
def from_config(
    config: dict[str, Any],
    working_dir: str = os.getcwd(),
    dataset_keys: Optional[list[str]] = None,
):
    from sevenn.sevenn_logger import Logger
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
        'root': working_dir,
        'process_num_cores': config[KEY.PREPROCESS_NUM_CORES],
        **config[KEY.DATA_FORMAT_ARGS]
    }

    datasets = {}
    for dk in dataset_keys:
        if not (paths := config[dk]):
            continue
        name = dk.split('_')[1].strip()
        if (len(paths) == 1
           and 'processed_7net' in paths[0]
           and paths[0].endswith('.pt')):
            dataset_args.update(filename2args(paths[0]))
        else:
            dataset_args.update({'files': paths, 'processed_name': name})
        datasets[name] = SevenNetGraphDataset(**dataset_args)

    train_set = datasets['trainset']

    # print statistics of each dataset
    for name, dataset in datasets.items():
        dataset.run_stat()
        Logger().bar()
        Logger().writeline(f'{name} distribution:')
        Logger().statistic_write(dataset.statistics)
        Logger().format_k_v('# atoms (node)', dataset.natoms, write=True)
        Logger().format_k_v('# structures (graph)', len(dataset), write=True)
    Logger().bar()

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
    # sorted to alphabetical order (which is same as before)
    chem_keys = [KEY.CHEMICAL_SPECIES, KEY.NUM_SPECIES, KEY.TYPE_MAP]
    if all([config[ck] == 'auto' for ck in chem_keys]):  # see parse_input.py
        Logger().writeline('Known species are obtained from the dataset')
        chem_species = sorted(train_set.species)
        config.update(util.chemical_species_preprocess(chem_species))

    """
    if 'validset' not in dataset_keys:
        Logger().writeline('As validset is not given, I use random split')
        Logger().writeline('Note that statistics computed BEFORE the random split!')
        ratio = float(config[KEY.RATIO])
        train, valid = random_split(datasets['dataset'], (1.0 - ratio, ratio))
        datasets['dataset'] = train
        datasets['validset'] = valid
    """

    return datasets
