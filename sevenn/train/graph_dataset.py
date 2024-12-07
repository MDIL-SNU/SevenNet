import os
import warnings
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.serialization
import torch.utils.data
import yaml
from ase.data import chemical_symbols
from torch_geometric.data import Data
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from tqdm import tqdm

import sevenn._keys as KEY
import sevenn.train.dataload as dataload
import sevenn.util as util
from sevenn import __version__
from sevenn._const import NUM_UNIV_ELEMENT
from sevenn.atom_graph_data import AtomGraphData
from sevenn.sevenn_logger import Logger

if torch.__version__.split()[0] >= '2.4.0':
    # load graph without error
    torch.serialization.add_safe_globals([AtomGraphData])

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


def pt_to_args(pt_filename: str):
    """
    Return arg dict of root and processed_name from path to .pt
    Usage:
        dataset = SevenNetGraphDataset(
            **pt_to_args({path}/sevenn_data/dataset.pt)
        )
    """
    processed_dir, basename = os.path.split(pt_filename)
    return {
        'root': os.path.dirname(processed_dir),
        'processed_name': os.path.basename(basename),
    }


def _run_stat(
    graph_list,
    y_keys: List[str] = [KEY.ENERGY, KEY.PER_ATOM_ENERGY, KEY.FORCE, KEY.STRESS],
) -> Dict[str, Any]:
    """
    Loop over dataset and init any statistics might need
    """
    n_neigh = []
    natoms_counter = Counter()
    composition = torch.zeros((len(graph_list), NUM_UNIV_ELEMENT))
    stats: Dict[str, Any] = {y: {'_array': []} for y in y_keys}

    for i, graph in tqdm(
        enumerate(graph_list), desc='run_stat', total=len(graph_list)
    ):
        z_tensor = graph[KEY.ATOMIC_NUMBERS]
        natoms_counter.update(z_tensor.tolist())
        composition[i] = torch.bincount(z_tensor, minlength=NUM_UNIV_ELEMENT)
        n_neigh.append(torch.unique(graph[KEY.EDGE_IDX][0], return_counts=True)[1])
        for y, dct in stats.items():
            dct['_array'].append(
                graph[y].reshape(
                    -1,
                )
            )

    stats.update({'num_neighbor': {'_array': n_neigh}})
    for y, dct in stats.items():
        array = torch.cat(dct['_array'])
        if array.dtype == torch.int64:  # because of n_neigh
            array = array.to(torch.float)
        try:
            median = torch.quantile(array, q=0.5)
        except RuntimeError:
            warnings.warn(f'skip median due to too large tensor size: {y}')
            median = torch.nan
        dct.update(
            {
                'mean': float(torch.mean(array)),
                'std': float(torch.std(array, correction=0)),
                'median': float(median),
                'max': float(torch.max(array)),
                'min': float(torch.min(array)),
                'count': array.numel(),
                '_array': array,
            }
        )

    natoms = {chemical_symbols[int(z)]: cnt for z, cnt in natoms_counter.items()}
    natoms['total'] = sum(list(natoms.values()))
    stats.update({'_composition': composition, 'natoms': natoms})
    return stats


def _elemwise_reference_energies(composition: np.ndarray, energies: np.ndarray):
    from sklearn.linear_model import Ridge

    c = composition
    y = energies
    zero_indices = np.all(c == 0, axis=0)
    c_reduced = c[:, ~zero_indices]
    # will not 100% reproduce, as it is sorted by Z
    # train/dataset.py was sorted by alphabets of chemical species
    coef_reduced = Ridge(alpha=0.1, fit_intercept=False).fit(c_reduced, y).coef_
    full_coeff = np.zeros(NUM_UNIV_ELEMENT)
    full_coeff[~zero_indices] = coef_reduced
    return full_coeff.tolist()  # ex: full_coeff[1] = H_reference_energy


class SevenNetGraphDataset(InMemoryDataset):
    """
    Replacement of AtomGraphDataset. (and .sevenn_data)
    Extends InMemoryDataset of PyG. From given 'files', and 'cutoff',
    build graphs for training SevenNet model. Preprocessed graphs are saved to
    f'{root}/sevenn_data/{processed_name}.pt

    TODO: Save meta info (cutoff) by overriding .save and .load
    TODO: 'tag' is not used yet, but initialized
    'tag' is replacement for 'label', and each datapoint has it as integer
    'tag' is usually parsed from if the structure_list of load_dataset

    Args:
        root: path to save/load processed PyG dataset
        cutoff: edge cutoff of given AtomGraphData
        files: list of filenames to initialize dataset:
               ASE readable (with proper extension), structure_list, .sevenn_data
        process_num_cores: # of cpu cores to build graph
        processed_name: save as {root}/sevenn_data/{processed_name}.pt
        pre_transfrom: optional transform for each graph: def (graph) -> graph
        pre_filter: optional filtering function for each graph: def (graph) -> graph
        force_reload: if True, reload dataset from files even if there exist
                      {root}/sevenn_data/{processed_name}
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
        if files is None:
            files = []
        elif isinstance(files, str):
            files = [files]  # user convenience
        files = [os.path.abspath(file) for file in files]
        self._files = files
        self._full_file_list = []
        if not processed_name.endswith('.pt'):
            processed_name += '.pt'
        self._processed_names = [
            processed_name,  # {root}/sevenn_data/{name}.pt
            processed_name.replace('.pt', '.yaml'),
        ]

        root = root or './'
        _pdir = os.path.join(root, 'sevenn_data')
        _pt = os.path.join(_pdir, self._processed_names[0])
        if not os.path.exists(_pt) and len(self._files) == 0:
            raise ValueError(
                (
                    f'{_pt} not found and no files to process. '
                    + 'If you copied only .pt file, please copy '
                    + 'whole sevenn_data dir without changing its name.'
                    + ' They all work together.'
                )
            )

        _yam = os.path.join(_pdir, self._processed_names[1])
        if not os.path.exists(_yam) and len(self._files) == 0:
            raise ValueError(f'{_yam} not found and no files to process')

        self.process_num_cores = process_num_cores
        self.process_kwargs = process_kwargs

        self.tag_map = {}
        self.statistics = {}
        self.finalized = False

        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            log=log,
            force_reload=force_reload,
        )  # Internally calls 'process'
        self.load(self.processed_paths[0])  # load pt, saved after process

    def load(self, path: str, data_cls=Data) -> None:
        super().load(path, data_cls)
        if len(self) == 0:
            warnings.warn(f'No graphs found {self.processed_paths[0]}')
        self._load_meta()

    def _load_meta(self) -> None:
        with open(self.processed_paths[1], 'r') as f:
            meta = yaml.safe_load(f)

        if meta['sevennet_version'] == '0.10.0':
            self._save_meta(list(self))
            with open(self.processed_paths[1], 'r') as f:
                meta = yaml.safe_load(f)

        cutoff = float(meta['cutoff'])
        if float(meta['cutoff']) != self.cutoff:
            warnings.warn(
                (
                    'Loaded dataset is built with different cutoff length: '
                    + f'{cutoff} != {self.cutoff}, dataset cutoff will be'
                    + f'overwritten to {cutoff}'
                )
            )
        self.cutoff = cutoff
        self._files = meta['files']
        self.statistics = meta['statistics']

    @property
    def raw_file_names(self) -> List[str]:
        return self._files

    @property
    def processed_file_names(self) -> List[str]:
        return self._processed_names

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'sevenn_data')

    @property
    def full_file_list(self) -> List[str]:
        return self._full_file_list

    def process(self):
        graph_list: list[AtomGraphData] = []
        for file in self.raw_file_names:
            tmplist = SevenNetGraphDataset.file_to_graph_list(
                filename=file,
                cutoff=self.cutoff,
                num_cores=self.process_num_cores,
                **self.process_kwargs,
            )
            self._full_file_list.extend([os.path.abspath(file)] * len(tmplist))
            graph_list.extend(tmplist)

        processed_graph_list = []
        for data in graph_list:
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            processed_graph_list.append(data)

        if len(processed_graph_list) == 0:
            # Can not save at all if there is no graph (error in PyG), raise an error
            raise ValueError('Zero graph found after filtering')

        # save graphs, handled by torch_geometrics
        self.save(processed_graph_list, self.processed_paths[0])
        self._save_meta(processed_graph_list)
        if self.log:
            Logger().writeline(f'Dataset is saved: {self.processed_paths[0]}')

    def _save_meta(self, graph_list) -> None:
        stats = _run_stat(graph_list)
        stats['elemwise_reference_energies'] = _elemwise_reference_energies(
            stats['_composition'].numpy(), stats[KEY.ENERGY]['_array'].numpy()
        )
        self.statistics = stats

        stats_save = {}
        for label, dct in self.statistics.items():
            if label.startswith('_'):
                continue
            stats_save[label] = {}
            if not isinstance(dct, dict):
                stats_save[label] = dct
            else:
                for k, v in dct.items():
                    if k.startswith('_'):
                        continue
                    stats_save[label][k] = v

        meta = {
            'sevennet_version': __version__,
            'cutoff': self.cutoff,
            'when': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'files': self._files,
            'statistics': stats_save,
            'species': self.species,
            'num_graphs': self.statistics[KEY.ENERGY]['count'],
            'per_atom_energy_mean': self.per_atom_energy_mean,
            'force_rms': self.force_rms,
            'per_atom_energy_std': self.per_atom_energy_std,
            'avg_num_neigh': self.avg_num_neigh,
            'sqrt_avg_num_neigh': self.sqrt_avg_num_neigh,
        }

        with open(self.processed_paths[1], 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)

    @property
    def species(self):
        return [z for z in self.statistics['natoms'].keys() if z != 'total']

    @property
    def natoms(self):
        return self.statistics['natoms']

    @property
    def per_atom_energy_mean(self):
        return self.statistics[KEY.PER_ATOM_ENERGY]['mean']

    @property
    def elemwise_reference_energies(self):
        return self.statistics['elemwise_reference_energies']

    @property
    def force_rms(self):
        mean = self.statistics[KEY.FORCE]['mean']
        std = self.statistics[KEY.FORCE]['std']
        return float((mean**2 + std**2) ** (0.5))

    @property
    def per_atom_energy_std(self):
        return self.statistics['per_atom_energy']['std']

    @property
    def avg_num_neigh(self):
        return self.statistics['num_neighbor']['mean']

    @property
    def sqrt_avg_num_neigh(self):
        return self.avg_num_neigh**0.5

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
        filename: str, cutoff: float, num_cores: int = 1
    ) -> list[AtomGraphData]:
        datadct = dataload.structure_list_reader(filename)
        graph_list = []
        for tag, atoms_list in datadct.items():
            tmp = dataload.graph_build(atoms_list, cutoff, num_cores)
            graph_list.extend(_tag_graphs(tmp, tag))
        return graph_list

    @staticmethod
    def _read_ase_readable(
        filename: str, cutoff: float, num_cores: int = 1, tag: str = '', **ase_kwargs
    ) -> list[AtomGraphData]:
        atoms_list = dataload.ase_reader(filename, **ase_kwargs)
        graph_list = dataload.graph_build(atoms_list, cutoff, num_cores)
        if tag != '':
            graph_list = _tag_graphs(graph_list, tag)
        return graph_list

    @staticmethod
    def _read_graph_dataset(
        filename: str, cutoff: float, **kwargs
    ) -> list[AtomGraphData]:
        meta_f = filename.replace('.pt', '.yaml')
        orig_cutoff = cutoff
        if not os.path.exists(filename):
            raise FileNotFoundError(f'No such file: {filename}')
        if not os.path.exists(meta_f):
            warnings.warn('No meta info found, beware of cutoff...')
        else:
            with open(meta_f, 'r') as f:
                meta = yaml.safe_load(f)
            orig_cutoff = float(meta['cutoff'])
            if orig_cutoff != cutoff:
                warnings.warn(
                    f'{filename} has different cutoff length: '
                    + f'{cutoff} != {orig_cutoff}'
                )
        ds_args: dict[str, Any] = dict({'cutoff': orig_cutoff})
        ds_args.update(pt_to_args(filename))
        ds_args.update(kwargs)
        dataset = SevenNetGraphDataset(**ds_args)
        # TODO: hard coded. consult with inference.py
        glist = [g.fit_dimension() for g in dataset]  # type: ignore
        for g in glist:
            if KEY.STRESS in g:
                # (1, 6) is what we want
                g[KEY.STRESS] = g[KEY.STRESS].unsqueeze(0)
        return glist

    @staticmethod
    def file_to_graph_list(
        filename: str, cutoff: float, num_cores: int = 1, **kwargs
    ) -> List[AtomGraphData]:
        """
        kwargs: if file is ase readable, passed to ase.io.read
        """
        if not os.path.isfile(filename):
            raise ValueError(f'No such file: {filename}')
        graph_list: list[AtomGraphData]
        if filename.endswith('.pt'):
            graph_list = SevenNetGraphDataset._read_graph_dataset(filename, cutoff)
        elif filename.endswith('.sevenn_data'):
            graph_list, cutoff_other = SevenNetGraphDataset._read_sevenn_data(
                filename
            )
            if cutoff_other != cutoff:
                warnings.warn(f'Given {filename} has different {cutoff_other}!')
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
        'root': working_dir,
        'process_num_cores': config[KEY.PREPROCESS_NUM_CORES],
        **config[KEY.DATA_FORMAT_ARGS],
    }

    datasets = {}
    for dk in dataset_keys:
        if not (paths := config[dk]):
            continue
        if isinstance(paths, str):
            paths = [paths]
        name = dk.split('_')[1].strip()
        if (
            len(paths) == 1
            and 'sevenn_data' in paths[0]
            and paths[0].endswith('.pt')
        ):
            dataset_args.update(pt_to_args(paths[0]))
        else:
            dataset_args.update({'files': paths, 'processed_name': name})
        dataset_path = os.path.join(working_dir, 'sevenn_data', f'{name}.pt')
        if os.path.exists(dataset_path) and 'force_reload' not in dataset_args:
            log.writeline(
                f'Dataset will be loaded from {dataset_path}, without update.'
            )
            log.writeline(
                'If you have changed your files to read, put force_reload=True'
                + ' under the data_format_args key'
            )
        datasets[name] = SevenNetGraphDataset(**dataset_args)

    train_set = datasets['trainset']

    chem_species = set(train_set.species)
    # print statistics of each dataset
    for name, dataset in datasets.items():
        log.bar()
        log.writeline(f'{name} distribution:')
        log.statistic_write(dataset.statistics)
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

    if 'validset' not in datasets and config.get(KEY.RATIO, 0.0) > 0.0:
        log.writeline('Use validation set as random split from the training set')
        log.writeline(
            'Note that statistics, shift, scale, and conv_denominator are '
            + 'computed before random split.\n If you want these after random '
            + 'split, please preprocess dataset and set it as load_trainset_path '
            + 'and load_validset_path explicitly.'
        )

        ratio = float(config[KEY.RATIO])
        train, valid = torch.utils.data.random_split(
            datasets['trainset'], (1.0 - ratio, ratio)
        )
        datasets['trainset'] = train
        datasets['validset'] = valid

    return datasets
