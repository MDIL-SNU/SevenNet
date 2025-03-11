import bisect
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.logger import Logger


def _arrange_paths_by_modality(paths: List[dict]):
    modal_dct = {}
    for path in paths:
        if isinstance(path, dict):
            if KEY.DATA_MODALITY not in path:
                raise ValueError(f'{KEY.DATA_MODALITY} is missing')
            modal = path.pop(KEY.DATA_MODALITY)
        else:
            raise TypeError(f'{path} is not dict or str')
        if modal not in modal_dct:
            modal_dct[modal] = []
        modal_dct[modal].append(path)
    return modal_dct


def combined_variance(
    means: np.ndarray, stds: np.ndarray, sample_sizes: np.ndarray, ddof: int = 0
) -> float:
    """
    Calculate the combined variance for multiple datasets.
    """
    assert len(means) == len(stds) and len(stds) == len(sample_sizes)
    # Total number of samples
    total_samples = np.sum(sample_sizes)

    # Combined mean
    combined_mean = np.sum(sample_sizes * means) / total_samples

    # Combined variance calculation
    variance_terms = (sample_sizes - ddof) * (stds**2)
    mean_diff_terms = sample_sizes * ((means - combined_mean) ** 2)
    combined_variance = (np.sum(variance_terms) + np.sum(mean_diff_terms)) / (
        total_samples - ddof
    )

    return combined_variance


def combined_std(
    means: List[float], stds: List[float], sample_sizes: List[int]
) -> float:
    """
    Calculate the combined std for multiple datasets.
    """
    assert len(means) == len(stds) and len(stds) == len(sample_sizes)
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    sample_sizes_arr = np.array(sample_sizes)

    cv = combined_variance(means_arr, stds_arr, sample_sizes_arr)
    return np.sqrt(cv)


def combined_mean(means: List[float], sample_sizes: List[int]) -> float:
    """
    Calculate the combined mean for multiple datasets.
    """
    assert len(means) == len(sample_sizes)
    means_arr = np.array(means)
    sample_sizes_arr = np.array(sample_sizes)

    return np.sum(sample_sizes_arr * means_arr) / np.sum(sample_sizes_arr)


def combined_rms(
    means: List[float], stds: List[float], sample_sizes: List[int]
) -> float:
    """
    Calculate the combined RMS for multiple datasets.
    """
    assert len(means) == len(stds) and len(stds) == len(sample_sizes)
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    sample_sizes_arr = np.array(sample_sizes)

    cm = combined_mean(means, sample_sizes)
    cv = combined_variance(means_arr, stds_arr, sample_sizes_arr)

    # Combined RMS calculation
    return np.sqrt(cm**2 + cv)


class SevenNetMultiModalDataset(ConcatDataset):
    def __init__(
        self,
        modal_dataset_dict: Dict[str, Dataset],
    ):
        datasets = []
        modals = []
        for modal, dataset in modal_dataset_dict.items():
            modals.append(modal)
            datasets.append(dataset)
        self.modals = modals
        super().__init__(datasets)

    def __getitem__(self, idx):
        graph = super().__getitem__(idx)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        modality = self.modals[dataset_idx]
        graph[KEY.DATA_MODALITY] = modality
        return graph

    def _modal_wise_property(self, attribute_name: str):
        dct = {}
        for modal, dataset in zip(self.modals, self.datasets):
            try:
                if hasattr(dataset, attribute_name):
                    dct[modal] = getattr(dataset, attribute_name)
            except AttributeError:
                dct[modal] = None
        return dct

    @property
    def dataset_dict(self):
        arr = {}
        for idx, modality in enumerate(self.modals):
            arr[modality] = self.datasets[idx]
        return arr

    @property
    def species(self):
        dct = self._modal_wise_property('species')
        tot = set()
        for sp in dct.values():
            tot.update(sp)
        dct['total'] = list(tot)
        return dct

    @property
    def natoms(self):
        return self._modal_wise_property('natoms')

    @property
    def per_atom_energy_mean(self):
        dct = self._modal_wise_property('per_atom_energy_mean')
        try:
            means = []
            sample_sizes = []
            for modality, mean in dct.items():
                means.append(mean)
                sample_sizes.append(
                    self.statistics[modality][KEY.PER_ATOM_ENERGY]['count']
                )
            cm = combined_mean(means, sample_sizes)
            dct['total'] = cm
        except KeyError:
            pass
        return dct

    @property
    def elemwise_reference_energies(self):
        # total is not supported (it is expensive and complex, but useless)
        return self._modal_wise_property('elemwise_reference_energies')

    @property
    def force_rms(self):
        dct = self._modal_wise_property('force_rms')
        try:
            means = []
            sample_sizes = []
            stds = []
            for modality in dct:
                means.append(self.statistics[modality][KEY.FORCE]['mean'])
                sample_sizes.append(self.statistics[modality][KEY.FORCE]['count'])
                stds.append(self.statistics[modality][KEY.FORCE]['std'])
            cm = combined_rms(means, stds, sample_sizes)
            dct['total'] = cm
        except KeyError:
            pass
        return dct

    @property
    def per_atom_energy_std(self):
        dct = self._modal_wise_property('per_atom_energy_std')
        try:
            means = []
            sample_sizes = []
            stds = []
            for modality in dct:
                means.append(self.statistics[modality][KEY.PER_ATOM_ENERGY]['mean'])
                sample_sizes.append(
                    self.statistics[modality][KEY.PER_ATOM_ENERGY]['count']
                )
                stds.append(self.statistics[modality][KEY.PER_ATOM_ENERGY]['std'])
            cm = combined_std(means, stds, sample_sizes)
            dct['total'] = cm
        except KeyError:
            pass
        return dct

    @property
    def avg_num_neigh(self):
        dct = self._modal_wise_property('avg_num_neigh')
        try:
            means = []
            sample_sizes = []
            for modality, mean in dct.items():
                means.append(mean)
                sample_sizes.append(
                    self.statistics[modality]['num_neighbor']['count']
                )
            cm = combined_mean(means, sample_sizes)
            dct['total'] = cm
        except KeyError:
            pass
        return dct

    @property
    def sqrt_avg_num_neigh(self):
        avg_nn = self.avg_num_neigh
        return {k: v**0.5 for k, v in avg_nn.items()}

    @property
    def statistics(self):
        return self._modal_wise_property('statistics')

    @staticmethod
    def as_graph_dataset(
        paths: List[dict],
        **graph_dataset_kwargs,
    ):
        import sevenn.train.graph_dataset as gd

        modal_paths = _arrange_paths_by_modality(paths)
        dataset_dct = {}
        for modality, paths in modal_paths.items():
            kwargs = deepcopy(graph_dataset_kwargs)
            if (dataset := gd.from_single_path(paths, **kwargs)) is None:
                pname = kwargs.pop('processed_name', 'graph').replace('.pt', '')
                dataset = gd.SevenNetGraphDataset(
                    files=paths,
                    processed_name=f'{pname}_{modality}.pt',
                    **kwargs,
                )
            dataset_dct[modality] = dataset
        return SevenNetMultiModalDataset(dataset_dct)


def from_config(
    config: Dict[str, Any],
    working_dir: str = os.getcwd(),
    dataset_keys: Optional[List[str]] = None,
):
    log = Logger()
    if dataset_keys is None:
        dataset_keys = [
            k for k in config if (k.startswith('load_') and k.endswith('_path'))
        ]

    if KEY.LOAD_TRAINSET not in dataset_keys:
        raise ValueError(f'{KEY.LOAD_TRAINSET} must be present in config')

    dataset_args = {
        'cutoff': config[KEY.CUTOFF],
        'root': working_dir,
        'process_num_cores': config.get(KEY.PREPROCESS_NUM_CORES, 1),
        'use_data_weight': config.get(KEY.USE_WEIGHT, False),
        **config[KEY.DATA_FORMAT_ARGS],
    }

    datasets = {}
    for dk in dataset_keys:
        if not (paths := config[dk]):
            continue
        if isinstance(paths, str):
            paths = [paths]
        name = '_'.join([nn.strip() for nn in dk.split('_')[1:-1]])
        dataset_args.update({'processed_name': name})
        datasets[name] = SevenNetMultiModalDataset.as_graph_dataset(
            paths,  # type: ignore
            **dataset_args,
        )

    train_set = datasets['trainset']

    modals_dataset = set()
    chem_species = set()
    # print statistics of each dataset
    for name, dataset in datasets.items():
        for idx, modality in enumerate(dataset.modals):
            log.bar()
            log.writeline(f'{name} - {modality} distribution:')
            log.statistic_write(dataset.statistics[modality])
            log.format_k_v(
                '# structures (graph)', len(dataset.datasets[idx]), write=True
            )
            modals_dataset.update([modality])
        chem_species.update(dataset.species['total'])
    log.bar()

    if (modal_map := config.get(KEY.MODAL_MAP, None)) is None:
        modals = sorted(list(modals_dataset))
        modal_map = {modal: i for i, modal in enumerate(modals)}
        config[KEY.MODAL_MAP] = modal_map

    modals = list(modal_map.keys())
    if not modals_dataset.issubset(modal_map):
        raise ValueError(
            f'Found modalities in datasets: {modals_dataset} are not subset of'
            + f' {modals}. Use sevenn_cp tool to append/assign modality'
        )

    log.writeline(f'Modalities of this model: {modals}')

    config[KEY.NUM_MODALITIES] = len(modal_map)

    # initialize known species from dataset if 'auto'
    # sorted to alphabetical order (which is same as before)
    chem_keys = [KEY.CHEMICAL_SPECIES, KEY.NUM_SPECIES, KEY.TYPE_MAP]
    if all([config[ck] == 'auto' for ck in chem_keys]):  # see parse_input.py
        log.writeline('Known species are obtained from the dataset')
        config.update(util.chemical_species_preprocess(sorted(list(chem_species))))

    # retrieve shift, scale, conv_denominaotrs from user input (keyword)
    init_from_stats_candid = [KEY.SHIFT, KEY.SCALE, KEY.CONV_DENOMINATOR]
    init_from_stats = [
        k for k in init_from_stats_candid if isinstance(config[k], str)
    ]

    for k in init_from_stats:
        input = config[k]
        if not hasattr(train_set, input):
            raise NotImplementedError(input)
        modal_stat = getattr(train_set, input)
        try:
            if k == KEY.CONV_DENOMINATOR and 'total' in modal_stat:
                # conv_denominator is not modal-wise
                var = modal_stat['total']
            elif k == KEY.SHIFT and config[KEY.USE_MODAL_WISE_SHIFT]:
                modal_stat.pop('total', None)
                var = modal_stat
            elif k == KEY.SHIFT and not config[KEY.USE_MODAL_WISE_SHIFT]:
                var = modal_stat['total']
            elif k == KEY.SCALE and config[KEY.USE_MODAL_WISE_SCALE]:
                modal_stat.pop('total', None)
                var = modal_stat
            elif k == KEY.SCALE and not config[KEY.USE_MODAL_WISE_SCALE]:
                var = modal_stat['total']
            else:
                raise NotImplementedError(f'Failed to init {k} from statistics')
        except KeyError as e:
            if e.args[0] == 'total':
                raise NotImplementedError(
                    f'{k}: {input} does not support total statistics. '
                    + f'Set use_modal_wise_{k} True or specify numbers manually'
                )
            else:
                raise e
        config.update({k: var})
        log.writeline(f'{k} is obtained from statistics')

    return datasets
