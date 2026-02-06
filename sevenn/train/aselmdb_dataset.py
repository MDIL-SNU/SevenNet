from __future__ import annotations

import os
import os.path as osp
from glob import glob
import time
import warnings
from collections import Counter
from pathlib import Path
import bisect
import zlib
import typing
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch.distributed as dist
import lmdb
import orjson

import ase
from ase.data import chemical_symbols
from ase.db.core import Database, now, ops
from ase.db.row import AtomsRow
from tqdm import tqdm

import sevenn._keys as KEY
import sevenn.util as util
from sevenn._const import NUM_UNIV_ELEMENT
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.atoms_dataset import SevenNetAtomsDataset
from sevenn.train.dataload import _set_atoms_y



class LMDBDatabase(Database):
    """
    Same class develop by Meta

    Copyright (c) Meta, Inc. and its affiliates.

    This source code is modified from the ASE db json backend
    and is thus licensed under the corresponding LGPL2.1 license

    The ASE notice for the LGPL2.1 license is available here:
    https://gitlab.com/ase/ase/-/blob/master/LICENSE
    """
    def __init__(
        self,
        filename: str | Path | None = None,
        create_indices: bool = True,
        use_lock_file: bool = False,
        serial: bool = False,
        readonly: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        For the most part, this is identical to the standard ase db initiation
        arguments, except that we add a readonly flag.
        """
        super().__init__(
            Path(filename),
            create_indices,
            use_lock_file,
            serial,
            *args,
            **kwargs,
        )

        # Add a readonly mode for when we're only training
        # to make sure there's no parallel locks
        self.readonly = readonly

        if self.readonly:
            # Open a new env
            self.env = lmdb.open(
                str(self.filename),
                subdir=False,
                meminit=False,
                map_async=True,
                readonly=True,
                lock=False,
            )

            # Open a transaction and keep it open for fast read/writes!
            self.txn = self.env.begin(write=False)

        else:
            # Open a new env with write access
            self.env = lmdb.open(
                str(self.filename),
                map_size=1099511627776 * 2,
                subdir=False,
                meminit=False,
                map_async=True,
            )

            self.txn = self.env.begin(write=True)

        # Load all ids based on keys in the DB.
        self.ids = []
        self.deleted_ids = []
        self._load_ids()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self.close()

    def close(self) -> None:
        # Close the lmdb environment and transaction
        self.txn.commit()
        self.env.close()

    def _write(
        self,
        atoms: ase.Atoms | AtomsRow,
        key_value_pairs: dict,
        data: dict | None,
        idx: int | None = None,
    ) -> None:
        Database._write(self, atoms, key_value_pairs, data)

        mtime = now()

        if isinstance(atoms, AtomsRow):
            row = atoms
        else:
            row = AtomsRow(atoms)
            row.ctime = mtime
            row.user = os.getenv("USER")

        dct = {}
        for key in row.__dict__:
            if key[0] == "_" or key in row._keys or key == "id":
                continue
            dct[key] = row[key]

        dct["mtime"] = mtime

        if key_value_pairs:
            dct["key_value_pairs"] = key_value_pairs

        if data:
            dct["data"] = data

        constraints = row.get("constraints")
        if constraints:
            dct["constraints"] = [constraint.todict() for constraint in constraints]

        # json doesn't like Cell objects, so make it an array
        dct["cell"] = np.asarray(dct["cell"])

        if idx is None:
            idx = self._nextid
            nextid = idx + 1
        else:
            data = self.txn.get(f"{idx}".encode("ascii"))
            assert data is not None

        # Add the new entry
        self.txn.put(
            f"{idx}".encode("ascii"),
            zlib.compress(orjson.dumps(dct, option=orjson.OPT_SERIALIZE_NUMPY)),
        )
        # only append if idx is not in ids
        if idx not in self.ids:
            self.ids.append(idx)
            self.txn.put(
                "nextid".encode("ascii"),
                zlib.compress(orjson.dumps(nextid, option=orjson.OPT_SERIALIZE_NUMPY)),
            )
        # check if id is in removed ids and remove accordingly
        if idx in self.deleted_ids:
            self.deleted_ids.remove(idx)
            self._write_deleted_ids()

        return idx

    def _update(
        self,
        idx: int,
        key_value_pairs: dict | None = None,
        data: dict | None = None,
    ):
        # hack this to play nicely with ASE code
        row = self._get_row(idx, include_data=True)
        if data is not None or key_value_pairs is not None:
            self._write(atoms=row, idx=idx, key_value_pairs=key_value_pairs, data=data)

    def _write_deleted_ids(self):
        self.txn.put(
            "deleted_ids".encode("ascii"),
            zlib.compress(
                orjson.dumps(self.deleted_ids, option=orjson.OPT_SERIALIZE_NUMPY)
            ),
        )

    def delete(self, ids: list[int]) -> None:
        for idx in ids:
            self.txn.delete(f"{idx}".encode("ascii"))
            self.ids.remove(idx)

        self.deleted_ids += ids
        self._write_deleted_ids()

    def _get_row(self, idx: int, include_data: bool = True):
        if idx is None:
            assert len(self.ids) == 1
            idx = self.ids[0]
        data = self.txn.get(f"{idx}".encode("ascii"))

        if data is not None:
            dct = orjson.loads(zlib.decompress(data))
        else:
            raise KeyError(f"Id {idx} missing from the database!")

        if not include_data:
            dct.pop("data", None)

        dct["id"] = idx
        return AtomsRow(dct)

    def _get_row_by_index(self, index: int, include_data: bool = True):
        """Auxiliary function to get the ith entry, rather than a specific id"""
        data = self.txn.get(f"{self.ids[index]}".encode("ascii"))

        if data is not None:
            dct = orjson.loads(zlib.decompress(data))
        else:
            raise KeyError(f"Id {id} missing from the database!")

        if not include_data:
            dct.pop("data", None)

        dct["id"] = id
        return AtomsRow(dct)

    def _select(
        self,
        keys,
        cmps: list[tuple[str, str, str]],
        explain: bool = False,
        verbosity: int = 0,
        limit: int | None = None,
        offset: int = 0,
        sort: str | None = None,
        include_data: bool = True,
        columns: str = "all",
    ):
        if explain:
            yield {"explain": (0, 0, 0, "scan table")}
            return

        if sort is not None:
            if sort[0] == "-":
                reverse = True
                sort = sort[1:]
            else:
                reverse = False

            rows = []
            missing = []
            for row in self._select(keys, cmps):
                key = row.get(sort)
                if key is None:
                    missing.append((0, row))
                else:
                    rows.append((key, row))

            rows.sort(reverse=reverse, key=lambda x: x[0])
            rows += missing

            if limit:
                rows = rows[offset : offset + limit]
            for _, row in rows:
                yield row
            return

        if not limit:
            limit = -offset - 1

        cmps = [(key, ops[op], val) for key, op, val in cmps]
        n = 0
        for idx in self.ids:
            if n - offset == limit:
                return
            row = self._get_row(idx, include_data=include_data)

            for key in keys:
                if key not in row:
                    break
            else:
                for key, op, val in cmps:
                    if isinstance(key, int):
                        value = np.equal(row.numbers, key).sum()
                    else:
                        value = row.get(key)
                        if key == "pbc":
                            assert op in [ops["="], ops["!="]]
                            value = "".join("FT"[x] for x in value)
                    if value is None or not op(value, val):
                        break
                else:
                    if n >= offset:
                        yield row
                    n += 1

    @property
    def metadata(self):
        """Load the metadata from the DB if present"""
        if self._metadata is None:
            metadata = self.txn.get("metadata".encode("ascii"))
            if metadata is None:
                self._metadata = {}
            else:
                self._metadata = orjson.loads(zlib.decompress(metadata))

        return self._metadata.copy()

    @metadata.setter
    def metadata(self, dct):
        self._metadata = dct

        # Put the updated metadata dictionary
        self.txn.put(
            "metadata".encode("ascii"),
            zlib.compress(orjson.dumps(dct, option=orjson.OPT_SERIALIZE_NUMPY)),
        )

    @property
    def _nextid(self):
        """Get the id of the next row to be written"""
        # Get the nextid
        nextid_data = self.txn.get("nextid".encode("ascii"))
        return orjson.loads(zlib.decompress(nextid_data)) if nextid_data else 1

    def count(self, selection=None, **kwargs) -> int:
        """Count rows.

        See the select() method for the selection syntax.  Use db.count() or
        len(db) to count all rows.
        """
        if selection is not None:
            n = 0
            for _row in self.select(selection, **kwargs):
                n += 1
            return n
        else:
            # Fast count if there's no queries! Just get number of ids
            return len(self.ids)

    def _load_ids(self) -> None:
        """Load ids from the DB

        Since ASE db ids are mostly 1-N integers, but can be missing entries
        if ids have been deleted. To save space and operating under the assumption
        that there will probably not be many deletions in most OCP datasets,
        we just store the deleted ids.
        """

        # Load the deleted ids
        deleted_ids_data = self.txn.get("deleted_ids".encode("ascii"))
        if deleted_ids_data is not None:
            self.deleted_ids = orjson.loads(zlib.decompress(deleted_ids_data))

        # Reconstruct the full id list
        self.ids = [i for i in range(1, self._nextid) if i not in set(self.deleted_ids)]


class AseDBDataset:
    """
    Modified, combined code for fairchem AseDBDataset w/o dependency
    """
    def __init__(
        self,
        src: Union[List[str], str],
    ):
        if isinstance(src, list):
            filepaths = []
            for path in src:
                if os.path.isdir(path):
                    filepaths.extend(glob(f"{path}/*"))
                elif os.path.isfile(path):
                    filepaths.append(path)
                else:
                    raise RuntimeError(f"Error reading dataset in {path}!")
        elif os.path.isfile(src):
            filepaths = [src]
        elif os.path.isdir(src):
            filepaths = glob(f'{src}/*')
        else:
            filepaths = glob(src)

        self.dbs = []

        for path in sorted(filepaths):
            try:
                self.dbs.append(self.connect_db(path))
            except ValueError:
                pass


        # In order to get all of the unique IDs using the default ASE db interface
        # we have to load all the data and check ids using a select. This is extremely
        # inefficient for large dataset. If the db we're using already presents a list of
        # ids and there is no query, we can just use that list instead and save ourselves
        # a lot of time!
        self.db_ids = []
        for db in self.dbs:
            if hasattr(db, "ids"):
                self.db_ids.append(db.ids)
            else:
                # this is the slow alternative
                self.db_ids.append([row.id for row in db.select()])

        idlens = [len(ids) for ids in self.db_ids]
        self._idlen_cumulative = np.cumsum(idlens).tolist()

        self.ids = list(range(sum(idlens)))
        self.num_samples = len(self.ids)

        if self.num_samples == 0:
            raise ValueError(f"No valid ase data found, check src {src}!")


    def __len__(self) -> int:
        return self.num_samples


    def connect_db(self, path: str) -> Database:
        if any(path.endswith(ext) for ext in ['aselmdb', 'lmdb']):
            return LMDBDatabase(path, readonly=True)

        return ase.db.connect(path)


    def get_atoms(self, idx: int) -> ase.Atoms:
        db_idx = bisect.bisect(self._idlen_cumulative, idx)

        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._idlen_cumulative[db_idx - 1]
        assert el_idx >= 0

        atoms_row = self.dbs[db_idx]._get_row(self.db_ids[db_idx][el_idx])
        atoms = atoms_row.toatoms()

        if isinstance(atoms_row.data, dict):
            atoms.info.update(atoms_row.data)

        return atoms


class SevenNetASElmdbDataset(SevenNetAtomsDataset):
    def __init__(
        self,
        cutoff: float,
        files: Union[str, List[str]],
        #sequence: Optional[List[int]] = None,
        stat_sequence_info: Union[str, float, int] = 10000,
        is_auto_mode: bool = False,
        atoms_filter: Optional[Callable] = None,  # not used yet
        atoms_transform: Optional[Callable] = None,
        graph_transform: Optional[Callable] = None,
        **process_kwargs,
    ):
        self.cutoff = cutoff
        if isinstance(files, str):
            files = [files]  # user convenience
        files = [osp.abspath(file) for file in files]

        self._files = files
        self.is_auto_mode = is_auto_mode
        self.atoms_filter = atoms_filter
        self.atoms_trasform = atoms_transform
        self.graph_trasform = graph_transform
        self._scanned = False
        self._avg_num_neigh_approx = None
        self.statistics = {}

        self._dataset = AseDBDataset(src=files)
        """
        if sequence:
            self.total_sequence = np.array(sequence)
        else:
            _seq = list(range(len(self._dataset)))
            np.random.shuffle(_seq)
            self.total_sequence = _seq
        """

        if isinstance(stat_sequence_info, str) and osp.exists(stat_sequence_info):
            self.stat_sequence = np.load(stat_sequence_info)

        elif isinstance(stat_sequence_info, int):
            #sample_num = min(len(self.total_sequence), stat_sequence_info)
            sample_num = min(len(self), stat_sequence_info)
            """
            self.stat_sequence = np.random.choice(
                self.total_sequence, sample_num, replace=False
            )
            """
            self.stat_sequence = np.random.choice(
                np.arange(len(self)), sample_num, replace=False
            )

        elif isinstance(stat_sequence_info, float):
            #sample_num = int(len(self.total_sequence) * stat_sequence_info)
            sample_num = int(len(self) * stat_sequence_info)
            """
            self.stat_sequence = np.random.choice(
                self.total_sequence, sample_num, replace=False
            )
            """
            self.stat_sequence = np.random.choice(
                np.arange(len(self)), sample_num, replace=False
            )

        else:
            raise ValueError(
                'stat_sequence_info should be one of str, int, float, '
                + f'but got {type(stat_sequence_info)}'
            )
        #self._run_sequence = self.total_sequence

    def __len__(self):
        # total, run_sequence deprecated.
        # Should only be used in OrderedSampler.__init__
        #return len(self._run_sequence)
        return len(self._dataset)

    def __getitem__(self, index):
        #idx = self._run_sequence[index]
        #atoms = self.set_atoms_y_with_idx(idx)
        atoms = self.set_atoms_y_with_idx(index)
        if self.atoms_trasform is not None:
            atoms = self.atoms_trasform(atoms)

        graph = self._graph_build(atoms)
        if self.graph_trasform is not None:
            graph = self.graph_trasform(graph)

        return AtomGraphData.from_numpy_dict(graph)

    def preload(self):
        print('Start preload..., fulfilling OS cache')
        print('Quick if it was already loaded into cache. Check "free -h"')
        print('May useless if the RAM is smaller than dataset size')
        print(f'Number of files: {len(self._dataset.dbs)}', flush=True)
        end = time.time()
        for lmdbdb in tqdm(self._dataset.dbs):
            lmdb_env = lmdbdb.env
            with lmdb_env.begin(write=False) as txn:
                with txn.cursor() as cursor:
                    if cursor.first():  # move cursor to start, skip 0 len db
                        while cursor.next():
                            _, _ = cursor.key(), cursor.value()
                            # dct = orjson.loads(zlib.decompress(v))
        print(f'Preload elapsed (sec): {time.time() - end:.4f}', flush=True)

    def set_atoms_y_with_idx(self, idx):
        atoms = self._dataset.get_atoms(idx)
        atoms = _set_atoms_y([atoms])[0]
        return atoms

    """
    def continue_from_data_progress(
        self,
        total_data_num: int = -1,
        current_data_index: int = 0,
        sequence: Optional[List[int]] = None,
    ):
        if total_data_num < 0:  # Nothing to continue
            return
        elif total_data_num != len(self._dataset):
            raise ValueError(
                'data_progress is not compatible with the dataset'
                + 'set reset_data_progress: True to fresh start'
            )
        if sequence is not None:
            assert len(sequence) == len(self._dataset)
            self.total_sequence = sequence
        self._truncated_sequence(current_data_index)

    def set_epoch(self, epoch: int, is_ddp: bool):
        '''
        Should be called before every epoch
        Mimic behavior of distributed sampler
        '''
        # TODO: rngkey things based on epoch?
        self.shuffle_sequence(is_ddp)
        self._run_sequence = self.total_sequence

    def _truncated_sequence(self, index):
        '''
        Used to start from middle of index (for continuing large-data training)
        Also changes __len__ of the dataset
        '''
        self._run_sequence = self.total_sequence[index:]

    def shuffle_sequence(self, broadcast=False):
        # ambiguous as both run_sequence and total_sequece can be random idx
        # assume total_sequnce is the one that is shuffled every epoch and
        # run_sequence is simlpy for truncation of total_sequence for continue
        shuffled = np.random.permutation(self.total_sequence)
        if broadcast:
            shuffled_bcast = [shuffled]
            dist.broadcast_object_list(shuffled_bcast, src=0)
            shuffled = shuffled_bcast[0]
        self.total_sequence = shuffled

    def save_sequence(self, filename):  # should not used
        np.save(filename, self.total_sequence)
    """
    @property
    def species(self):
        mode = (
            'total' if self.is_auto_mode else 'stat'
        )  # species should be fully scanned only in `auto` mode
        self.run_stat(mode=mode)
        return [z for z in self.statistics['_natoms'].keys() if z != 'total']

    @property
    def avg_num_neigh(self, n_sample=10000):
        if self._avg_num_neigh_approx is None:
            if len(self.stat_sequence) > n_sample:
                warnings.warn(
                    """SevenNetASElmdbDataset does not provide correct avg_num_neigh
                    as it does not build graph. We will compute only random 10000
                    structures graph to approximate this value. If you want more
                    precise avg_num_neigh, use SevenNetGraphDataset. If it is not
                    viable due to memory limit, you need online algorithm to do this
                    , which is not yet implemented in the SevenNet"""
                )
            n_sample = min(len(self.stat_sequence), n_sample)
            indices = np.random.choice(self.stat_sequence, n_sample, replace=False)
            n_neigh = []
            for i in indices:
                atoms = self.set_atoms_y_with_idx(i)
                graph = self._graph_build(atoms)
                _, nn = np.unique(graph[KEY.EDGE_IDX][0], return_counts=True)
                n_neigh.append(nn)
            n_neigh = np.concatenate(n_neigh)
            self._avg_num_neigh_approx = np.mean(n_neigh)
        return self._avg_num_neigh_approx

    def run_stat(self, mode='stat'):
        """
        Loop over dataset and init any statistics might need
        Unlink SevenNetGraphDataset, neighbors count is not computed as
        it requires to build graph
        """
        if self._scanned is True:
            return  # statistics already computed
        target_sequence = (
            #self.stat_sequence if mode == 'stat' else self.total_sequence
            self.stat_sequence if mode == 'stat' else np.arange(len(self))
        )
        y_keys: List[str] = [KEY.ENERGY, KEY.PER_ATOM_ENERGY, KEY.FORCE, KEY.STRESS]
        natoms_counter = Counter()
        composition = np.zeros((len(target_sequence), NUM_UNIV_ELEMENT))
        stats: Dict[str, Dict[str, Any]] = {y: {'_array': []} for y in y_keys}

        for i, atom_idx in enumerate(
            tqdm(target_sequence, desc='run_stat', total=len(target_sequence))
        ):
            atoms = self._dataset.get_atoms(atom_idx)
            atoms = _set_atoms_y([atoms])[0]
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


def _get_keys_from_config(config, start, end):
    keys = []
    for k in config:
        if k.startswith(start) and k.endswith(end):
            keys.append(k)
    return keys


def from_config(
    config: dict[str, Any],
    working_dir: str = os.getcwd(),
    dataset_keys: Optional[list[str]] = None,
    sequence_keys: Optional[list[str]] = None,
):
    from sevenn.sevenn_logger import Logger

    log = Logger()
    if dataset_keys is None:
        dataset_keys = _get_keys_from_config(config, 'load_', '_path')

    if sequence_keys is None:
        sequence_keys = _get_keys_from_config(config, 'load_', '_sequence')

    if KEY.LOAD_TRAINSET not in dataset_keys:
        raise ValueError(f'{KEY.LOAD_TRAINSET} must be present in config')

    # initialize arguments for loading dataset
    dataset_args = {
        'cutoff': config[KEY.CUTOFF],
        **config[KEY.DATA_FORMAT_ARGS],
    }

    chem_keys = [KEY.CHEMICAL_SPECIES, KEY.NUM_SPECIES, KEY.TYPE_MAP]
    is_auto_mode = all([config[ck] == 'auto' for ck in chem_keys])

    datasets: Dict[str, SevenNetASElmdbDataset] = {}
    for dk in dataset_keys:
        if not (paths := config[dk]):
            continue
        if isinstance(paths, str):
            paths = [paths]
        name = dk.split('_')[1].strip()
        sk = dk.replace('_path', '_sequence')

        total_sequence_path = None
        stat_sequence_info = 10000
        if sk in config:
            #total_sequence_path = config[sk].get('total_sequence_path', None)
            stat_sequence_info = config[sk].get('stat_sequence_info', None)
        dataset_args.update(
            {
                'files': paths,
                #'sequence_file': total_sequence_path,
                'stat_sequence_info': stat_sequence_info,
                'is_auto_mode': is_auto_mode,
            }
        )
        datasets[name] = SevenNetASElmdbDataset(**dataset_args)

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
        log.writeline(f'{name} distribution (may subsampled):')
        log.statistic_write(dataset.statistics)
        log.format_k_v('# atoms (node)', dataset.natoms, write=True)
        # log.format_k_v('# structures (graph)', len(dataset), write=True)
        log.format_k_v('# total structures in db', len(dataset._dataset), write=True)

        chem_species.update(dataset.species)
    log.bar()

    # initialize known species from dataset if 'auto'
    # sorted to alphabetical order (which is same as before)
    if is_auto_mode:  # see parse_input.py
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
