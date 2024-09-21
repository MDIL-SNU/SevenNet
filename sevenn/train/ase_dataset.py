from functools import partial
from typing import Any, Callable, Optional

import ase.db.row
import numpy as np
from ase.db.core import parse_selection
from ase.db.sqlite import SQLite3Database
from torch.utils.data import IterableDataset, get_worker_info

import sevenn.train.dataload as dataload
from sevenn.atom_graph_data import AtomGraphData


class _AtomsSQLite3DatabaseLazy(SQLite3Database):

    def lazy_selection(self, selection=None, limit=None, offset=0,
                       sort=None, include_data=True, columns='all',
                       batch=12, filter_fn=None, **kwargs):
        keys, cmps = parse_selection(selection, **kwargs)
        values = np.array([None for _ in range(27)])
        values[25] = '{}'
        values[26] = 'null'
        if columns == 'all':
            columnindex = list(range(26))
        else:
            columnindex = [c for c in range(26)
                           if self.columnnames[c] in columns]
        if include_data:
            columnindex.append(26)
        if sort:
            if sort[0] == '-':
                order = 'DESC'
                sort = sort[1:]
            else:
                order = 'ASC'
            if sort in ['id', 'energy', 'username', 'calculator',
                        'ctime', 'mtime', 'magmom', 'pbc',
                        'fmax', 'smax', 'volume', 'mass', 'charge', 'natoms']:
                sort_table = 'systems'
            else:
                for dct in self._select(keys + [sort], cmps=[], limit=1,
                                        include_data=False,
                                        columns=['key_value_pairs']):
                    if isinstance(dct['key_value_pairs'][sort], str):
                        sort_table = 'text_key_values'
                    else:
                        sort_table = 'number_key_values'
                    break
                else:
                    # No rows.  Just pick a table:
                    sort_table = 'number_key_values'
        else:
            order = None
            sort_table = None
        what = ', '.join('systems.' + name
                         for name in
                         np.array(self.columnnames)[np.array(columnindex)])
        sql, args = self.create_select_statement(keys, cmps, sort, order,
                                                 sort_table, what)
        if limit:
            sql += f'\nLIMIT {limit}'
        if offset:
            sql += self.get_offset_string(offset, limit=limit)

        with self.managed_connection() as con:
            cur = con.cursor()
            cur.execute(sql, args)
            # while (shortvalues := cur.fetchone()) is not None:
            buffer = []
            while (shortvalues_list := cur.fetchmany(batch * 10)) is not None:
                for shortvalues in shortvalues_list:
                    values[columnindex] = shortvalues
                    row = self._convert_tuple_to_row(tuple(values))
                    if filter_fn is None or filter_fn(row):
                        buffer.append(row)
                    if len(buffer) == batch:
                        yield buffer
                        buffer = []
            if buffer:
                yield buffer


def _default_atoms_to_graph(atoms, cutoff):
    np_dct = dataload.atoms_to_graph(
        atoms, cutoff, transfer_info=False, y_from_calc=True
    )
    return AtomGraphData.from_numpy_dict(np_dct)


class AtomsSQLite3Dataset(IterableDataset):
    """
    Iterable dataset that yields AtomGraphData from ase db
    Currently only for sqlite3 ase db

    Args:
        db_path: path to ase.db
        cutoff: cutoff in float, if given, will be passed to atoms_to_graph_fn
        selection: ASE db selection
        atoms_to_graph: callable that converts atoms to PyG data (numpy dict)
        atoms_row_transform: callable for atoms_row
        selection_kwargs:
    """

    def __init__(
        self,
        db_path: str,
        cutoff: Optional[float] = None,
        selection: Any = None,
        atoms_to_graph: Callable = _default_atoms_to_graph,
        atoms_row_transform: Optional[Callable] = None,
        record_ids: bool = True,
        **selection_kwargs,
    ):
        super().__init__()

        self.db = _AtomsSQLite3DatabaseLazy(db_path)
        self._db_path = db_path
        if cutoff is None and atoms_to_graph == _default_atoms_to_graph:
            raise ValueError('Default atoms_to_graph requires cutoff!')
        self.cutoff = cutoff

        self.selection = selection
        self.selection_kwargs = selection_kwargs

        if cutoff is not None:
            atoms_to_graph = partial(atoms_to_graph, cutoff=cutoff)
        self.atoms_to_graph = atoms_to_graph
        self.atom_row_transform = atoms_row_transform

        self.sel = self.db.lazy_selection(**self.selection_kwargs)

        self.record_ids = record_ids
        self.ids = []

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            fetch_n = 1
            me = 0
        else:
            fetch_n = worker_info.num_workers
            me = worker_info.id
        self.selection_kwargs.update({'batch': fetch_n})
        self.new_select(self.selection, **self.selection_kwargs)

        try:
            while True:
                # TODO: this is naive way to read dataset.
                # All worker reads data at the same time with redundancy
                atoms_row = next(self.sel)
                if len(atoms_row) < me + 1:
                    raise StopIteration()
                atoms_row = atoms_row[me]

                if self.record_ids:
                    self.ids.append(atoms_row.id)

                if self.atom_row_transform is not None:
                    atoms_row = self.atom_row_transform(atoms_row)

                if isinstance(atoms_row, ase.db.row.AtomsRow):
                    atoms = atoms_row.toatoms()
                elif isinstance(atoms_row, ase.Atoms):
                    atoms = atoms_row
                else:
                    raise ValueError()

                yield self.atoms_to_graph(atoms)
        except StopIteration:
            print('stop called')
            pass

    def new_select(self, selection, **selection_kwargs):
        if self.sel is not None:
            self.sel.close()

        self.selection = selection
        self.selection_kwargs = selection_kwargs
        self.sel = self.db.lazy_selection(selection=selection, **selection_kwargs)
