import logging
import os
import uuid
from collections import Counter
from copy import deepcopy
from typing import Literal

import ase.calculators.singlepoint as singlepoint
import ase.io
import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from torch_geometric.loader import DataLoader

import sevenn.train.dataload as dl
import sevenn.train.graph_dataset as ds
from sevenn._const import NUM_UNIV_ELEMENT
from sevenn.atom_graph_data import AtomGraphData

cutoff = 4.0

_samples = {
    'bulk': bulk('NaCl', 'rocksalt', a=5.63),
    'mol': molecule('H2O'),
    'isolated': molecule('H'),
}


_nedges_c4 = {'bulk': 36, 'mol': 6, 'isolated': 0}


def get_atoms(
    atoms_type: Literal['bulk', 'mol', 'isolated'],
    init_y_as: Literal['calc', 'info', 'none'],
):
    """
    Return atoms w, w/o reference values with its
    # of edges for 4.0 cutoff length
    """
    assert atoms_type in _samples
    atoms = deepcopy(_samples[atoms_type])
    natoms = len(atoms)
    if init_y_as == 'calc':
        results = {
            'energy': np.random.rand(1),
            'forces': np.random.rand(natoms, 3),
            'stress': np.random.rand(6),
        }
        if not atoms.pbc.all():
            del results['stress']
        calc = singlepoint.SinglePointCalculator(atoms, **results)
        atoms = calc.get_atoms()
    elif init_y_as == 'info':
        atoms.info['y_energy'] = np.random.rand(1)
        atoms.arrays['y_force'] = np.random.rand(natoms, 3)
        atoms.info['y_stress'] = np.random.rand(6)
        if not atoms.pbc.all():
            del atoms.info['y_stress']
    return atoms, _nedges_c4[atoms_type]


@pytest.mark.parametrize('init_y_as', ['calc', 'info'])
@pytest.mark.parametrize('atoms_type', ['bulk', 'mol', 'isolated'])
def test_atoms_to_graph(atoms_type, init_y_as):
    atoms, nedges = get_atoms(atoms_type, init_y_as)
    is_stress = atoms.pbc.all()
    y_from_calc = init_y_as == 'calc'

    graph = dl.atoms_to_graph(atoms, cutoff=cutoff, y_from_calc=y_from_calc)

    essential = {
        'atomic_numbers': ((len(atoms),), int),
        'pos': ((len(atoms), 3), float),
        'edge_index': ((2, nedges), int),
        'edge_vec': ((nedges, 3), float),
        'total_energy': ((), float),
        'force_of_atoms': ((len(atoms), 3), float),
        'cell_volume': ((), float),
        'num_atoms': ((), int),
        'per_atom_energy': ((), float),
        'stress': ((1, 6), float),
    }

    for k, (shape, dtype) in essential.items():
        assert k in graph, f'{k} missing in graph'
        assert isinstance(
            graph[k], np.ndarray
        ), f'{k}: {type(graph[k])} is not np.ndarray'
        assert graph[k].shape == shape, f'{k} shape {graph[k].shape} != {shape}'
        if not is_stress and k == 'stress':
            assert np.isnan(graph[k]).all()
        else:
            assert graph[k].dtype == dtype, f'{k} dtype {graph[k].dtype} != {dtype}'

    assert graph['per_atom_energy'] == (graph['total_energy'] / len(atoms))
    assert graph['num_atoms'] == len(atoms)
    if not is_stress:
        assert graph['cell_volume'] == np.finfo(float).eps


@pytest.mark.parametrize('atoms_type', ['bulk', 'mol', 'isolated'])
def test_unlabeled_atoms_to_graph(atoms_type):
    atoms, nedges = get_atoms(atoms_type, 'none')

    graph = dl.unlabeled_atoms_to_graph(atoms, cutoff=cutoff)

    essential = {
        'atomic_numbers': ((len(atoms),), int),
        'pos': ((len(atoms), 3), float),
        'edge_index': ((2, nedges), int),
        'edge_vec': ((nedges, 3), float),
        'cell_volume': ((), float),
        'num_atoms': ((), int),
    }

    for k, (shape, dtype) in essential.items():
        assert k in graph, f'{k} missing in graph'
        assert isinstance(
            graph[k], np.ndarray
        ), f'{k}: {type(graph[k])} is not np.ndarray'
        assert graph[k].dtype == dtype, f'{k} dtype {graph[k].dtype} != {dtype}'
        assert graph[k].shape == shape, f'{k} shape {graph[k].shape} != {shape}'

    assert graph['num_atoms'] == len(atoms)
    if not atoms.pbc.all():
        assert graph['cell_volume'] == np.finfo(float).eps


@pytest.mark.parametrize('init_y_as', ['calc', 'info'])
@pytest.mark.parametrize('atoms_type', ['bulk', 'mol', 'isolated'])
def test_atom_graph_data(atoms_type, init_y_as):
    atoms, nedges = get_atoms(atoms_type, init_y_as)
    y_from_calc = init_y_as == 'calc'
    is_stress = atoms.pbc.all()
    np_graph = dl.atoms_to_graph(atoms, cutoff=cutoff, y_from_calc=y_from_calc)
    graph = AtomGraphData.from_numpy_dict(np_graph)

    essential = {
        'atomic_numbers': ((len(atoms),), int),
        'edge_index': ((2, nedges), int),
        'edge_vec': ((nedges, 3), float),
    }
    auxilaray = {
        'x': ((len(atoms),), int),
        'pos': ((len(atoms), 3), float),
        'num_atoms': ((), int),
        'cell_volume': ((), float),
        'total_energy': ((), float),
        'per_atom_energy': ((), float),
        'force_of_atoms': ((len(atoms), 3), float),
        'stress': ((1, 6), float),
    }

    for k, (shape, dtype) in essential.items():
        assert k in graph, f'{k} missing in graph'
        assert isinstance(
            graph[k], torch.Tensor
        ), f'{k}: {type(graph[k])} is not an tensor'
        assert graph[k].is_floating_point() == (dtype is float)
        assert graph[k].shape == shape, f'{k} shape {graph[k].shape} != {shape}'

    for k, (shape, dtype) in auxilaray.items():
        if k not in graph:
            continue
        assert isinstance(
            graph[k], torch.Tensor
        ), f'{k}: {type(graph[k])} is not an tensor'
        assert graph[k].shape == shape, f'{k} shape {graph[k].shape} != {shape}'
        if not is_stress and k == 'stress':
            assert torch.isnan(graph[k]).all()
        else:
            assert graph[k].is_floating_point() == (dtype is float)


def test_graph_build():
    """
    Compare parallel implementation, should preserve order
    """
    atoms_list = [
        get_atoms(t, 'calc')[0]  # type: ignore
        for t in list(_samples.keys())
    ]
    one_core = dl.graph_build(atoms_list, cutoff, num_cores=1, y_from_calc=True)
    two_core = dl.graph_build(atoms_list, cutoff, num_cores=2, y_from_calc=True)

    assert len(one_core) == len(two_core)
    for g1, g2 in zip(one_core, two_core):
        assert set(g1.keys()) == set(g2.keys())
        for k in g1.keys():
            if not isinstance(g1[k], torch.Tensor):
                continue
            if k == 'stress':  # TODO: robust way to test it
                assert torch.allclose(g1[k], g2[k]) or (
                    torch.isnan(g1[k]).all() == torch.isnan(g2[k]).all()
                )
            else:
                assert torch.allclose(g1[k], g2[k])


@pytest.fixture(scope='module')
def graph_dataset_tuple():
    tmpdir = os.getenv('TMPDIR', '/tmp')
    randstr = uuid.uuid4().hex
    assert os.access(tmpdir, os.W_OK), f'{tmpdir} is not writable'

    root = tmpdir
    files = f'{root}/{randstr}.extxyz'
    atoms_list = [
        get_atoms(atype, 'calc')[0]  # type: ignore
        for atype in ['bulk', 'mol', 'isolated']
    ]
    ase.io.write(files, atoms_list, 'extxyz')

    dataset = ds.SevenNetGraphDataset(
        cutoff=cutoff,
        root=root,
        files=files,
        processed_name=f'{randstr}.pt',
    )
    assert os.path.isfile(f'{root}/sevenn_data/{randstr}.pt'), 'dataset not written'
    return dataset, atoms_list


def test_sevenn_graph_dataset_properties(graph_dataset_tuple):
    dataset, atoms_list = graph_dataset_tuple

    species = set()
    natoms = Counter()
    elist = []
    e_per_list = []
    flist = []
    slist = []
    for at in atoms_list:
        chems = at.get_chemical_symbols()
        species.update(chems)
        natoms.update(chems)
        elist.append(at.get_potential_energy())
        e_per_list.append(at.get_potential_energy() / len(at))
        flist.extend(at.get_forces())
        try:
            slist.append(at.get_stress())
        except NotImplementedError:
            slist.append(np.full(6, np.nan))

    elist = np.array(elist)
    e_per_list = np.array(e_per_list)
    flist = np.array(flist)
    slist = np.array(slist)

    natoms['total'] = sum([cnt for cnt in list(natoms.values())])

    assert set(dataset.species) == species
    assert dataset.natoms == natoms
    assert np.allclose(dataset.per_atom_energy_mean, e_per_list.mean())
    assert np.allclose(dataset.force_rms, np.sqrt((flist**2).mean()))


def test_sevenn_graph_dataset_elemwise_energies(graph_dataset_tuple):
    logger = logging.getLogger(__name__)

    dataset, atoms_list = graph_dataset_tuple

    ref_e = dataset.elemwise_reference_energies
    assert len(ref_e) == NUM_UNIV_ELEMENT
    z_set = set()
    for atoms in atoms_list:
        inferred_e = 0
        atomic_numbers = atoms.get_atomic_numbers()
        z_set.update(atomic_numbers)
        for z in atomic_numbers:
            inferred_e += ref_e[z]
        # it never be same, but should be similar
        logger.info('elemwise energy should be similar:')
        logger.info(f'{inferred_e:4f} {atoms.get_potential_energy()[0]:4f}')

    for z in range(NUM_UNIV_ELEMENT):
        if z not in z_set:
            assert ref_e[z] == 0


# def test_sevenn_graph_dataset_num_neigh(graph_dataset_tuple):
#    pass


def test_sevenn_graph_dataset_statistics(graph_dataset_tuple):
    dataset, atoms_list = graph_dataset_tuple

    elist = []
    e_per_list = []
    flist = []
    slist = []
    for at in atoms_list:
        elist.append(at.get_potential_energy())
        e_per_list.append(at.get_potential_energy() / len(at))
        flist.extend(at.get_forces())
        try:
            slist.append(at.get_stress())
        except NotImplementedError:
            slist.append(np.full(6, np.nan))

    dct = {
        'total_energy': np.array(elist),
        'per_atom_energy': np.array(e_per_list),
        'force_of_atoms': np.array(flist).flatten(),
        # 'stress': np.array(slist),  # TODO: it may have nan
    }

    for key in dct:
        assert np.allclose(dataset.statistics[key]['mean'], dct[key].mean()), key
        assert np.allclose(dataset.statistics[key]['std'], dct[key].std(ddof=0)), key
        assert np.allclose(
            dataset.statistics[key]['median'], np.median(dct[key])
        ), key
        assert np.allclose(dataset.statistics[key]['max'], dct[key].max()), key
        assert np.allclose(dataset.statistics[key]['min'], dct[key].min()), key


@pytest.mark.parametrize(
    'a_types,init_ys', [(['bulk', 'mol', 'isolated'], ['calc', 'calc', 'calc'])]
)
def test_7net_graph_dataset_batch_shape(a_types, init_ys, tmp_path):
    assert len(a_types) == len(init_ys)
    n_graph = len(a_types)
    atoms_list = []
    tot_edges = 0
    tot_atoms = 0
    for a_type, init_y in zip(a_types, init_ys):
        atoms, n_edge = get_atoms(a_type, init_y)
        tot_edges += n_edge
        tot_atoms += len(atoms)
        atoms_list.append(atoms)
    ase.io.write(tmp_path / 'tmp', atoms_list, format='extxyz')
    dataset = ds.SevenNetGraphDataset(cutoff, tmp_path, str(tmp_path / 'tmp'))
    loader = DataLoader(dataset, batch_size=n_graph)
    graph = next(iter(loader))

    essential = {
        'x': ((tot_atoms,), int),
        'atomic_numbers': ((tot_atoms,), int),
        'pos': ((tot_atoms, 3), float),
        'edge_index': ((2, tot_edges), int),
        'edge_vec': ((tot_edges, 3), float),
        'total_energy': ((n_graph,), float),
        'force_of_atoms': ((tot_atoms, 3), float),
        'cell_volume': ((n_graph,), float),
        'num_atoms': ((n_graph,), int),
        'per_atom_energy': ((n_graph,), float),
        'stress': ((n_graph, 6), float),
        'batch': ((tot_atoms,), int),  # from PyG
    }

    for k, (shape, dtype) in essential.items():
        assert k in graph, f'{k} missing in graph'
        assert isinstance(
            graph[k], torch.Tensor
        ), f'{k}: {type(graph[k])} is not an tensor'
        assert graph[k].is_floating_point() == (dtype is float)
        assert graph[k].shape == shape, f'{k} shape {graph[k].shape} != {shape}'
