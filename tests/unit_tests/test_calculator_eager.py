"""
SevenNetCalculator eager calls (cuda_graph off) with LunaTP: the results equal
a model call on fresh device inputs in the same edge order, also when the
topology changes between consecutive calls (as GCMC), and the LunaTP graph is
built once per topology (the device edge index of an unchanged topology is
reused). Model: the omni-v2 checkpoint (train/omni/checkpoint_1.pth, modal mpa).
"""
import pathlib

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule

import sevenn._keys as KEY
import sevenn.nn.lunatp_helper as lunatp_helper
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator, _sort_edges
from sevenn.train.dataload import unlabeled_atoms_to_graph

CKPT = pathlib.Path(__file__).parents[2] / 'train' / 'omni' / 'checkpoint_1.pth'

pytestmark = [
    pytest.mark.skipif(not CKPT.is_file(), reason=f'{CKPT} not found'),
    pytest.mark.skipif(
        not lunatp_helper.is_lunatp_available(), reason='lunatp not available'
    ),
]


def _reference(calc, atoms):
    """model call on fresh device tensors (no cached topology), same edge order"""
    data = AtomGraphData.from_numpy_dict(
        unlabeled_atoms_to_graph(atoms, calc.cutoff, with_shift=False)
    )
    data[KEY.DATA_MODALITY] = calc.modal
    _sort_edges(data)
    data.to(calc.device)
    out = calc.model(data)
    return calc.output_to_results(out)


def _moved(atoms, i):
    """same topology, other positions (up to 0.01 A per coordinate)"""
    a = atoms.copy()
    a.positions += np.random.default_rng(i).uniform(-0.01, 0.01, a.positions.shape)
    return a


@pytest.fixture
def deterministic():
    """deterministic scatter/index_add (no fp32 atomics): repeated calls of
    the same inputs give the same bits, also for periodic cells"""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_eager_topology_reuse(monkeypatch, deterministic):
    calc = SevenNetCalculator(str(CKPT), modal='mpa', enable_lunatp=True)
    calc.cuda_graph = False
    builds = []
    make_graph = lunatp_helper.make_graph

    def counting_make_graph(*args):
        builds.append(1)
        return make_graph(*args)

    monkeypatch.setattr(lunatp_helper, 'make_graph', counting_make_graph)

    dimer = Atoms('Si2', positions=[[0, 0, 0], [0, 0, 2.3]])
    trimer = Atoms('Si3', positions=[[0, 0, 0], [0, 0, 2.3], [0, 2.2, 1.0]])
    h2o = molecule('H2O')
    si8 = bulk('Si', 'diamond', a=5.43, cubic=True)
    # (atoms, LunaTP graph built by this call)
    sequence = [
        (dimer, True), (dimer, False), (trimer, True), (trimer, False),
        (dimer, True), (h2o, True), (h2o, False), (si8, True), (si8, False),
    ]
    for i, (atoms, new_topology) in enumerate(sequence):
        atoms = _moved(atoms, i)
        n_before = len(builds)
        calc.calculate(atoms)
        assert len(builds) - n_before == int(new_topology), i
        got = {k: np.copy(v) for k, v in calc.results.items()}
        ref = _reference(calc, atoms)
        assert got.keys() == ref.keys()
        assert got['num_edges'] == ref['num_edges']
        for k in ('energy', 'free_energy', 'energies', 'forces', 'stress'):
            assert np.asarray(got[k]).dtype == np.asarray(ref[k]).dtype, k
            assert np.array_equal(got[k], ref[k]), (i, k)

