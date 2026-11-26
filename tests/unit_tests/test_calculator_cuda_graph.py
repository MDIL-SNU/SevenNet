from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

import sevenn.calculator as sevenn_calculator
import sevenn.nn.lunatp_helper as lunatp
from sevenn.calculator import SevenNetCalculator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


@pytest.fixture(scope='module', params=['e3nn', 'lunatp'])
def calculator(request):
    enable_lunatp = request.param == 'lunatp'
    if enable_lunatp and not lunatp.is_lunatp_available():
        pytest.skip('LunaTP not available')
    return SevenNetCalculator(
        '7net-0_11July2024', device='cuda', enable_lunatp=enable_lunatp,
        fused=False, radial_table=False,
    )


@pytest.fixture
def calc(calculator):
    calculator._graphs.clear()
    calculator._graphs_seen.clear()
    calculator._n_captures = calculator._n_replays = calculator._n_refreshes = 0
    calculator._pad = None
    return calculator


def _results(calc, atoms, graph):
    enabled, calc.cuda_graph = calc.cuda_graph, graph
    try:
        calc.calculate(atoms)
        return {key: np.copy(value) for key, value in calc.results.items()}
    finally:
        calc.cuda_graph = enabled


def _assert_replay_matches_eager(calc, atoms):
    actual, expected = _results(calc, atoms, True), _results(calc, atoms, False)
    assert actual.keys() == expected.keys()
    for key, ref in expected.items():
        assert actual[key].dtype == ref.dtype
        scale = 1e-5 if key == 'stress' else 1e-6
        np.testing.assert_allclose(
            actual[key], ref, atol=1e-5 + scale * np.abs(ref).max(), rtol=1e-6,
            err_msg=key,
        )
    assert actual['num_edges'] == expected['num_edges']


def _chain(n):
    return Atoms('Si' * n, positions=[[2.3 * i, 0.2 * (i % 2), 0] for i in range(n)])


def _rattled_si(seed):
    atoms = bulk('Si', 'diamond', a=6.0, cubic=True)
    atoms.rattle(0.03, seed=seed)
    return atoms


def test_default_replay_refreshes_topology(calc):
    assert calc.cuda_graph
    counts = set()
    for seed in range(1, 6):
        _assert_replay_matches_eager(calc, _rattled_si(seed))
        counts.add(calc.results['num_edges'])
    assert len(counts) > 1 and calc._pad is True
    assert calc._n_captures == 1 and calc._n_refreshes >= 2
    assert calc._n_replays == 3


def test_capture_failure_falls_back(calc, monkeypatch):
    output = calc.model._modules['force_output']
    forward = output.forward

    def synchronizing_forward(data):
        torch.cuda.synchronize()
        return forward(data)

    with monkeypatch.context() as patch:
        patch.setattr(output, 'forward', synchronizing_forward)
        _assert_replay_matches_eager(calc, _chain(2))
        with pytest.warns(UserWarning, match='CUDA graph capture failed'):
            _assert_replay_matches_eager(calc, _chain(2))
        _assert_replay_matches_eager(calc, _chain(2))
    assert calc._n_captures == 1 and calc._n_replays == 0
    for _ in range(3):
        _assert_replay_matches_eager(calc, _chain(3))
    assert calc._n_captures == 2 and calc._n_replays == 1


def test_cyclic_topologies_capture_bound(calc, monkeypatch):
    monkeypatch.setattr(calc, '_MAX_GRAPHS', 1)
    monkeypatch.setattr(calc, '_FREE_CAPTURES', 2)
    for _ in range(4):
        for n in (2, 3):
            _assert_replay_matches_eager(calc, _chain(n))
    assert calc._n_captures == 2 and calc._n_replays == 2


def test_graph_capacity_growth(calc):
    atoms = bulk('Si', 'diamond', a=5.43, cubic=True)
    atoms.rattle(0.1, seed=3)
    for lattice in (5.43, 4.95, 4.55):
        compressed = atoms.copy()
        compressed.set_cell(atoms.cell * lattice / 5.43, scale_atoms=True)
        for _ in range(3):
            _assert_replay_matches_eager(calc, compressed)
    assert calc._pad is True
    assert calc._n_captures == 3 and calc._n_replays >= 3
    assert len(calc._graphs) == (2 if calc._segment else 1)


def test_padding_fallback(calc, monkeypatch):
    pad = sevenn_calculator._pad_edges
    monkeypatch.setattr(
        sevenn_calculator, '_pad_edges',
        lambda tensors, n, length, segment: pad(tensors, n, 5.9, segment),
    )
    atoms = _rattled_si(1)
    _assert_replay_matches_eager(calc, atoms)
    with pytest.warns(UserWarning, match='padded edges change'):
        _assert_replay_matches_eager(calc, atoms)
    assert calc._pad is False
    for _ in range(3):
        _assert_replay_matches_eager(calc, atoms)
    assert calc._n_captures == calc._n_replays == 1


@pytest.mark.skipif(not lunatp.is_lunatp_available(), reason='LunaTP unavailable')
def test_eager_reuses_topology(monkeypatch):
    calc = SevenNetCalculator(
        '7net-0_11July2024', device='cuda', enable_lunatp=True, cuda_graph=False,
        fused=False, radial_table=False,
    )
    assert not calc.cuda_graph and not calc.fused
    make_graph = Mock(wraps=lunatp.make_graph)
    monkeypatch.setattr(lunatp, 'make_graph', make_graph)
    dimer, trimer = _chain(2), _chain(3)
    for expected_calls, atoms in enumerate((dimer, trimer, dimer), start=1):
        calc.calculate(atoms)
        displaced = atoms.copy()
        displaced.positions[0, 0] += 0.01
        calc.calculate(displaced)
        assert make_graph.call_count == expected_calls


def test_multimodal_model_has_no_host_sync():
    from sevenn import _keys as KEY
    from sevenn.atom_graph_data import AtomGraphData
    from sevenn.train.dataload import unlabeled_atoms_to_graph

    calc = SevenNetCalculator('7net-mf-0', modal='R2SCAN', device='cuda')
    atoms = _rattled_si(1)
    data = AtomGraphData.from_numpy_dict(
        unlabeled_atoms_to_graph(atoms, calc.cutoff)
    )
    data[KEY.DATA_MODALITY] = calc.modal
    data.to(calc.device)
    calc.model(data.clone())
    mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode('error')
    try:
        assert calc.model(data)[KEY.PRED_FORCE].shape == (len(atoms), 3)
    finally:
        torch.cuda.set_sync_debug_mode(mode)
