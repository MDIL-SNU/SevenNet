"""
SevenNetCalculator(cuda_graph=True): a repeated topology replays a captured CUDA
graph with the eager results; another topology runs eagerly and stays correct;
a return to the first topology replays its graph; a failed capture falls back to
eager; topologies cycling through a full cache stop being captured once captures
do not pay back; along MD, the edges are padded to the kept graph's number of
edges, the edge index (and the LunaTP graph tables) is copied in, and the
replay equals eager.
Model: 7net-0 (e3nn and LunaTP backends).
"""
import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import sevenn.calculator as sevenn_calculator
from sevenn.calculator import SevenNetCalculator
from sevenn.nn.lunatp_helper import is_lunatp_available
from sevenn.util import pretrained_name_to_path

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA'),
]

BACKENDS = [
    'e3nn',
    pytest.param(
        'lunatp',
        marks=pytest.mark.skipif(
            not is_lunatp_available(), reason='lunatp not available'
        ),
    ),
]


@pytest.fixture(scope='module', params=BACKENDS)
def calc(request):
    return SevenNetCalculator(
        pretrained_name_to_path('7net-0_11July2024'),
        enable_lunatp=request.param == 'lunatp',
        cuda_graph=True,
    )


@pytest.fixture
def counters(calc, monkeypatch):
    """number of captures and replays"""
    n = {'capture': 0, 'replay': 0}
    capture = calc._capture
    replay = torch.cuda.CUDAGraph.replay

    def count_capture(*args):
        entry = capture(*args)
        n['capture'] += entry is not None
        return entry

    def count_replay(self):
        n['replay'] += 1
        return replay(self)

    def reset():  # the calculator is shared by the tests of a module
        calc._graphs.clear()
        calc._graphs_seen.clear()
        calc._n_captures = calc._n_replays = calc._n_refreshes = 0
        calc._pad = None

    monkeypatch.setattr(calc, '_capture', count_capture)
    monkeypatch.setattr(torch.cuda.CUDAGraph, 'replay', count_replay)
    reset()
    yield n
    reset()


def _results(calc, atoms, graph: bool):
    calc.cuda_graph = graph
    try:
        calc.calculate(atoms.copy())
    finally:
        calc.cuda_graph = True
    return {k: np.copy(v) for k, v in calc.results.items()}


def _assert_eager_equal(calc, atoms):
    """graph-path result == eager result of the same atoms"""
    got = _results(calc, atoms, graph=True)
    ref = _results(calc, atoms, graph=False)
    assert got.keys() == ref.keys()
    for k in ('energy', 'free_energy', 'energies', 'forces', 'stress'):
        assert np.asarray(got[k]).dtype == np.asarray(ref[k]).dtype, k
        assert np.shape(got[k]) == np.shape(ref[k]), k
        # fp32 atomic adds: eager itself repeats only to ~1e-6 of the largest
        # entry (non-periodic stress: volume 1e-3 A^3, entries up to ~1e4).
        # Stress: a sum of edge virials with cancellation; padded edges change
        # the kernels' work split over edges (H2O: 2e-6 of the largest entry)
        rel = 1e-5 if k == 'stress' else 1e-6
        atol = 1e-5 + rel * np.abs(ref[k]).max()
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-6, atol=atol, err_msg=k)
    assert got['num_edges'] == ref['num_edges']
    return got


def _systems():
    return {
        'dimer': Atoms('Si2', positions=[[0, 0, 0], [0, 0, 2.3]]),
        'h2o': molecule('H2O'),
        'si8': bulk('Si', 'diamond', a=5.43, cubic=True),
    }


def _moved(atoms, i):
    """same topology, other positions (up to 0.01 A per coordinate; the Si8
    shells at 4.50 and 5.43 A stay on their side of the 5 A cutoff)"""
    a = atoms.copy()
    a.positions += np.random.default_rng(i).uniform(-0.01, 0.01, a.positions.shape)
    return a


@pytest.mark.parametrize('name', ['dimer', 'h2o', 'si8'])
def test_graph_equals_eager(calc, counters, name):
    atoms = _systems()[name]
    for i in range(4):  # eager, capture + replay, replay, replay
        _assert_eager_equal(calc, _moved(atoms, i))
    assert counters['capture'] == 1
    assert counters['replay'] == 3


def test_topology_change(calc, counters):
    dimer = _systems()['dimer']
    trimer = Atoms('Si3', positions=[[0, 0, 0], [0, 0, 2.3], [0, 2.2, 1.0]])
    for i in range(3):
        _assert_eager_equal(calc, _moved(dimer, i))
    assert (counters['capture'], counters['replay']) == (1, 2)
    _assert_eager_equal(calc, trimer)  # other atom count: eager, still correct
    assert (counters['capture'], counters['replay']) == (1, 2)
    _assert_eager_equal(calc, _moved(dimer, 7))  # back: replays the first graph
    assert (counters['capture'], counters['replay']) == (1, 3)


def test_capture_failure_falls_back(calc, counters, monkeypatch):
    force_output = calc.model._modules['force_output']
    forward = force_output.forward

    def syncing_forward(data):
        torch.cuda.synchronize()  # not allowed while a stream is capturing
        return forward(data)

    monkeypatch.setattr(force_output, 'forward', syncing_forward)
    dimer = _systems()['dimer']
    _assert_eager_equal(calc, dimer)
    with pytest.warns(UserWarning, match='CUDA graph capture failed') as rec:
        _assert_eager_equal(calc, _moved(dimer, 1))  # capture fails
        _assert_eager_equal(calc, _moved(dimer, 2))  # eager, no second capture
    assert sum('CUDA graph capture failed' in str(w.message) for w in rec) == 1
    assert (counters['capture'], counters['replay']) == (0, 0)
    monkeypatch.setattr(force_output, 'forward', forward)
    h2o = _systems()['h2o']  # capture works afterwards
    for i in range(3):
        _assert_eager_equal(calc, _moved(h2o, i))
    assert (counters['capture'], counters['replay']) == (1, 2)


def test_cyclic_topologies_capture_bound(calc, counters, monkeypatch):
    """
    NEB-like: 5 topologies in turn through a 2-graph cache. The cache evicts
    each graph before its next use; without the thrash guard 20 of the 50
    calls capture, with it 8 (2 free captures, then 2 replays per capture)
    """
    monkeypatch.setattr(calc, '_MAX_GRAPHS', 2)
    monkeypatch.setattr(calc, '_FREE_CAPTURES', 2)
    chains = [
        Atoms(f'Si{n}', positions=[[2.3 * i, 0.4 * (i % 2), 0] for i in range(n)])
        for n in range(2, 7)
    ]
    for cycle in range(10):
        for atoms in chains:
            _assert_eager_equal(calc, _moved(atoms, cycle))
    assert (calc._n_captures, calc._n_replays) == (8, 14)
    assert counters['capture'] == 8


def _langevin(calc, atoms, steps):
    """
    Langevin MD (1000 K, 2 fs) driven by the graph path; every step equals
    eager. Returns the counts of eager calls, captures, refreshed and pure
    replays
    """
    kinds = dict.fromkeys(('eager', 'capture', 'refresh', 'replay'), 0)
    rng = np.random.default_rng(0)
    MaxwellBoltzmannDistribution(atoms, temperature_K=1000, rng=rng)
    dt, gamma, kt = 2 * units.fs, 0.02 / units.fs, 1000 * units.kB
    m = atoms.get_masses()[:, None]
    v = atoms.get_velocities()
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1**2) * kt / m)
    for _ in range(steps):
        before = (calc._n_captures, calc._n_replays, calc._n_refreshes)
        f = _assert_eager_equal(calc, atoms)['forces']
        kinds[_kind(calc, before)] += 1
        v = v + 0.5 * dt * f / m
        atoms.positions += 0.5 * dt * v
        v = c1 * v + c2 * rng.standard_normal(v.shape)
        atoms.positions += 0.5 * dt * v
        v = v + 0.5 * dt * f / m
    return kinds


def _kind(calc, before):
    """eager / capture / refresh / replay: what the last graph-path call did"""
    if calc._n_captures > before[0]:
        return 'capture'
    if calc._n_refreshes > before[2]:
        return 'refresh'
    return 'replay' if calc._n_replays > before[1] else 'eager'


def _si8_at_cutoff():
    """Si8 whose third neighbour shell (a sqrt(11)/4 = 4.97 A, pairs of two
    atoms) lies at the 5 A cutoff of 7net-0, rattled so that edges cross the
    cutoff along MD (the fourth shell, at a, is an atom and its own image:
    its length does not change)"""
    atoms = bulk('Si', 'diamond', a=6.0, cubic=True)
    atoms.rattle(0.03, seed=1)
    return atoms


def test_md_refresh_equals_eager(calc, counters):
    """
    Langevin MD of Si8 (a = 6.0 A, rattled: its third neighbour shell lies at
    the 5 A cutoff; 1000 K, 2 fs, 30 steps): the edge index and the
    number of edges change; every call after the capture is padded to the kept
    graph's number of edges (self-loops of length r_c, where every edge term
    vanishes), the edge index (and LunaTP graph tables) is copied in, and the
    replay equals eager (same edge order). One capture serves the trajectory
    """
    kinds = _langevin(calc, _si8_at_cutoff(), 30)
    print('Si8 Langevin', kinds)
    assert calc._pad is True  # padding stayed on (pads checked neutral)
    assert kinds['capture'] == 1 and kinds['eager'] == 1
    assert kinds['refresh'] >= 10


def test_edges_beyond_padding_and_segment_change(calc, counters):
    """
    Si8 (rattled) compressed from a = 5.43 to 4.40 A and back in 28 calls:
    the fourth and fifth neighbour shells enter the 5 A cutoff, so the number
    of edges grows past the kept graph's padded number of edges (that call
    captures anew with more edges, up to the LunaTP cap of 32 per source
    segment), and the LunaTP source segments change (other keys) while the
    edges would fit the graph of the previous call. No call fails, every call
    equals eager
    """
    base = bulk('Si', 'diamond', a=5.43, cubic=True)
    base.rattle(0.1, seed=3)
    lattice = list(np.linspace(5.43, 4.40, 14))
    kinds = dict.fromkeys(('eager', 'capture', 'refresh', 'replay'), 0)
    trace = []  # (number of edges, kept graphs {segments: padded edges})
    for a in lattice + lattice[::-1]:
        atoms = base.copy()
        atoms.set_cell(base.cell * a / 5.43, scale_atoms=True)
        before = (calc._n_captures, calc._n_replays, calc._n_refreshes)
        got = _assert_eager_equal(calc, atoms)
        kinds[_kind(calc, before)] += 1
        kept = {k[-1][1]: e['num_edges'] for k, e in calc._graphs.items() if e}
        trace.append((got['num_edges'], kept))
    print('Si8 compression', kinds, [t[0] for t in trace])
    # a call with more edges than its kept graph: the graph of a key grew
    padded = {}
    for _, kept in trace:
        for key, n in kept.items():
            padded.setdefault(key, set()).add(n)
    assert any(len(v) >= 2 for v in padded.values()), padded
    if calc._segment:  # LunaTP: several segment counts, one graph each
        assert len(trace[-1][1]) >= 2
        assert kinds['replay'] + kinds['refresh'] >= 10


def test_padding_fallback(calc, counters, monkeypatch):
    """
    Padded edges that change the outputs (here: length 5.9 A instead of r_c,
    beyond the cutoff, where the polynomial envelope of 7net-0 is not 0):
    the neutrality check turns padding off with one warning, the number of
    edges is in the key again, and Langevin MD stays equal to eager
    """
    pad_edges = sevenn_calculator._pad_edges
    monkeypatch.setattr(
        sevenn_calculator, '_pad_edges',
        lambda tensors, n, length, segment: pad_edges(tensors, n, 5.9, segment),
    )
    with pytest.warns(UserWarning, match='padded edges change') as rec:
        kinds = _langevin(calc, _si8_at_cutoff(), 30)
    print('Si8 Langevin, padding off', kinds)
    assert sum('padded edges change' in str(w.message) for w in rec) == 1
    assert calc._pad is False
    assert kinds['capture'] >= 2  # one graph per number of edges
