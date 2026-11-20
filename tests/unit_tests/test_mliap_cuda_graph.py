"""
LAMMPS ML-IAP CUDA graph (sevenn/nn/_mliap_cuda_graph.py) on the emulated
lmp_data of test_parallel_consistency.py, one rank, fp32 on GPU: along a
sequence of structures (changing pair lists) the replayed call gives the
energy, forces and stress of the serial model; one capture serves the
sequence (the other calls are replays with the pair list copied in); a model
that cannot be captured (ffn_moe) declines; more pairs than the capture holds are captured again; with two ranks (ghosts owned
by another rank) the graph turns itself off.
"""
import numpy as np
import pytest
import torch

from sevenn.nn._mliap_cuda_graph import MLIAPCudaGraph
from tests.unit_tests.test_arch_options import ARCHS
from tests.unit_tests.test_parallel_consistency import (
    GPU_CONFIGS,
    MODAL,
    _assert_close_f32,
    _atoms,
    _backend_available,
    _cf,
    _default_dtype,
    _FakeLmpData,
    _mliap_domains,
    _run_mliap,
    _save_checkpoint,
    _serial_f32,
    _World,
)
from tests.unit_tests.test_model import cutoff

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='needs CUDA'
)


def _graph(cp, backend):
    from sevenn.util import model_from_checkpoint

    kwargs = {} if backend == 'e3nn' else {f'enable_{backend}': True}
    model, _ = model_from_checkpoint(str(cp), **kwargs)
    model.set_is_batch_data(False)
    model.delete_module_by_key('force_output')
    model.requires_grad_(False)
    model.to('cuda').eval()
    modal = MODAL if model.modal_map else None
    return MLIAPCudaGraph(model, torch.device('cuda'), cutoff, modal=modal)


def _compute(graph, lmp):
    assert graph.compute(lmp)
    # the graph's static output holds the padded edges too
    lmp.pair_forces = lmp.pair_forces[: lmp.npairs].clone()


def _moved(atoms, step):
    out = atoms.copy()
    out.rattle(0.03 * step, seed=10 + step)
    return out


@pytest.mark.parametrize('arch', ARCHS)
@pytest.mark.parametrize('backend', ['e3nn', 'lunatp'])
@pytest.mark.parametrize('name', GPU_CONFIGS)
def test_mliap_cuda_graph(tmp_path, name, backend, arch):
    if backend != 'e3nn' and not _backend_available(backend):
        pytest.skip(f'{backend} not available')
    cp = _save_checkpoint(_cf(name, arch), tmp_path / 'cp.pth')
    atoms = _atoms((2, 2, 2))
    n_pairs = []
    with _default_dtype(torch.float32):
        graph = _graph(cp, backend)
        if name == 'moe':  # torch.nonzero per expert: a host sync, no capture
            lmp = _FakeLmpData(
                _World(_mliap_domains(atoms, (1, 1, 1), device='cuda')), 0, atoms
            )
            with pytest.warns(UserWarning, match='capture failed'):
                assert not graph.compute(lmp)
            assert not graph.enabled
            return
        for step in range(5):
            moved = _moved(atoms, step)
            par, ranks, _ = _run_mliap(
                [graph], moved, (1, 1, 1), compute=_compute, device='cuda'
            )
            n_pairs.append(len(ranks[0]['pi']))
            _assert_close_f32(par, _serial_f32(cp, moved), (name, backend, step))
    assert len(set(n_pairs)) > 1, n_pairs  # the pair list changed
    assert graph.enabled
    assert graph.n_captures == 1 and graph.n_replays == 4, (
        graph.n_captures, graph.n_replays, n_pairs
    )
    assert graph.n_refreshes >= 3


@pytest.mark.parametrize('backend', ['e3nn', 'lunatp'])
def test_mliap_cuda_graph_outgrown(tmp_path, backend):
    """more pairs than the captured call holds: captured again"""
    if backend != 'e3nn' and not _backend_available(backend):
        pytest.skip(f'{backend} not available')
    arch = ARCHS[0]
    cp = _save_checkpoint(_cf(GPU_CONFIGS[0], arch), tmp_path / 'cp.pth')
    atoms = _atoms((2, 2, 2))
    squeezed = atoms.copy()
    squeezed.set_cell(atoms.cell * 0.9, scale_atoms=True)
    with _default_dtype(torch.float32):
        graph = _graph(cp, backend)
        graph.MIN_EDGE_ROOM = 8
        for n, a in enumerate([atoms, atoms, squeezed, squeezed, atoms]):
            par, _, _ = _run_mliap(
                [graph], a, (1, 1, 1), compute=_compute, device='cuda'
            )
            _assert_close_f32(par, _serial_f32(cp, a), (backend, n))
    assert graph.n_captures == 2 and graph.n_replays == 3


def test_mliap_cuda_graph_two_ranks_off(tmp_path):
    """a ghost owned by another rank: the graph declines and stays off"""
    cp = _save_checkpoint(_cf(GPU_CONFIGS[0], ARCHS[0]), tmp_path / 'cp.pth')
    atoms = _atoms((2, 2, 2))
    with _default_dtype(torch.float32):
        graph = _graph(cp, 'e3nn')
        ranks = _mliap_domains(atoms, (2, 1, 1), device='cuda')

        class OneOfTwo(_FakeLmpData):
            # rank 0 alone: the ghosts of rank 1 return rank 1's rows
            def forward_exchange(self, feats, out, vec_len):
                n = self.nlocal
                out.copy_(feats)
                for ro, (g, no) in self.rk['ghost_src'].items():
                    out[n + g] = feats[:n][no] if ro == self.rank else -7.0

        lmp = OneOfTwo(_World(ranks), 0, atoms)
        with pytest.warns(UserWarning, match='owned by another rank'):
            assert not graph.compute(lmp)
        assert not graph.enabled and not graph.compute(lmp)
    assert np.all(lmp.eatoms == 0.0)
