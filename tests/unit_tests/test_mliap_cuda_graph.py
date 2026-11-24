import numpy as np
import pytest
import torch
from _mliap_fixture import (
    MODAL,
    FakeLmpData,
    assert_efs,
    atoms_with_ghosts,
    cutoff,
    evaluate,
    reference,
    save_checkpoint,
)

from sevenn.nn._mliap_cuda_graph import MLIAPCudaGraph
from sevenn.nn.lunatp_helper import is_lunatp_available
from sevenn.util import model_from_checkpoint

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='needs CUDA'
)


def _graph(cp, backend, modal=None):
    if backend == 'lunatp' and not is_lunatp_available():
        pytest.skip('LunaTP not available')
    model, _ = model_from_checkpoint(str(cp), enable_lunatp=backend == 'lunatp')
    model.set_is_batch_data(False)
    model.delete_module_by_key('force_output')
    model.requires_grad_(False)
    model.to('cuda').eval()
    return MLIAPCudaGraph(model, torch.device('cuda'), cutoff, modal=modal)


def _compute(graph, atoms, index_dtype=np.int32):
    lmp = FakeLmpData(atoms, index_dtype)
    assert graph.compute(lmp)
    return lmp.efs(atoms), lmp.npairs


def _moved(atoms, step):
    out = atoms.copy()
    out.rattle(0.03 * step, seed=10 + step)
    return out


@pytest.mark.parametrize('backend', ['e3nn', 'lunatp'])
@pytest.mark.parametrize('modal', [None, MODAL])
def test_mliap_cuda_graph(tmp_path, backend, modal):
    cp = save_checkpoint(tmp_path / 'cp.pth', modal=bool(modal))
    graph = _graph(cp, backend, modal)
    ref = reference(cp, modal)
    atoms = atoms_with_ghosts()
    n_pairs = []
    for step in range(5):
        moved = _moved(atoms, step)
        out, npairs = _compute(graph, moved, index_dtype=np.int64)
        assert_efs(out, evaluate(ref, moved))
        n_pairs.append(npairs)

    assert len(set(n_pairs)) > 1, n_pairs
    assert graph.enabled
    assert graph.n_captures == 1 and graph.n_replays == 4
    assert graph.n_refreshes >= 3


@pytest.mark.parametrize('backend', ['e3nn', 'lunatp'])
def test_mliap_cuda_graph_outgrown(tmp_path, backend):
    cp = save_checkpoint(tmp_path / 'cp.pth')
    graph = _graph(cp, backend)
    graph.MIN_EDGE_ROOM = 8
    ref = reference(cp)
    atoms = atoms_with_ghosts()
    squeezed = atoms.copy()
    squeezed.set_cell(atoms.cell * 0.9, scale_atoms=True)
    for moved in [atoms, atoms, squeezed, squeezed, atoms]:
        out, _ = _compute(graph, moved)
        assert_efs(out, evaluate(ref, moved))

    assert graph.n_captures == 2 and graph.n_replays == 3


def test_mliap_cuda_graph_foreign_ghost(tmp_path):
    graph = _graph(save_checkpoint(tmp_path / 'cp.pth'), 'e3nn')
    lmp_data = FakeLmpData(atoms_with_ghosts(), foreign=True)
    with pytest.warns(UserWarning, match='owned by another rank'):
        assert not graph.compute(lmp_data)
    assert not graph.enabled and not graph.compute(lmp_data)
    assert not lmp_data.eatoms.any()
