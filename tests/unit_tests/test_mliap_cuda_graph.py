import numpy as np
import pytest
import torch
from _mliap_fixture import (
    MODAL,
    FakeLmpData,
    assert_efs,
    atoms_with_ghosts,
    compute_graph,
    cutoff,
    evaluate,
    graph_model,
    moved,
    reference,
    save_checkpoint,
)

from sevenn.nn._mliap_cuda_graph import MLIAPCudaGraph
from sevenn.nn.lunatp_helper import is_lunatp_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='needs CUDA'
)


def _graph(cp, backend, modal=None):
    if backend == 'lunatp' and not is_lunatp_available():
        pytest.skip('LunaTP not available')
    model = graph_model(cp, enable_lunatp=backend == 'lunatp')
    return MLIAPCudaGraph(model, torch.device('cuda'), cutoff, modal=modal)


@pytest.mark.parametrize('backend, modal', [('e3nn', None), ('lunatp', MODAL)])
def test_mliap_cuda_graph(tmp_path, backend, modal):
    cp = save_checkpoint(tmp_path / 'cp.pth', modal=bool(modal))
    graph = _graph(cp, backend, modal)
    ref = reference(cp, modal)
    atoms = atoms_with_ghosts()
    n_pairs = []
    for step in range(5):
        structure = moved(atoms, step)
        out, npairs = compute_graph(graph, structure, index_dtype=np.int64)
        assert_efs(out, evaluate(ref, structure))
        n_pairs.append(npairs)

    assert len(set(n_pairs)) > 1, n_pairs
    assert graph.enabled
    assert graph.n_captures == 1 and graph.n_replays == 4
    assert graph.n_refreshes >= 3


def test_mliap_cuda_graph_outgrown(tmp_path):
    cp = save_checkpoint(tmp_path / 'cp.pth')
    graph = _graph(cp, 'lunatp')
    graph.MIN_EDGE_ROOM = 8
    ref = reference(cp)
    atoms = atoms_with_ghosts()
    squeezed = atoms.copy()
    squeezed.set_cell(atoms.cell * 0.9, scale_atoms=True)
    for structure in [atoms, atoms, squeezed, squeezed, atoms]:
        out, _ = compute_graph(graph, structure)
        assert_efs(out, evaluate(ref, structure))

    assert graph.n_captures == 2 and graph.n_replays == 3


def test_mliap_cuda_graph_foreign_ghost(tmp_path):
    graph = _graph(save_checkpoint(tmp_path / 'cp.pth'), 'e3nn')
    lmp_data = FakeLmpData(atoms_with_ghosts(), foreign=True)
    with pytest.warns(UserWarning, match='owned by another rank'):
        assert not graph.compute(lmp_data)
    assert not graph.enabled and not graph.compute(lmp_data)
    assert not lmp_data.eatoms.any()


@pytest.mark.parametrize('replay', [False, True])
@pytest.mark.parametrize('folding', [False, True])
def test_ghost_folding_is_independent(tmp_path, monkeypatch, replay, folding):
    pytest.importorskip('lammps.mliap.mliap_unified_abc')
    from sevenn.mliap import SevenNetMLIAPWrapper

    for var in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE',
                'PMIX_UNIV_SIZE', 'SLURM_STEP_NUM_TASKS'):
        monkeypatch.setenv(var, '1')
    monkeypatch.setenv('SEVENN_MLIAP_CUDA_GRAPH', str(int(replay)))
    monkeypatch.delenv('SEVENN_MLIAP_FOLD_EAGER', raising=False)
    if folding:  # Exercise the default with replay both enabled and disabled.
        monkeypatch.delenv('SEVENN_MLIAP_GHOST_FOLDING', raising=False)
    else:
        monkeypatch.setenv('SEVENN_MLIAP_GHOST_FOLDING', '0')

    cp = save_checkpoint(tmp_path / 'ghost.pth')
    model = SevenNetMLIAPWrapper(str(cp), use_radial_table=False)
    ref = reference(cp)
    atoms = atoms_with_ghosts()
    structures = (atoms, atoms, atoms, moved(atoms, 3))
    for structure in structures:
        lmp = FakeLmpData(structure)
        lmp.elems = torch.as_tensor(lmp.elems, device='cuda')

        def reverse_exchange(source, output, width):
            output.copy_(source)
            owners = torch.as_tensor(lmp.ghost_owner, device=source.device)
            output[:lmp.nlocal].index_add_(0, owners, source[lmp.nlocal:])

        lmp.reverse_exchange = reverse_exchange
        model.compute_forces(lmp)
        assert_efs(lmp.efs(structure), evaluate(ref, structure))

    assert (model.graph is not None) == folding
    if folding:
        assert model.graph.eager_fold == (not replay)
    else:
        assert (model._unfolded_graph is not None) == replay
        if replay:
            assert model._unfolded_graph.replays > 0


@pytest.mark.parametrize(
    'options, fused, radial_table',
    [
        ({}, True, True),
        ({'use_fused': False}, False, True),
        ({'use_radial_table': False}, True, False),
    ],
    ids=['defaults', 'disable-fused', 'disable-radial-table'],
)
def test_parallel_inference_uses_eager(
    tmp_path, monkeypatch, options, fused, radial_table
):
    if not is_lunatp_available():
        pytest.skip('LunaTP not available')
    pytest.importorskip('lammps.mliap.mliap_unified_abc')
    from sevenn.mliap import SevenNetMLIAPWrapper
    from sevenn.nn.lunatp_fused import LunaTPFusedRadialConvolution
    from sevenn.nn.radial_table import RadialTable

    monkeypatch.setenv('OMPI_COMM_WORLD_SIZE', '2')
    cp = save_checkpoint(tmp_path / 'parallel.pth')
    model = SevenNetMLIAPWrapper(
        str(cp), use_lunatp=True, use_cuda_graph=True, **options,
    )
    ref = reference(cp)
    atoms = atoms_with_ghosts()
    atoms.pbc = False
    counts = set()
    structures = (atoms, atoms, atoms[:0], moved(atoms, 2), moved(atoms, 3))
    for structure in structures:
        lmp = FakeLmpData(structure)
        lmp.elems = torch.as_tensor(lmp.elems, device='cuda')
        lmp.reverse_exchange = lambda send, recv, width: recv.copy_(send)
        model.compute_forces(lmp)
        if not len(structure):
            assert lmp.energy == 0.0
            continue
        assert_efs(lmp.efs(structure), evaluate(ref, structure))
        assert model.graph is None
        counts.add(lmp.npairs)
    assert len(counts) > 1
    fused_layers = [
        layer for layer in model.model.modules()
        if isinstance(layer, LunaTPFusedRadialConvolution)
    ]
    assert bool(fused_layers) == fused
    assert (
        any(layer.knots for layer in fused_layers)
        or any(isinstance(layer, RadialTable) for layer in model.model.modules())
    ) == radial_table
