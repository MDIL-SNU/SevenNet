import numpy as np
import pytest
import torch
from _mliap_fixture import (
    MODAL,
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

from sevenn.calculator import SevenNetCalculator
from sevenn.nn._mliap_cuda_graph import MLIAPCudaGraph
from sevenn.nn.lunatp_helper import is_lunatp_available

pytestmark = pytest.mark.skipif(
    not is_lunatp_available(), reason='needs CUDA and LunaTP'
)


def _fused_graph(cp, knots=0):
    from sevenn.nn.inference_specialization import fold_modal
    from sevenn.nn.lunatp_fused import collapse_first_layer, fuse_radial_conv
    from sevenn.util import model_from_checkpoint

    model = graph_model(cp, enable_lunatp=True)
    names = fuse_radial_conv(model, knots=knots)
    modal = MODAL if model.modal_map else None
    if modal is not None:
        fold_modal(model, modal)
    if knots:
        reference, _ = model_from_checkpoint(str(cp))
        reference.delete_module_by_key('force_output')
        names = names + collapse_first_layer(model, reference, modal)
    model.prune_species = True
    return MLIAPCudaGraph(model, torch.device('cuda'), cutoff, modal=modal), names


@pytest.mark.parametrize('self_connection, knots', [('linear', 0), ('nequip', 4096)])
def test_fused_mliap_cuda_graph(tmp_path, self_connection, knots):
    modal = MODAL if knots else None
    cp = save_checkpoint(tmp_path / 'cp.pth', self_connection, modal=bool(modal))
    atoms = atoms_with_ghosts()
    if knots:
        atoms.symbols[:4] = 'NaClNaCl'
    graph, names = _fused_graph(cp, knots=knots)
    ref = reference(cp, modal)
    assert names
    for step in range(4):
        structure = moved(atoms, step)
        out, _ = compute_graph(graph, structure)
        assert_efs(out, evaluate(ref, structure))

    assert graph.enabled
    assert graph.n_captures == 1 and graph.n_replays == 3
    assert graph._prep is not None and 'flags' in graph._entry
    if knots:
        from sevenn.nn.inference_specialization import PrunedSelfConnectionIntro
        from sevenn.nn.lunatp_fused import FusedFirstLayer

        n_species = len(set(atoms.numbers))
        pruned = [
            m for m in graph.model.modules()
            if isinstance(m, PrunedSelfConnectionIntro)
        ]
        assert pruned and all(m.columns.numel() == n_species for m in pruned)
        first = [
            m for m in graph.model.modules() if isinstance(m, FusedFirstLayer)
        ]
        assert first and len(first[0].columns) == n_species


def test_fused_edge_geometry():
    import sevenn._keys as KEY
    from sevenn.nn.edge_embedding import SphericalEncoding
    from sevenn.nn.lunatp_fused import FusedEdgeGeometry

    class _Emb:
        spherical = SphericalEncoding(4)

    geo = FusedEdgeGeometry(_Emb())
    torch.manual_seed(0)
    v = (torch.randn(500, 3, device='cuda') * 2.0).requires_grad_(True)
    out = geo({KEY.EDGE_VEC: v})
    r, y = out[KEY.EDGE_LENGTH], out[KEY.EDGE_ATTR]
    gr, gy = torch.randn_like(r), torch.randn_like(y)
    (g,) = torch.autograd.grad([r, y], v, [gr, gy])
    v64 = v.detach().double().requires_grad_(True)
    y64 = _Emb.spherical.double().to('cuda')(v64)
    r64 = v64.norm(dim=-1)
    (g64,) = torch.autograd.grad([r64, y64], v64, [gr.double(), gy.double()])
    assert torch.allclose(y.double(), y64, atol=2e-6)
    assert torch.allclose(r.double(), r64, atol=2e-6)
    assert torch.allclose(g.double(), g64, atol=1e-5 * float(g64.abs().max()))


def test_default_fused_calculator(tmp_path):
    from sevenn.nn.lunatp_fused import (
        FusedFirstLayer,
        LunaTPFusedRadialConvolution,
    )

    cp = save_checkpoint(tmp_path / 'cp.pth')
    plain = SevenNetCalculator(
        str(cp), device='cpu', compute_atomic_virial=True, radial_table=False,
    )
    fused = SevenNetCalculator(
        str(cp), device='cuda', enable_lunatp=True, compute_atomic_virial=True,
    )
    assert fused.cuda_graph and not plain.cuda_graph
    convs = [
        conv for conv in fused.model.modules()
        if isinstance(conv, LunaTPFusedRadialConvolution)
    ]
    assert convs
    assert {(conv.knots, conv.table_r0) for conv in convs} == {(2048, 0.0)}
    atoms = atoms_with_ghosts()
    fewer_species = atoms[atoms.numbers != 1]
    empty_edges = atoms[:2]
    empty_edges.set_cell([30.0, 30.0, 30.0])
    empty_edges.positions[:] = [[5, 5, 5], [25, 25, 25]]
    for structure in (fewer_species, empty_edges, atoms, fewer_species):
        assert_efs(evaluate(fused, structure), evaluate(plain, structure))
        np.testing.assert_allclose(
            fused.results['stresses'], plain.results['stresses'], atol=1e-5
        )
    assert len(fused._fused_species) == 4
    first = next(m for m in fused.model.modules() if isinstance(m, FusedFirstLayer))
    assert first.table.numel() and torch.isfinite(first.table).all()

    # Replay with a different number of real edges in the same padded buffers.
    three = atoms[:3]
    three.set_cell([30.0, 30.0, 30.0])
    three.positions[:] = [[5, 5, 5], [7, 5, 5], [5 + cutoff, 5, 5]]
    for separation in (cutoff + 0.01, cutoff + 0.01, cutoff - 0.01):
        three.positions[2, 0] = 5 + separation
        fused.calculate(three)
        actual = [fused.results[key] for key in ('energy', 'forces', 'stress')]
        assert_efs(actual, evaluate(plain, three))
    assert fused._pad and fused._n_replays == fused._n_refreshes == 1


@pytest.mark.parametrize('fused', [False, True])
def test_ase_fusion_and_interpolation_switches(tmp_path, fused):
    from sevenn.nn.lunatp_fused import LunaTPFusedRadialConvolution
    from sevenn.nn.radial_table import RadialTable

    cp = save_checkpoint(tmp_path / 'cp.pth')
    candidate = SevenNetCalculator(
        str(cp), device='cuda', enable_lunatp=True, cuda_graph=False,
        fused=fused, radial_table=not fused, radial_knots=1024, radial_r_min=0.20,
    )
    conv = candidate.model._modules['0_convolution']
    assert not candidate.cuda_graph
    if fused:
        assert isinstance(conv, LunaTPFusedRadialConvolution) and conv.knots == 0
    else:
        assert isinstance(conv.weight_nn, RadialTable)
        assert conv.weight_nn.r0 == 0.20 and conv.weight_nn.coef.shape[0] == 1023
    atoms = atoms_with_ghosts()
    expected = evaluate(reference(cp), atoms)
    assert_efs(evaluate(candidate, atoms), expected)
    if fused:
        candidate.cuda_graph = True
        for _ in range(4):
            candidate.calculate(atoms)
        assert candidate._n_replays > 0
        actual = [candidate.results[key] for key in ('energy', 'forces', 'stress')]
        assert_efs(actual, expected)
