import numpy as np
import pytest
import torch
from _mliap_fixture import (
    MODAL,
    assert_efs,
    atoms_with_ghosts,
    cutoff,
    evaluate,
    reference,
    save_checkpoint,
)
from test_mliap_cuda_graph import _compute, _moved

from sevenn.nn._mliap_cuda_graph import MLIAPCudaGraph
from sevenn.nn.lunatp_helper import is_lunatp_available

pytestmark = pytest.mark.skipif(
    not is_lunatp_available(), reason='needs CUDA and LunaTP'
)


def _fused_graph(cp, knots=0):
    from sevenn.nn.inference_specialization import fold_modal
    from sevenn.nn.lunatp_fused import fuse_radial_conv
    from sevenn.util import model_from_checkpoint

    model, _ = model_from_checkpoint(str(cp), enable_lunatp=True)
    model.set_is_batch_data(False)
    model.delete_module_by_key('force_output')
    model.requires_grad_(False)
    model.to('cuda').eval()
    names = fuse_radial_conv(model, knots=knots)
    modal = MODAL if model.modal_map else None
    if modal is not None:
        fold_modal(model, modal)
    if knots:  # the first layer as per-species radial tables, where it applies
        from sevenn.nn.lunatp_fused import collapse_first_layer

        reference, _ = model_from_checkpoint(str(cp))
        reference.delete_module_by_key('force_output')
        names = names + collapse_first_layer(model, reference, modal)
    model.prune_species = True
    return MLIAPCudaGraph(model, torch.device('cuda'), cutoff, modal=modal), names


@pytest.mark.parametrize('knots', [0, 2048])
@pytest.mark.parametrize('self_connection', ['nequip', 'linear'])
def test_fused_mliap_cuda_graph(tmp_path, self_connection, knots):
    cp = save_checkpoint(tmp_path / 'cp.pth', self_connection, modal=True)
    atoms = atoms_with_ghosts()
    graph, names = _fused_graph(cp, knots=knots)
    ref = reference(cp, MODAL)
    assert names
    for step in range(4):
        moved = _moved(atoms, step)
        out, _ = _compute(graph, moved)
        assert_efs(out, evaluate(ref, moved))

    assert graph.enabled
    assert graph.n_captures == 1 and graph.n_replays == 3
    assert graph._prep is not None and 'flags' in graph._entry
    if self_connection == 'nequip':
        from sevenn.nn.inference_specialization import PrunedSelfConnectionIntro
        from sevenn.nn.lunatp_fused import FusedFirstLayer

        n_species = len(set(atoms.numbers))
        pruned = [
            m for m in graph.model.modules()
            if isinstance(m, PrunedSelfConnectionIntro)
        ]
        assert pruned and all(m.columns.numel() == n_species for m in pruned)
        if knots:
            first = [
                m for m in graph.model.modules() if isinstance(m, FusedFirstLayer)
            ]
            assert first and len(first[0].columns) == n_species


@pytest.mark.parametrize('lmax', [1, 2, 3, 4])
def test_fused_edge_geometry(lmax):
    """one-kernel edge length + spherical harmonics and their backward"""
    import sevenn._keys as KEY
    from sevenn.nn.edge_embedding import SphericalEncoding
    from sevenn.nn.lunatp_fused import FusedEdgeGeometry

    class _Emb:
        spherical = SphericalEncoding(lmax)

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


def _md(efs, atoms, steps, dt):
    """velocity Verlet (unit masses, at rest): potential, total energy, x"""
    atoms = atoms.copy()
    v = np.zeros((len(atoms), 3))
    e, f = efs(atoms)[:2]
    e_start = e
    for _ in range(steps):
        v += 0.5 * dt * f
        atoms.set_positions(atoms.get_positions() + dt * v)
        e, f = efs(atoms)[:2]
        v += 0.5 * dt * f
    return e_start, e, e + 0.5 * float((v * v).sum()), atoms.get_positions()


@pytest.mark.parametrize('knots', [0, 2048])
def test_fused_md_energy(tmp_path, knots):
    cp = save_checkpoint(tmp_path / 'cp.pth')
    atoms = atoms_with_ghosts()
    calc = reference(cp)
    graph, names = _fused_graph(cp, knots=knots)
    assert names

    def ref(a):
        return evaluate(calc, a)

    def new(a):
        return _compute(graph, a)[0]

    f0 = np.abs(ref(atoms)[1]).max()
    dt = float(np.sqrt(2.0 * 2e-3 / f0))
    e0, e_ref, tot_ref, x_ref = _md(ref, atoms, 20, dt)
    _, e_new, tot_new, x_new = _md(new, atoms, 20, dt)
    assert abs(e_ref - e0) > 1e-3
    assert abs(e_new - e_ref) < 1e-4
    assert abs(tot_new - tot_ref) < 1e-4
    np.testing.assert_allclose(x_new, x_ref, atol=1e-4, rtol=0)


def test_fused_calculator(tmp_path):
    from sevenn.calculator import SevenNetCalculator

    cp = save_checkpoint(tmp_path / 'cp.pth')
    plain = SevenNetCalculator(str(cp), device='cpu', compute_atomic_virial=True)
    fused = SevenNetCalculator(
        str(cp), device='cuda', enable_lunatp=True, fused=True,
        compute_atomic_virial=True,
    )
    assert any(name.startswith('0_convolution') for name in fused.fused)
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
