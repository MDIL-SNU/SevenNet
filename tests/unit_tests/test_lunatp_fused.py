"""
Frozen-model LunaTP path (sevenn/nn/lunatp_fused.py) in the LAMMPS ML-IAP
CUDA graph, on the emulated lmp_data of test_parallel_consistency.py: fused
radial convolutions (weight rows per distinct edge length, dE/dr contracted
in the backward kernel, hidden layers from a table), species self connection
pruned to the species present, modal input folded. Energy, forces and stress
of the serial model along a sequence of structures, from one capture; fp16
weight rows are close but not equal.
"""
import numpy as np
import pytest
import torch

from sevenn.nn._mliap_cuda_graph import MLIAPCudaGraph
from tests.unit_tests.test_arch_options import ARCHS
from tests.unit_tests.test_mliap_cuda_graph import _compute, _moved
from tests.unit_tests.test_model import cutoff
from tests.unit_tests.test_parallel_consistency import (
    GPU_CONFIGS,
    MODAL,
    _assert_close_f32,
    _atoms,
    _backend_available,
    _cf,
    _default_dtype,
    _graph,
    _run_mliap,
    _save_checkpoint,
    _serial_f32,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _backend_available('lunatp')),
    reason='needs CUDA and LunaTP',
)


def _fused_graph(cp, half_weights=False, knots=0):
    from sevenn.nn.lunatp_fused import fold_modal, fuse_radial_conv
    from sevenn.util import model_from_checkpoint

    model, _ = model_from_checkpoint(str(cp), enable_lunatp=True)
    model.set_is_batch_data(False)
    model.delete_module_by_key('force_output')
    model.requires_grad_(False)
    model.to('cuda').eval()
    names = fuse_radial_conv(model, half_weights=half_weights, knots=knots)
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


# nequip_sc: the species (fully connected) self connection, pruned
# knots: weight rows per distinct edge length (0) / in-kernel weight table
@pytest.mark.parametrize('knots', [0, 2048])
@pytest.mark.parametrize('name', ['nequip_sc'] + GPU_CONFIGS)
def test_fused_mliap_cuda_graph(tmp_path, name, knots):
    if name == 'moe':  # no capture (torch.nonzero per expert)
        pytest.skip('ffn_moe is not captured')
    cp = _save_checkpoint(_cf(name, ARCHS[0]), tmp_path / 'cp.pth')
    atoms = _atoms((2, 2, 2))
    with _default_dtype(torch.float32):
        graph, names = _fused_graph(cp, knots=knots)
        if not names:
            pytest.skip('no convolution of this config is fused')
        for step in range(4):
            moved = _moved(atoms, step)
            par, _, _ = _run_mliap(
                [graph], moved, (1, 1, 1), compute=_compute, device='cuda'
            )
            _assert_close_f32(par, _serial_f32(cp, moved), (name, knots, step))
    assert graph.enabled
    assert graph.n_captures == 1 and graph.n_replays == 3
    if name == 'nequip_sc':
        from sevenn.nn.lunatp_fused import PrunedSelfConnectionIntro

        pruned = [
            m for m in graph.model.modules()
            if isinstance(m, PrunedSelfConnectionIntro)
        ]
        n_species = len(set(atoms.numbers.tolist()))
        assert pruned and all(m.columns.numel() == n_species for m in pruned)
        if knots:
            from sevenn.nn.lunatp_fused import FusedFirstLayer

            first = [
                m for m in graph.model.modules() if isinstance(m, FusedFirstLayer)
            ]
            assert first and len(first[0].columns) == n_species


def test_fused_half_weights(tmp_path):
    """fp16 weight rows: the forces of the serial model to ~1e-3"""
    cp = _save_checkpoint(_cf(GPU_CONFIGS[0], ARCHS[0]), tmp_path / 'cp.pth')
    atoms = _moved(_atoms((2, 2, 2)), 1)
    with _default_dtype(torch.float32):
        graph, names = _fused_graph(cp, half_weights=True)
        if not names:
            pytest.skip('no convolution of this config is fused')
        par, _, _ = _run_mliap(
            [graph], atoms, (1, 1, 1), compute=_compute, device='cuda'
        )
        ref = _serial_f32(cp, atoms)
    err = float(abs(par[1] - ref[1]).max())
    assert 0.0 < err < 2e-2 * float(abs(ref[1]).max())


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


# the trajectory of the optimized path against the serial model's: forces and
# energies together, and what their differences add up to over the steps
@pytest.mark.parametrize('knots', [0, 2048])
def test_fused_md_energy(tmp_path, knots):
    from sevenn.util import model_from_checkpoint

    cp = _save_checkpoint(_cf('nequip_sc', ARCHS[0]), tmp_path / 'cp.pth')
    atoms = _atoms((2, 2, 2))
    steps = 40
    with _default_dtype(torch.float32):
        serial, _ = model_from_checkpoint(str(cp))
        serial.set_is_batch_data(False)
        if serial.modal_map:
            serial.prepare_modal_deploy(MODAL)
        serial.eval()

        def ref(a):
            g = _graph(a)
            for k, v in g.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    g[k] = v.float()
            out = serial(g)
            return (
                out['inferred_total_energy'].item(),
                out['inferred_force'].detach().numpy().astype(np.float64),
            )

        graph, names = _fused_graph(cp, knots=knots)
        assert names

        def new(a):
            return _run_mliap(
                [graph], a, (1, 1, 1), compute=_compute, device='cuda'
            )[0][:2]

        f0 = np.abs(ref(atoms)[1]).max()
        dt = float(np.sqrt(2.0 * 2e-3 / f0))  # first step: 2e-3 A at most
        e0, e_ref, tot_ref, x_ref = _md(ref, atoms, steps, dt)
        _, e_new, tot_new, x_new = _md(new, atoms, steps, dt)
    # fp32 energies of ~600 eV; measured: 1e-6 eV, 8e-6 A after atoms moved
    # by up to ~1 A and the potential energy changed by 0.08 eV
    tol = 5e-8 * max(1.0, abs(e_ref))
    assert abs(e_ref - e0) > 100 * tol  # the atoms moved
    assert abs(e_new - e_ref) <= tol, (e_new, e_ref)
    assert abs(tot_new - tot_ref) <= tol, (tot_new, tot_ref)
    assert np.abs(x_new - x_ref).max() <= 1e-4
