"""Apple GPU execution (sevenn/mps/). MPS tests are skipped
where no Apple GPU exists: a skip is not a pass."""

import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from e3nn.nn import Gate
from e3nn.o3 import Irreps, Linear, TensorProduct

import sevenn._keys as KEY
import sevenn.train.dataload as dl
from sevenn import mps
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.model_build import build_E3_equivariant_model
from sevenn.mps import runtime as mps_runtime
from sevenn.mps.kernels import generate_msl, probe_cg
from sevenn.mps.modules import mps_torch_supported
from sevenn.mps.padding import pad, padding_is_safe, unpad
from sevenn.util import chemical_species_preprocess

needs_mps = pytest.mark.skipif(
    not (
        torch.backends.mps.is_available()
        and mps_torch_supported()
        and hasattr(torch.mps, 'compile_shader')
    ),
    reason='needs an Apple GPU and PyTorch 2.7 or later (fused kernels)',
)
KEYS = (
    KEY.PRED_TOTAL_ENERGY,
    KEY.PRED_FORCE,
    KEY.PRED_STRESS,
    KEY.ATOMIC_ENERGY,
)


def _config(
    lmax=2,
    channel=4,
    T=3,
    cutoff=4.0,
    species=('Na', 'Cl'),
    **kw,
):
    """A small SevenNet model."""
    cfg = {
        'cutoff': cutoff,
        'channel': channel,
        'radial_basis': {'radial_basis_name': 'bessel'},
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'interaction_type': 'nequip',
        'lmax': lmax,
        'is_parity': True,
        'num_convolution_layer': T,
        'weight_nn_hidden_neurons': [64, 64],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': 30.0,
        'train_denominator': False,
        'self_connection_type': 'nequip',
        'shift': -10.0,
        'scale': 10.0,
        'train_shift_scale': False,
        'irreps_manual': False,
        'lmax_edge': -1,
        'lmax_node': -1,
        'readout_as_fcn': False,
        'use_bias_in_linear': False,
        '_normalize_sph': True,
    }
    cfg.update(chemical_species_preprocess(list(species)))
    cfg.update(kw)
    return cfg


def _build(cfg, seed=0):
    torch.manual_seed(seed)
    return build_E3_equivariant_model(cfg)


def _graph(seed=1, sheared=False):
    at = bulk('NaCl', 'rocksalt', a=5.63) * (2, 1, 1)
    if sheared:  # strongly anisotropic cell
        strain = [[1.0, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.3]]
        at.set_cell(at.cell[:] @ torch.tensor(strain).numpy(), scale_atoms=True)
    at.rattle(0.15 if sheared else 0.1, seed=seed)
    return AtomGraphData.from_numpy_dict(dl.unlabeled_atoms_to_graph(at, 4.0))


def _run(model, graph):
    g = graph.clone().to('mps')
    out = model(g)
    return {k: out[k].detach().cpu() for k in KEYS if k in out}


def _close(a, b, rel=2e-5):
    assert a.keys() == b.keys()
    for k in a:
        scale = max(float(a[k].abs().max()), 1.0)
        assert float((a[k] - b[k]).abs().max()) < rel * scale, k


def _fused(model):
    return any(type(m).__name__.startswith('Fused') for m in model.modules())


def _e3nn_on_gpu(model):
    """A model on the Apple GPU running the reference e3nn code: the tests switch the
    kernels on themselves."""
    model.to('mps')
    mps.restore(model)
    return model


def _model(**kw):
    model = _e3nn_on_gpu(_build(_config(**kw)))
    model.set_is_batch_data(False)
    return model.eval()


def _tp():
    ir_x = Irreps('4x0e+4x1o+4x2e')
    ir_y = Irreps('1x0e+1x1o+1x2e')
    ir_mid, ins = [], []
    for i, (m, a) in enumerate(ir_x):
        for j, (_, b) in enumerate(ir_y):
            for c in a * b:
                ins.append((i, j, len(ir_mid), 'uvu', True))
                ir_mid.append((m, c))
    return TensorProduct(
        ir_x, ir_y, Irreps(ir_mid), ins, shared_weights=False, internal_weights=False
    )


# ---- no GPU needed ----


def test_probed_coefficients_are_sparse_with_no_borderline_values():
    for P in probe_cg(_tp()):
        g = np.abs(P['g'])
        assert np.all((g < 1e-7) | (g > 1e-3))


def test_generated_source_has_all_kernels():
    src = generate_msl(_tp())
    for k in ('uvu_fwd', 'uvu_adj_x', 'uvu_adj_yw', 'uvu_sum_y'):
        assert f'kernel void {k}(' in src


def test_pad_shapes_noop_and_unpad():
    def graph():
        return AtomGraphData.from_numpy_dict(
            dl.unlabeled_atoms_to_graph(bulk('NaCl', 'rocksalt', a=5.63), 4.0)
        )

    plan = SimpleNamespace(expect_padded=None)
    g = graph()
    n = g[KEY.EDGE_IDX].shape[1]
    for bucket in (0, 1, n):
        assert pad(g, 4.0, bucket, plan) is None and g[KEY.EDGE_IDX].shape[1] == n
    assert plan.expect_padded is None
    assert pad(g, 4.0, n + 7, plan) == n
    assert g[KEY.EDGE_IDX].shape[1] == n + 7 and g[KEY.EDGE_VEC].shape[0] == n + 7
    assert plan.expect_padded == (n, n + 7)
    assert bool((g[KEY.EDGE_IDX][:, n:] == 0).all())
    assert bool((g[KEY.EDGE_VEC][n:, 0] == 4.0).all())
    ref = graph()
    unpad(g, n)
    for key in (KEY.EDGE_IDX, KEY.EDGE_VEC):
        assert torch.equal(g[key], ref[key])


def test_padding_is_safe_for_default_model():
    assert padding_is_safe(_build(_config()), 4.0)


def test_padding_is_not_safe_with_sigmoid_radial_activation():
    model = _build(_config(act_radial='sigmoid'))
    assert not padding_is_safe(model, 4.0)
    assert not padding_is_safe(model, 4.0, all_fused=True)


def test_large_radial_weights_only_matter_with_a_convolution_on_e3nn():
    model = _build(_config())
    with torch.no_grad():
        for m in model.modules():
            if getattr(m, 'weight_nn', None) is not None:
                for p in m.weight_nn.parameters():
                    p *= 1e4
    assert padding_is_safe(model, 4.0, all_fused=True)


def test_cuda_and_cpu_paths_do_not_import_the_apple_code():
    code = (
        'import sys, numpy, sevenn.calculator, sevenn.model_build, '
        'sevenn.checkpoint, sevenn.scripts.deploy\n'
        "m = sevenn.calculator.SevenNetCalculator('7net-0', device='cpu').model\n"
        "m.to('cpu').double().float()\n"
        "assert not [m for m in sys.modules if m.startswith('sevenn.mps')]\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)


# ---- Apple GPU ----


@needs_mps
def test_fused_equals_e3nn():
    from sevenn.mps import modules as mps_opt

    model = _model()
    graph = _graph()
    ref = _run(model, graph)
    rep = mps_opt.enable(model)
    assert rep['conv_fused'] == 3 and rep['conv_e3nn'] == 0 and rep['gate_e3nn'] == 0
    _close(ref, _run(model, graph))
    mps_opt.disable(model)
    _close(ref, _run(model, graph))  # MPS itself is not bitwise reproducible


@needs_mps
def test_state_dict_unchanged_and_strict_round_trip():
    from sevenn.mps import modules as mps_opt

    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mps_opt.enable(model)
    after = model.state_dict()
    assert list(before) == list(after)
    for k in before:
        assert torch.equal(before[k], after[k])
    model.load_state_dict(before, strict=True)


@needs_mps
def test_load_state_dict_after_enable_refreshes_the_tables():
    from sevenn.mps import modules as mps_opt

    graph = _graph()
    other = _e3nn_on_gpu(_build(_config(), seed=7))
    other.set_is_batch_data(False)
    other.eval()
    ref = _run(other, graph)
    model = _model()
    mps_opt.enable(model)
    model.load_state_dict(other.state_dict(), strict=True)
    _close(ref, _run(model, graph))


@needs_mps
def test_tensor_product_kernels_equal_e3nn_on_random_graph_with_unsorted_edges():
    from sevenn.mps.kernels import EdgePlan, FusedTP, FusedUvu

    tp = _tp().to('mps')
    K = FusedTP(tp)
    N, E = 17, 400
    torch.manual_seed(0)
    x = torch.randn(N, tp.irreps_in1.dim, device='mps', requires_grad=True)
    y = torch.randn(E, tp.irreps_in2.dim, device='mps', requires_grad=True)
    w = torch.randn(E, tp.weight_numel, device='mps', requires_grad=True)
    idx = torch.randint(
        0, N - 2, (2, E), device='mps'
    )  # repeated, unsorted, two nodes without edges
    ref = torch.zeros(N, K.nout, device='mps').index_add_(
        0, idx[0], tp(x[idx[1]], y, w)
    )
    out = FusedUvu.apply(x, y, w, K, EdgePlan().get(idx, N))
    assert float((ref - out).detach().abs().max()) < 1e-5
    v = torch.randn_like(out)
    for a, b in zip(
        torch.autograd.grad((ref * v).sum(), [x, y, w]),
        torch.autograd.grad((out * v).sum(), [x, y, w]),
    ):
        assert float((a - b).abs().max()) < 1e-4


def _tp_unequal():
    ir_x = Irreps('8x0e+4x1o+2x2e')
    ir_y = Irreps('1x0e+1x1o+1x2e')
    ir_mid, ins = [], []
    for i, (m, a) in enumerate(ir_x):
        for j, (_, b) in enumerate(ir_y):
            for c in a * b:
                ins.append((i, j, len(ir_mid), 'uvu', True))
                ir_mid.append((m, c))
    ir_mid = Irreps(ir_mid)
    ir_sorted, p, _ = ir_mid.sort()
    ins = sorted(((i, j, p[k], c, t) for i, j, k, c, t in ins), key=lambda z: z[2])
    return TensorProduct(
        ir_x, ir_y, ir_sorted, ins, shared_weights=False, internal_weights=False
    )


@needs_mps
@pytest.mark.parametrize('tp_fn', [_tp, _tp_unequal])
def test_kernels_equal_e3nn_to_third_order(tp_fn):
    from sevenn.mps.kernels import EdgePlan, FusedTP, FusedUvu

    torch.manual_seed(0)
    tp = tp_fn().to('mps')
    N, E = 7, 23
    # unsorted edges, an atom with no edges, and repeated pairs
    idx = torch.randint(0, N - 1, (2, E), device='mps')
    x = torch.randn(N, tp.irreps_in1.dim, device='mps', requires_grad=True)
    y = torch.randn(E, tp.irreps_in2.dim, device='mps', requires_grad=True)
    w = torch.randn(E, tp.weight_numel, device='mps', requires_grad=True)

    def ref(x, y, w):
        return torch.zeros(N, tp.irreps_out.dim, device='mps').index_add(
            0, idx[0], tp(x[idx[1]], y, w)
        )

    def fused(x, y, w):
        return FusedUvu.apply(x, y, w, FusedTP(tp), EdgePlan().get(idx, N))

    probe = torch.randn(N, tp.irreps_out.dim, device='mps')
    got = {}
    for name, f in (('ref', ref), ('fused', fused)):
        out = f(x, y, w)
        g1 = torch.autograd.grad((out * probe).sum(), (x, y, w), create_graph=True)
        s2 = sum(g.pow(2).sum() for g in g1)
        g2 = torch.autograd.grad(s2, (x, y, w), create_graph=True)
        g3 = torch.autograd.grad(sum((g * g.detach()).sum() for g in g2), (x, y, w))
        got[name] = [out.detach()] + [g.detach() for g in g1 + g2 + g3]
    for k, (a, b) in enumerate(zip(got['ref'], got['fused'])):
        assert float((a - b).abs().max()) < 2e-5 * max(float(a.abs().max()), 1e-6), k


@needs_mps
@pytest.mark.parametrize('tp_fn', [_tp, _tp_unequal])
def test_kernels_mixed_derivatives_through_noncontiguous_inputs(tp_fn):
    """x, y and w as transposed views of leaves: mixed derivatives to third order
    reach the leaves and equal e3nn's (a contiguous copy saved inside the forward
    would cut the graph to the caller's tensor)."""
    from sevenn.mps.kernels import EdgePlan, FusedTP, FusedUvu

    torch.manual_seed(5)
    tp = tp_fn().to('mps')
    N, E = 7, 23
    idx = torch.randint(0, N - 1, (2, E), device='mps')
    X = torch.randn(tp.irreps_in1.dim, N, device='mps', requires_grad=True)
    Y = torch.randn(tp.irreps_in2.dim, E, device='mps', requires_grad=True)
    W = torch.randn(tp.weight_numel, E, device='mps', requires_grad=True)
    K = FusedTP(tp)

    def ref(x, y, w):
        return torch.zeros(N, tp.irreps_out.dim, device='mps').index_add(
            0, idx[0], tp(x[idx[1]], y, w)
        )

    def fused(x, y, w):
        return FusedUvu.apply(x, y, w, K, EdgePlan().get(idx, N))

    probe = torch.randn(N, tp.irreps_out.dim, device='mps')
    got = {}
    for name, f in (('ref', ref), ('fused', fused)):
        out = f(X.t(), Y.t(), W.t())
        # out is linear in x: square it, so that every level below depends on X
        (gx,) = torch.autograd.grad((out.pow(2) * probe).sum(), X, create_graph=True)
        gyw = torch.autograd.grad(gx.pow(2).sum(), (Y, W), create_graph=True)
        (gxx,) = torch.autograd.grad(sum(g.pow(2).sum() for g in gyw), X)
        got[name] = [out.detach(), gx.detach(), *[g.detach() for g in gyw], gxx]
    for k, (a, b) in enumerate(zip(got['ref'], got['fused'])):
        assert float((a - b).abs().max()) < 2e-5 * max(float(a.abs().max()), 1e-6), k


@needs_mps
def test_kernels_with_unequal_multiplicities_equal_e3nn():
    from sevenn.mps.kernels import EdgePlan, FusedTP, FusedUvu

    tp = _tp_unequal().to('mps')
    K = FusedTP(tp)
    N, E = 13, 300
    torch.manual_seed(1)
    x = torch.randn(N, tp.irreps_in1.dim, device='mps', requires_grad=True)
    y = torch.randn(E, tp.irreps_in2.dim, device='mps', requires_grad=True)
    w = torch.randn(E, tp.weight_numel, device='mps', requires_grad=True)
    idx = torch.randint(0, N - 1, (2, E), device='mps')
    ref = torch.zeros(N, K.nout, device='mps').index_add_(
        0, idx[0], tp(x[idx[1]], y, w)
    )
    out = FusedUvu.apply(x, y, w, K, EdgePlan().get(idx, N))
    assert float((ref - out).detach().abs().max()) < 1e-5
    v = torch.randn_like(out)
    for a, b in zip(
        torch.autograd.grad((ref * v).sum(), [x, y, w]),
        torch.autograd.grad((out * v).sum(), [x, y, w]),
    ):
        assert float((a - b).abs().max()) < 1e-4
    # pad edges at the end are skipped and get zero gradients
    plan = EdgePlan()
    idx = idx.clone()
    idx[:, E - 5 :] = 0
    plan.expect_padded = (E - 5, E)
    plan.get(idx, N)
    w0 = w.detach().clone()
    w0[E - 5 :] = 0.0
    w0.requires_grad_(True)
    out = FusedUvu.apply(x, y, w0, K, plan)
    gy, gw = torch.autograd.grad((out * v).sum(), [y, w0])
    assert (
        float(gy[E - 5 :].abs().max()) == 0.0
        and float(gw[E - 5 :].abs().max()) == 0.0
    )


@needs_mps
def test_model_with_unequal_channels_runs_fully_fused():
    from sevenn.mps import modules as mps_opt

    model = _model(
        channel=[8, 8, 8, 8],
        lmax=2,
        irreps_manual=['8x0e', '8x0e+4x1o+2x2e', '8x0e+4x1o+2x2e', '8x0e'],
    )
    graph = _graph()
    ref = _run(model, graph)
    rep = mps_opt.enable(model)
    assert rep['conv_e3nn'] == 0 and rep['conv_fused'] == 3
    _close(ref, _run(model, graph))


@needs_mps
def test_kernels_respect_views_with_storage_offset():
    from sevenn.mps.kernels import CsrApply, CsrMap

    x = torch.arange(12.0, device='mps').reshape(4, 3)
    M = CsrMap(
        [(np.array([0, 2]), np.array([0, 1]), np.array([1.0, 1.0]))],
        np.zeros(2),
        3,
        2,
        'mps',
    )
    out = CsrApply.apply(x[1:], M, None)  # view with an offset
    assert out.cpu().tolist() == [[3.0, 5.0], [6.0, 8.0], [9.0, 11.0]]
    out = CsrApply.apply(
        x[:, 1:].reshape(4, 2).repeat(1, 2)[:, :3], M, None
    )  # non-contiguous source
    assert out.shape == (4, 2)


@needs_mps
def test_large_linear_layers_stay_on_e3nn(monkeypatch):
    from sevenn.mps import modules as mps_opt

    model = _model()
    graph = _graph()
    ref = _run(model, graph)
    monkeypatch.setattr(mps_opt, 'LINEAR_MAX_NNZ', 0)
    rep = mps_opt.enable(model)
    assert rep['linear_fused'] == 0 and rep['linear_e3nn'] > 0
    _close(ref, _run(model, graph))
    mps_opt.disable(model)


@needs_mps
def test_fused_linear_with_bias_and_gradients():
    from sevenn.mps import modules as mps_opt

    lin = Linear(Irreps('3x0e+2x1o'), Irreps('2x0e+1x1o'), biases=True).to('mps')
    x = torch.randn(5, lin.irreps_in.dim, device='mps', requires_grad=True)
    ref = lin(x)
    gref = torch.autograd.grad(ref.pow(2).sum(), x)[0]
    mps_opt._fuse(lin, mps_opt.FusedLinear)
    out = lin(x)
    assert float((ref - out).abs().max()) < 1e-6
    assert (
        float((gref - torch.autograd.grad(out.pow(2).sum(), x)[0]).abs().max())
        < 1e-5
    )


@needs_mps
@pytest.mark.parametrize('scale', [1.0, 100.0])
def test_fused_gate_tables_and_gradients(scale):
    # scale 100: exp and tanh of large arguments stay finite in the Metal kernels
    from sevenn.mps import modules as mps_opt

    gate = Gate(
        '3x0e+2x0o',
        [torch.nn.functional.silu, torch.tanh],
        '2x0e+2x0o',
        [torch.nn.functional.silu, torch.tanh],
        '2x1o+2x2e',
    ).to('mps')
    x = (torch.randn(6, gate.irreps_in.dim, device='mps') * scale).requires_grad_()
    ref = gate(x)
    gref = torch.autograd.grad(ref.pow(2).sum(), x)[0]
    gate._tables = mps_opt._gate_tables(gate, 'mps')
    gate.__class__ = mps_opt.FusedGate
    out = gate(x)
    gout = torch.autograd.grad(out.pow(2).sum(), x)[0]
    assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(gout).all())
    assert float((ref - out).abs().max()) < 1e-6 * max(float(ref.abs().max()), 1.0)
    tol = 1e-5 * max(float(gref.abs().max()), 1.0)
    assert float((gref - gout).abs().max()) < tol


@needs_mps
def test_unsupported_gate_activation_keeps_e3nn_on_the_gpu():
    from sevenn.mps import modules as mps_opt

    model = _model(act_scalar={'e': 'ssp', 'o': 'tanh'})
    graph = _graph()
    ref = _run(model, graph)
    with pytest.warns(UserWarning, match='gate keeps the e3nn code'):
        rep = mps_opt.enable(model)
    assert (
        rep['gate_fused'] == 0 and rep['gate_e3nn'] == 3 and rep['conv_fused'] == 3
    )
    _close(ref, _run(model, graph))


@needs_mps
@pytest.mark.parametrize('bucket', [37, 67, 1024])
def test_pad_edges_are_silent_in_fused_kernels(bucket):
    from sevenn.mps import modules as mps_opt

    model = _model()
    mps_opt.enable(model)
    at = bulk('NaCl', 'rocksalt', a=5.63).repeat((2, 1, 1))
    at.rattle(0.05, seed=3)
    g0 = dl.unlabeled_atoms_to_graph(at, 4.0)
    ref = _run(model, AtomGraphData.from_numpy_dict(dict(g0)))

    def padded(plan):
        g = AtomGraphData.from_numpy_dict(dict(g0)).to('mps')
        assert pad(g, 4.0, bucket, plan) == g0['edge_index'].shape[1]
        return g

    g = padded(model._mps_plan)  # the plan is told which edges are pads
    out = model(g)
    out = {k: out[k].detach().cpu() for k in KEYS if k in out}
    for k in (KEY.PRED_TOTAL_ENERGY, KEY.PRED_FORCE, KEY.PRED_STRESS):
        assert float((ref[k] - out[k]).abs().max()) < 3e-5 * max(
            float(ref[k].abs().max()), 1.0
        ), k
    # padded but treated as real edges: same result, and all derivatives finite
    out = _run(model, padded(SimpleNamespace(expect_padded=None)))
    for k, v in out.items():
        assert torch.isfinite(v).all(), k
    _close(ref, out)


def _calc_pair(bucket=None):
    """Reference and fused calculators, on two instances of the same model."""
    a = _calc(_build(_config()), off=True)
    b = _calc(_build(_config()))
    if bucket is not None:
        mps.prepare(b.model, bucket=bucket)
    return a, b


@needs_mps
@pytest.mark.parametrize('cutoff', [4.0, 5.3, 6.0])
def test_default_calculator_really_pads_on_the_gpu(cutoff):
    model = _build(_config(cutoff=cutoff))
    calc = SevenNetCalculator(model, file_type='model_instance', device='mps')
    assert _fused(calc.model) and calc.model._mps.bucket == 256
    at = bulk('NaCl', 'rocksalt', a=5.63).repeat((2, 1, 1))
    at.rattle(0.05, seed=3)
    at.calc = calc
    at.get_forces()
    plan = calc.model._mps_plan
    n = dl.unlabeled_atoms_to_graph(at, cutoff)['edge_index'].shape[1]
    assert plan.num_edges % 256 == 0 and plan.num_real == n < plan.num_edges


@needs_mps
def test_calculator_fuses_the_instance_it_is_given_and_drops_the_pad_edges():
    model = _build(_config())
    b = _calc(model)
    a = _calc(_build(_config()), off=True)
    assert b.model is model and _fused(model)  # moving it to 'mps' switched it
    assert mps.report(model)['conv_fused'] == 3 and mps.report(a.model) == {}
    at = bulk('NaCl', 'rocksalt', a=5.63).repeat((2, 1, 1))
    at.rattle(0.05, seed=3)
    seen = []
    b.model.register_forward_hook(
        lambda m, args, out: seen.append(int(out[KEY.EDGE_IDX].shape[1]))
    )
    for dx in (0.0, 0.03, 0.2):  # the edge count changes
        at.positions += np.random.default_rng(1).normal(
            0, dx + 1e-3, at.positions.shape
        )
        at.calc = a
        fa, ea = at.get_forces(), at.get_potential_energy()
        at.calc = b
        fb, eb = at.get_forces(), at.get_potential_energy()
        n = dl.unlabeled_atoms_to_graph(at, 4.0)['edge_index'].shape[1]
        assert seen[-1] == n  # the output carries the real edges only
        assert abs(ea - eb) < 1e-3 and np.abs(fa - fb).max() < 2e-4


@needs_mps
@pytest.mark.parametrize('bucket', [0, 256])
def test_isolated_atoms_without_edges(bucket):
    a, b = _calc_pair(bucket=bucket)
    at = Atoms(
        'NaCl',
        positions=[[0, 0, 0], [30.0, 0, 0]],
        cell=[60.0, 60.0, 60.0],
        pbc=True,
    )
    at.calc = a
    ea, fa = at.get_potential_energy(), at.get_forces()
    at.calc = b
    eb, fb = at.get_potential_energy(), at.get_forces()
    assert abs(ea - eb) < 1e-4 and np.all(fa == 0.0) and np.all(fb == 0.0)
    assert b.model._mps.n_real is None  # nothing to pad


@needs_mps
def test_train_mode_runs_the_kernels_and_supports_force_training(
    monkeypatch
):
    from sevenn.mps import modules as mps_opt

    ref = _e3nn_on_gpu(_build(_config()))
    ref.set_is_batch_data(False)
    model = _e3nn_on_gpu(_build(_config()))
    model.set_is_batch_data(False)
    rep = mps_opt.enable(model)
    assert rep['conv_fused'] > 0 and rep['conv_e3nn'] == 0
    calls = []
    apply = mps_opt.FusedUvu.apply
    monkeypatch.setattr(
        mps_opt.FusedUvu, 'apply', lambda *a: calls.append(1) or apply(*a)
    )
    graph = _graph()
    ref.train()
    model.train()
    losses = []
    for m in (ref, model):
        out = m(graph.clone().to('mps'))
        loss = out[KEY.PRED_FORCE].pow(2).mean()
        loss.backward()
        losses.append(float(loss.detach().cpu()))
    assert abs(losses[0] - losses[1]) < 1e-5 * losses[0]
    for (n, p), (_, q) in zip(ref.named_parameters(), model.named_parameters()):
        if p.grad is not None:
            assert float((p.grad - q.grad).abs().max()) < 1e-4 * max(
                float(p.grad.abs().max()), 1e-3
            ), n
    assert len(calls) == rep['conv_fused']
    model.eval()  # back to the kernels; weights changed in place are picked up
    with torch.no_grad():
        for p, q in zip(ref.parameters(), model.parameters()):
            p += 1e-3
            q += 1e-3
    ref.eval()
    _close(_run(ref, graph), _run(model, graph))


@needs_mps
def test_train_mode_has_second_derivatives_also_with_unequal_channels():
    from sevenn.mps import modules as mps_opt

    graph = _graph()
    for kw in (
        {},
        dict(
            channel=[8, 8, 8, 8],
            lmax=2,
            irreps_manual=['8x0e', '8x0e+4x1o+2x2e', '8x0e+4x1o+2x2e', '8x0e'],
        ),
    ):
        ref = _e3nn_on_gpu(_build(_config(**kw)))
        model = _e3nn_on_gpu(_build(_config(**kw)))
        for m in (ref, model):
            m.set_is_batch_data(False)
        rep = mps_opt.enable(model)
        assert rep['conv_fused'] > 0 and rep['conv_e3nn'] == 0
        ref.train()
        model.train()
        g = []
        for m in (ref, model):
            out = m(graph.clone().to('mps'))
            f = out[KEY.PRED_FORCE]
            # a second derivative through the forces, as a force loss needs
            v = torch.autograd.grad(
                f.pow(2).sum(), out[KEY.EDGE_VEC], create_graph=True
            )[0]
            w = next(p for p in m.parameters() if p.requires_grad and p.dim() > 1)
            g.append(torch.autograd.grad(v.sum(), w, allow_unused=True)[0])
        assert g[0] is not None and g[1] is not None
        assert float((g[0] - g[1]).abs().max()) < 1e-4 * max(
            float(g[0].abs().max()), 1e-3
        )


@needs_mps
@pytest.mark.parametrize('all_types', [8, 0])
def test_train_mode_node_layers_avoid_e3nn_and_equal_it(all_types, monkeypatch):
    from e3nn.nn import Gate as E3Gate
    from e3nn.o3 import FullyConnectedTensorProduct

    from sevenn.mps import modules as mps_opt

    graph = _graph()
    ref = _e3nn_on_gpu(_build(_config()))
    model = _e3nn_on_gpu(_build(_config()))
    for m in (ref, model):
        m.set_is_batch_data(False)
        m.train()
    mps_opt.enable(model)
    monkeypatch.setattr(
        mps_opt, 'SELFCONN_ALL_TYPES', all_types
    )  # 0: the loop over atom types present
    grads, fell_back = [], []
    for m, fused in ((ref, False), (model, True)):
        # record which classes call e3nn's own forward, with the shape of their input
        if fused:
            for cls in (Linear, FullyConnectedTensorProduct, E3Gate):

                def spy(self, *a, _orig=cls.forward, **k):
                    fell_back.append((type(self).__name__, a[0].dim()))
                    return _orig(self, *a, **k)

                monkeypatch.setattr(cls, 'forward', spy)
        out = m(graph.clone().to('mps'))
        (
            out[KEY.PRED_FORCE].pow(2).mean()
            + out[KEY.PRED_TOTAL_ENERGY].pow(2).mean()
        ).backward()
        grads.append(
            {
                n: p.grad.detach().cpu().clone()
                for n, p in m.named_parameters()
                if p.grad is not None
            }
        )
    # a fused layer may fall back only for inputs its kernels do not take
    assert not [
        f
        for f in fell_back
        if f[0] in ('FusedLinear', 'FusedSelfConnection', 'FusedGate') and f[1] == 2
    ], fell_back
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        assert float((grads[0][n] - grads[1][n]).abs().max()) < 1e-4 * max(
            float(grads[0][n].abs().max()), 1e-3
        ), n


def test_weight_map_refuses_a_map_that_is_not_one_weight_per_entry():
    from sevenn.mps.modules import _weight_map

    w = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    with pytest.raises(NotImplementedError, match='one weight times a constant'):
        _weight_map(w, lambda: torch.stack([w[0] * w[1], w[2]]))


@needs_mps
def test_trainer_switches_on_only_for_the_apple_gpu(monkeypatch):
    from sevenn.mps import modules as mps_opt
    from sevenn.train.trainer import Trainer

    for device, env, expect in (
        ('mps', None, True),
        ('mps', '1', False),
        ('cpu', None, False),
    ):
        if env is None:
            monkeypatch.delenv('SEVENN_DISABLE_MPS', raising=False)
        else:
            monkeypatch.setenv('SEVENN_DISABLE_MPS', env)
        tr = Trainer(_build(_config()), [], torch.optim.Adam, device=device)
        assert (
            any(isinstance(m, mps_opt.FusedConvolution) for m in tr.model.modules())
            == expect
        ), (device, env)


@needs_mps
def test_second_derivatives_raise_in_the_linear_kernel():
    from sevenn.mps import modules as mps_opt

    # the convolution kernels have them (see the third-order test); linear and gate
    # kernels do not
    lin = Linear(Irreps('3x0e+2x1o'), Irreps('2x0e+1x1o')).to('mps').eval()
    mps_opt._fuse(lin, mps_opt.FusedLinear)
    x = torch.randn(5, lin.irreps_in.dim, device='mps', requires_grad=True)
    with pytest.raises(
        RuntimeError, match='linear and gate layers have first derivatives only'
    ):
        torch.autograd.grad(lin(x).pow(2).sum(), x, create_graph=True)


@needs_mps
def test_batched_data_equals_single_graphs():
    from torch_geometric.loader.dataloader import Collater

    from sevenn.mps import modules as mps_opt

    model = _model()
    mps_opt.enable(model)
    g1, g2 = _graph(seed=1), _graph(seed=2)
    r1, r2 = _run(model, g1), _run(model, g2)
    model.set_is_batch_data(True)
    out = model(Collater([g1, g2])([g1.clone(), g2.clone()]).to('mps'))
    F = out[KEY.PRED_FORCE].detach().cpu()
    assert (
        float((torch.cat([r1[KEY.PRED_FORCE], r2[KEY.PRED_FORCE]]) - F).abs().max())
        < 3e-5
    )
    E = out[KEY.PRED_TOTAL_ENERGY].detach().cpu()
    assert (
        abs(float(E[0] - r1[KEY.PRED_TOTAL_ENERGY])) < 1e-3
        and abs(float(E[1] - r2[KEY.PRED_TOTAL_ENERGY])) < 1e-3
    )


@needs_mps
def test_stale_padding_announcement_does_not_cut_edges():
    from sevenn.mps.kernels import EdgePlan

    idx = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device='mps')
    p = EdgePlan()
    p.expect_padded = (2, 4)  # same count, but the tail is not pad edges
    p.get(idx, 3)
    assert p.num_real == 4
    p.expect_padded = (2, 9)  # other count
    p.get(idx.clone(), 3)
    assert p.num_real == 4 and p.expect_padded is None


@needs_mps
def test_instance_under_default_file_type():
    model = _build(_config())
    calc = SevenNetCalculator(model, device='mps')
    assert _fused(calc.model) and mps.report(calc.model)['conv_fused'] == 3


@needs_mps
def test_deepcopy_after_load_state_dict_uses_the_new_weights():
    import copy

    from sevenn.mps import modules as mps_opt

    graph = _graph()
    other = _e3nn_on_gpu(_build(_config(), seed=7))
    other.set_is_batch_data(False)
    other.eval()
    ref = _run(other, graph)
    model = _model()
    mps_opt.enable(model)
    model.load_state_dict(other.state_dict(), strict=True)
    _close(ref, _run(copy.deepcopy(model), graph))


@needs_mps
def test_enable_leaves_the_global_random_generator_alone():
    from sevenn.mps import modules as mps_opt

    model = _model()
    torch.manual_seed(123)
    a = torch.randn(3)
    torch.manual_seed(123)
    mps_opt.enable(model)
    assert torch.equal(a, torch.randn(3))


@needs_mps
def test_too_large_graph_is_refused():
    from sevenn.mps.kernels import EdgePlan, FusedTP, FusedUvu

    K = FusedTP(_tp().to('mps'))
    plan = EdgePlan()
    plan.num_edges, plan.num_nodes = 2**28, 10
    x = torch.zeros(1, K.nx, device='mps')
    with pytest.raises(RuntimeError, match='too large'):
        FusedUvu.apply(x, x, x, K, plan)


@needs_mps
def test_edge_plan_notices_new_tensors_and_in_place_changes():
    from sevenn.mps.kernels import EdgePlan

    idx = torch.tensor([[0, 1, 1], [1, 0, 2]], device='mps')
    p = EdgePlan()
    p.get(idx, 3)
    assert p.dst.cpu().tolist() == [0, 1, 1]
    idx[0, 0] = 2
    p.get(idx, 3)
    assert p.dst.cpu().tolist() == [2, 1, 1]
    p.get(torch.tensor([[1, 1], [0, 2]], device='mps'), 3)
    assert p.num_edges == 2


def _calc(model, off=False, **kw):
    """A calculator on the Apple GPU; off: the opt-out (reference e3nn code)."""
    with mock.patch.dict(os.environ, {'SEVENN_DISABLE_MPS': '1' if off else ''}):
        return SevenNetCalculator(
            model, file_type='model_instance', device='mps', **kw
        )


def _fused_against_off(at, **kw):
    results = []
    for off in (True, False):
        calc = _calc(_build(_config(**kw)), off=off)
        a = at.copy()
        a.calc = calc
        results.append(
            (a.get_potential_energy(), a.get_forces(), a.get_stress(), calc)
        )
    return results


@needs_mps
@pytest.mark.parametrize(
    'name, at',
    [
        (
            'no_edges',
            Atoms(
                'Na2', positions=[[0, 0, 0], [20, 0, 0]], cell=[40, 40, 40], pbc=True
            ),
        ),
        (
            'one_atom_no_edges',
            Atoms('Cl', positions=[[1, 1, 1]], cell=[30, 30, 30], pbc=True),
        ),
        (
            'isolated_atom_beside_a_dimer',
            Atoms(
                'NaClCl',
                positions=[[0, 0, 0], [2, 0, 0], [15, 15, 15]],
                cell=[30, 30, 30],
                pbc=True,
            ),
        ),
        (
            'one_atom_self_images',
            Atoms('Na', positions=[[0, 0, 0]], cell=[3.1, 3.1, 3.1], pbc=True),
        ),
    ],
)
def test_few_or_no_edges(name, at):
    (e0, f0, s0, _), (e1, f1, s1, _) = _fused_against_off(at)
    assert np.isfinite(e1) and np.isfinite(f1).all() and np.isfinite(s1).all()
    assert abs(e0 - e1) < 1e-4
    assert np.abs(f0 - f1).max() < 1e-4
    assert np.abs(s0 - s1).max() < 1e-4


@needs_mps
def test_lmax_4():
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=2)
    (e0, f0, s0, _), (e1, f1, s1, calc) = _fused_against_off(
        at, lmax=4, channel=4
    )
    report = mps.report(calc.model)
    assert report['conv_e3nn'] == 0 and report['conv_fused'] > 0
    assert abs(e0 - e1) < 1e-3
    assert np.abs(f0 - f1).max() < 1e-4
    assert np.abs(s0 - s1).max() < 1e-4


@needs_mps
def test_weights_loaded_on_the_cpu_are_used_after_moving_back():
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=2)
    calc = _calc(_build(_config()))
    a = at.copy()
    a.calc = calc
    e0 = a.get_potential_energy()
    calc.model.to('cpu')
    names = set(dict(calc.model.named_parameters()))
    state = {
        k: v.clone() * 1.01 if k in names else v.clone()
        for k, v in calc.model.state_dict().items()
    }
    calc.model.load_state_dict(state)
    calc.model.to('mps')
    calc.reset()  # ASE would return the cached result for the same geometry
    a = at.copy()
    a.calc = calc
    ref = _calc(_build(_config()), off=True)
    ref.model.load_state_dict({k: v.to('mps') for k, v in state.items()})
    b = at.copy()
    b.calc = ref
    assert abs(a.get_potential_energy() - e0) > 1e-2  # the weights did change
    assert abs(a.get_potential_energy() - b.get_potential_energy()) < 1e-3
    assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-4


@needs_mps
def test_enable_refuses_other_float_types():
    from sevenn.mps.modules import enable

    model = _e3nn_on_gpu(_build(_config())).to(torch.float16)
    with pytest.raises(ValueError, match='float32'):
        enable(model)


@needs_mps
def test_failed_call_leaves_no_padding_note_for_the_next_graph():
    calc = _calc(_build(_config()))
    ref = _calc(_build(_config()), off=True)
    a = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    a.rattle(0.05, seed=5)
    a.calc = calc

    def boom(data):
        raise RuntimeError('boom')

    good = calc.model.forward
    calc.model.forward = boom
    with pytest.raises(RuntimeError, match='boom'):
        a.get_potential_energy()
    plan = calc.model._mps_plan
    assert plan.expect_padded is not None  # the hook ran, the call failed
    calc.model.forward = good
    b = a.copy()
    b.calc = ref
    assert abs(a.get_potential_energy() - b.get_potential_energy()) < 1e-3
    assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-4
    assert plan.expect_padded is None


@needs_mps
def test_no_padding_with_a_convolution_left_on_e3nn(monkeypatch):
    real = mps_runtime.enable

    def partly(model, parts='all'):
        report = real(model, 'linear+selfconn+gate')
        report['conv_e3nn'] = 1
        return report

    monkeypatch.setattr(mps_runtime, 'enable', partly)
    model = _build(_config())
    with pytest.warns(UserWarning, match='padding disabled'):
        model.to('mps')
    assert model._mps.bucket == 0 and not model._forward_pre_hooks


@needs_mps
def test_autocast_never_reaches_the_kernels(monkeypatch):
    from sevenn.mps import kernels

    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=2)
    calc = _calc(_build(_config()))
    at.calc = calc
    calls = []
    for name in ('_UvuFwd', 'CsrApply', 'GateApply'):
        fn = getattr(kernels, name)
        monkeypatch.setattr(
            fn, 'apply', lambda *a, _n=name, _f=fn.apply: calls.append(_n) or _f(*a)
        )
    real_pad = mps_runtime.pad
    monkeypatch.setattr(
        mps_runtime, 'pad', lambda *a, **k: calls.append('pad') or real_pad(*a, **k)
    )
    forces = None
    with torch.autocast(device_type='mps', dtype=torch.float16):
        try:
            forces = at.get_forces()
        except RuntimeError:
            pass  # as the e3nn code does
    assert calls == []  # neither the kernels nor the padding, whatever the outcome
    if forces is not None:
        assert np.isfinite(forces).all()


def _fused_convolution():
    calc = _calc(_build(_config()))
    conv = calc.model._modules['0_convolution']
    K = conv._kernels
    idx = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], device='mps')
    x = torch.randn(4, K.nx, device='mps')
    y = torch.randn(4, K.ny, device='mps')
    w = torch.randn(4, K.nw, device='mps')
    return conv, K, idx, x, y, w


@needs_mps
def test_kernels_refuse_other_types_and_shapes():
    from sevenn.mps.kernels import EdgePlan, FusedUvu

    _, K, idx, x, y, w = _fused_convolution()
    plan = EdgePlan().get(idx, 4)
    FusedUvu.apply(x, y, w, K, plan)
    for a, b, c in (
        (x, y.half(), w),
        (x, y, w.half()),
        (x[:3], y, w),
        (x, y[:3], w),
        (x, y, w[:, :-1]),
    ):
        with pytest.raises(ValueError, match='expected float32'):
            FusedUvu.apply(a, b, c, K, plan)


@needs_mps
def test_float16_edge_input_takes_the_reference_path():
    conv, K, idx, x, y, _ = _fused_convolution()
    y16 = y.half()
    wi = torch.randn(4, conv.weight_nn.hs[0], device='mps')
    mixed = conv(
        {
            KEY.EDGE_IDX: idx,
            KEY.NODE_FEATURE: x,
            KEY.EDGE_ATTR: y16,
            KEY.EDGE_EMBEDDING: wi,
        }
    )[KEY.NODE_FEATURE]
    same = conv(
        {
            KEY.EDGE_IDX: idx,
            KEY.NODE_FEATURE: x,
            KEY.EDGE_ATTR: y16.float(),
            KEY.EDGE_EMBEDDING: wi,
        }
    )[KEY.NODE_FEATURE]
    assert float((mixed - same).abs().max()) < 1e-4


def test_edge_plan_refuses_indices_out_of_range():
    from sevenn.mps.kernels import EdgePlan

    for bad in ([[0, 4], [1, 0]], [[0, -1], [1, 0]], [[0, 1], [5, 0]]):
        with pytest.raises(ValueError, match='out of range'):
            EdgePlan().get(torch.tensor(bad), 4)


@needs_mps
def test_module_without_tables_stays_as_it_was():
    from e3nn.o3 import FullyConnectedTensorProduct

    from sevenn.mps.modules import FusedSelfConnection, _fuse

    tp = FullyConnectedTensorProduct('4x0e', '2x0e', '4x1o')  # no paths
    with pytest.raises(NotImplementedError):
        _fuse(tp, FusedSelfConnection)
    assert type(tp) is FullyConnectedTensorProduct


@needs_mps
def test_output_irrep_with_more_channels_is_refused():
    from sevenn.mps.kernels import FusedTP

    tp = TensorProduct(
        '2x0e',
        '1x0e',
        '2x0e+4x1o',
        [(0, 0, 0, 'uvu', True)],
        internal_weights=False,
        shared_weights=False,
    )
    with pytest.raises(NotImplementedError, match='more channels'):
        FusedTP(tp)


@needs_mps
def test_self_connection_uses_e3nn_when_types_carry_gradients():
    calc = _calc(_build(_config()))
    tp = calc.model._modules['0_self_connection_intro'].fc_tensor_product
    x = torch.randn(3, tp.irreps_in1.dim, device='mps')
    y = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device='mps', requires_grad=True
    )
    tp(x, y).sum().backward()
    assert y.grad is not None and float(y.grad.abs().max()) > 0


# ---- edge cases ----


@needs_mps
def test_enable_twice_is_the_same_as_once():
    from sevenn.mps import modules as mps_opt

    model = _model()
    graph = _graph()
    ref = _run(model, graph)
    first = mps_opt.enable(model, parts='conv')
    second = mps_opt.enable(model)
    third = mps_opt.enable(model)
    assert first['linear_fused'] == 0 and second == third
    assert second['selfconn_fused'] > 0 and second['selfconn_e3nn'] == 0
    convs = [m for m in model.modules() if isinstance(m, mps_opt.FusedConvolution)]
    assert convs and all(m._plan is model._mps_plan for m in convs)
    _close(ref, _run(model, graph))


@needs_mps
def test_declined_self_connection_is_left_as_it_was():
    from sevenn.mps import modules as mps_opt
    from sevenn.nn.self_connection import SelfConnectionIntro

    class Custom(torch.nn.Module):
        def forward(self, x, y):
            return x

    model = _model()
    sc = [m for m in model.modules() if type(m) is SelfConnectionIntro][0]
    custom = Custom()
    sc.fc_tensor_product = custom
    with pytest.warns(UserWarning, match='self-connection keeps'):
        mps_opt.enable(model)
    assert sc.fc_tensor_product is custom and type(custom) is Custom


@needs_mps
def test_float64_shift_and_scale_go_to_float32_on_mps():
    # SevenNet 0.13.1 keeps the energy shift and scale in float64
    model = _build(_config())
    rescale = [m for m in model.modules() if hasattr(m, 'shift')]
    assert rescale
    ref = {}
    for m in rescale:
        m.shift.data = m.shift.data.double()
        m.scale.data = m.scale.data.double()
        ref[m] = (m.shift.data.clone(), m.scale.data.clone())
    model.to('mps')
    for m in rescale:
        assert m.shift.dtype == m.scale.dtype == torch.float32
        assert m.shift.device.type == 'mps'
        assert torch.allclose(m.shift.cpu().double(), ref[m][0])
    model.to('cpu')
    model.to(torch.float64)  # elsewhere float64 stays
    assert all(m.shift.dtype == torch.float64 for m in rescale)


@needs_mps
def test_conversions_drop_the_derived_tables():
    import io

    from sevenn.mps import modules as mps_opt

    graph = _graph()
    # a float16 round trip rounds the weights in place, keeping identity and version
    model = _model()
    mps_opt.enable(model)
    _run(model, graph)
    model.half()
    model.float()
    ref = _e3nn_on_gpu(_build(_config())).eval()
    ref.set_is_batch_data(False)
    ref.load_state_dict(model.state_dict())
    _close(_run(ref, graph), _run(model, graph))
    # a pickled fused model loaded on the CPU and moved back, eval and train
    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    back = torch.load(buf, map_location='cpu', weights_only=False).to('mps')
    _close(_run(ref, graph), _run(back, graph))
    back.train()
    ref.train()
    _close(_run(ref, graph), _run(back, graph))


@needs_mps
def test_autocast_takes_the_reference_path_in_node_layers():
    from sevenn.mps import modules as mps_opt

    species = ('H', 'Li', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Cl')
    model = _e3nn_on_gpu(_build(_config(species=species))).train()
    mps_opt.enable(model)
    sc = [m for m in model.modules() if isinstance(m, mps_opt.FusedSelfConnection)]
    assert sc and sc[0].irreps_in2.dim > mps_opt.SELFCONN_ALL_TYPES
    x = torch.randn(6, sc[0].irreps_in1.dim, device='mps')
    y = torch.zeros(6, sc[0].irreps_in2.dim, device='mps')
    y[:, 3] = 1.0
    with torch.autocast('mps', dtype=torch.float16):
        out = sc[0](x, y)
        want = mps_opt.FullyConnectedTensorProduct.forward(sc[0], x, y)
    assert out.dtype == want.dtype and torch.allclose(out, want)


@needs_mps
@pytest.mark.parametrize('many_types', [False, True])
def test_eval_after_training_uses_the_new_weights(many_types):
    from sevenn.mps import modules as mps_opt

    species = (
        ('H', 'Li', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Cl')
        if many_types
        else ('Na', 'Cl')
    )
    cfg = _config(species=species, use_bias_in_linear=True)
    model = _e3nn_on_gpu(_build(cfg))
    model.set_is_batch_data(False)
    mps_opt.enable(model)
    graph = _graph()
    ref = _e3nn_on_gpu(_build(cfg)).eval()
    ref.set_is_batch_data(False)
    for _ in range(2):  # the second epoch only updates the values of the map
        model.train()
        out = model(graph.clone().to('mps'))
        out[KEY.PRED_TOTAL_ENERGY].sum().backward()  # builds the train tables
        with torch.no_grad():  # an optimizer step
            for p in model.parameters():
                p.add_(0.01 * torch.randn_like(p))
        model.eval()
        ref.load_state_dict(model.state_dict())
        _close(_run(ref, graph), _run(model, graph))
    trained = [
        m
        for m in model.modules()
        if isinstance(m, (mps_opt.FusedLinear, mps_opt.FusedSelfConnection))
    ]
    # the rebuild used the train tables (layers called with 3-d input have none)
    with_tables = sum(m.__dict__.get('_train') is not None for m in trained)
    assert with_tables > len(trained) // 2


@needs_mps
def test_trainer_keeps_e3nn_without_compile_shader(monkeypatch):
    from sevenn.train.trainer import Trainer

    monkeypatch.delenv('SEVENN_DISABLE_MPS', raising=False)
    monkeypatch.delattr(torch.mps, 'compile_shader')
    with pytest.warns(UserWarning, match='compile_shader'):
        tr = Trainer(_build(_config()), [], torch.optim.Adam, device='mps')
    assert not _fused(tr.model) and mps.report(tr.model) == {}


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='needs an Apple GPU'
)
def test_old_pytorch_is_refused_with_a_clear_error(monkeypatch):
    # PyTorch before 2.6 has no scatter_reduce on MPS: refused up front
    from sevenn.mps import modules as mps_opt
    from sevenn.train.trainer import Trainer

    monkeypatch.setattr(mps_opt._policy, '_TORCH_OK', False)
    for off in (False, True):
        with pytest.raises(RuntimeError, match='PyTorch 2.6 or later'):
            _calc(_build(_config()), off=off)
    with pytest.raises(RuntimeError, match='PyTorch 2.6 or later'):
        Trainer(_build(_config()), [], torch.optim.Adam, device='mps')


@needs_mps
def test_type_index_must_be_int32():
    from sevenn.mps.kernels import CsrApply, CsrMap

    eye = (np.arange(2), np.arange(2), np.ones(2))
    M = CsrMap([eye, eye], np.zeros(2), 2, 2, torch.device('mps'))
    x = torch.ones(4, 2, device='mps')
    with pytest.raises(ValueError, match='int32'):
        CsrApply.apply(x, M, torch.ones(4, dtype=torch.int64, device='mps'))
    assert CsrApply.apply(x, M, torch.ones(4, dtype=torch.int32, device='mps')).shape


@needs_mps
@pytest.mark.parametrize('opt', ['fused', 'off'])
def test_sevenn_inference_on_the_apple_gpu(opt, tmp_path, monkeypatch):
    from unittest import mock

    from sevenn.main.sevenn_inference import main as inference_main

    data = Path(__file__).parents[1] / 'data'
    if opt == 'off':
        monkeypatch.setenv('SEVENN_DISABLE_MPS', '1')
    else:
        monkeypatch.delenv('SEVENN_DISABLE_MPS', raising=False)
    reports = []
    enable = mps_runtime.enable
    monkeypatch.setattr(
        mps_runtime, 'enable', lambda m: reports.append(enable(m)) or reports[-1]
    )
    out = tmp_path / 'inference_results'
    device = ['--device', 'mps']
    argv = ['sevenn_inference', '--output', str(out), '--batch', '2', *device,
            '7net-0', str(data / 'systems' / 'hfo2.extxyz')]
    with mock.patch('sys.argv', argv):
        inference_main()

    def errors(path):
        lines = (path / 'errors.txt').read_text().splitlines()
        return np.array([float(ln.split(':')[-1]) for ln in lines])

    want = errors(data / 'inferences' / 'snet0_on_hfo2')
    assert np.allclose(errors(out), want, rtol=1e-4, atol=1e-6)
    if opt != 'off':
        assert len(reports) == 1 and reports[0]['conv_fused'] > 0
        assert reports[0]['conv_e3nn'] == 0
    else:
        assert not reports


@needs_mps
@pytest.mark.parametrize('kind', ['convolution', 'scatter_gather'])
def test_cuda_only_tensor_products_are_refused_on_the_apple_gpu(kind):
    from sevenn.nn.convolution import IrrepsScatterGatterFusedConvolution

    model = _build(_config())
    name = next(n for n, m in model._modules.items() if hasattr(m, 'convolution'))
    if kind == 'convolution':  # stands for a patched cuEquivariance convolution
        model._modules[name].convolution = torch.nn.Identity()
    else:  # FlashTP, OpenEquivariance, LunaTP, fused cuEquivariance
        fused = IrrepsScatterGatterFusedConvolution.__new__(
            IrrepsScatterGatterFusedConvolution
        )
        torch.nn.Module.__init__(fused)
        model._modules[name] = fused
    with pytest.raises(ValueError, match='build it without enable_cueq'):
        model.to('mps')
    # refused before anything moved
    assert all(p.device.type == 'cpu' for p in model.parameters())


@needs_mps
def test_sevenn_train_on_the_apple_gpu(tmp_path, monkeypatch):
    from unittest import mock

    import yaml

    from sevenn.logger import Logger
    from sevenn.main.sevenn import main as sevenn_main

    monkeypatch.delenv('SEVENN_DISABLE_MPS', raising=False)
    reports = []
    enable = mps_runtime.enable
    monkeypatch.setattr(
        mps_runtime, 'enable', lambda m: reports.append(enable(m)) or reports[-1]
    )
    root = Path(__file__).parents[2]
    data = str(root / 'tests' / 'data' / 'systems' / 'hfo2.extxyz')
    cfg = yaml.safe_load(open(root / 'sevenn' / 'presets' / 'base.yaml'))
    cfg['train']['epoch'] = 1
    cfg['train']['device'] = 'mps'
    cfg['data']['load_trainset_path'] = data
    cfg['data'].pop('load_testset_path', None)
    cfg['data']['load_validset_path'] = data
    yml = tmp_path / 'input.yaml'
    yml.write_text(yaml.dump(cfg))
    Logger().switch_file(str(tmp_path / 'log.sevenn'))
    argv = ['sevenn', 'train', '-w', str(tmp_path), str(yml)]
    with mock.patch('sys.argv', argv):
        sevenn_main()
    assert len(reports) == 1 and reports[0]['conv_fused'] > 0
    ckpt = torch.load(tmp_path / 'checkpoint_best.pth', weights_only=False)
    assert ckpt['config']['device'] == 'mps'


# ---- padded batches: training and `sevenn inference` ----


def _batch(n=3):
    from torch_geometric.data import Batch

    return Batch.from_data_list(
        [_graph(seed=s, sheared=s % 2 == 1) for s in range(1, n + 1)]
    )


def _step(model, batch):
    """Outputs and parameter gradients of a force-and-stress loss, as in training."""
    model.zero_grad()
    out = model(batch.clone().to('mps'))
    loss = (
        out[KEY.PRED_TOTAL_ENERGY].pow(2).mean()
        + out[KEY.PRED_FORCE].pow(2).mean()
        + out[KEY.PRED_STRESS].pow(2).mean()
    )
    loss.backward()
    vals = {
        k: out[k].detach().cpu()
        for k in KEYS
        if k in out and k != KEY.ATOMIC_ENERGY
    }
    grads = {
        n: p.grad.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return vals, grads, int(out[KEY.EDGE_IDX].shape[1])


@needs_mps
@pytest.mark.parametrize('lmax', [1, 2])
def test_padded_batch_gives_the_same_training_step(lmax):
    model = _build(_config(lmax=lmax)).to('mps')
    model.set_is_batch_data(True)
    model.train()
    assert mps.prepare(model, bucket=0)['conv_e3nn'] == 0  # fused, not padded
    batch = _batch()
    n_real = int(batch[KEY.EDGE_IDX].shape[1])
    assert n_real % 64 != 0
    ref, gref, _ = _step(model, batch)
    mps.prepare(model, bucket=64)
    got, ggot, n_out = _step(model, batch)
    plan = model._mps_plan
    assert plan.num_edges % 64 == 0 and plan.num_edges > n_real
    assert plan.num_real == n_real  # the kernels saw only the real edges
    assert n_out == n_real  # and the output has them only
    _close(ref, got, rel=1e-5)
    assert gref.keys() == ggot.keys()
    for n in gref:
        scale = max(float(gref[n].abs().max()), 1e-3)
        assert float((gref[n] - ggot[n]).abs().max()) < 1e-5 * scale, n


@needs_mps
def test_trainer_pads_every_batch(monkeypatch):
    from torch_geometric.loader import DataLoader

    from sevenn.train.trainer import Trainer

    monkeypatch.delenv('SEVENN_DISABLE_MPS', raising=False)
    tr = Trainer(_build(_config()), [], torch.optim.Adam, device='mps')
    assert tr.model._mps.bucket == 256
    seen = []
    fwd = tr.model.forward
    monkeypatch.setattr(
        tr.model,
        'forward',
        lambda d: seen.append(d[KEY.EDGE_IDX].shape[1]) or fwd(d),
    )
    graphs = [_graph(seed=s, sheared=s % 2 == 1) for s in range(1, 6)]
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    real = [int(b[KEY.EDGE_IDX].shape[1]) for b in loader]

    class _Recorder:  # what losses and metrics are given: the real edges only
        def __init__(self):
            self.edges = []

        def update(self, out, **kwargs):
            self.edges.append(int(out[KEY.EDGE_IDX].shape[1]))

    rec = _Recorder()
    tr.run_one_epoch(loader, is_train=False, error_recorder=rec)  # type: ignore
    assert seen and all(e % 256 == 0 for e in seen), seen
    assert rec.edges == real, (rec.edges, real)


@needs_mps
def test_output_and_input_keep_the_real_edges_only():
    model = _build(_config()).to('mps')
    mps.prepare(model, bucket=64)
    model.set_is_batch_data(True)
    model.eval()
    batch = _batch().to('mps')
    n_real = int(batch[KEY.EDGE_IDX].shape[1])
    n_atoms = int(batch[KEY.POS].shape[0])
    idx = batch[KEY.EDGE_IDX].clone()
    out = model(batch)
    assert model._mps_plan.num_edges % 64 == 0  # it was padded inside
    assert out[KEY.EDGE_IDX].shape[1] == n_real
    assert out[KEY.EDGE_VEC].shape[0] == n_real
    assert out[KEY.POS].shape[0] == n_atoms
    assert torch.equal(out[KEY.EDGE_IDX], idx)
    assert batch[KEY.EDGE_IDX].shape[1] == n_real  # the input object too


@needs_mps
def test_second_derivatives_raise_in_the_gate_kernel():
    from sevenn.mps import modules as mps_opt

    gate = (
        Gate('2x0e', [torch.nn.functional.silu], '1x0e', [torch.tanh], '1x1o')
        .to('mps')
        .eval()
    )
    gate._tables = mps_opt._gate_tables(gate, 'mps')
    gate.__class__ = mps_opt.FusedGate
    x = torch.randn(5, gate.irreps_in.dim, device='mps', requires_grad=True)
    with pytest.raises(
        RuntimeError, match='linear and gate layers have first derivatives only'
    ):
        torch.autograd.grad(gate(x).pow(2).sum(), x, create_graph=True)


@needs_mps
@pytest.mark.parametrize('modal', ['R2SCAN', 'PBE'])
def test_multi_fidelity_model_equals_e3nn(modal):
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=3)
    res = {}
    for off in (False, True):
        with mock.patch.dict(os.environ, {'SEVENN_DISABLE_MPS': '1' if off else ''}):
            calc = SevenNetCalculator('7net-mf-0', device='mps', modal=modal)
        assert (mps.report(calc.model).get('conv_fused', 0) > 0) != off
        a = at.copy()
        a.calc = calc
        res[off] = (a.get_potential_energy(), a.get_forces(), a.get_stress())
    assert abs(res[False][0] - res[True][0]) < 1e-4
    assert np.abs(res[False][1] - res[True][1]).max() < 1e-4
    assert np.abs(res[False][2] - res[True][2]).max() < 1e-6


# ---- the model hook: moving to and from the Apple GPU ----


@needs_mps
def test_leaving_the_apple_gpu_restores_a_scriptable_model():
    from e3nn.util import jit

    model = _build(_config())
    model.to('mps')
    assert _fused(model) and model._forward_pre_hooks and model._forward_hooks
    model.to('cpu')
    assert not _fused(model) and '_mps' not in model.__dict__
    assert not model._forward_pre_hooks and not model._forward_hooks
    model.set_is_batch_data(False)
    jit.script(model)  # as sevenn get_model does


@needs_mps
def test_a_copy_of_a_prepared_model_has_its_own_state():
    import copy

    model = _build(_config()).to('mps')
    twin = copy.deepcopy(model)
    assert _fused(twin) and twin._mps is not model._mps
    assert twin._mps_plan is not model._mps_plan
    twin.to('cpu')
    assert _fused(model) and model._forward_pre_hooks  # the original is untouched
    graph = _graph()
    model.set_is_batch_data(False)
    ref = _e3nn_on_gpu(_build(_config()))
    ref.set_is_batch_data(False)
    _close(_run(ref.eval(), graph), _run(model.eval(), graph))


@needs_mps
def test_float64_parameters_become_float32_and_keep_their_identity():
    model = _build(_config())
    shift = next(p for n, p in model.named_parameters() if 'shift' in n)
    shift.data = shift.data.double()
    opt = torch.optim.Adam(model.parameters())
    model.to('mps')
    assert shift.dtype == torch.float32 and shift.device.type == 'mps'
    assert any(p is shift for g in opt.param_groups for p in g['params'])
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert _fused(model)


@needs_mps
def test_train_and_eval_switch_on_a_prepared_model():
    model = _build(_config()).to('mps')
    model.set_is_batch_data(False)
    ref = _e3nn_on_gpu(_build(_config()))
    ref.set_is_batch_data(False)
    graph = _graph()
    for mode in ('train', 'eval', 'train'):
        getattr(model, mode)()
        getattr(ref, mode)()
        _close(_run(ref, graph), _run(model, graph))


@needs_mps
def test_half_precision_on_the_apple_gpu_keeps_the_reference_code():
    model = _build(_config()).to('mps')
    with pytest.warns(UserWarning, match='need float32 weights'):
        model.half()
    assert not _fused(model) and mps.report(model) == {}


# ---- selection: device and opt-out ----


@pytest.mark.parametrize(
    'value, off',
    [('1', True), ('true', True), (' Yes ', True), ('TRUE', True),
     ('0', False), ('', False), ('no', False), ('off', False)],
)  # fmt: skip
def test_opt_out_values(value, off, monkeypatch):
    monkeypatch.setenv('SEVENN_DISABLE_MPS', value)
    assert mps.disabled() is off


# ---- atomic virial, reEWC, radial tables, devices ----


class _EnergyLoss:
    def get_loss(self, out, model=None):
        return out[KEY.PRED_TOTAL_ENERGY].pow(2).mean()


@needs_mps
def test_atomic_virial_equals_the_cpu():
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=4)
    res = {}
    for dev in ('mps', 'cpu'):
        a = at.copy()
        a.calc = SevenNetCalculator('7net-0', device=dev, compute_atomic_virial=True)
        a.get_potential_energy()
        res[dev] = np.asarray(a.calc.results['stresses'])
    assert res['mps'].shape == (len(at), 6)
    assert np.abs(res['mps'] - res['cpu']).max() < 1e-5


def test_ewc_state_keeps_its_dtype_and_the_callers_dicts_on_the_cpu():
    from sevenn.train.reewc.loss import EWCLoss

    model = _build(_config())
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    f64 = {n: p.detach().double().clone() for n, p in named.items()}
    fisher, opt = dict(f64), {n: v.clone() for n, v in f64.items()}
    loss = EWCLoss(fisher, opt)
    assert float(loss.get_loss({}, model)) == pytest.approx(0.0, abs=1e-6)
    name = next(iter(named))
    assert loss.fisher_dict[name].dtype == torch.float64  # CPU and CUDA: as before
    assert loss.fisher_dict is not fisher and loss.opt_params_dict is not opt
    assert fisher[name] is f64[name]  # the caller's dict is not modified


@needs_mps
def test_ewc_state_in_float64_follows_the_model_to_the_apple_gpu():
    from sevenn.train.reewc.loss import EWCLoss

    model = _build(_config())
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    f64 = {n: p.detach().double().clone() for n, p in named.items()}
    loss = EWCLoss(dict(f64), {n: v.clone() for n, v in f64.items()})
    model.to('mps')
    assert float(loss.get_loss({}, model)) == pytest.approx(0.0, abs=1e-6)
    name = next(iter(named))
    assert loss.fisher_dict[name].dtype == torch.float32
    assert loss.opt_params_dict[name].device.type == 'mps'


@needs_mps
def test_reewc_trainer_pads_the_memory_batches_too(monkeypatch):
    from torch_geometric.loader import DataLoader

    from sevenn.train.reewc.trainer import ReewcTrainer

    monkeypatch.delenv('SEVENN_DISABLE_MPS', raising=False)
    graphs = [_graph(seed=s, sheared=s % 2 == 1) for s in range(1, 6)]
    tr = ReewcTrainer(
        _build(_config()),
        [(_EnergyLoss(), 1.0)],
        torch.optim.Adam,
        device='mps',
        memory_loader=DataLoader(graphs[3:], batch_size=2),
    )
    assert tr.model._mps.bucket == 256
    seen = []
    fwd = tr.model.forward
    monkeypatch.setattr(
        tr.model,
        'forward',
        lambda d: seen.append(d[KEY.EDGE_IDX].shape[1]) or fwd(d),
    )
    tr.run_one_epoch(DataLoader(graphs[:3], batch_size=2), is_train=True)
    assert len(seen) == 4 and all(e % 256 == 0 for e in seen), seen


@needs_mps
def test_no_radial_tables_on_the_apple_gpu():
    calc = SevenNetCalculator('7net-0', device='mps')
    assert _fused(calc.model) and mps.report(calc.model)['conv_e3nn'] == 0
    with pytest.raises(ValueError, match='radial tables are not available'):
        SevenNetCalculator('7net-0', device='mps', radial_table=True)
    model = _build(_config())
    with pytest.raises(ValueError, match='radial tables are not available'):
        SevenNetCalculator(
            model, file_type='model_instance', device='mps', radial_table=True
        )
    # refused before the move: the caller's model is untouched
    assert all(p.device.type == 'cpu' for p in model.parameters())
    assert '_mps' not in model.__dict__


@needs_mps
def test_cpu_fallback_of_pytorch_is_warned_about(monkeypatch):
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    with pytest.warns(UserWarning, match='PYTORCH_ENABLE_MPS_FALLBACK'):
        _build(_config()).to('mps')


# ---- device='auto' ----


@pytest.mark.parametrize(
    'cuda, apple, value, want',
    [(True, True, '', 'cuda'), (False, True, '', 'mps'), (False, True, '1', 'cpu'),
     (False, False, '', 'cpu'), (True, False, '1', 'cuda')],
)  # fmt: skip
def test_auto_device(cuda, apple, value, want, monkeypatch):
    from sevenn.util import auto_device

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: apple)
    monkeypatch.setattr(mps._policy, '_TORCH_OK', True)
    monkeypatch.setenv('SEVENN_DISABLE_MPS', value)
    assert auto_device() == want
    # nor with radial tables, which the Apple GPU does not have
    assert auto_device(apple_gpu=False) == ('cuda' if cuda else 'cpu')
    # nor does any model with a PyTorch too old for it (before 2.6)
    monkeypatch.setattr(mps._policy, '_TORCH_OK', False)
    assert auto_device() == ('cuda' if cuda else 'cpu')


@needs_mps
def test_auto_and_explicit_devices_on_a_mac(monkeypatch):
    monkeypatch.delenv('SEVENN_DISABLE_MPS', raising=False)
    calc = SevenNetCalculator(_build(_config()), file_type='model_instance')
    assert calc.device.type == 'mps' and _fused(calc.model)
    calc = SevenNetCalculator(
        _build(_config()), file_type='model_instance', device='cpu'
    )
    assert calc.device.type == 'cpu' and '_mps' not in calc.model.__dict__
    monkeypatch.setenv('SEVENN_DISABLE_MPS', '1')
    calc = SevenNetCalculator(_build(_config()), file_type='model_instance')
    assert calc.device.type == 'cpu'
    calc = SevenNetCalculator(
        _build(_config()), file_type='model_instance', device='mps'
    )  # opt-out on 'mps': the reference code, still in float32 on the GPU
    assert calc.device.type == 'mps' and not _fused(calc.model)
    assert all(p.dtype == torch.float32 for p in calc.model.parameters())
    assert not calc.model._forward_pre_hooks


# ---- lifecycle edge cases: failed moves, no_grad, float64, retried graphs ----


@needs_mps
def test_a_float64_model_moves_in_float32():
    model = _build(_config()).double()
    model.to('mps')
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert _fused(model)


@needs_mps
def test_a_float64_default_dtype_still_fuses_every_module():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = _build(_config())
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # no module may keep the e3nn code
            model.to('mps')
        assert all(p.dtype == torch.float32 for p in model.parameters())
        rep = mps.report(model)
        assert rep and not any(v for k, v in rep.items() if k.endswith('_e3nn'))
    finally:
        torch.set_default_dtype(old)


@needs_mps
def test_move_inside_no_grad_prepares_the_model():
    model = _build(_config())
    with torch.no_grad():
        model.to('mps')
    assert _fused(model) and model._mps.bucket == 256


@needs_mps
def test_a_failed_conversion_leaves_the_model_prepared():
    model = _build(_config()).to('mps')
    with pytest.raises(Exception):
        model.double()  # MPS has no float64
    assert _fused(model) and model._forward_pre_hooks and '_mps' in model.__dict__


@needs_mps
def test_the_same_graph_after_a_failed_call_is_not_padded_twice():
    model = _build(_config()).to('mps')
    model.set_is_batch_data(False)
    model.eval()
    ref = _e3nn_on_gpu(_build(_config()))
    ref.set_is_batch_data(False)
    ref.eval()
    graph = _graph().to('mps')
    n = int(graph[KEY.EDGE_IDX].shape[1])
    want = _run(ref, graph)
    good = model.forward

    def boom(data):
        raise RuntimeError('boom')

    model.forward = boom
    with pytest.raises(RuntimeError, match='boom'):
        model(graph)
    assert int(graph[KEY.EDGE_IDX].shape[1]) > n  # left padded by the failure
    model.forward = good
    out = model(graph)
    assert int(out[KEY.EDGE_IDX].shape[1]) == n
    assert model._mps_plan.num_real == n
    _close(want, {k: out[k].detach().cpu() for k in KEYS if k in out})
