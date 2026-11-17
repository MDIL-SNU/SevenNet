# TODO: add gradient test from total loss after double precision.
#       so far, it is empirically checked by seeing learning curves
import copy
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from ase.build import bulk
from torch_geometric.loader.dataloader import Collater

import sevenn
import sevenn._keys as KEY
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.lunatp_helper import is_lunatp_available
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.util import chemical_species_preprocess, model_from_checkpoint

cutoff = 4.0

_atoms = bulk('NaCl', 'rocksalt', a=4.00) * (2, 2, 2)
_avg_num_neigh = 30.0
_atoms.rattle()

_graph = AtomGraphData.from_numpy_dict(dl.unlabeled_atoms_to_graph(_atoms, cutoff))


def get_graphs(batched):
    # batch size 2
    cloned = [_graph.clone().to('cuda'), _graph.clone().to('cuda')]
    if not batched:
        return cloned
    else:
        return Collater(cloned)(cloned)


def get_model_config():
    config = {
        'cutoff': cutoff,
        'channel': 32,
        'lmax': 2,
        'is_parity': True,
        'num_convolution_layer': 3,
        'self_connection_type': 'nequip',  # not NequIp
        'interaction_type': 'nequip',
        'radial_basis': {
            'radial_basis_name': 'bessel',
        },
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'weight_nn_hidden_neurons': [64, 64],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': _avg_num_neigh,
        'train_denominator': False,
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
    chems = set()
    chems.update(_atoms.get_chemical_symbols())
    config.update(**chemical_species_preprocess(list(chems)))
    return config


def get_model(config_overwrite=None, use_lunatp=False):
    cf = get_model_config()
    if config_overwrite is not None:
        cf.update(config_overwrite)

    cf['use_lunatp'] = use_lunatp

    model = build_E3_equivariant_model(cf, parallel=False)
    assert isinstance(model, AtomGraphSequential)
    model.to('cuda')
    return model


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize(
    'cf',
    [
        ({}),
        ({'lmax': 3}),
        ({'num_interaction_layer': 2}),
        ({'num_interaction_layer': 4}),
    ],
)
def test_model_output(cf):
    torch.manual_seed(777)
    model_e3nn = get_model(cf)
    torch.manual_seed(777)
    model_lunatp = get_model(cf, use_lunatp=True)

    model_e3nn.set_is_batch_data(True)
    model_lunatp.set_is_batch_data(True)

    e3nn_out = model_e3nn._preprocess(get_graphs(batched=True))
    lunatp_out = model_lunatp._preprocess(get_graphs(batched=True))

    for k, e3nn_f in model_e3nn._modules.items():
        lunatp_f = model_lunatp._modules[k]
        e3nn_out = e3nn_f(e3nn_out)  # type: ignore
        lunatp_out = lunatp_f(lunatp_out)  # type: ignore
        assert torch.allclose(e3nn_out.x, lunatp_out.x, atol=1e-6), (
            f'{k} \n\n {e3nn_f} \n\n {lunatp_f}'
        )

    assert torch.allclose(
        e3nn_out.inferred_total_energy, lunatp_out.inferred_total_energy
    )
    assert torch.allclose(e3nn_out.atomic_energy, lunatp_out.atomic_energy)
    assert torch.allclose(
        e3nn_out.inferred_force, lunatp_out.inferred_force, atol=1e-5
    )
    assert torch.allclose(
        e3nn_out.inferred_stress, lunatp_out.inferred_stress, atol=1e-5
    )


@pytest.mark.filterwarnings('ignore:.*is not found from.*')
@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize(
    'start_from_lunatp',
    [
        (True),
        (False),
    ],
)
def test_checkpoint_convert(tmp_path, start_from_lunatp):
    torch.manual_seed(123)
    model_from = get_model(use_lunatp=start_from_lunatp)

    cfg = get_model_config()
    cfg.update(
        {
            'use_lunatp': start_from_lunatp,
            'version': sevenn.__version__,
        }
    )
    torch.save(
        {'model_state_dict': model_from.state_dict(), 'config': cfg},
        tmp_path / 'cp_from.pth',
    )

    model_to, _ = model_from_checkpoint(
        str(tmp_path / 'cp_from.pth'), enable_lunatp=(not start_from_lunatp)
    )
    model_to.to('cuda')

    model_from.set_is_batch_data(True)
    model_to.set_is_batch_data(True)

    from_out = model_from(get_graphs(batched=True))
    to_out = model_to(get_graphs(batched=True))

    assert torch.allclose(
        from_out.inferred_total_energy, to_out.inferred_total_energy
    )
    assert torch.allclose(from_out.atomic_energy, to_out.atomic_energy)
    assert torch.allclose(from_out.inferred_force, to_out.inferred_force, atol=1e-5)
    assert torch.allclose(
        from_out.inferred_stress, to_out.inferred_stress, atol=1e-5
    )


@pytest.mark.filterwarnings('ignore:.*is not found from.*')
@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize(
    'start_from_lunatp',
    [
        (True),
        (False),
    ],
)
def test_checkpoint_convert_no_batch(tmp_path, start_from_lunatp):
    torch.manual_seed(123)
    model_from = get_model(use_lunatp=start_from_lunatp)

    cfg = get_model_config()
    cfg.update(
        {
            'use_lunatp': start_from_lunatp,
            'version': sevenn.__version__,
        }
    )
    torch.save(
        {'model_state_dict': model_from.state_dict(), 'config': cfg},
        tmp_path / 'cp_from.pth',
    )

    model_to, _ = model_from_checkpoint(
        str(tmp_path / 'cp_from.pth'), enable_lunatp=(not start_from_lunatp)
    )
    model_to.to('cuda')

    model_from.set_is_batch_data(False)
    model_to.set_is_batch_data(False)

    from_out = model_from(get_graphs(batched=False)[0])
    to_out = model_to(get_graphs(batched=False)[0])

    assert torch.allclose(
        from_out.inferred_total_energy, to_out.inferred_total_energy
    )
    assert torch.allclose(from_out.atomic_energy, to_out.atomic_energy)
    assert torch.allclose(from_out.inferred_force, to_out.inferred_force, atol=1e-5)
    assert torch.allclose(
        from_out.inferred_stress, to_out.inferred_stress, atol=1e-5
    )


def assert_atoms(atoms1, atoms2, rtol=1e-5, atol=1e-6):
    def acl(a, b, rtol=rtol, atol=atol):
        return np.allclose(a, b, rtol=rtol, atol=atol)

    assert len(atoms1) == len(atoms2)
    assert acl(atoms1.get_cell(), atoms2.get_cell())
    assert acl(atoms1.get_potential_energy(), atoms2.get_potential_energy())
    assert acl(atoms1.get_forces(), atoms2.get_forces(), rtol * 10, atol * 10)
    assert acl(
        atoms1.get_stress(voigt=False),
        atoms2.get_stress(voigt=False),
        rtol * 10,
        atol * 10,
    )
    # assert acl(atoms1.get_potential_energies(), atoms2.get_potential_energies())


@pytest.mark.filterwarnings('ignore:.*is not found from.*')
@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_calculator(tmp_path):
    lunatp = True
    model = get_model(use_lunatp=lunatp)
    ref_calc = SevenNetCalculator(model, file_type='model_instance')
    atoms = copy.deepcopy(_atoms)
    atoms.calc = ref_calc

    cfg = get_model_config()
    cfg.update(
        {'use_lunatp': lunatp, 'version': sevenn.__version__}
    )

    cp_path = str(tmp_path / 'cp.pth')
    torch.save(
        {'model_state_dict': model.state_dict(), 'config': cfg},
        cp_path,
    )

    calc2 = SevenNetCalculator(cp_path, enable_lunatp=False)
    atoms2 = copy.deepcopy(_atoms)
    atoms2.calc = calc2

    assert_atoms(atoms, atoms2)


_CALCULATOR_CALLS = """
import sys
from ase import Atoms
from ase.build import bulk
from sevenn.calculator import SevenNetCalculator

for cuda_graph in (False, True):
    calc = SevenNetCalculator(sys.argv[1], enable_lunatp=True, cuda_graph=cuda_graph)
    dimer = Atoms('NaCl', positions=[[0, 0, 0], [0, 0, 2.4]])
    for atoms in (dimer, bulk('NaCl', 'rocksalt', a=4.0)):
        for _ in range(3):  # cuda_graph: eager, capture, replay
            atoms.calc = calc
            atoms.get_potential_energy()
            atoms.get_forces()
            calc.results['stress']
"""


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.skipif(
    os.environ.get('SEVENNET_TEST_PRECOMPILE') != '1',
    reason='compiles 44 kernels (7-15 min); set SEVENNET_TEST_PRECOMPILE=1',
)
def test_precompile_covers_calculator(tmp_path, monkeypatch):
    """
    After precompile into an empty kernel cache, a new process running
    SevenNetCalculator (frozen weights; eager and cuda_graph; energy, forces,
    stress) adds no LunaEquivariance kernel library to the cache.
    """
    from sevenn.nn.lunatp_helper import precompile

    # set first: a045a88 compiles the convolution forward kernels in build_model
    cache = tmp_path / 'lunaeq'
    monkeypatch.setenv('LUNAEQ_CACHE_DIR', str(cache))
    cfg = get_model_config()
    cfg.update({
        'channel': 4, 'lmax': 1, 'num_convolution_layer': 2,
        'use_lunatp': True, 'version': sevenn.__version__,
    })
    model = build_E3_equivariant_model(cfg, parallel=False)
    cp_path = str(tmp_path / 'cp.pth')
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, cp_path)

    precompile(cp_path)
    built = {str(p) for p in cache.rglob('*.so')}  # 0.2.0: <digest>/<name>_v*.so
    assert built
    subprocess.run(
        [sys.executable, '-c', _CALCULATOR_CALLS, cp_path], check=True
    )
    added = {str(p) for p in cache.rglob('*.so')} - built
    assert not added, f'kernels built after precompile: {sorted(added)}'


_LOAD_EQU_NORM = """
import sys
import torch
from e3nn.o3 import Irreps
import sevenn.nn.lunatp_helper
from LunaEquivariance.nn.equ_norm import EquNormOperator

for irreps in sys.argv[1:]:
    EquNormOperator(irreps)(torch.randn(2, Irreps(irreps).dim, device='cuda'))
"""


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_kernel_cache_independent_of_load_order(tmp_path):
    """
    A process loads two kernel libraries of one base name (equ_norm) in the
    order (X, Y) into an empty cache; a second process loads them in the
    order (Y, X) and compiles nothing (LunaEquivariance 0.2.0 names them by
    the per-process load count, see _DigestVersionedCppExtension).
    """
    env = {**os.environ, 'LUNAEQ_CACHE_DIR': str(tmp_path)}
    configs = ['2x0e+1x1o', '3x0e+1x1o']

    def libraries_after(irreps):
        cmd = [sys.executable, '-c', _LOAD_EQU_NORM, *irreps]
        subprocess.run(cmd, check=True, env=env)
        return sorted(str(p) for p in tmp_path.rglob('*.so'))

    built = libraries_after(configs)
    assert len(built) == 2
    assert libraries_after(configs[::-1]) == built


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_unsorted_edges():
    # LunaTP needs edges sorted by destination, LunaTPConvolution sorts them
    torch.manual_seed(777)
    model_e3nn = get_model()
    torch.manual_seed(777)
    model_lunatp = get_model(use_lunatp=True)
    model_e3nn.set_is_batch_data(False)
    model_lunatp.set_is_batch_data(False)

    graph = _graph.clone().to('cuda')
    perm = torch.randperm(graph.edge_index.shape[1], device='cuda')
    graph.edge_index = graph.edge_index[:, perm]
    graph.edge_vec = graph.edge_vec[perm]
    dst = graph.edge_index[0]
    assert not bool(torch.all(dst[1:] >= dst[:-1]))

    e3nn_out = model_e3nn(graph.clone())
    lunatp_out = model_lunatp(graph.clone())
    assert torch.allclose(
        e3nn_out.inferred_total_energy, lunatp_out.inferred_total_energy
    )
    assert torch.allclose(
        e3nn_out.inferred_force, lunatp_out.inferred_force, atol=1e-5
    )
    assert torch.allclose(
        e3nn_out.inferred_stress, lunatp_out.inferred_stress, atol=1e-5
    )


def test_no_fallback(monkeypatch):
    import sevenn.nn.lunatp_helper as lunatp_helper
    from sevenn.checkpoint import _solve_accelerator

    monkeypatch.setattr(lunatp_helper, 'is_lunatp_available', lambda: False)
    with pytest.raises(ImportError):
        _solve_accelerator(False, False, False, True)
    with pytest.raises(ValueError):
        _solve_accelerator(False, False, True, True)


_metaformer = {
    'interaction_type': 'metaformer',
    'self_connection_type': 'identity',
    'irreps_manual': ['16x0e', '16x0e+8x1o+4x2e', '16x0e+8x1o+4x2e', '16x0e'],
    'use_final_norm': True,
}


def _norm_passes(module, x0, v0, r, s):
    """forward, force-like first derivative (create_graph), loss backward"""
    x = x0.clone().requires_grad_(True)
    v = v0.clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    y = module({'x': x})['x']
    (g,) = torch.autograd.grad(y, x, v, create_graph=True)
    ((y * r).sum() + (g * s).sum()).backward()
    out = [y.detach(), g.detach(), x.grad, v.grad]
    return out + [q.grad for q in module.parameters()]


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize(
    'irreps,centering,affine',
    [
        ('16x0e+8x1o+4x2e+4x3o', True, True),
        ('3x1o+5x0e+2x2e', True, True),  # 0e block not first
        ('16x0e', True, True),
        ('4x1o+2x2e', True, True),  # no scalars: no centering
        ('5x0e+3x1o', False, True),
        ('5x0e+3x1o', True, False),
    ],
)
def test_fused_rmsnorm_module(irreps, centering, affine):
    from sevenn.nn.lunatp_helper import (
        LunaTPEquivariantRMSNorm,
        _run_once,
        patch_rms_norm,
    )
    from sevenn.nn.normalization import EquivariantRMSNorm

    torch.manual_seed(0)
    ref = EquivariantRMSNorm(irreps, 'x', centering=centering, affine=affine).cuda()
    if affine:
        with torch.no_grad():
            ref.weight.uniform_(0.5, 1.5)
            ref.bias.normal_()
    fused = patch_rms_norm(ref)
    assert isinstance(fused, LunaTPEquivariantRMSNorm)
    assert fused.state_dict().keys() == ref.state_dict().keys()
    x0 = torch.randn(37, ref.irreps.dim, device='cuda') * 3.0 + 0.5
    v0, r, s = (torch.randn_like(x0) for _ in range(3))
    expected = _norm_passes(ref, x0, v0, r, s)
    actual = _norm_passes(fused, x0, v0, r, s)
    for a, b in zip(expected, actual):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-5)
    _run_once(fused, x0.device, x0.dtype, 2)  # precompile path


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_fused_rmsnorm_needs_one_scalar_block():
    from sevenn.nn.lunatp_helper import patch_rms_norm
    from sevenn.nn.normalization import EquivariantRMSNorm

    norm = EquivariantRMSNorm('4x0e+2x1o+3x0e', 'x')  # centered jointly
    assert patch_rms_norm(norm) is norm


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_fused_rmsnorm_model():
    from sevenn.nn.lunatp_helper import LunaTPEquivariantRMSNorm
    from sevenn.nn.normalization import EquivariantRMSNorm

    torch.manual_seed(777)
    model_e3nn = get_model(_metaformer)
    model_lunatp = get_model(_metaformer, use_lunatp=True)  # fused by default
    # opt-out keeps the PyTorch norms
    plain = get_model({**_metaformer, 'lunatp_fused_norm': False}, use_lunatp=True)
    assert not any(
        isinstance(m, LunaTPEquivariantRMSNorm) for m in plain.modules()
    )
    # same norm parameter / buffer names: checkpoints load across backends
    # (the convolution keys differ between backends independently of this)
    norm_keys = [
        {k for k in m.state_dict() if '_norm' in k}
        for m in (model_e3nn, model_lunatp)
    ]
    assert norm_keys[0] == norm_keys[1] and norm_keys[0]
    norms = [m for k, m in model_lunatp.named_children() if 'norm' in k]
    assert len(norms) == 2 * 3 + 1  # 3 layers x (norm_1, norm_2) + final_norm
    assert all(isinstance(m, LunaTPEquivariantRMSNorm) for m in norms)
    with torch.no_grad():
        for m in norms:
            m.weight.uniform_(0.5, 1.5)
            m.bias.normal_()

    # reference: the same LunaTP model with the PyTorch norms
    model_ref = get_model({**_metaformer, 'lunatp_fused_norm': False},
                          use_lunatp=True)
    for k, m in list(model_ref.named_children()):
        if isinstance(m, LunaTPEquivariantRMSNorm):
            plain = EquivariantRMSNorm(m.irreps, m.key_x, eps=m.eps)
            model_ref._modules[k] = plain.cuda()
    model_ref.load_state_dict(model_lunatp.state_dict())

    def run(model):
        model.train()
        model.set_is_batch_data(True)
        model.zero_grad(set_to_none=True)
        out = model(get_graphs(batched=True))
        e, f, s = (
            out.inferred_total_energy, out.inferred_force, out.inferred_stress
        )
        (e.square().sum() + f.square().sum() + s.square().sum()).backward()
        grads = {n: q.grad for n, q in model.named_parameters()}
        return e.detach(), f.detach(), s.detach(), grads

    e1, f1, s1, g1 = run(model_ref)
    e2, f2, s2, g2 = run(model_lunatp)
    assert torch.allclose(e1, e2, rtol=1e-5, atol=1e-5)
    assert torch.allclose(f1, f2, atol=1e-5)
    assert torch.allclose(s1, s2, atol=1e-5)
    for n, g in g1.items():
        if g is None:
            assert g2[n] is None, n
            continue
        # fp32 noise: relative to the gradient tensor's norm (entries near 0)
        assert (g - g2[n]).norm() <= 1e-4 * g.norm() + 1e-8, n


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize('irreps', ['6x0e+3x1o+2x2e+2x3o', '6x0e'])
def test_fused_rmsnorm_gradcheck_float64(irreps):
    """first and second derivatives (force training) of the fused kernel"""
    from sevenn.nn.lunatp_helper import patch_rms_norm
    from sevenn.nn.normalization import EquivariantRMSNorm

    torch.manual_seed(0)
    fused = patch_rms_norm(EquivariantRMSNorm(irreps, 'x').cuda().double())
    dim, n0 = fused.irreps.dim, fused.num_scalars
    kw = dict(device='cuda', dtype=torch.float64, requires_grad=True)
    x = (torch.randn(5, dim, **kw) * 2 + 0.3).detach().requires_grad_(True)
    w = (1 + 0.3 * torch.randn(fused.irreps.num_irreps, **kw)).detach()
    b = (0.2 * torch.randn(dim, **kw)).detach()

    def f(x, w, b):
        return fused.equ_norm(x, w, b.narrow(0, fused.scalar_start, n0))

    inputs = [t.requires_grad_(True) for t in (x, w, b)]
    tol = dict(eps=1e-6, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradcheck(f, inputs, **tol)
    assert torch.autograd.gradgradcheck(f, inputs, **tol)


_bilinear = {
    'gate_type': 'bilinear',
    'lmax': 3,
    'self_connection_type': 'linear',
    'irreps_manual': [
        '16x0e', '16x0e+8x1o+8x2e+8x3o', '16x0e+8x1o+8x2e+8x3o', '16x0e'
    ],
}


def _randomize_bilinear(model):
    from sevenn.nn.bilinear_gate import BilinearGate

    n = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, BilinearGate):
                m.linear.weight.normal_()
                n += 1
    assert n > 0


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize(
    'cf', [_bilinear, {**_bilinear, 'interaction_type': 'metaformer',
                       'self_connection_type': 'identity'},
           {**_bilinear, 'bilinear_b': 'shared'}]
)
def test_bilinear_gate_output(cf):
    from sevenn.nn.bilinear_gate import BilinearGate
    from sevenn.nn.lunatp_helper import LunaTPTensorProduct

    torch.manual_seed(777)
    model_e3nn = get_model(cf)
    _randomize_bilinear(model_e3nn)
    model_lunatp = get_model(cf, use_lunatp=True)
    missing, unexpected = model_lunatp.load_state_dict(
        model_e3nn.state_dict(), strict=False
    )
    # only the convolutions differ (e3nn constants, LunaTP's empty tp weight)
    assert all(k.endswith('tp_conv.tp.weight') for k in missing), missing
    assert all('_convolution.convolution.' in k for k in unexpected), unexpected
    gate = model_lunatp._modules['1_equivariant_gate']
    assert isinstance(gate, BilinearGate)
    assert isinstance(gate.tp, LunaTPTensorProduct)

    model_e3nn.set_is_batch_data(True)
    model_lunatp.set_is_batch_data(True)
    e3nn_out = model_e3nn._preprocess(get_graphs(batched=True))
    lunatp_out = model_lunatp._preprocess(get_graphs(batched=True))
    for k, e3nn_f in model_e3nn._modules.items():
        e3nn_out = e3nn_f(e3nn_out)  # type: ignore
        lunatp_out = model_lunatp._modules[k](lunatp_out)  # type: ignore
        assert torch.allclose(e3nn_out.x, lunatp_out.x, atol=1e-5), k
    assert torch.allclose(
        e3nn_out.inferred_total_energy, lunatp_out.inferred_total_energy
    )
    assert torch.allclose(
        e3nn_out.inferred_force, lunatp_out.inferred_force, atol=1e-5
    )
    assert torch.allclose(
        e3nn_out.inferred_stress, lunatp_out.inferred_stress, atol=1e-5
    )


@pytest.mark.filterwarnings('ignore:.*is not found from.*')
@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize('start_from_lunatp', [True, False])
def test_bilinear_checkpoint_convert(tmp_path, start_from_lunatp):
    torch.manual_seed(123)
    model_from = get_model(_bilinear, use_lunatp=start_from_lunatp)
    _randomize_bilinear(model_from)
    cfg = get_model_config()
    cfg.update(
        {**_bilinear, 'use_lunatp': start_from_lunatp, 'version': sevenn.__version__}
    )
    torch.save(
        {'model_state_dict': model_from.state_dict(), 'config': cfg},
        tmp_path / 'cp_from.pth',
    )
    model_to, _ = model_from_checkpoint(
        str(tmp_path / 'cp_from.pth'), enable_lunatp=(not start_from_lunatp)
    )
    model_to.to('cuda')
    model_from.set_is_batch_data(True)
    model_to.set_is_batch_data(True)
    from_out = model_from(get_graphs(batched=True))
    to_out = model_to(get_graphs(batched=True))
    # fp32: the e3nn scatter is not deterministic (~5e-6 eV run to run here)
    assert torch.allclose(
        from_out.inferred_total_energy, to_out.inferred_total_energy, atol=2e-5
    )
    assert torch.allclose(from_out.inferred_force, to_out.inferred_force, atol=1e-5)
    assert torch.allclose(
        from_out.inferred_stress, to_out.inferred_stress, atol=1e-5
    )


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize('b', ['own', 'shared'])
def test_bilinear_gate_double_backward(b):
    """second derivatives (force-loss backward) of the LunaTP BilinearGate equal
    e3nn's in float64, and torch gradcheck/gradgradcheck pass on the LunaTP op"""
    from e3nn.o3 import Irreps

    from sevenn._const import ACTIVATION_DICT
    from sevenn.nn.bilinear_gate import BilinearGate
    from sevenn.nn.lunatp_helper import patch_bilinear_gate

    act = {'e': ACTIVATION_DICT['e']['silu'], 'o': ACTIVATION_DICT['o']['tanh']}
    irreps = Irreps('8x0e+4x1o+4x2e+4x3o')
    torch.manual_seed(0)
    ref = BilinearGate(irreps, act, act, b=b)
    with torch.no_grad():
        ref.linear.weight.normal_()
    luna = patch_bilinear_gate(BilinearGate(irreps, act, act, b=b))
    luna.load_state_dict(ref.state_dict(), strict=True)
    ref, luna = ref.cuda().double(), luna.cuda().double()
    x = torch.randn(
        6, ref.get_gate_irreps_in().dim, device='cuda', dtype=torch.float64
    )

    def f(m):
        return lambda t: m({KEY.NODE_FEATURE: t})[KEY.NODE_FEATURE]

    res = []
    for m in (ref, luna):
        xx = x.clone().requires_grad_(True)
        y = f(m)(xx)
        (g,) = torch.autograd.grad(y, xx, torch.ones_like(y), create_graph=True)
        loss = g.square().sum() + y.square().sum()
        res.append(torch.autograd.grad(loss, (xx, m.linear.weight)))
    # LunaTP's float64 path agrees with e3nn to ~2e-8 relative to the largest
    # entry (fp32-precision constants); a wrong derivative would differ at O(1)
    for a, b in zip(*res):
        assert (a - b).abs().max() < 1e-6 * b.abs().max()
    xx = x[:3].clone().requires_grad_(True)
    # atomic accumulation (index_select backward) is not bitwise reproducible
    assert torch.autograd.gradcheck(f(luna), (xx,), nondet_tol=1e-12)
    assert torch.autograd.gradgradcheck(f(luna), (xx,), nondet_tol=1e-12)


_s2ffn = {
    'interaction_type': 'metaformer',
    'self_connection_type': 'identity',
    'lmax': 3,
    'irreps_manual': ['16x0e', '16x0e+8x1o+4x2e+4x3o', '16x0e+8x1o+4x2e+4x3o',
                      '16x0e'],
    'ffn_type': 's2_bilinear',
}


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_s2_bilinear_ffn_lunatp():
    """S2 grid FFN on the LunaTP model (patched linears around the plain-torch
    grid GEMMs): E/F/S and the force-loss parameter gradients equal e3nn's"""
    from sevenn.nn.ffn_variants import S2BilinearGate

    torch.manual_seed(777)
    model_e3nn = get_model(_s2ffn)
    model_lunatp = get_model(_s2ffn, use_lunatp=True)
    missing, unexpected = model_lunatp.load_state_dict(
        model_e3nn.state_dict(), strict=False
    )
    assert all(k.endswith('tp_conv.tp.weight') for k in missing), missing
    assert all('_convolution.convolution.' in k for k in unexpected), unexpected
    assert isinstance(model_lunatp._modules['1_ffn_gate'], S2BilinearGate)

    def run(model):
        model.train()
        model.set_is_batch_data(True)
        model.zero_grad(set_to_none=True)
        out = model(get_graphs(batched=True))
        e, f, s = (
            out.inferred_total_energy, out.inferred_force, out.inferred_stress
        )
        (e.square().sum() + f.square().sum() + s.square().sum()).backward()
        grads = {n: q.grad for n, q in model.named_parameters() if 'ffn' in n}
        return e.detach(), f.detach(), s.detach(), grads

    e1, f1, s1, g1 = run(model_e3nn)
    e2, f2, s2, g2 = run(model_lunatp)
    assert torch.allclose(e1, e2, rtol=1e-5, atol=2e-5)
    assert torch.allclose(f1, f2, atol=1e-5)
    assert torch.allclose(s1, s2, atol=1e-5)
    assert g1 and g1.keys() == g2.keys()
    for n, g in g1.items():
        assert (g - g2[n]).norm() <= 1e-4 * g.norm() + 1e-8, n


def _attention_models(opts):
    """e3nn, new LunaTP (default) and old LunaTP (lunatp_fast_attention False)
    attention models with the same random weights (zero-initialized radial
    bias, type biases and output gate randomized)"""
    cf = {**_metaformer, 'conv_attention': opts}
    torch.manual_seed(777)
    model_e3nn = get_model(cf)
    with torch.no_grad():
        for n, p in model_e3nn.named_parameters():
            if 'edge_attention' in n and (
                n.split('.')[-2] in ('rb', 'og') or n.endswith(('t_src', 't_dst'))
            ):
                p.normal_(0, 0.3)
    models = {'e3nn': model_e3nn}
    for name, fast in (('new', True), ('old', False)):
        model = get_model({**cf, 'lunatp_fast_attention': fast}, use_lunatp=True)
        missing, _ = model.load_state_dict(model_e3nn.state_dict(), strict=False)
        assert all(k.endswith('tp_conv.tp.weight') for k in missing), missing
        models[name] = model
    return models


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize(
    'opts', [{}, {'skip_first': False, 'null_logit': False, 'out_gate': False}]
)
def test_attention_lunatp_shared_graph(opts, monkeypatch):
    """
    LunaTP attention (default): one LunaTP graph per forward (the old path
    rebuilt it in every attention layer) and one attention envelope; E/F/S
    and the force-loss parameter gradients equal the old LunaTP path and e3nn
    """
    import sevenn.nn.lunatp_helper as lunatp_helper
    from sevenn.nn.convolution import IrrepsScatterGatterFusedConvolution
    from sevenn.nn.edge_attention import EdgeAttention

    models = _attention_models(opts)
    new, old = models['new'], models['old']
    n_attn = sum(isinstance(m, EdgeAttention) for m in new.modules())
    assert n_attn == (3 if opts else 2)
    assert all(
        isinstance(m, lunatp_helper.LunaTPEdgeAttention)
        for m in new.modules() if isinstance(m, EdgeAttention)
    )
    shared = lunatp_helper.LunaTPSharedGraphConvolution
    assert isinstance(new._modules['1_convolution'], shared)
    assert isinstance(new._modules['0_convolution'], shared)  # also w/o attention
    assert type(old._modules['1_edge_attention']) is EdgeAttention
    assert type(old._modules['1_convolution']) is IrrepsScatterGatterFusedConvolution

    calls = []
    make_graph = lunatp_helper.make_graph
    monkeypatch.setattr(
        lunatp_helper, 'make_graph',
        lambda *a, **k: calls.append(1) or make_graph(*a, **k),
    )

    def run(model):
        model.train()
        model.set_is_batch_data(True)
        model.zero_grad(set_to_none=True)
        out = model(get_graphs(batched=True))
        e, f, s = (
            out.inferred_total_energy, out.inferred_force, out.inferred_stress
        )
        (e.square().sum() + f.square().sum() + s.square().sum()).backward()
        grads = {
            n: q.grad for n, q in model.named_parameters() if q.grad is not None
        }
        return e.detach(), f.detach(), s.detach(), grads

    results = {}
    for name, model in models.items():
        calls.clear()
        results[name] = run(model)
        if name == 'new':
            assert len(calls) == 1
        elif name == 'old':  # + layer 0 if it has no attention
            assert len(calls) == n_attn + (0 if opts else 1)
    for name in ('old', 'e3nn'):  # fp32 atomics: run-to-run noise ~1e-6
        e1, f1, s1, g1 = results[name]
        e2, f2, s2, g2 = results['new']
        assert torch.allclose(e1, e2, rtol=1e-5, atol=2e-5), name
        assert torch.allclose(f1, f2, atol=1e-5), name
        assert torch.allclose(s1, s2, atol=1e-5), name
        assert g1 and g1.keys() == g2.keys()
        # the old path alone repeats the type-bias gradients only to ~1e-4
        # (fp32 atomics, sums with cancellation)
        for n, g in g1.items():
            assert (g - g2[n]).norm() <= 1e-3 * g.norm() + 1e-8, (name, n)


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_attention_lunatp_envelope_not_stale():
    """a second forward on the same (moved) data recomputes the envelope"""
    model = _attention_models({})['new']
    model.eval()
    model.set_is_batch_data(False)
    graph = _graph.clone().to('cuda')
    model(graph)
    assert 'attn_env2' in graph
    graph.edge_vec = graph.edge_vec.detach() * 1.02  # same data object
    out = model(graph)
    ref = _graph.clone().to('cuda')
    ref.edge_vec = ref.edge_vec * 1.02
    ref_out = model(ref)
    assert torch.allclose(
        out.inferred_total_energy, ref_out.inferred_total_energy, rtol=1e-6
    )
    assert torch.allclose(out.inferred_force, ref_out.inferred_force, atol=1e-6)


# ------------------------------------------- metaformer extensions (mf-ext)
_mfext = {  # A4-like: metaformer, pair radial MLP, SiLU readout, XPLOR, modal
    'interaction_type': 'metaformer',
    'self_connection_type': 'identity',
    'lmax': 3,
    'irreps_manual': ['16x0e', '16x0e+8x1o+4x2e+4x3o', '16x0e+8x1o+4x2e+4x3o',
                      '16x0e'],
    'drop_unused_tp_irreps': True,
    'radial_pair_embedding': True,
    'readout_activation': 'silu',
    'cutoff_function': {'cutoff_function_name': 'XPLOR', 'cutoff_on': 3.5},
    'use_modality': True,
    KEY.NUM_MODALITIES: 2,
    KEY.MODAL_MAP: {'mpa': 0, 'omat24': 1},
    'use_modal_node_embedding': False,
    'use_modal_self_inter_intro': True,
    'use_modal_self_inter_outro': True,
    'use_modal_output_block': True,
    'use_modal_wise_shift': True,
    'use_modal_wise_scale': False,
    'shift': [-1.5, -2.5],
    'scale': 1.0,
}
# config name -> (overrides of _mfext, substrings of zero-initialized
# parameters to randomize before the comparison)
MFEXT_CONFIGS = {
    'a4': ({}, ('table_',)),
    'no_prenorm': ({'metaformer_norm': False}, ('table_',)),
    'readout_block': ({'readout_activation': None,
                       'readout_block': {'type': 'swiglu'}}, ('table_',)),
    'readout_block_gate': ({'readout_activation': None, 'readout_block': {}},
                           ('table_', 'readout_norm')),
    'per_layer_readout': ({'per_layer_readout': True},
                          ('table_', 'layer_readout')),
}


def _mfext_run(model, v, dtype=torch.float32):
    """E, F, S, Hessian-vector product -d(F.v)/dpos (double backward, as the
    force loss) and the parameter gradients of a loss of all of them"""
    model.train()
    model.zero_grad(set_to_none=True)
    g = _graph.clone().to('cuda')
    for k, t in g.items():
        if torch.is_tensor(t) and t.is_floating_point():
            g[k] = t.to(dtype)
    pos = g[KEY.POS].clone().requires_grad_(True)
    src, dst = g[KEY.EDGE_IDX]
    shift = (g[KEY.EDGE_VEC] - (g[KEY.POS][dst] - g[KEY.POS][src])).detach()
    g[KEY.EDGE_VEC] = pos[dst] - pos[src] + shift
    out = model(g)
    e = out[KEY.PRED_TOTAL_ENERGY]
    f = out[KEY.PRED_FORCE]
    s = out[KEY.PRED_STRESS]
    (hv,) = torch.autograd.grad((f * v.to(dtype)).sum(), pos, create_graph=True)
    loss = e.square().sum() + f.square().sum() + s.square().sum()
    (loss + hv.square().sum()).backward()
    grads = {
        n: q.grad.double() for n, q in model.named_parameters()
        if q.grad is not None
    }
    return [t.detach().double() for t in (e, f, s, -hv)], grads


def _to_double_hook(module, inputs, data):
    for k in (KEY.NODE_FEATURE, KEY.NODE_ATTR):  # one-hot is float32
        if k in data and data[k].is_floating_point():
            data[k] = data[k].double()
    return data


def _mfext_backends(name):
    """e3nn fp32, LunaTP fp32 and e3nn fp64 (reference) models, same weights"""
    over, rand = MFEXT_CONFIGS[name]
    cf = {**_mfext, **over}
    torch.manual_seed(777)
    model_e3nn = get_model(cf)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, q in model_e3nn.named_parameters():
            if any(r in n for r in rand):
                q.copy_(0.3 * torch.randn(q.shape, generator=gen))
    model_lunatp = get_model(cf, use_lunatp=True)
    missing, unexpected = model_lunatp.load_state_dict(
        model_e3nn.state_dict(), strict=False
    )
    assert all(k.endswith('tp_conv.tp.weight') for k in missing), missing
    assert all('_convolution.convolution.' in k for k in unexpected), unexpected
    model_64 = copy.deepcopy(model_e3nn).double()
    model_64._modules['onehot_idx_to_onehot'].register_forward_hook(
        _to_double_hook
    )
    for m in (model_e3nn, model_lunatp, model_64):
        m.prepare_modal_deploy('mpa')
    return model_e3nn, model_lunatp, model_64


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize('name', list(MFEXT_CONFIGS))
def test_mfext_lunatp(name):
    """
    every new IrrepsLinear / EquivariantGate / EquivariantRMSNorm of the
    metaformer extensions is LunaTP-patched; E/F/S, a Hessian-vector product
    (double backward) and the parameter gradients of a loss of all of them
    agree with the e3nn model's to fp32 precision (both compared with an
    fp64 e3nn reference)
    """
    from LunaEquivariance import LunaEqGate, LunaEqLinear

    from sevenn.nn.equivariant_gate import EquivariantGate
    from sevenn.nn.lunatp_helper import LunaTPEquivariantRMSNorm
    from sevenn.nn.linear import IrrepsLinear
    from sevenn.nn.normalization import EquivariantRMSNorm

    model_e3nn, model_lunatp, model_64 = _mfext_backends(name)
    for k, m in model_lunatp._modules.items():
        if isinstance(m, IrrepsLinear):
            assert isinstance(m.linear, LunaEqLinear), k
        elif isinstance(m, EquivariantGate):
            assert isinstance(m.gate, LunaEqGate), k
        elif isinstance(m, EquivariantRMSNorm):
            assert isinstance(m, LunaTPEquivariantRMSNorm), k
    gen = torch.Generator().manual_seed(2)
    v = torch.randn(_graph[KEY.POS].shape, generator=gen).to('cuda')
    ref64, g64 = _mfext_run(model_64, v, torch.float64)
    ref, g_ref = _mfext_run(model_e3nn, v)
    out, g_out = _mfext_run(model_lunatp, v)
    # LunaTP fp32 is as close to the fp64 e3nn reference as e3nn fp32 is
    # (a missing patch or a wrong derivative would be off at O(1))
    for a64, a, b, tag in zip(ref64, ref, out, ('E', 'F', 'S', 'Hv')):
        noise = (a - a64).abs().max().item()
        err = (b - a64).abs().max().item()
        floor = 1e-6 * max(1.0, a64.abs().max().item())
        assert err <= 10 * noise + floor, (tag, err, noise)
    assert ref64[3].abs().max() > 1e-3  # non-trivial Hessian
    assert g64 and g64.keys() == g_ref.keys() == g_out.keys()
    for n, g in g64.items():
        noise = (g_ref[n] - g).norm().item()
        err = (g_out[n] - g).norm().item()
        assert err <= 10 * noise + 1e-6 * g.norm().item() + 1e-10, (n, err, noise)
