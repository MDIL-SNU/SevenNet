"""
Physics consistency of every architecture option (float64, CPU). Required
for each new op before it is used in training:
- forces = -dE/dx (central finite differences along random directions)
- stress = (1/V) dE/d(strain) (central finite differences, ASE convention)
- position Hessian from double backward (the path the force loss trains
  through) is symmetric, obeys the acoustic sum rule and matches finite
  differences of the inference-path forces
- dimer energy is C2 at the cutoff (energy, force and force constant go to
  their far-field values continuously) for the polynomial envelope
"""
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

import sevenn._keys as KEY
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.util import dtype_correct
from tests.unit_tests.test_arch_options import _metaformer, _partial
from tests.unit_tests.test_model import cutoff, get_model

_film_rand = {**_metaformer, 'ffn_film': True}
CONFIGS = {
    'nequip': {},
    'partial_linear_sc': _partial,
    'metaformer': _metaformer,
    'metaformer_noffn_finalnorm': {**_metaformer, 'use_ffn': False,
                                   'use_final_norm': True},
    'xplor': {**_metaformer,
              'cutoff_function': {'cutoff_function_name': 'XPLOR',
                                  'cutoff_on': 3.5}},
    'frozen_bessel': {**_metaformer, 'radial_basis': {
        'radial_basis_name': 'bessel', 'bessel_basis_num': 8,
        'trainable_coeff': False}},
    'pair_radial': {**_metaformer, 'radial_pair_embedding': True},
    'silu_readout': {**_metaformer, 'readout_activation': 'silu'},
    'bilinear_gate': {**_metaformer, 'gate_type': 'bilinear'},
    'bilinear_gate_l3': {**_partial, 'gate_type': 'bilinear', 'lmax': 3,
                         'irreps_manual': ['4x0e', '4x0e+4x1o+4x2e+4x3o',
                                           '4x0e+4x1o+4x2e+4x3o', '4x0e']},
    'bilinear_gate_same_l': {**_metaformer, 'gate_type': 'bilinear',
                             'bilinear_paths': 'same_l'},
    'bilinear_gate_shared': {**_partial, 'gate_type': 'bilinear', 'lmax': 3,
                             'bilinear_b': 'shared',
                             'irreps_manual': ['4x0e', '4x0e+4x1o+4x2e+4x3o',
                                               '4x0e+4x1o+4x2e+4x3o', '4x0e']},
    'swiglu': {**_metaformer, 'ffn_type': 'swiglu'},
    'layer_scale': {**_metaformer, 'layer_scale_init': 0.1},
    'film': _film_rand,
    'agnesi': {**_metaformer, 'distance_transform': 'agnesi'},
    'sigmoid_gate': {**_metaformer, 'act_gate': {'e': 'sigmoid', 'o': 'tanh'}},
    'attention': {**_metaformer, 'conv_attention': {}},
    'attention_pair': {**_metaformer, 'conv_attention': {},
                       'radial_pair_embedding': True},
    'radial_node_cond': {**_metaformer, 'radial_pair_embedding': True,
                         'radial_node_cond': True},
    'moe': {**_metaformer, 'ffn_moe': True},
    's2_bilinear': {**_metaformer, 'ffn_type': 's2_bilinear'},
    's2_bilinear_l3': {**_metaformer, 'ffn_type': 's2_bilinear', 'lmax': 3,
                       'irreps_manual': ['4x0e', '4x0e+4x1o+4x2e+4x3o',
                                         '4x0e+4x1o+4x2e+4x3o', '4x0e']},
    'irreps_conv': {**_metaformer,
                    'irreps_conv': ['4x0e', '2x0e+2x1o+2x2e', '2x0e+2x1o+2x2e']},
    'no_prenorm': {**_metaformer, 'metaformer_norm': False},
    'readout_block': {**_metaformer, 'readout_block': {}},
    'readout_block_swiglu': {**_metaformer, 'radial_pair_embedding': True,
                             'readout_block': {'type': 'swiglu',
                                               'hidden_scale': 2}},
    'per_layer_readout': {**_metaformer, 'radial_pair_embedding': True,
                          'per_layer_readout': True},
    'per_layer_readout_nequip': {**_partial, 'per_layer_readout': True},
    'radial_mlp_block': {**_metaformer, 'radial_pair_embedding': True,
                         'radial_mlp_block': {}},
    'radial_mlp_block_mlp_layer': {
        **_metaformer, 'radial_pair_embedding': True,
        'radial_mlp_block': {'type': 'mlp', 'norm': 'layer', 'num_blocks': 2}},
    'radial_mlp_block_node_cond': {
        **_metaformer, 'radial_pair_embedding': True, 'radial_node_cond': True,
        'radial_mlp_block': {'final_norm': False}},
}


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _to_double_hook(module, inputs, data):
    for k in (KEY.NODE_FEATURE, KEY.NODE_ATTR):  # one-hot is created as float32
        if k in data and data[k].is_floating_point():
            data[k] = data[k].double()
    return data


def _model(cf):
    torch.manual_seed(0)
    model = get_model(cf).double()
    model.onehot_idx_to_onehot.register_forward_hook(_to_double_hook)
    with torch.no_grad():  # zero-initialized tables would hide their derivatives
        for n, p in model.named_parameters():
            if 'film' in n or 'equivariant_gate.linear' in n:  # bilinear term
                p.normal_(0, 0.3)
            if ('edge_attention' in n and (
                n.split('.')[-2] in ('rb', 'og') or n.endswith(('t_src', 't_dst'))
            )) or 'radial_node_cond' in n or 'layer_readout' in n:
                p.normal_(0, 0.3)   # zero-initialized radial bias / output gate
            if cf.get('radial_mlp_block') is not None and 'table_' in n:
                p.normal_(0, 0.3)   # element-pair tables (zero-initialized)
    model.set_is_batch_data(False)
    return model


def _graph(atoms):
    dct = dl.unlabeled_atoms_to_graph(atoms, cutoff)
    for k, v in dct.items():  # keep float64 from numpy (from_numpy_dict casts)
        if isinstance(v, np.ndarray):
            dct[k] = dtype_correct(v, float_dtype=torch.float64)
    return AtomGraphData(**dct)


def _efs(model, atoms):
    model.eval()
    out = model(_graph(atoms))
    e = out[KEY.PRED_TOTAL_ENERGY].item()
    f = out[KEY.PRED_FORCE].detach().numpy()
    s = -out[KEY.PRED_STRESS].detach().numpy()[[0, 1, 2, 4, 5, 3]]  # ASE voigt
    return e, f, s


def _voigt_to_mat(s):
    xx, yy, zz, yz, xz, xy = s
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _atoms():
    atoms = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    rng = np.random.default_rng(0)
    atoms.set_cell(atoms.cell @ (np.eye(3) + 0.03 * rng.normal(size=(3, 3))),
                   scale_atoms=True)
    atoms.rattle(0.08, seed=1)
    return atoms


def _hessian_autograd(model, atoms):
    """-dF/dx through double backward (training path)"""
    model.train()
    g = _graph(atoms)
    pos = g[KEY.POS].clone().requires_grad_(True)
    src, dst = g[KEY.EDGE_IDX]
    shift = (g[KEY.EDGE_VEC] - (g[KEY.POS][dst] - g[KEY.POS][src])).detach()
    g[KEY.EDGE_VEC] = pos[dst] - pos[src] + shift
    f = model(g)[KEY.PRED_FORCE].reshape(-1)
    rows = [
        -torch.autograd.grad(f[i], pos, retain_graph=True)[0].reshape(-1)
        for i in range(f.numel())
    ]
    model.eval()
    return torch.stack(rows).detach().numpy()


def _fd4(f, h):
    """df/dt at 0, 4-point central difference (truncation error O(h^4))"""
    return (8 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12 * h)


@pytest.mark.parametrize('name', list(CONFIGS))
def test_force_stress_hessian(name):
    model = _model(CONFIGS[name])
    atoms = _atoms()
    e0, f0, s0 = _efs(model, atoms)
    assert np.abs(f0).max() > 1e-6
    rng = np.random.default_rng(1)
    h = 1e-4

    for _ in range(3):  # forces
        d = rng.normal(size=f0.shape)
        d /= np.linalg.norm(d)
        de = _fd4(lambda t: _efs(
            model, Atoms(atoms, positions=atoms.positions + t * d))[0], h)
        assert abs(de + (f0 * d).sum()) < 1e-6 * max(1.0, abs(de)), name

    vol = atoms.get_volume()
    for _ in range(2):  # stress
        eps = rng.normal(size=(3, 3))
        eps = 0.5 * (eps + eps.T)
        eps /= np.linalg.norm(eps)

        def e_strain(t):
            a = atoms.copy()
            a.set_cell(atoms.cell @ (np.eye(3) + t * eps), scale_atoms=True)
            return _efs(model, a)[0]

        de = _fd4(e_strain, h)
        assert abs(de - vol * (_voigt_to_mat(s0) * eps).sum()) < 1e-6 * max(
            1.0, abs(de)
        ), name

    H = _hessian_autograd(model, atoms)
    scale = np.abs(H).max()
    assert scale > 0
    assert np.abs(H - H.T).max() < 1e-8 * scale, name  # symmetric
    n = len(atoms)
    asr = H.reshape(n, 3, n, 3).sum(axis=2)  # translation invariance
    assert np.abs(asr).max() < 1e-8 * scale, name
    for _ in range(2):  # matches finite differences of inference forces
        d = rng.normal(size=f0.shape)
        d /= np.linalg.norm(d)
        fp = _efs(model, Atoms(atoms, positions=atoms.positions + h * d))[1]
        fm = _efs(model, Atoms(atoms, positions=atoms.positions - h * d))[1]
        hd = -(fp - fm).reshape(-1) / (2 * h)
        assert np.abs(hd - H @ d.reshape(-1)).max() < 1e-5 * scale, name


def _dimer_tail(model, h=1e-3):
    """(E - E_far, dE/dr, d2E/dr2) of a Na-Cl dimer at r_c - 0.02 and r_c - 0.01"""
    def e(x):
        return _efs(model, Atoms('NaCl', positions=[[0, 0, 0], [x, 0, 0]]))[0]

    e_far = e(cutoff + 1.0)
    out = []
    for delta in (0.02, 0.01):
        r = cutoff - delta
        out.append(np.array([
            e(r) - e_far,
            (e(r + h) - e(r - h)) / (2 * h),
            (e(r + h) - 2 * e(r) + e(r - h)) / h**2,
        ]))
    return out


@pytest.mark.parametrize('name', ['metaformer', 'agnesi', 'pair_radial', 'attention',
                                  'radial_node_cond',
                                  'swiglu', 'film', 'bilinear_gate',
                                  'bilinear_gate_l3', 's2_bilinear',
                                  's2_bilinear_l3', 'radial_mlp_block',
                                  'radial_mlp_block_mlp_layer',
                                  'radial_mlp_block_node_cond'])
def test_dimer_c2_at_cutoff(name):
    """
    poly_cut (p=6) vanishes as (r_c - r)^3: halving the distance to r_c must
    shrink E - E_far ~8x, the force ~4x and the force constant ~2x (C2).
    """
    far, near = _dimer_tail(_model(CONFIGS[name]))
    # components already ~0 at r_c - 0.02 (e.g. env^2-weighted attention: higher
    # order than C2) pass; the others must shrink as (r_c - r)^3, ^2, ^1
    tiny = 1e-8
    for i, thr in enumerate((0.2, 0.35, 0.6)):
        assert abs(near[i]) < tiny or abs(near[i]) < thr * abs(far[i]), (far, near)


def _k_jump(model, r0, delta=0.002, h=1e-3):
    """relative change of the dimer force constant across r0"""
    def k(r):
        e = [_efs(model, Atoms('NaCl', positions=[[0, 0, 0], [x, 0, 0]]))[0]
             for x in (r - h, r, r + h)]
        return (e[0] - 2 * e[1] + e[2]) / h**2

    km, kp = k(r0 - delta), k(r0 + delta)
    return abs(kp - km) / max(abs(kp), abs(km))


def test_xplor_force_constant_jumps_at_r_on():
    """
    XPLOR switching S(r) is C1 at r_on: S'' jumps from 0 to
    -24 r_on^2 / (r_c^2 - r_on^2)^2, so the pair force constant is
    discontinuous where a neighbor distance crosses r_on (here 3.5 A).
    At r_c the Bessel basis itself vanishes, so the product is C2 there.
    The polynomial envelope has no such point.
    """
    r_on = CONFIGS['xplor']['cutoff_function']['cutoff_on']
    xplor, smooth = _model(CONFIGS['xplor']), _model(CONFIGS['metaformer'])
    # a discontinuity does not shrink with the step, smooth variation does
    j1, j2 = (_k_jump(xplor, r_on, delta=d) for d in (0.002, 0.001))
    assert j1 > 0.05 and j2 > 0.8 * j1, (j1, j2)
    s1, s2 = (_k_jump(smooth, r_on, delta=d, h=5e-4) for d in (0.002, 0.001))
    assert s2 < 0.7 * s1, (s1, s2)
    far, near = _dimer_tail(_model(CONFIGS['xplor']))
    ratio = np.abs(near / far)
    assert ratio[0] < 0.2 and ratio[1] < 0.35 and ratio[2] < 0.6, ratio
