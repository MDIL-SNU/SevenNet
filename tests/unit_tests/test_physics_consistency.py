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
- learned pair repulsion: positive, strictly decreasing, C2 at its own pair
  cutoff r_c,ij (exactly zero beyond), for random parameters
- parameters that start at a value that switches their path off (zero
  tables, zero weights, magnitude residual exponent 0) are randomized, so
  every path is differentiated
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
    's2_inv': {**_metaformer, 'ffn_type': 's2_inv'},
    's2_inv_l3': {**_metaformer, 'ffn_type': 's2_inv', 'lmax': 3,
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
    'conv_gate_off': {**_metaformer, 'conv_gate': False},
    'parallel_block': {**_metaformer, 'parallel_block': True},
    'light_conv': {**_metaformer, 'conv_gate': False, 'parallel_block': True,
                   'radial_pair_embedding': True},
    'parallel_block_variants': {
        **_metaformer, 'parallel_block': [False, True, True],
        'layer_scale_init': 0.1, 'ffn_film': True, 'ffn_type': 'swiglu'},
    'short_radial': {**_metaformer, 'radial_pair_embedding': True,
                     'radial_short_basis': {}},
    'short_radial_block_node_cond': {
        **_metaformer, 'radial_pair_embedding': True, 'radial_node_cond': True,
        'radial_mlp_block': {}, 'radial_short_basis': {'cutoff': 3.2,
                                                       'num_basis': 4}},
    'pair_repulsion': {**_metaformer, 'radial_pair_embedding': True,
                       'pair_repulsion': {'cutoff_scale': 1.4,
                                          'amp_init': 1.0}},
    'pair_repulsion_nequip_3terms': {
        **_partial, 'pair_repulsion': {'num_terms': 3, 'cutoff_scale': 1.4,
                                       'max_cutoff': 3.5, 'amp_init': 1.0}},
    'norm_magnitude': {**_metaformer, 'radial_pair_embedding': True,
                       'norm_magnitude': True},
    'norm_magnitude_variants': {
        **_metaformer, 'norm_magnitude': True, 'use_final_norm': True,
        'readout_block': {}, 'parallel_block': [False, True, True],
        'irreps_manual': ['4x0e', '4x0e+2x0o+4x1o+2x1e+4x2e+2x2o',
                          '4x0e+2x0o+4x1o+2x1e+4x2e+2x2o', '4x0e']},
    'short_range_both': {**_metaformer, 'radial_pair_embedding': True,
                         'ffn_type': 's2_bilinear', 'norm_magnitude': True,
                         'pair_repulsion': {'cutoff_scale': 1.4,
                                            'amp_init': 1.0}},
    'magnitude_residual': {**_metaformer, 'radial_pair_embedding': True,
                           'ffn_type': 's2_bilinear',
                           'magnitude_residual': True},
    'magnitude_residual_conv': {**_metaformer, 'radial_pair_embedding': True,
                                'magnitude_residual': {'branches': 'conv'}},
    'magnitude_residual_variants': {
        **_metaformer, 'magnitude_residual': {'gamma_max': 0.8, 'rho_ref': 1.5},
        'readout_block': {}, 'per_layer_readout': True, 'conv_gate': False,
        'layer_scale_init': 0.1, 'norm_magnitude': True,
        'irreps_manual': ['4x0e', '4x0e+2x0o+4x1o+2x1e+4x2e+2x2o',
                          '4x0e+2x0o+4x1o+2x1e+4x2e+2x2o', '4x0e']},
    'exp_radial': {**_metaformer, 'radial_pair_embedding': True,
                   'ffn_type': 's2_bilinear', 'radial_exp_basis': {}},
    'exp_radial_short_block_node_cond': {
        **_metaformer, 'radial_pair_embedding': True, 'radial_node_cond': True,
        'radial_mlp_block': {}, 'radial_short_basis': {'cutoff': 3.2,
                                                       'num_basis': 4},
        'radial_exp_basis': {'scales': [0.3, 0.9], 'cutoff': 2.8}},
    'light_conv_full_parity': {
        **_metaformer, 'conv_gate': False, 'parallel_block': True,
        'irreps_manual': ['4x0e', '4x0e+2x0o+4x1o+2x1e+4x2e+2x2o',
                          '4x0e+2x0o+4x1o+2x1e+4x2e+2x2o', '4x0e']},
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
            )) or 'radial_node_cond' in n or 'layer_readout' in n or (
                'w_short' in n or 'w_exp' in n
            ):
                p.normal_(0, 0.3)   # zero-initialized radial bias / output gate
            if cf.get('radial_mlp_block') is not None and 'table_' in n:
                p.normal_(0, 0.3)   # element-pair tables (zero-initialized)
            if n.startswith('pair_repulsion.') and 'radius' not in n:
                # element-dependent pairs; effective spread 0.3 (stored / c)
                c = model._modules['pair_repulsion'].param_scale
                p.add_(0.3 * torch.randn_like(p) / c)
            if 'mag_weight' in n:
                p.normal_(0, 0.3)   # zero-initialized norm magnitude feature
            if n.endswith('_outro.gamma'):
                p.uniform_(0.2, 0.9)   # magnitude residual exponent (init 0)
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
                                  's2_bilinear_l3', 's2_inv', 's2_inv_l3',
                                  'radial_mlp_block',
                                  'radial_mlp_block_mlp_layer',
                                  'radial_mlp_block_node_cond', 'short_radial',
                                  'short_radial_block_node_cond',
                                  'pair_repulsion', 'norm_magnitude',
                                  'short_range_both', 'magnitude_residual',
                                  'magnitude_residual_variants', 'exp_radial',
                                  'exp_radial_short_block_node_cond'])
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


@pytest.mark.parametrize('name,opt,weight', [
    ('short_radial', 'radial_short_basis', 'w_short'),
    ('short_radial_block_node_cond', 'radial_short_basis', 'w_short'),
    ('exp_radial', 'radial_exp_basis', 'w_exp'),
    ('exp_radial_short_block_node_cond', 'radial_exp_basis', 'w_exp'),
    ('exp_radial_short_block_node_cond', 'radial_short_basis', 'w_short'),
])
def test_short_radial_c2_at_inner_cutoff(name, opt, weight):
    """
    the short-range (Bessel) and exponential basis terms vanish C2 at their
    own cutoff r_s: for a dimer, dE = E - E(their weights zeroed) shrinks
    like (r_s - r)^3 (dE ~8x, its force ~4x, its force constant ~2x when the
    distance to r_s halves), is exactly zero beyond r_s, and the force
    constant is continuous across r_s
    """
    import copy

    cf = CONFIGS[name]
    rs = cf[opt].get('cutoff', 3.0)
    model = _model(cf)
    ref = copy.deepcopy(model)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if weight in n:
                p.zero_()

    def de(x):
        at = Atoms('NaCl', positions=[[0, 0, 0], [x, 0, 0]])
        return _efs(model, at)[0] - _efs(ref, at)[0]

    h = 1e-3
    out = []
    for delta in (0.02, 0.01):
        r = rs - delta
        out.append(np.array([
            de(r), (de(r + h) - de(r - h)) / (2 * h),
            (de(r + h) - 2 * de(r) + de(r - h)) / h**2,
        ]))
    far, near = out
    assert abs(de(rs - 0.5)) > 1e-4  # the short term is active inside r_s
    for i, thr in enumerate((0.2, 0.35, 0.6)):
        assert abs(near[i]) < 1e-8 or abs(near[i]) < thr * abs(far[i]), (far, near)
    for x in (rs + 1e-3, rs + 0.1, rs + 0.5):
        assert de(x) == 0.0
    s1, s2 = (_k_jump(model, rs, delta=d, h=5e-4) for d in (0.002, 0.001))
    assert s2 < 0.7 * s1 or s2 < 1e-6, (s1, s2)


def test_exp_radial_contact_slope():
    """
    the Bessel basis is even in r: a dimer's energy E(r) is an even analytic
    function near r = 0, so its force -dE/dr vanishes linearly at contact
    (halves when r halves); with the exponential basis (f_k'(0) = -1/a_k)
    the force tends to a non-zero limit. Both energies stay finite.
    """
    import copy

    model = _model(CONFIGS['exp_radial'])
    base = copy.deepcopy(model)
    with torch.no_grad():
        for n, p in base.named_parameters():
            if 'w_exp' in n:
                p.zero_()

    def force(m, x):
        f = _efs(m, Atoms('NaCl', positions=[[0, 0, 0], [x, 0, 0]]))[1]
        return f[1, 0]

    fb = [force(base, x) for x in (0.004, 0.002, 0.001)]
    fe = [force(model, x) for x in (0.004, 0.002, 0.001)]
    assert abs(fb[2]) < 0.6 * abs(fb[1]) < 0.6**2 * abs(fb[0]), fb  # ~ r
    assert abs(fe[2]) > 1e-3, fe
    assert abs(fe[2] - fe[1]) < 0.1 * abs(fe[2]), fe  # finite limit
    for x in (1e-3, 0.3):
        at = Atoms('NaCl', positions=[[0, 0, 0], [x, 0, 0]])
        assert np.isfinite(_efs(model, at)[0])


def _pair_term(model, x, sym='NaCl'):
    """dimer pair repulsion energy phi(x) (sum of the per-atom terms)"""
    model.eval()
    at = Atoms(sym, positions=[[0, 0, 0], [x, 0, 0]])
    return model(_graph(at))[KEY.PAIR_REPULSION_ENERGY].sum().item()


@pytest.mark.parametrize('name', ['pair_repulsion', 'pair_repulsion_nequip_3terms'])
def test_pair_repulsion_dimer(name):
    """
    the pair term of a dimer is positive and strictly decreasing up to its
    pair cutoff r_c = min(s (R_Na + R_Cl), r_max), diverges as r -> 0, is
    exactly zero beyond r_c and C2 at r_c (phi, phi', phi'' shrink like
    (r_c - r)^3, ^2, ^1); the force constant of the whole model is continuous
    across r_c
    """
    from ase.data import atomic_numbers, covalent_radii

    cf = CONFIGS[name]
    opts = cf['pair_repulsion']
    rc = opts['cutoff_scale'] * (covalent_radii[atomic_numbers['Na']]
                                 + covalent_radii[atomic_numbers['Cl']])
    rc = min(rc, opts.get('max_cutoff', cutoff))
    model = _model(cf)
    grid = np.concatenate([np.geomspace(0.02, 0.5, 30),
                           np.linspace(0.5, rc - 1e-3, 120)[1:]])
    phi = np.array([_pair_term(model, x) for x in grid])
    assert (phi > 0).all()
    assert (np.diff(phi) < 0).all()
    assert phi[0] > 20 * phi[29]  # 1/r: grows without bound as r -> 0
    for x in (rc + 1e-4, rc + 0.05, rc + 0.3):
        assert _pair_term(model, x) == 0.0
    h = 1e-3
    out = []
    for delta in (0.02, 0.01):
        r = rc - delta
        e = [_pair_term(model, x) for x in (r - h, r, r + h)]
        out.append(np.array([e[1], (e[2] - e[0]) / (2 * h),
                             (e[2] - 2 * e[1] + e[0]) / h**2]))
    far, near = out
    assert abs(_pair_term(model, rc - 0.5)) > 1e-4  # active inside r_c
    for i, thr in enumerate((0.2, 0.35, 0.6)):
        assert abs(near[i]) < 1e-9 or abs(near[i]) < thr * abs(far[i]), (far, near)
    s1, s2 = (_k_jump(model, rc, delta=d, h=5e-4) for d in (0.002, 0.001))
    assert s2 < 0.7 * s1 or s2 < 1e-6, (s1, s2)


def test_pair_repulsion_guarantee_random_parameters():
    """
    positivity and monotonicity hold by construction for any parameters:
    20 draws with a wide spread of log amplitudes and decay constants, all
    element pairs, dense grid 0.01 A .. r_c
    """
    from sevenn.nn.pair_repulsion import PairRepulsion

    tm = {1: 0, 8: 1, 29: 2, 68: 3, 18: 4}
    mod = PairRepulsion(tm, 5, num_terms=3, cutoff_scale=1.2, max_cutoff=6.0)
    c = mod.param_scale  # stored = effective / c
    gen = torch.Generator().manual_seed(0)
    r = torch.linspace(0.01, 6.0, 3000)
    for _ in range(20):
        with torch.no_grad():  # effective lambda, theta ~ N(0, 3^2)
            mod.log_amp.copy_(3.0 * torch.randn(5, 3, generator=gen) / c)
            mod.decay_raw.copy_(3.0 * torch.randn(5, 3, generator=gen) / c)
        for i in range(5):
            for j in range(5):
                ti = torch.full_like(r, i, dtype=torch.long)
                tj = torch.full_like(r, j, dtype=torch.long)
                phi = mod.pair_energy(r, ti, tj)
                assert torch.equal(phi, mod.pair_energy(r, tj, ti))  # symmetric
                rc = min(1.2 * float(mod.radius[i] + mod.radius[j]), 6.0)
                inside = r < rc - 1e-3
                p = phi[inside]
                assert (p > 0).all() and (p[1:] <= p[:-1]).all()
                d = p[1:] - p[:-1]
                big = p[1:] > 1e-30  # strict where it does not underflow
                assert (d[big] < 0).all()
                assert (phi[r >= rc] == 0).all()
