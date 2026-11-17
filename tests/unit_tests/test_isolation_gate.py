"""
Isolated-atom gate (sevenn/nn/isolation_gate.py, model config architecture:
isolation_gate, built by sevenn/model_build_gate.py): an atom without
neighbours gets E = shift exactly. A config without the key builds the
original model (test_architecture_selection). float64, CPU, random weights
(zero-initialized paths randomized, a constant must not hide behind them). FD forces / stress /
Hessian of more options: test_physics_consistency.py; parallel MD:
test_parallel_consistency.py. The segments and ML-IAP are also checked here on a
sparse cell: its atoms have s well below 1 and different per atom, in the bulk
of those tests s = 1 hides a wrong ghost gate.
"""
import copy

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule
from torch_geometric.loader.dataloader import Collater

import sevenn._keys as KEY
from sevenn.nn.isolation_gate import ISOLATION_C, IsolationGate
from sevenn.nn.scale import ModalWiseRescale
from tests.unit_tests.test_arch_options import _metaformer, _partial
from tests.unit_tests.test_model import cutoff, get_model
from tests.unit_tests.test_parallel_consistency import (
    _A0,
    _A4,
    MODAL,
    _assert_close,
    _config,
    _double,
    _e3gnn_parallel,
    _mliap_model,
    _randomize,
    _run_mliap,
    _segments,
    _serial_efs,
)
from tests.unit_tests.test_physics_consistency import _graph

GATE = {KEY.ARCHITECTURE: 'isolation_gate'}
_A18 = {**_A4, 'ffn_type': 's2_bilinear', 'ffn_s2_channels': 8,
        'edge_gate': {'dim': 4}, **GATE}
_FULL = {
    **_A18, 'ffn_film': True, 'magnitude_residual': True,
    'radial_exp_basis': {}, 'coord_scale': {},
    'pair_repulsion': {'cutoff_scale': 1.4, 'amp_init': 1.0},
    'per_layer_readout': True,
}
CONFIGS = {
    'a18': _A18,
    'full': _FULL,
    'A0_nequip_linear_sc': _A0,
    'nequip_scalar_shift': {},  # test_model.get_model_config: plain nequip
    'per_layer_nequip': {**_A0, 'per_layer_readout': True},
    's2_inv_readout_block': {
        **_A4, 'ffn_type': 's2_inv', 'readout_activation': None,
        'readout_block': {}},
    'final_norm': {**_A4, 'use_final_norm': True, 'norm_magnitude': True},
    'parallel_block': {**_A4, 'parallel_block': [False, True, True]},
    'linear_residual': {**_A4, 'self_connection_type': 'linear'},
    'modal_node_embedding': {**_A4, 'use_modal_node_embedding': True},
    'modal_scale_species_shift': {
        **_A4, 'use_modal_wise_shift': False, 'use_modal_wise_scale': True,
        'shift': -3.0, 'scale': [1.5, 0.7]},
    'metaformer_species_shift': {
        **_metaformer, 'shift': [-1.0, -2.0, -3.0, -4.0],
        'scale': [1.0, 2.0, 0.5, 1.5]},
    'pair_radial_partial': {**_partial, 'radial_pair_embedding': True},
    'bias_in_linear': {**_A4, 'use_bias_in_linear': True},
    'readout_fcn': {
        **_metaformer, 'readout_as_fcn': True,
        'readout_fcn_hidden_neurons': [4, 4], 'readout_fcn_activation': 'relu'},
    # x_0 has other multiplicities than x_L: the copies must be carried along
    'irreps_change': {
        **_metaformer,
        'irreps_manual': ['6x0e', '4x0e+2x1o', '4x0e+2x1o', '5x0e']},
}
CONFIGS = {k: {**v, **GATE} for k, v in CONFIGS.items()}


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _model(cf):
    torch.manual_seed(0)
    model = _double(get_model(copy.deepcopy(cf)))
    _randomize(model)
    with torch.no_grad():  # zero-initialized paths would hide a constant
        for n, p in model.named_parameters():
            if 'layer_readout' in n or n.endswith('_edge_gate.theta'):
                p.normal_(0, 0.3)
    return model


def _modals(model):
    return list(model.modal_map) if model.modal_map else [None]


def _g(atoms, modal=None):
    g = _graph(atoms)
    if modal is not None:
        g[KEY.DATA_MODALITY] = modal
    return g


def _batch(items):
    gs = [_g(a, m) for a, m in items]
    return Collater(gs)(gs)


def _single(model, modal):
    m = copy.deepcopy(model)
    m.eval()
    m.set_is_batch_data(False)
    if modal is not None:
        m.prepare_modal_deploy(modal)
    return m


def _isolated(z):
    return Atoms(numbers=[z], positions=[[0.0, 0.0, 0.0]], cell=[20, 20, 20],
                 pbc=True)


def _shift(model, z, modal):
    rescale = model._modules['rescale_atomic_energy']
    t = model.type_map[z]
    sh = rescale.shift.detach()
    if isinstance(rescale, ModalWiseRescale) and rescale.use_modal_wise_shift:
        return sh[model.modal_map[modal], t].item()
    return sh.reshape(-1)[t if sh.numel() > 1 else 0].item()


def _cell(seed=1):
    at = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    at.rattle(0.05, seed=seed)
    sym = at.get_chemical_symbols()
    sym[0], sym[3] = 'O', 'H'
    at.set_chemical_symbols(sym)
    at.set_cell(at.cell @ (np.eye(3) + 0.02 * np.array(
        [[0.3, -1, 0.2], [0.5, 0.4, 0.1], [-0.2, 0.6, -0.7]])), scale_atoms=True)
    return at


def _dimer(r, modal_vacuum=20.0):
    return Atoms('NaCl', positions=[[0, 0, 0], [r, 0, 0]],
                 cell=[modal_vacuum] * 3, pbc=True)


def _sparse_atoms():
    """about 2-5 neighbours per atom, spread over two ranks along a"""
    at = bulk('NaCl', 'rocksalt', a=7.3, cubic=True) * (2, 1, 1)
    at.rattle(0.35, seed=3)
    sym = at.get_chemical_symbols()
    sym[1], sym[6] = 'O', 'H'
    at.set_chemical_symbols(sym)
    at.wrap()
    return at


def _gate_values(model, atoms):
    data = _single(model, _modals(model)[0])(_g(atoms))
    return data[KEY.NODE_GATE].detach().numpy()


@pytest.mark.parametrize('name', list(CONFIGS))
def test_isolated_atom_is_shift(name):
    """every species and modality, bit-exact, batched next to real structures
    (training mode) and alone (eval, non-batched), no force and no stress"""
    model = _model(CONFIGS[name])
    modals = _modals(model)
    zs = sorted(model.type_map)
    items = [(_isolated(z), m) for m in modals for z in zs]
    real = [(_cell(), modals[0]), (molecule('H2O'), modals[-1])]
    real[1][0].cell = [12, 12, 12]
    real[1][0].pbc = True
    batch = _batch(real[:1] + items + real[1:])
    model.train()
    model.set_is_batch_data(True)
    out = model(batch)
    e = out[KEY.PRED_TOTAL_ENERGY].detach()[1:-1]
    ref = torch.tensor([_shift(model, z, m) for m in modals for z in zs])
    assert torch.equal(e, ref), (e - ref)
    assert torch.all(out[KEY.PRED_FORCE][8: 8 + len(items)] == 0)

    for at, m in items:
        out = _single(model, m)(_g(at))
        assert out[KEY.PRED_TOTAL_ENERGY].item() == _shift(
            model, int(at.numbers[0]), m
        )
        assert torch.all(out[KEY.NODE_GATE] == 0)
        assert torch.all(out[KEY.PRED_FORCE] == 0)


@pytest.mark.parametrize('name', ['a18', 'full', 'A0_nequip_linear_sc'])
def test_far_dimer_and_c1_at_cutoff(name):
    """
    r just above the cutoff: sum of the shifts, bit-exact. r just below: E, and
    dE/dr by FD, go to the far values with the envelope. XPLOR: w ~ (r_c - r)^2,
    so E - E_far shrinks ~4x and dE/dr ~2x when r_c - r halves.
    """
    model = _model(CONFIGS[name])
    na, cl = 11, 17
    for m in _modals(model):
        mm = _single(model, m)

        def e(r):
            return mm(_g(_dimer(r)))[KEY.PRED_TOTAL_ENERGY].item()

        far = _shift(model, na, m) + _shift(model, cl, m)
        assert e(cutoff + 1e-3) == far
        assert e(cutoff + 1.0) == far
        h = 1e-5
        vals = []
        for d in (0.02, 0.01, 0.005):
            r = cutoff - d
            vals.append((e(r) - far, (e(r + h) - e(r - h)) / (2 * h)))
        assert abs(e(cutoff - 0.3) - far) > 1e-5, 'dimer has no interaction'
        for (e0, d0), (e1, d1) in zip(vals[:-1], vals[1:]):
            assert abs(e1) < 0.35 * abs(e0)
            assert abs(d1) < 0.6 * abs(d0)
        # analytic force = FD, and it ends in zero across the cutoff
        at = _dimer(cutoff - 1e-3)
        out = mm(_g(at))
        assert abs(out[KEY.PRED_FORCE].detach()[1, 0].item()) < 0.2 * abs(vals[0][1])
        f0 = mm(_g(_dimer(cutoff - 0.01)))[KEY.PRED_FORCE].detach()[1, 0].item()
        assert abs(f0 + vals[1][1]) < 1e-5 * abs(f0) + 1e-7


@pytest.mark.parametrize('name', ['a18', 'full', 'A0_nequip_linear_sc'])
@pytest.mark.parametrize('kind', ['bulk', 'sparse'])
def test_fd_forces_stress(name, kind):
    """analytic against central FD, modal on, periodic; sparse: s far from 1,
    so dE/ds * ds/dr is a large part of the force"""
    model = _single(_model(CONFIGS[name]), MODAL)
    atoms = _cell() if kind == 'bulk' else _sparse_atoms()

    def e(at):
        return model(_g(at))[KEY.PRED_TOTAL_ENERGY].item()

    out = model(_g(atoms))
    f = out[KEY.PRED_FORCE].detach().numpy()
    s = -out[KEY.PRED_STRESS].detach().numpy()[[0, 1, 2, 4, 5, 3]]
    assert np.abs(f).max() > 1e-4
    h = 1e-4
    rng = np.random.default_rng(0)
    for _ in range(6):
        v = rng.normal(size=atoms.positions.shape)
        v /= np.linalg.norm(v)
        p, m = atoms.copy(), atoms.copy()
        p.positions += h * v
        m.positions -= h * v
        fd = (e(p) - e(m)) / (2 * h)
        assert abs(fd + np.sum(f * v)) < 1e-6 * max(1.0, abs(fd)), (fd, f)
    vol = atoms.get_volume()
    voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for k, (a, b) in enumerate(voigt):
        de = []
        for sgn in (1, -1):
            eps = np.eye(3)
            eps[a, b] += sgn * h / (1 if a == b else 2)
            if a != b:
                eps[b, a] += sgn * h / 2
            at = atoms.copy()
            at.set_cell(atoms.cell @ eps, scale_atoms=True)
            de.append(e(at))
        fd = (de[0] - de[1]) / (2 * h * vol)
        assert abs(fd - s[k]) < 1e-6 * max(1.0, abs(fd)), (k, fd, s[k])


@pytest.mark.parametrize('name', ['a18', 'full', 'A0_nequip_linear_sc'])
def test_o3_invariance(name):
    model = _single(_model(CONFIGS[name]), MODAL)
    atoms = _sparse_atoms()
    e0 = model(_g(atoms))[KEY.PRED_TOTAL_ENERGY].item()
    f0 = model(_g(atoms))[KEY.PRED_FORCE].detach().numpy()
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    for R in (q * np.sign(np.linalg.det(q)), -q * np.sign(np.linalg.det(q))):
        at = atoms.copy()
        at.set_cell(atoms.cell @ R.T, scale_atoms=False)
        at.positions = atoms.positions @ R.T
        out = model(_g(at))
        assert abs(out[KEY.PRED_TOTAL_ENERGY].item() - e0) < 1e-9
        assert np.abs(out[KEY.PRED_FORCE].detach().numpy() - f0 @ R.T).max() < 1e-8


@pytest.mark.parametrize('name', ['a18', 'full', 'A0_nequip_linear_sc'])
def test_batch_against_single(name):
    """one-atom frames between real ones: real frames do not see them"""
    model = _model(CONFIGS[name])
    modals = _modals(model)
    frames = [
        (_cell(), modals[0]), (_isolated(8), modals[0]),
        (_sparse_atoms(), modals[-1]), (_isolated(17), modals[-1]),
        (_dimer(2.4), modals[-1]),
    ]
    model.eval()
    model.set_is_batch_data(True)
    out = model(_batch(frames))
    i = 0
    for k, (at, m) in enumerate(frames):
        s = _single(model, m)(_g(at))
        e, f = s[KEY.PRED_TOTAL_ENERGY].item(), s[KEY.PRED_FORCE].detach()
        assert abs(out[KEY.PRED_TOTAL_ENERGY][k].item() - e) < 1e-9
        fb = out[KEY.PRED_FORCE].detach()[i: i + len(at)]
        assert torch.allclose(fb, f, atol=1e-9)
        sb = out[KEY.PRED_STRESS].detach()[k]
        assert torch.allclose(sb, s[KEY.PRED_STRESS].detach().squeeze(), atol=1e-9)
        i += len(at)


def test_gate_values():
    """s = 1 - exp(-c n): a18 envelope (XPLOR 5.5 - 6.0), dimer, bulk"""
    gate = IsolationGate()
    assert gate.c == ISOLATION_C == 1.0
    w = torch.tensor([1.0, 1.0, 1.0])
    data = {
        KEY.NODE_FEATURE: torch.zeros(3),
        KEY.EDGE_ENVELOPE: w,
        KEY.EDGE_IDX: torch.tensor([[0, 0, 1], [1, 1, 0]]),
    }
    s = gate(data)[KEY.NODE_GATE]
    assert s[2].item() == 0.0  # no edge
    assert s[1].item() == pytest.approx(1 - np.exp(-1.0))
    assert s[0].item() == pytest.approx(1 - np.exp(-2.0))
    # float32 model: exact 0 and no underflow near the cutoff
    d32 = {k: v.float() if v.is_floating_point() else v for k, v in data.items()}
    d32[KEY.EDGE_ENVELOPE] = torch.tensor([1e-9, 0.0, 0.0], dtype=torch.float32)
    s = gate(d32)[KEY.NODE_GATE]
    assert s[0].item() == pytest.approx(1e-9, rel=1e-5)


def test_bulk_gate_and_force_of_s():
    """
    Dense cell (n of about 26, as a typical atom at r_c = 6 A or less): s = 1
    and s gives no force. Sparse cell: the force of s (forces with s detached
    differ) is real.
    """
    model = _model(CONFIGS['a18'])
    dense = _cell()
    dense.set_cell(dense.cell * 0.78, scale_atoms=True)
    s = _gate_values(model, dense)
    assert s.min() > 0.999999

    class _Detached(IsolationGate):
        def forward(self, data):
            data = super().forward(data)
            data[KEY.NODE_GATE] = data[KEY.NODE_GATE].detach()
            return data

    frozen = copy.deepcopy(model)
    frozen._modules['node_gate'] = _Detached()
    bulk_f = [
        _single(m, MODAL)(_g(dense))[KEY.PRED_FORCE].detach().numpy()
        for m in (model, frozen)
    ]
    assert np.abs(bulk_f[0] - bulk_f[1]).max() < 1e-5 * np.abs(bulk_f[0]).max()
    sparse = _sparse_atoms()
    s = _gate_values(model, sparse)
    assert s.min() < 0.99 and s.max() - s.min() > 0.01, s
    sparse_f = [
        _single(m, MODAL)(_g(sparse))[KEY.PRED_FORCE].detach().numpy()
        for m in (model, frozen)
    ]
    assert np.abs(sparse_f[0] - sparse_f[1]).max() > 1e-3


@pytest.mark.parametrize('name', ['a18', 'full', 'A0_nequip_linear_sc',
                                  'modal_node_embedding', 'per_layer_nequip'])
def test_parallel_sparse(name):
    """e3gnn/parallel and ML-IAP where s differs per atom"""
    cf = CONFIGS[name]
    model = _single(_model(cf), MODAL)
    atoms = _sparse_atoms()
    s = model(_g(atoms))[KEY.NODE_GATE].detach().numpy()
    assert s.min() < 0.99
    ser = _serial_efs(model, atoms)
    segs = _segments(model, cf)
    par, ranks = _e3gnn_parallel(segs, atoms, _config(cf), (2, 1, 1))
    assert all(len(rk['ghost']) > 0 for rk in ranks)
    assert ranks[0]['x_dims'][0] == model._modules['0_convolution']._comm_size
    _assert_close(par, ser, name)
    res, _, _ = _run_mliap([_mliap_model(model) for _ in range(2)], atoms, (2, 1, 1))
    _assert_close(res, ser, name)


def test_parallel_needs_the_exchange_of_layer_0():
    """why layer 0 is exchanged: the conv input of a neighbour depends on its s
    (a ghost atom's s is only known to its owner), s = 1 there changes E"""
    cf = CONFIGS['a18']
    model = _single(_model(cf), MODAL)
    atoms = _sparse_atoms()
    ser = _serial_efs(model, atoms)
    frozen = copy.deepcopy(model)

    class _NoGate0(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, data):
            s = data[KEY.NODE_GATE]
            data[KEY.NODE_GATE] = torch.ones_like(s)
            data = self.module(data)
            data[KEY.NODE_GATE] = s
            return data

    frozen._modules['0_self_interaction_1'] = _NoGate0(
        frozen._modules['0_self_interaction_1']
    )
    assert abs(_serial_efs(frozen, atoms)[0] - ser[0]) > 1e-4


def test_readout_activation_with_constant_is_refused():
    with pytest.raises(ValueError, match='act\\(0\\)'):
        get_model({**_A4, 'readout_activation': 'sigmoid', **GATE})
    get_model({**_A4, 'readout_activation': 'sigmoid'})  # original model: no check


def test_torchscript_serial():
    import e3nn.util.jit

    for name in ('a18', 'A0_nequip_linear_sc'):
        model = _single(_model(CONFIGS[name]), MODAL)
        model.delete_module_by_key('force_output')
        model.eval_type_map = False
        scripted = torch.jit.freeze(e3nn.util.jit.script(model))
        for at in (_cell(), _isolated(11), _dimer(5.0)):
            g = _g(at)
            d = {k: v for k, v in g.items() if torch.is_tensor(v)}
            d[KEY.NODE_FEATURE] = torch.tensor(
                [model.type_map[int(z)] for z in d[KEY.ATOMIC_NUMBERS]],
                dtype=torch.int64,
            )
            d2 = dict(d)
            e = scripted(d)[KEY.PRED_TOTAL_ENERGY].item()
            e0 = model(d2)[KEY.PRED_TOTAL_ENERGY].item()
            assert abs(e - e0) < 1e-10
            if len(at) == 1:
                assert e == _shift(model, int(at.numbers[0]), MODAL)


def test_checkpoint_calculator_deploy(tmp_path):
    """checkpoint -> calculator (float32) -> sevenn deploy, isolated atom = shift"""
    import sevenn
    from sevenn.calculator import SevenNetCalculator
    from sevenn.scripts.deploy import deploy
    from sevenn.util import load_checkpoint

    torch.set_default_dtype(torch.float32)
    cf = _config(CONFIGS['a18'])
    torch.manual_seed(0)
    model = get_model(copy.deepcopy(cf))
    _randomize(model)
    cfg = copy.deepcopy(cf)
    cfg.update({'version': sevenn.__version__, KEY.USE_MODALITY: True})
    path = tmp_path / 'cp.pth'
    torch.save({'config': cfg, 'model_state_dict': model.state_dict(),
                'epoch': 1}, path)
    built = load_checkpoint(str(path)).build_model()
    assert torch.equal(built._modules['rescale_atomic_energy'].shift,
                       model._modules['rescale_atomic_energy'].shift)
    calc = SevenNetCalculator(str(path), modal='omat24', device='cpu')
    iso = _isolated(17)
    iso.calc = calc
    assert iso.get_potential_energy() == _shift(model, 17, 'omat24')
    assert np.all(iso.get_forces() == 0)
    fname = str(tmp_path / 'deployed_serial.pt')
    deploy(str(path), fname, modal='omat24')
    assert torch.jit.load(fname) is not None


@pytest.mark.parametrize('name', ['a18', 'A0_nequip_linear_sc', 'final_norm',
                                  'per_layer_nequip', 'linear_residual'])
def test_architecture_selection(name):
    """
    No key or 'sevennet': the original model, none of the gate modules, every
    module of the shared classes; isolation_gate: the same parameters (state
    dict keys and shapes), the gate on top
    """
    from sevenn.nn.ffn_variants import AddToResidual
    from sevenn.nn.isolation_gate import (
        GateBlend,
        GatedIrrepsLinear,
        GateReference,
        NodeReference,
        ReferenceDelta,
    )
    from sevenn.nn.linear import IrrepsLinear
    from sevenn.nn.normalization import MagnitudeResidualOutro
    from sevenn.nn.self_connection import SelfConnectionOutro

    gated_cf = CONFIGS[name]
    plain_cf = {k: v for k, v in gated_cf.items() if k != KEY.ARCHITECTURE}
    gated = get_model(copy.deepcopy(gated_cf))
    for cf in (plain_cf, {**plain_cf, KEY.ARCHITECTURE: 'sevennet'}):
        plain = get_model(copy.deepcopy(cf))
        assert 'node_gate' not in plain._modules
        for k, m in plain._modules.items():
            assert not isinstance(
                m, (IsolationGate, NodeReference, ReferenceDelta, GateBlend,
                    GateReference)
            ), k
            if isinstance(m, IrrepsLinear):
                assert not isinstance(m, GatedIrrepsLinear), k
            for cls in (SelfConnectionOutro, AddToResidual, MagnitudeResidualOutro):
                if isinstance(m, cls):
                    assert type(m) is cls, k
        sd_p, sd_g = plain.state_dict(), gated.state_dict()
        assert list(sd_p) == list(sd_g)
        assert all(sd_p[k].shape == sd_g[k].shape for k in sd_p)
    assert isinstance(gated._modules['node_gate'], IsolationGate)
    with pytest.raises(ValueError, match='architecture'):
        get_model({**plain_cf, KEY.ARCHITECTURE: 'gate'})


def test_continue_keeps_architecture():
    """continue: the checkpoint's architecture, a missing key is the original"""
    from sevenn.scripts.processing_continue import _keep_architecture

    class _Log:
        def writeline(self, _):
            pass

    for cp_arch, yaml_arch, out in (
        (None, 'isolation_gate', 'sevennet'),
        ('isolation_gate', 'sevennet', 'isolation_gate'),
        ('sevennet', None, 'sevennet'),
    ):
        config_cp = {} if cp_arch is None else {KEY.ARCHITECTURE: cp_arch}
        config = {} if yaml_arch is None else {KEY.ARCHITECTURE: yaml_arch}
        _keep_architecture(config, config_cp, _Log())
        assert config[KEY.ARCHITECTURE] == out
