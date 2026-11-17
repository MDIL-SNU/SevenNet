"""
Isolated-atom anchor (config anchor_isolated_atoms, sevenn/nn/isolated_anchor.py):
E_i = shift + scale (f(h_i) - f(h0_Z)), so an atom without neighbours gets
exactly the shift, in training mode (virtual isolated nodes) and after
folding the constant into the shift (inference, deploy, parallel MD).
float64, CPU. Physics (FD forces / stress / Hessian, C2) and parallel MD
(e3gnn/parallel, ML-IAP) of anchored configs: CONFIGS of
test_physics_consistency.py / test_parallel_consistency.py.
"""
import copy

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule
from torch_geometric.loader.dataloader import Collater

import sevenn._keys as KEY
from sevenn.nn.isolated_anchor import (
    AnchorSubtract,
    AnchorVirtualNodes,
    fold_into_rescale,
    fold_isolated_anchor,
    is_anchored,
    isolated_scaled_energy,
)
from sevenn.nn.scale import ModalWiseRescale, Rescale, SpeciesWiseRescale
from sevenn.scripts.deploy import build_parallel_model_list
from tests.unit_tests.test_arch_options import _metaformer, _partial
from tests.unit_tests.test_model import get_model
from tests.unit_tests.test_parallel_consistency import (
    _A0,
    _A4,
    _config,
    _double,
    _randomize,
)
from tests.unit_tests.test_physics_consistency import _graph

_ANCHOR = {KEY.ANCHOR_ISOLATED_ATOMS: True}
_FULL = {  # options of the current / next runs together
    **_A4, 'ffn_type': 's2_bilinear', 'ffn_s2_channels': 8,
    'ffn_film': True, 'magnitude_residual': True,
    'radial_exp_basis': {}, 'edge_gate': {'dim': 4}, 'coord_scale': {},
    'pair_repulsion': {'cutoff_scale': 1.4, 'amp_init': 1.0},
    'per_layer_readout': True,
}
CONFIGS = {
    # multi-modal (mpa + omat24, modal-wise shift, species-wise scale)
    'A4': _A4,
    'A0_nequip': _A0,
    'full': _FULL,
    's2_inv_readout_block': {
        **_A4, 'ffn_type': 's2_inv', 'readout_activation': None,
        'readout_block': {}},
    'modal_scale_species_shift': {
        **_A4, 'use_modal_wise_shift': False, 'use_modal_wise_scale': True,
        'shift': -3.0, 'scale': [1.5, 0.7]},
    # single modal: scalar shift (Rescale) and species-wise shift
    'nequip_scalar_shift': {},
    'metaformer_species_shift': {
        **_metaformer, 'shift': [-1.0, -2.0, -3.0, -4.0],
        'scale': [1.0, 2.0, 0.5, 1.5]},
    'pair_radial_partial': {**_partial, 'radial_pair_embedding': True},
}


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _model(cf, anchor=True):
    torch.manual_seed(0)
    cf = {**copy.deepcopy(cf), **(_ANCHOR if anchor else {})}
    model = _double(get_model(cf))
    _randomize(model)
    with torch.no_grad():  # zero-initialized paths would hide the constant
        for n, p in model.named_parameters():
            if 'layer_readout' in n or n.endswith('_edge_gate.theta'):
                p.normal_(0, 0.3)
    return model


def _copy_weights(src, dst):
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=True)
    assert not missing and not unexpected


def _modals(model):
    return list(model.modal_map) if model.modal_map else [None]


def _g(atoms, modal):
    g = _graph(atoms)
    if modal is not None:
        g[KEY.DATA_MODALITY] = modal
    return g


def _batch(items):
    gs = [_g(a, m) for a, m in items]
    return Collater(gs)(gs)


def _isolated(sym):
    return Atoms(sym, positions=[[0.0, 0.0, 0.0]], cell=[20, 20, 20], pbc=True)


def _shift(model, z, modal):
    """fixed shift of atomic number z in modality modal (DFT isolated energy)"""
    rescale = model._modules['rescale_atomic_energy']
    t = model.type_map[z]
    sh = rescale.shift.detach()
    if isinstance(rescale, ModalWiseRescale) and rescale.use_modal_wise_shift:
        return sh[model.modal_map[modal], t].item()
    return sh.reshape(-1)[t if sh.numel() > 1 else 0].item()


def _structures():
    nacl = bulk('NaCl', 'rocksalt', a=5.63, cubic=True)
    nacl.rattle(0.05, seed=1)
    sym = nacl.get_chemical_symbols()
    sym[0], sym[3] = 'O', 'H'
    nacl.set_chemical_symbols(sym)
    dimer = Atoms('NaCl', positions=[[0, 0, 0], [2.6, 0.1, 0]],
                  cell=[15, 15, 15], pbc=True)
    h2o = molecule('H2O')
    h2o.cell = [12, 12, 12]
    h2o.pbc = True
    # an atom without neighbours next to a bonded pair, in one structure
    mixed = Atoms('NaClH', positions=[[0, 0, 0], [2.5, 0, 0], [10, 10, 10]],
                  cell=[20, 20, 20], pbc=True)
    return {'bulk': nacl, 'dimer': dimer, 'molecule': h2o, 'mixed': mixed}


def _d(model, g):
    """TorchScript input (as deploy / LAMMPS): tensors only, type indices"""
    d = {k: v for k, v in g.items() if torch.is_tensor(v)}
    tm = model.type_map
    d[KEY.NODE_FEATURE] = torch.tensor(
        [tm[int(z)] for z in d[KEY.ATOMIC_NUMBERS]], dtype=torch.int64)
    return d


def _efs(out):
    return (out[KEY.PRED_TOTAL_ENERGY].detach().clone(),
            out[KEY.PRED_FORCE].detach().clone(),
            out[KEY.PRED_STRESS].detach().clone())


def test_default_off_and_old_configs():
    """no key / false: no anchor modules, the same model as before"""
    base = get_model(_A4)
    off = get_model({**_A4, KEY.ANCHOR_ISOLATED_ATOMS: False})
    on = get_model({**_A4, **_ANCHOR})
    assert list(base._modules) == list(off._modules)
    assert not is_anchored(base) and not is_anchored(off) and is_anchored(on)
    names = list(on._modules)
    assert names[0] == 'anchor_virtual_nodes'
    assert names[names.index('rescale_atomic_energy') - 1] == 'anchor_subtract'
    # parameter-free: identical state dicts, checkpoints load either way
    assert list(base.state_dict()) == list(on.state_dict())
    from sevenn._const import DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG, model_defaults

    assert DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG[KEY.ANCHOR_ISOLATED_ATOMS] is False
    assert model_defaults({})[KEY.ANCHOR_ISOLATED_ATOMS] is False


def test_config_validation():
    from sevenn.parse_input import init_model_config

    cf = {'chemical_species': ['H', 'O'], KEY.ANCHOR_ISOLATED_ATOMS: True}
    assert init_model_config(dict(cf))[KEY.ANCHOR_ISOLATED_ATOMS] is True
    assert init_model_config({'chemical_species': ['H']})[
        KEY.ANCHOR_ISOLATED_ATOMS] is False
    with pytest.raises(ValueError):
        init_model_config({**cf, KEY.ANCHOR_ISOLATED_ATOMS: 'yes'})


@pytest.mark.parametrize('name', list(CONFIGS))
def test_isolated_atom_is_shift(name):
    """
    every species in every modality, batched training mode (virtual nodes,
    also next to bulk atoms of other modalities), non-batched eval mode and
    folded model: E(isolated) == shift (bitwise before folding)
    """
    model = _model(CONFIGS[name])
    modals = _modals(model)
    zs = sorted(model.type_map)
    items = [(_isolated(Atoms(numbers=[z]).get_chemical_symbols()[0]), m)
             for m in modals for z in zs]
    st = _structures()
    mixed = [(st['bulk'], modals[0]), (st['molecule'], modals[-1])]
    batch = _batch(mixed[:1] + items + mixed[1:])
    model.train()
    model.set_is_batch_data(True)
    out = model(batch)
    e = out[KEY.PRED_TOTAL_ENERGY].detach()[1:-1]
    ref = torch.tensor([_shift(model, z, m) for m in modals for z in zs])
    assert torch.equal(e, ref), (e - ref)
    assert out[KEY.PRED_FORCE].shape[0] == batch[KEY.NUM_ATOMS].sum()
    assert torch.all(out[KEY.PRED_FORCE][8 : 8 + len(items)] == 0)

    model.eval()
    model.set_is_batch_data(False)
    for at, m in items[:: max(1, len(items) // 5)]:
        z = int(at.numbers[0])
        model_m = copy.deepcopy(model)
        if m is not None:
            model_m.prepare_modal_deploy(m)
        out = model_m(_g(at, None))
        assert out[KEY.PRED_TOTAL_ENERGY].item() == _shift(model, z, m)

    folded = fold_isolated_anchor(copy.deepcopy(model))
    assert not is_anchored(folded)
    folded.set_is_batch_data(True)
    out = folded(_batch(items))
    assert torch.allclose(out[KEY.PRED_TOTAL_ENERGY].detach(), ref,
                          rtol=0, atol=1e-12)


@pytest.mark.parametrize('name', list(CONFIGS))
def test_folded_equals_unfolded(name):
    """bulk / dimer / molecule / mixed: E, F, S of the folded model equal the
    training-mode model (batched, all modalities) and the non-batched one"""
    model = _model(CONFIGS[name])
    folded = fold_isolated_anchor(copy.deepcopy(model))
    modals = _modals(model)
    items = [(at, m) for m in modals for at in _structures().values()]
    model.train()
    folded.train()
    for mdl in (model, folded):
        mdl.set_is_batch_data(True)
    e1, f1, s1 = _efs(model(_batch(items)))
    e2, f2, s2 = _efs(folded(_batch(items)))
    assert torch.allclose(e1, e2, rtol=1e-14, atol=1e-12), (e1 - e2)
    assert torch.allclose(f1, f2, rtol=0, atol=1e-12)
    assert torch.allclose(s1, s2, rtol=0, atol=1e-12)
    # non-batched eval, one modality (deploy-like)
    for mdl in (model, folded):
        mdl.eval()
        mdl.set_is_batch_data(False)
        if modals[0] is not None:
            mdl.prepare_modal_deploy(modals[-1])
    for at in _structures().values():
        a1, b1, c1 = _efs(model(_g(at, None)))
        a2, b2, c2 = _efs(folded(_g(at, None)))
        assert abs(a1.item() - a2.item()) < 1e-12
        assert torch.allclose(b1, b2, rtol=0, atol=1e-12)
        assert torch.allclose(c1, c2, rtol=0, atol=1e-12)


@pytest.mark.parametrize('name', ['A4', 'full', 'nequip_scalar_shift'])
def test_anchor_only_removes_a_constant(name):
    """
    same weights with and without the anchor: forces / stress equal (virtual
    nodes never reach real atoms), E_anchored = E_plain - sum_i scale f0(Z_i, m)
    in value AND parameter gradients (the constant is trained through)
    """
    anch = _model(CONFIGS[name])
    plain = _model(CONFIGS[name], anchor=False)
    _copy_weights(anch, plain)
    modals = _modals(anch)
    st = _structures()
    items = [(st['bulk'], modals[0]), (st['dimer'], modals[-1]),
             (st['mixed'], modals[0])]
    zs = sorted(anch.type_map)
    iso = [(_isolated(Atoms(numbers=[z]).get_chemical_symbols()[0]), m)
           for m in modals for z in zs]
    for mdl in (anch, plain):
        mdl.train()
        mdl.set_is_batch_data(True)
    out_a = anch(_batch(items))
    out_p = plain(_batch(items))
    out_i = plain(_batch(iso))  # E_iso(plain) - shift = scale f0
    c = {(m, z): out_i[KEY.PRED_TOTAL_ENERGY][k] - _shift(plain, z, m)
         for k, (m, z) in enumerate([(m, z) for m in modals for z in zs])}
    corr = torch.stack([
        sum(c[(m, int(z))] for z in at.numbers) for at, m in items
    ])
    target = out_p[KEY.PRED_TOTAL_ENERGY] - corr
    e_a = out_a[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e_a, target, rtol=1e-14, atol=1e-11)
    assert torch.allclose(out_a[KEY.PRED_FORCE], out_p[KEY.PRED_FORCE],
                          rtol=0, atol=1e-12)
    assert torch.allclose(out_a[KEY.PRED_STRESS], out_p[KEY.PRED_STRESS],
                          rtol=0, atol=1e-12)
    w = torch.linspace(0.5, 1.5, e_a.shape[0])
    loss_a = (w * e_a).sum() + out_a[KEY.PRED_FORCE].square().sum()
    loss_p = (w * target).sum() + out_p[KEY.PRED_FORCE].square().sum()
    pa = [(n, p) for n, p in anch.named_parameters() if p.requires_grad]
    pp = [p for p in plain.parameters() if p.requires_grad]
    assert len(pa) == len(pp)
    ga = torch.autograd.grad(loss_a, [p for _, p in pa], allow_unused=True,
                             retain_graph=True)
    gp = torch.autograd.grad(loss_p, pp, allow_unused=True, retain_graph=True)
    n_nonzero = 0
    for (n, _), a, p in zip(pa, ga, gp):
        if a is None or p is None:
            assert a is None and p is None, n
            continue
        assert torch.allclose(a, p, rtol=1e-9, atol=1e-11), n
        n_nonzero += int(a.abs().max() > 0)
    assert n_nonzero > 5
    # the constant is learned: the gradient differs from the plain energy's
    gq = torch.autograd.grad(
        (w * out_p[KEY.PRED_TOTAL_ENERGY]).sum()
        + out_p[KEY.PRED_FORCE].square().sum(),
        pp, allow_unused=True,
    )
    assert any(a is not None and q is not None and not torch.allclose(a, q)
               for a, q in zip(ga, gq))


def test_virtual_nodes_are_edge_free_and_dropped():
    """inside the model: virtual rows after the real ones, no edge touches
    them, one per (species, modality) of the batch; outputs have real sizes"""
    model = _model(_FULL)
    st = _structures()
    batch = _batch([(st['bulk'], 'mpa'), (st['molecule'], 'omat24'),
                    (st['dimer'], 'mpa')])
    n_real = int(batch[KEY.NUM_ATOMS].sum())
    seen = {}

    def hook(module, inputs, data):
        seen['x'] = data[KEY.NODE_FEATURE].clone()
        seen['batch'] = data[KEY.BATCH].clone()
        seen['modal'] = data[KEY.MODAL_TYPE].clone()
        seen['edge'] = data[KEY.EDGE_IDX].clone()
        seen['inv'] = data[KEY.ANCHOR_INDEX].clone()

    model._modules['anchor_virtual_nodes'].register_forward_hook(hook)
    model.train()
    model.set_is_batch_data(True)
    out = model(batch)
    x, b, modal, inv = seen['x'], seen['batch'], seen['modal'], seen['inv']
    n_v = x.shape[0] - n_real
    pairs = {(int(modal[b[i]]), int(x[i])) for i in range(n_real)}
    assert n_v == len(pairs)
    assert {(int(modal[b[i]]), int(x[i])) for i in range(n_real, x.shape[0])} \
        == pairs
    assert int(seen['edge'].max()) < n_real
    assert len(set(b[n_real:].tolist())) == n_v  # own dummy graph each
    assert int(b[n_real:].min()) == batch[KEY.NUM_ATOMS].shape[0]
    for i in range(n_real):  # twin: same species and modality
        j = n_real + int(inv[i])
        assert x[j] == x[i] and modal[b[j]] == modal[b[i]]
    for key in (KEY.NODE_FEATURE, KEY.NODE_ATTR, KEY.ATOM_TYPE, KEY.BATCH,
                KEY.ATOMIC_ENERGY, KEY.SCALED_ATOMIC_ENERGY, KEY.PRED_FORCE,
                KEY.PAIR_REPULSION_ENERGY):
        assert out[key].shape[0] == n_real, key
    assert out[KEY.MODAL_TYPE].shape[0] == 3
    assert out[KEY.MODAL_ATTR].shape[0] == 3


def test_rescale_fold_variants():
    """fold_into_rescale for every shift/scale module"""
    f0 = torch.tensor([[0.1, -0.2, 0.3]])
    r = fold_into_rescale(Rescale(-1.0, 2.0), f0)
    assert isinstance(r, SpeciesWiseRescale)
    assert torch.allclose(r.shift, torch.tensor([-1.2, -0.6, -1.6]))
    assert torch.allclose(r.scale, torch.tensor([2.0, 2.0, 2.0]))
    s = fold_into_rescale(SpeciesWiseRescale([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]),
                          f0)
    assert torch.allclose(s.shift, torch.tensor([-1.1, -1.6, -3.9]))
    f2 = torch.tensor([[0.1, -0.2, 0.3], [1.0, 2.0, 3.0]])
    m = ModalWiseRescale([-1.0, -2.0, -3.0], [[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]],
                         use_modal_wise_shift=False, use_modal_wise_scale=True)
    m = fold_into_rescale(m, f2)
    assert m.use_modal_wise_shift and m.shift.shape == (2, 3)
    assert torch.allclose(m.shift, torch.tensor([[-1.1, -1.6, -3.9],
                                                 [-3.0, -4.0, -4.5]]))
    with pytest.raises(ValueError):
        fold_into_rescale(SpeciesWiseRescale([-1.0, -2.0, -3.0], 1.0), f2)


def test_isolated_scaled_energy_matches_virtual_nodes():
    """offsets from the edge-less forward == the virtual nodes' f (training)"""
    model = _model(_FULL)
    f0 = isolated_scaled_energy(model)
    seen = {}

    def hook(module, inputs, data):
        inv = data[KEY.ANCHOR_INDEX]
        x = data[KEY.SCALED_ATOMIC_ENERGY]
        seen['virt'] = x[inv.shape[0]:].detach().clone().reshape(-1)
        seen['types'] = data[KEY.ATOM_TYPE][inv.shape[0]:].clone()

    model._modules['anchor_subtract'].register_forward_pre_hook(
        lambda m, args: hook(m, None, args[0]))
    st = _structures()
    model.train()
    model.set_is_batch_data(True)
    model(_batch([(st['bulk'], 'omat24')]))
    ref = f0[1, seen['types']]
    assert torch.allclose(seen['virt'], ref, rtol=0, atol=1e-13)
    # idempotent fold, plain models untouched
    folded = fold_isolated_anchor(copy.deepcopy(model))
    sh = folded._modules['rescale_atomic_energy'].shift.clone()
    fold_isolated_anchor(folded)
    assert torch.equal(sh, folded._modules['rescale_atomic_energy'].shift)
    assert torch.allclose(isolated_scaled_energy(folded), f0, atol=0, rtol=0)


@pytest.mark.parametrize('name', ['A4', 'full', 'nequip_scalar_shift'])
def test_torchscript(name):
    """unfolded anchored model scripted (non-batched) and the folded
    deploy model (script + freeze) give the eager energies / forces"""
    import e3nn.util.jit

    model = _model(CONFIGS[name])
    modal = _modals(model)[0]
    model.eval()
    model.set_is_batch_data(False)
    if modal is not None:
        model.prepare_modal_deploy(modal)
    at = _structures()['mixed']
    e0, f0, _ = _efs(model(_g(at, None)))
    ts_model = copy.deepcopy(model)
    ts_model.eval_type_map = False
    scripted = e3nn.util.jit.script(ts_model)
    e1, f1, _ = _efs(scripted(_d(model, _g(at, None))))
    assert abs(e0.item() - e1.item()) < 1e-12
    assert torch.allclose(f0, f1, rtol=0, atol=1e-12)
    folded = fold_isolated_anchor(copy.deepcopy(model))
    folded.eval_type_map = False
    frozen = torch.jit.freeze(e3nn.util.jit.script(folded))
    e2, f2, _ = _efs(frozen(_d(model, _g(at, None))))
    assert abs(e0.item() - e2.item()) < 1e-12
    assert torch.allclose(f0, f2, rtol=0, atol=1e-12)
    iso = _isolated('Na')
    e3 = frozen(_d(model, _g(iso, None)))[KEY.PRED_TOTAL_ENERGY].item()
    assert abs(e3 - _shift(model, 11, modal)) < 1e-12


@pytest.mark.parametrize('name', ['A4', 'full'])
def test_parallel_segments_folded(name):
    """e3gnn/parallel segments: no anchor modules, shift of the last segment
    folded (= shift - scale f0), from the config or from a given f0"""
    model = _model(CONFIGS[name])
    sd = model.state_dict()
    cfg = _config({**CONFIGS[name], **_ANCHOR})
    segs = build_parallel_model_list(sd, cfg, 'mpa')
    for seg in segs:
        assert not is_anchored(seg)
        assert not any(isinstance(m, (AnchorVirtualNodes, AnchorSubtract))
                       for m in seg.modules())
    f0 = isolated_scaled_energy(model)
    rescale = model._modules['rescale_atomic_energy']
    expect = rescale.shift.detach() - rescale.scale.detach() * f0
    got = segs[-1]._modules['rescale_atomic_energy'].shift.detach()
    assert torch.allclose(got, expect, rtol=0, atol=1e-12)
    segs2 = build_parallel_model_list(sd, cfg, 'mpa', isolated_f0=f0)
    assert torch.equal(
        segs2[-1]._modules['rescale_atomic_energy'].shift.detach(), got
    )
    # without the key: unchanged shift
    plain = build_parallel_model_list(sd, _config(CONFIGS[name]), 'mpa')
    assert torch.equal(plain[-1]._modules['rescale_atomic_energy'].shift,
                       rescale.shift)


def test_checkpoint_calculator_deploy(tmp_path):
    """checkpoint -> build_model (unfolded: the DFT shift stays, continue
    training sees it) -> SevenNetCalculator (folded) -> sevenn deploy (serial,
    folded) give the same energies; isolated atom = shift"""
    import sevenn
    from sevenn.calculator import SevenNetCalculator
    from sevenn.scripts.deploy import deploy
    from sevenn.util import load_checkpoint

    torch.set_default_dtype(torch.float32)
    cf = _config({**_A4, **_ANCHOR, 'ffn_type': 's2_bilinear'})
    torch.manual_seed(0)
    model = get_model(copy.deepcopy(cf))
    _randomize(model)
    cfg = copy.deepcopy(cf)
    cfg.update({'version': sevenn.__version__, KEY.USE_MODALITY: True})
    path = tmp_path / 'cp.pth'
    torch.save({'config': cfg, 'model_state_dict': model.state_dict(),
                'epoch': 1}, path)
    cp = load_checkpoint(str(path))
    built = cp.build_model()
    assert is_anchored(built)
    assert torch.equal(built._modules['rescale_atomic_energy'].shift,
                       model._modules['rescale_atomic_energy'].shift)
    calc = SevenNetCalculator(str(path), modal='omat24', device='cpu')
    assert not is_anchored(calc.model)
    at = _structures()['mixed']
    at.calc = calc
    e_calc = at.get_potential_energy()
    built.eval()
    built.set_is_batch_data(False)
    built.prepare_modal_deploy('omat24')
    g = _g(at, None)
    for k, v in g.items():
        if torch.is_tensor(v) and v.is_floating_point():
            g[k] = v.float()
    e_ref = built(g)[KEY.PRED_TOTAL_ENERGY].item()
    assert abs(e_calc - e_ref) < 1e-4
    iso = _isolated('Cl')
    iso.calc = calc
    e_iso = iso.get_potential_energy()
    assert abs(e_iso - _shift(model, 17, 'omat24')) < 1e-5
    fname = str(tmp_path / 'deployed_serial.pt')
    deploy(str(path), fname, modal='omat24')
    ts = torch.jit.load(fname)
    assert not any('anchor' in n for n, _ in ts.named_modules())
