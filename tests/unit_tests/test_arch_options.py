"""Tests for optional architecture modules (ablation switches) of ompa-v2."""
import torch
from e3nn.o3 import Irreps

import sevenn._keys as KEY
from tests.unit_tests.test_model import get_graphs, get_model

_partial = {
    'lmax': 2,
    'irreps_manual': ['4x0e', '4x0e+4x1o+4x2e', '4x0e+4x1o+4x2e', '4x0e'],
    'self_connection_type': 'linear',
}


def _convs(model):
    return {k: m for k, m in model.named_modules() if k.endswith('_convolution')}


def test_drop_unused_tp_irreps():
    full = get_model(_partial)
    drop = get_model({**_partial, KEY.DROP_UNUSED_TP_IRREPS: True})
    for (k, cf), (_, cd) in zip(_convs(full).items(), _convs(drop).items()):
        used = {ir for _, ir in Irreps(cf.convolution.irreps_out)}
        out_drop = {ir for _, ir in cd.convolution.irreps_out}
        assert out_drop <= used
        assert len(cd.convolution.instructions) <= len(cf.convolution.instructions)
    # the middle layer has 1e/2o outputs that the next linear never reads
    k1 = '1_convolution'
    assert {str(ir) for _, ir in _convs(full)[k1].convolution.irreps_out} > {
        str(ir) for _, ir in _convs(drop)[k1].convolution.irreps_out
    }
    out = drop(get_graphs(batched=True))
    assert torch.isfinite(out[KEY.PRED_FORCE]).all()


_metaformer = {
    **_partial,
    'interaction_type': 'metaformer',
    'self_connection_type': 'identity',
    KEY.DROP_UNUSED_TP_IRREPS: True,
}


def _rotated_pair(model):
    from e3nn.o3 import rand_matrix

    from tests.unit_tests.test_model import _graph_list

    g = _graph_list[0].clone()
    R = rand_matrix().to(g[KEY.POS].dtype)
    g_rot = g.clone()
    g_rot[KEY.POS] = g[KEY.POS] @ R.T
    if KEY.CELL in g:
        g_rot[KEY.CELL] = g[KEY.CELL] @ R.T
    g_rot[KEY.EDGE_VEC] = g[KEY.EDGE_VEC] @ R.T if KEY.EDGE_VEC in g else None
    model.set_is_batch_data(False)
    out = model(g.clone())
    out_rot = model(g_rot)
    return out, out_rot, R


import pytest  # noqa: E402


@pytest.mark.parametrize(
    'cf',
    [
        _metaformer,
        {**_metaformer, 'use_ffn': False},
        {**_metaformer, 'self_connection_type': 'linear'},
        {**_metaformer, 'use_final_norm': True, 'ffn_hidden_scale': 1},
    ],
)
def test_metaformer_equivariance(cf):
    model = get_model(cf)
    names = list(dict(model.named_children()))
    assert '1_norm_1' in names and '1_self_interaction_3' in names
    assert ('1_ffn_linear_1' in names) == cf.get('use_ffn', True)
    out, out_rot, R = _rotated_pair(model)
    e, e_r = out[KEY.PRED_TOTAL_ENERGY], out_rot[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4, rtol=1e-5)
    f, f_r = out[KEY.PRED_FORCE], out_rot[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    assert torch.isfinite(f).all() and f.abs().sum() > 0


def test_rmsnorm_and_residual_projection():
    from sevenn.nn.normalization import (
        EquivariantRMSNorm,
        SelfConnectionResidualIntro,
    )

    irreps = Irreps('8x0e+4x1o+2x2e')
    norm = EquivariantRMSNorm(irreps, data_key_x='x')
    x = torch.randn(5, irreps.dim) * 7.0
    y = norm({'x': x.clone()})['x']
    # each block weighted equally: balanced mean square is 1 after centering
    ms = (y.pow(2) * norm.balance).sum(-1)
    assert torch.allclose(ms, torch.ones(5), atol=1e-3)
    assert torch.allclose(y[:, :8].mean(-1), torch.zeros(5), atol=1e-5)
    # equivariance: rotating l>0 blocks commutes with the norm
    from e3nn.o3 import rand_matrix

    D = irreps.D_from_matrix(rand_matrix())
    y_rot = norm({'x': x @ D.T})['x']
    assert torch.allclose(y @ D.T, y_rot, atol=1e-4)

    res = SelfConnectionResidualIntro(Irreps('8x0e'), irreps, data_key_x='x')
    out = res({'x': x[:, :8]})[KEY.SELF_CONNECTION_TEMP]
    assert torch.equal(out[:, :8], x[:, :8]) and out[:, 8:].abs().sum() == 0
    down = SelfConnectionResidualIntro(irreps, Irreps('8x0e'), data_key_x='x')
    assert torch.equal(down({'x': x})[KEY.SELF_CONNECTION_TEMP], x[:, :8])


@pytest.mark.parametrize('cf', [{**_partial, 'radial_pair_embedding': True},
                                {**_metaformer, 'radial_pair_embedding': True}])
def test_radial_pair_embedding(cf):
    from sevenn.nn.radial_pair import PairRadialMLP

    model = get_model(cf)
    names = list(dict(model.named_children()))
    assert names.index('edge_pair_input') > names.index('onehot_idx_to_onehot')
    convs = _convs(model)
    assert all(isinstance(c.weight_nn, PairRadialMLP) for c in convs.values())
    # make pair tables non-trivial, then check equivariance and smooth cutoff
    with torch.no_grad():
        for c in convs.values():
            c.weight_nn.table_src.normal_()
            c.weight_nn.table_dst.normal_()
    out, out_rot, R = _rotated_pair(model)
    assert torch.allclose(
        out[KEY.PRED_TOTAL_ENERGY], out_rot[KEY.PRED_TOTAL_ENERGY], atol=1e-4
    )
    f, f_rot = out[KEY.PRED_FORCE], out_rot[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_rot, atol=1e-4)
    w = next(iter(convs.values())).weight_nn
    nb = w.num_basis
    inp = torch.randn(6, nb + 3)
    inp[:, nb] = 0.0  # envelope zero at cutoff
    inp[:, nb + 1 :] = torch.randint(0, w.table_src.shape[0], (6, 2)).float()
    assert torch.equal(w(inp), torch.zeros_like(w(inp)))


def test_new_modules_scriptable():
    from e3nn.util.jit import script

    cfs = [{**_metaformer, 'radial_pair_embedding': True, 'use_final_norm': True}]
    for cf in cfs:
        m = get_model(cf)
        m.set_is_batch_data(False)
        script(m.eval())


def test_readout_activation_keeps_modality():
    modal = {
        'use_modality': True,
        '_number_of_modalities': 2,
        '_modal_map': {'x1': 0, 'x2': 1},
        'use_modal_node_embedding': False,
        'use_modal_self_inter_intro': False,
        'use_modal_self_inter_outro': False,
        'use_modal_output_block': True,
        'use_modal_wise_shift': False,
        'use_modal_wise_scale': False,
    }
    base = get_model(modal)
    model = get_model({**modal, 'readout_activation': 'silu'})
    names = list(dict(model.named_children()))
    i = names.index('reduce_hidden_activation')
    assert names[i - 1] == 'reduce_input_to_hidden'
    assert names[i + 1] == 'reduce_hidden_to_energy'
    # modal output block (extra modal one-hot inputs) is still there
    n_base = sum(p.numel() for p in base.parameters())
    assert sum(p.numel() for p in model.parameters()) == n_base
    assert model.reduce_input_to_hidden.num_modalities == 2


def test_muon_routing_new_modules():
    from sevenn.train.muon import build_muon_param_groups

    model = get_model({**_metaformer, 'radial_pair_embedding': True})
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    muon = {name[id(p)] for p in groups[0]['params']}
    adam = {name[id(p)] for p in groups[1]['params']}
    assert '1_convolution.weight_nn.w1' in muon
    assert '1_ffn_linear_1.linear.weight' in muon
    assert '1_self_interaction_3.linear.weight' in muon
    assert '1_convolution.weight_nn.out.weight' in muon
    assert '1_convolution.weight_nn.hidden.0.weight' in muon
    assert {'1_convolution.weight_nn.table_src', '1_norm_1.weight'} <= adam


def test_frozen_bessel_moves_with_module():
    from sevenn.nn.edge_embedding import BesselBasis

    b = BesselBasis(6.0, 8, trainable_coeff=False)
    assert 'coeffs' not in b.state_dict()  # old checkpoints still load
    b = b.to(torch.float64)
    assert b(torch.rand(3, dtype=torch.float64) + 1.0).dtype == torch.float64
    if torch.cuda.is_available():
        assert b.cuda()(torch.rand(3, device='cuda') + 1.0).is_cuda
    model = get_model({**_partial, 'radial_basis': {
        'radial_basis_name': 'bessel', 'trainable_coeff': False}})
    model(get_graphs(batched=True))


def _bilinear_reference(g, x):
    """BilinearGate from its definition with plain torch ops:
    [act(s), act_g(g) * A] + Linear(A (x)_uuu (B * silu(w))), CG = sqrt(2l+1) w3j"""
    from e3nn import o3

    irreps_a = g.irreps_gated
    n_ch, dim_a = irreps_a.num_irreps, irreps_a.dim
    if g.shared:  # B = A, w appended to the plain gate input
        s, g_a, a = g.gate.sc(x[:, : g.dim_gate_in])
        b, w = a, x[:, g.dim_gate_in:]
    else:
        s, gw, ab = g.gate.sc(x)  # e3nn's own column extraction of the gate input
        a, b = ab[:, :dim_a], ab[:, dim_a:]
        g_a, w = gw[:, :n_ch], gw[:, n_ch:]
    starts = [sum(m for m, _ in irreps_a[:i]) for i in range(len(irreps_a))]
    ch = torch.cat([
        torch.arange(c, c + m).repeat_interleave(ir.dim)
        for c, (m, ir) in zip(starts, irreps_a)
    ])
    act_g = g.gate.act_gates.acts[0]  # e3nn-normalized gate activation
    base_out = torch.cat([g.gate.act_scalars(s), a * act_g(g_a)[:, ch]], 1)
    b = b * torch.nn.functional.silu(w)[:, ch]
    offs = [0]
    for m, ir in irreps_a:
        offs.append(offs[-1] + m * ir.dim)

    def blk(t, i):
        m, ir = irreps_a[i]
        return t[:, offs[i]:offs[i + 1]].reshape(-1, m, ir.dim)

    mids = []
    for k, (i, j) in enumerate(g.path_ij):
        l1, l2 = irreps_a[i].ir.l, irreps_a[j].ir.l
        ir3 = g.tp.irreps_out[k].ir
        cg = o3.wigner_3j(l1, l2, ir3.l) * (2 * ir3.l + 1) ** 0.5
        mids.append(torch.einsum('ijk,nui,nuj->nuk', cg, blk(a, i), blk(b, j))
                    .reshape(x.shape[0], -1))
    return base_out + g.linear(torch.cat(mids, 1))


@pytest.mark.parametrize('cf', [{**_partial, 'gate_type': 'bilinear'},
                                {**_metaformer, 'gate_type': 'bilinear'},
                                {**_partial, 'gate_type': 'bilinear',
                                 'irreps_manual': ['4x0e', '8x0e+4x1o+4x2e+4x1e',
                                                   '8x0e+4x1o+4x2e+4x1e', '4x0e']},
                                {**_partial, 'gate_type': 'bilinear',
                                 'bilinear_b': 'shared'},
                                {**_metaformer, 'gate_type': 'bilinear',
                                 'bilinear_b': 'shared'}])
def test_bilinear_gate(cf):
    from e3nn.util.jit import script

    from sevenn.nn.bilinear_gate import BilinearGate
    from sevenn.train.muon import build_muon_param_groups

    model = get_model(cf)
    gates = {
        k: m for k, m in model.named_children() if k.endswith('equivariant_gate')
    }
    assert isinstance(gates['1_equivariant_gate'], BilinearGate)
    assert not isinstance(gates['2_equivariant_gate'], BilinearGate)  # scalar out
    g = gates['1_equivariant_gate']
    # cross-l path exists with uniform widths (1o x 2e -> 1o)
    assert any(
        g.irreps_gated[i].ir != g.irreps_gated[j].ir for i, j in g.path_ij
    )
    si2 = dict(model.named_children())['1_self_interaction_2']
    assert si2.linear.irreps_out == g.get_gate_irreps_in()
    x = torch.randn(7, g.get_gate_irreps_in().dim)
    out = g({KEY.NODE_FEATURE: x.clone()})[KEY.NODE_FEATURE]
    ref = _bilinear_reference(g, x)
    assert torch.allclose(out, ref, atol=1e-6)  # zero-init: the plain gate
    with torch.no_grad():
        for m in gates.values():
            if isinstance(m, BilinearGate):
                m.linear.weight.normal_()
    out = g({KEY.NODE_FEATURE: x.clone()})[KEY.NODE_FEATURE]
    assert torch.allclose(out, _bilinear_reference(g, x), atol=1e-5)
    assert (out - g.gate(x[:, : g.dim_gate_in])[:, : g.dim_out]).abs().max() > 1e-2
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    build_muon_param_groups(model)
    # TorchScript (deploy path) gives the same energy and forces
    from tests.unit_tests.test_model import _graph_list

    model.eval()
    g0 = _graph_list[0].clone()
    o = model(g0.clone())  # before scripting (script converts in place)
    inp = {k: v for k, v in g0.to_dict().items() if isinstance(v, torch.Tensor)}
    inp[KEY.NODE_FEATURE] = model._atomic_numbers_to_onehot(inp[KEY.ATOMIC_NUMBERS])
    model.eval_type_map = False  # as sevenn deploy
    o_s = script(model)(inp)
    assert torch.allclose(o[KEY.PRED_TOTAL_ENERGY], o_s[KEY.PRED_TOTAL_ENERGY])
    assert torch.allclose(o[KEY.PRED_FORCE], o_s[KEY.PRED_FORCE], atol=1e-6)


@pytest.mark.parametrize('paths', ['same_l', 'lmax2', 'scalar'])
def test_bilinear_paths(paths):
    from sevenn.nn.bilinear_gate import BilinearGate

    cf = {**_partial, 'lmax': 3, 'gate_type': 'bilinear', 'irreps_manual': [
        '4x0e', '4x0e+4x1o+4x2e+4x3o', '4x0e+4x1o+4x2e+4x3o', '4x0e']}
    full = get_model(cf)
    model = get_model({**cf, 'bilinear_paths': paths})
    g = dict(model.named_children())['1_equivariant_gate']
    g_full = dict(full.named_children())['1_equivariant_gate']
    assert isinstance(g, BilinearGate)
    assert 0 < len(g.path_ij) < len(g_full.path_ij)
    keep = {
        'same_l': lambda i, j, ir: i == j,
        'lmax2': lambda i, j, ir: ir.l <= 2,
        'scalar': lambda i, j, ir: ir.l == 0,
    }[paths]
    assert all(
        keep(i, j, g.tp.irreps_out[k].ir) for k, (i, j) in enumerate(g.path_ij)
    )
    with torch.no_grad():
        g.linear.weight.normal_()
    x = torch.randn(5, g.get_gate_irreps_in().dim)
    out = g({KEY.NODE_FEATURE: x.clone()})[KEY.NODE_FEATURE]
    assert torch.allclose(out, _bilinear_reference(g, x), atol=1e-5)


def test_metaformer_ffn_per_layer():
    model = get_model({**_metaformer, 'use_ffn': [False, True, False]})
    names = list(dict(model.named_children()))
    assert '1_ffn_linear_1' in names
    assert '0_ffn_linear_1' not in names and '2_ffn_linear_1' not in names


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'ffn_type': 'swiglu'},
    {**_metaformer, 'layer_scale_init': 0.1},
    {**_metaformer, 'ffn_film': True},
    {**_metaformer, 'ffn_type': 'swiglu', 'layer_scale_init': 0.1, 'ffn_film': True,
     'radial_pair_embedding': True},
])
def test_ffn_variants(cf):
    from e3nn.util.jit import script

    from sevenn.nn.ffn_variants import ElementFiLM, LayerScale, SwiGLUGate
    from sevenn.train.muon import build_muon_param_groups

    model = get_model(cf)
    mods = dict(model.named_children())
    if cf.get('ffn_type') == 'swiglu':
        assert isinstance(mods['1_ffn_gate'], SwiGLUGate)
    if cf.get('layer_scale_init'):
        assert isinstance(mods['1_layer_scale_1'], LayerScale)
        assert isinstance(mods['1_layer_scale_2'], LayerScale)
    if cf.get('ffn_film'):
        assert isinstance(mods['1_ffn_film'], ElementFiLM)
        with torch.no_grad():  # make FiLM non-trivial before the checks
            for k, m in mods.items():
                if isinstance(m, ElementFiLM):
                    m.gamma.normal_(0, 0.3)
                    m.beta.normal_(0, 0.3)
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    adam = {name[id(p)] for p in groups[1]['params']}
    assert all(('film' in n or 'layer_scale' in n) <= (n in adam)
               for n in name.values())
    script(model.eval())


def test_film_identity_at_init():
    from sevenn.nn.ffn_variants import ElementFiLM

    film = ElementFiLM(Irreps('4x0e+2x1o'), num_species=3, data_key_x='x')
    x = torch.randn(5, 10)
    out = film({'x': x.clone(), KEY.ATOM_TYPE: torch.tensor([0, 1, 2, 1, 0])})
    assert torch.equal(out['x'], x)


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'distance_transform': 'agnesi'},
    {**_partial, 'distance_transform': 'agnesi'},
])
def test_agnesi_transform(cf):
    from e3nn.util.jit import script

    from sevenn.nn.edge_embedding import AgnesiTransform

    model = get_model(cf)
    emb = dict(model.named_children())['edge_embedding']
    assert isinstance(emb.distance_transform, AgnesiTransform)
    # transform is monotonically decreasing in r and lies in (0, 1)
    r = torch.linspace(0.5, 6.0, 50)
    t = emb.distance_transform(r, torch.zeros(2, 50, dtype=torch.long),
                               torch.zeros(1, dtype=torch.long))
    assert (t.diff() < 0).all() and (t > 0).all() and (t < 1).all()
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    script(model.eval())


def test_sigmoid_gate():
    cf = {**_metaformer, 'act_gate': {'e': 'sigmoid', 'o': 'tanh'}}
    model = get_model(cf)
    acts = [m.act_gates for k, m in model.named_modules()
            if k.endswith('equivariant_gate') and hasattr(m, 'act_gates')]
    out, out_rot, R = _rotated_pair(model)
    f, f_r = out[KEY.PRED_FORCE], out_rot[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    assert torch.isfinite(f).all()
    del acts


@pytest.mark.parametrize('base', [_metaformer, _partial])
def test_irreps_conv_narrow(base):
    conv = ['4x0e', '2x0e+2x1o+2x2e', '2x0e+2x1o+2x2e']
    model = get_model({**base, 'irreps_conv': conv})
    mods = dict(model.named_children())
    for t, ir in enumerate(conv):
        assert Irreps(mods[f'{t}_self_interaction_1'].irreps_out) == Irreps(ir)
        assert Irreps(mods[f'{t}_convolution'].convolution.irreps_in1) == Irreps(ir)
    # node features keep the full width (4x1o), the convolution runs on 2x1o
    assert Irreps(mods['1_self_interaction_2'].irreps_out).count('1o') == 4
    out, out_rot, R = _rotated_pair(model)
    f, f_r = out[KEY.PRED_FORCE], out_rot[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'conv_attention': {}},
    {**_metaformer, 'conv_attention': {'qk_dim': 8, 'out_gate': False},
     'radial_pair_embedding': True},
    {**_partial, 'conv_attention': {'null_logit': False}},
])
def test_conv_attention(cf):
    from e3nn.util.jit import script

    from sevenn.nn.edge_attention import AttentionOutGate, EdgeAttention

    model = get_model(cf)
    mods = dict(model.named_children())
    assert isinstance(mods['1_edge_attention'], EdgeAttention)
    assert '0_edge_attention' not in mods   # first layer keeps the plain sum
    names = list(mods)
    assert names.index('1_edge_attention') < names.index('1_convolution')
    if cf['conv_attention'].get('out_gate', True):
        assert isinstance(mods['1_attn_out_gate'], AttentionOutGate)
        assert names.index('1_attn_out_gate') == names.index('1_convolution') + 1
    assert float(mods['1_convolution'].denominator) == 1.0
    with torch.no_grad():  # make the attention non-trivial before the checks
        for m in mods.values():
            if isinstance(m, EdgeAttention):
                m.rb.weight.normal_(0, 0.3)
                m.og.weight.normal_(0, 0.3)
                m.t_src.normal_(0, 0.3)
                m.t_dst.normal_(0, 0.3)
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    assert torch.isfinite(f).all()
    script(model.eval())


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'radial_pair_embedding': True, 'radial_node_cond': True},
    {**_metaformer, 'radial_pair_embedding': True, 'radial_node_cond': True,
     'conv_attention': {}},
])
def test_radial_node_cond(cf):
    from e3nn.util.jit import script

    from sevenn.nn.radial_pair import NodeCondRadialInput

    model = get_model(cf)
    mods = dict(model.named_children())
    assert isinstance(mods['1_radial_node_cond'], NodeCondRadialInput)
    assert mods['1_convolution'].key_weight_input == '1_edge_pair_input_nc'
    with torch.no_grad():
        for m in mods.values():
            if isinstance(m, NodeCondRadialInput):
                m.p_src.weight.normal_(0, 0.3)
                m.p_dst.weight.normal_(0, 0.3)
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    script(model.eval())
