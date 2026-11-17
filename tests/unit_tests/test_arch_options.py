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


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'ffn_moe': True},
    {**_metaformer, 'ffn_moe': True, 'radial_pair_embedding': True,
     'readout_activation': 'silu', 'use_ffn': [True, False, True]},
])
def test_ffn_moe(cf):
    from e3nn.util.jit import script

    from sevenn.nn.moe_ffn import MoECombine, MoEDispatch, element_group
    from sevenn.train.muon import build_muon_param_groups

    assert [element_group(z) for z in (1, 8, 13, 26, 74, 64)] == [0, 1, 2, 3, 4, 5]
    model = get_model(cf)
    mods = dict(model.named_children())
    assert isinstance(mods['0_moe_dispatch'], MoEDispatch)
    assert isinstance(mods['0_moe_combine'], MoECombine)
    assert '0_ffn_linear_1_e0' in mods and '0_ffn_linear_2_e1' in mods
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    muon = {name[id(p)] for p in groups[0]['params']}
    assert any('_ffn_linear_1_e' in n for n in muon)
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    model.set_is_batch_data(True)
    out = model(get_graphs(batched=True))   # some experts empty in some graphs
    assert torch.isfinite(out[KEY.PRED_FORCE]).all()
    script(model.eval())


def test_s2_grid_quadrature():
    """Lebedev-38 rule == scipy's; Proj Y = I; exact up to degree 9."""
    import numpy as np

    from sevenn.nn.ffn_variants import lebedev_38, s2_grid_matrices

    grid = lebedev_38()
    assert grid.shape == (38, 4) and abs(grid[:, 3].sum() - 1) < 1e-15
    assert np.allclose((grid[:, :3] ** 2).sum(1), 1.0, atol=1e-15)
    try:
        from scipy.integrate import lebedev_rule
    except ImportError:
        lebedev_rule = None
    if lebedev_rule is not None:
        x, w = lebedev_rule(9)
        ref = np.concatenate([x.T, (w / w.sum())[:, None]], 1)
        dist = np.abs(ref[:, None, :] - grid[None, :, :]).max(-1)
        assert dist.min(0).max() < 1e-14 and dist.min(1).max() < 1e-14
    to_grid, from_grid = s2_grid_matrices(3)
    assert torch.allclose(to_grid @ from_grid, torch.eye(16, dtype=torch.float64),
                          atol=1e-14)
    # triple products Y_i Y_j Y_k (degree <= 9) integrate exactly: compare
    # with a Gauss-Legendre x uniform-phi product rule (exact to degree 19)
    from e3nn.o3 import spherical_harmonics

    ct, wt = np.polynomial.legendre.leggauss(10)
    phi = np.arange(20) * 2 * np.pi / 20
    st = np.sqrt(1 - ct**2)
    pts = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)),
                    np.outer(ct, np.ones_like(phi))], -1).reshape(-1, 3)
    wf = torch.from_numpy(np.outer(wt / 2, np.full(20, 1 / 20)).reshape(-1))
    yf = spherical_harmonics([0, 1, 2, 3], torch.from_numpy(pts), True,
                             normalization='component')
    g_ref = torch.einsum('p,pi,pj,pk->ijk', wf, yf, yf, yf)
    g_leb = torch.einsum('ip,jp,pk->ijk', to_grid, to_grid, from_grid)
    assert torch.allclose(g_leb, g_ref, atol=1e-13)


def test_s2_bilinear_gate_module():
    """exact O(3) equivariance (incl. inversion), Gaunt reference, moments"""
    import torch.nn.functional as F
    from e3nn.o3 import rand_matrix

    from sevenn.nn.ffn_variants import S2BilinearGate, s2_grid_matrices

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        irreps = Irreps('8x0e+12x1o+6x2e+6x3o')
        gate = S2BilinearGate(irreps, {'e': F.silu}, data_key_x='x')
        n0, C = 8, 6   # C = widest l > 0 block // 2
        assert gate.C == C and gate.n0 == n0
        assert gate.get_gate_irreps_in() == Irreps('8x0e+12x0e+12x1o+12x2e+12x3o')
        assert gate.irreps_out == Irreps('8x0e+6x0e+6x1o+6x2e+6x3o')
        assert S2BilinearGate(irreps, {'e': F.silu}, 4).C == 4
        torch.manual_seed(0)
        x = gate.irreps_in.randn(4000, -1)
        out = gate({'x': x.clone()})['x']
        for sl in gate.irreps_out.slices():   # ~unit second moment at init
            assert 0.75 < out[:, sl].pow(2).mean().item() < 1.3
        x = x[:20]
        out = out[:20]
        for R in (rand_matrix(), -rand_matrix(), -torch.eye(3)):
            d_in = gate.irreps_in.D_from_matrix(R)
            d_out = gate.irreps_out.D_from_matrix(R)
            out_r = gate({'x': x @ d_in.T})['x']
            assert torch.allclose(out_r, out @ d_out.T, atol=1e-12)
        # reference: Gaunt coefficients from the grid matrices, plain loops
        to_grid, from_grid = s2_grid_matrices(3)
        gaunt = torch.einsum('ip,jp,pk->ijk', to_grid, to_grid, from_grid)
        a = torch.zeros(20, C, 16)
        b = torch.zeros(20, C, 16)
        off = n0   # A|B region starts after the scalars s
        for l in range(4):
            d = 2 * l + 1
            blk = x[:, off: off + 2 * C * d].reshape(20, 2, C, d)
            a[..., l * l: (l + 1) ** 2] = blk[:, 0]
            b[..., l * l: (l + 1) ** 2] = blk[:, 1]
            off += 2 * C * d
        z = torch.einsum('nci,ncj,ijk->nck', a, b, gaunt)
        var = gaunt.pow(2).sum((0, 1))
        ref = [gate.act(x[:, :n0])]
        for l in range(4):
            sl = slice(l * l, (l + 1) ** 2)
            ref.append((z[..., sl] * var[sl].mean().rsqrt()).reshape(20, -1))
        assert torch.allclose(out, torch.cat(ref, -1), atol=1e-12)
        with pytest.raises(ValueError):   # pseudo-tensors are not allowed
            S2BilinearGate(Irreps('4x0e+2x1e'), {'e': F.silu})
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'ffn_type': 's2_bilinear'},
    {**_metaformer, 'ffn_type': 's2_bilinear', 'ffn_s2_channels': 2,
     'lmax': 3, 'irreps_manual': ['4x0e', '4x0e+4x1o+4x2e+4x3o',
                                  '4x0e+4x1o+4x2e+4x3o', '4x0e']},
    {**_metaformer, 'ffn_type': 's2_bilinear', 'ffn_film': True,
     'layer_scale_init': 0.1, 'radial_pair_embedding': True,
     'readout_activation': 'silu'},
])
def test_s2_bilinear_ffn(cf):
    from e3nn.util.jit import script

    from sevenn.nn.equivariant_gate import EquivariantGate
    from sevenn.nn.ffn_variants import ElementFiLM, S2BilinearGate
    from sevenn.train.muon import build_muon_param_groups

    model = get_model(cf)
    mods = dict(model.named_children())
    gate = mods['1_ffn_gate']
    assert isinstance(gate, S2BilinearGate)
    assert gate.C == cf.get('ffn_s2_channels', 4)   # hidden 8x1o // 2
    assert Irreps(mods['1_ffn_linear_1'].irreps_out) == gate.get_gate_irreps_in()
    assert Irreps(mods['1_ffn_linear_2'].irreps_in) == gate.irreps_out
    assert isinstance(mods['2_ffn_gate'], EquivariantGate)   # scalar-only FFN
    if cf.get('ffn_film'):
        with torch.no_grad():
            for m in mods.values():
                if isinstance(m, ElementFiLM):
                    m.gamma.normal_(0, 0.3)
                    m.beta.normal_(0, 0.3)
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    assert torch.isfinite(f).all() and f.abs().sum() > 0
    # inversion: E invariant, F odd
    from tests.unit_tests.test_model import _graph_list

    g = _graph_list[0].clone()
    g_inv = g.clone()
    g_inv[KEY.POS] = -g[KEY.POS]
    if KEY.CELL in g:
        g_inv[KEY.CELL] = -g[KEY.CELL]
    if KEY.EDGE_VEC in g:
        g_inv[KEY.EDGE_VEC] = -g[KEY.EDGE_VEC]
    o_i = model(g_inv)
    assert torch.allclose(o_i[KEY.PRED_TOTAL_ENERGY], e, atol=1e-4)
    assert torch.allclose(o_i[KEY.PRED_FORCE], -f, atol=1e-4)
    # grid buffers are not parameters; FFN linears stay on Muon
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    muon = {name[id(p)] for p in groups[0]['params']}
    assert any('1_ffn_linear_1' in n for n in muon)
    assert not any('to_grid' in n or 'from_grid' in n for n in name.values())
    model.set_is_batch_data(True)
    out = model(get_graphs(batched=True))
    assert torch.isfinite(out[KEY.PRED_FORCE]).all()
    # node-local and scriptable; grid matrices are buffers (deploy: script +
    # freeze, as for LAMMPS / ML-IAP)
    assert {'to_grid', 'from_grid', 'perm_in', 'perm_out'} <= {
        n for n, _ in gate.named_buffers()
    }
    model.set_is_batch_data(False)
    g = _graph_list[0].clone()
    ref = model.eval()(g.clone())   # before scripting (e3nn scripts in place)
    inp = {k: v for k, v in g.to_dict().items() if isinstance(v, torch.Tensor)}
    inp[KEY.NODE_FEATURE] = model._atomic_numbers_to_onehot(inp[KEY.ATOMIC_NUMBERS])
    model.eval_type_map = False  # as sevenn deploy
    out = torch.jit.freeze(script(model))(inp)
    assert torch.allclose(out[KEY.PRED_TOTAL_ENERGY], ref[KEY.PRED_TOTAL_ENERGY],
                          atol=1e-5)
    assert torch.allclose(out[KEY.PRED_FORCE], ref[KEY.PRED_FORCE], atol=1e-5)


@pytest.mark.parametrize('irreps, C', [
    ('8x0e+12x1o+6x2e+6x3o', None), ('16x0e+8x1o+4x2e+4x3o', 3),
    ('8x0e+12x1o+6x2e', None),
])
def test_s2_inv_gate_module(irreps, C):
    """
    s2_inv: act(s) and the l = 0 output equal S2BilinearGate's for the same
    s, A, B (incl. its degree-0 output scale); l > 0 outputs equal the e3nn
    Gate of A_l>0 with gates g; exact O(3) equivariance; unit second
    moments; same output irreps
    """
    import torch.nn.functional as F
    from e3nn.nn import Gate
    from e3nn.o3 import rand_matrix

    from sevenn.nn.ffn_variants import S2BilinearGate, S2InvGate

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        irreps = Irreps(irreps)
        act = {'e': F.silu, 'o': torch.tanh}
        s2 = S2BilinearGate(irreps, act, C, data_key_x='x')
        gate = S2InvGate(irreps, act, act, C, data_key_x='x')
        n0, C, L = gate.n0, gate.C, gate.lmax
        degrees = Irreps([(C, (l, (-1) ** l)) for l in range(L + 1)])
        assert C == s2.C and gate.n_g == L * C
        assert gate.irreps_out == s2.irreps_out
        assert gate.get_gate_irreps_in() == Irreps(
            [(n0, (0, 1)), (L * C, (0, 1))]) + degrees + degrees
        torch.manual_seed(0)
        n = 4000
        sc, g = torch.randn(n, n0), torch.randn(n, L * C)
        a, b = degrees.randn(n, -1), degrees.randn(n, -1)
        x = torch.cat([sc, g, a, b], 1)
        x2 = [sc]   # S2BilinearGate layout: [s | (A_l, B_l) per degree]
        for sl in degrees.slices():
            x2 += [a[:, sl], b[:, sl]]
        out = gate({'x': x.clone()})['x']
        ref = s2({'x': torch.cat(x2, 1)})['x']
        assert torch.equal(out[:, :n0], ref[:, :n0])
        assert torch.allclose(out[:, n0: n0 + C], ref[:, n0: n0 + C], atol=1e-12)
        e3gate = Gate('', [], f'{L * C}x0e', [F.silu], degrees[1:])
        ref_v = e3gate(torch.cat([g, a[:, C:]], 1))
        assert torch.allclose(out[:, n0 + C:], ref_v, atol=1e-12)
        for sl in gate.irreps_out.slices():   # ~unit second moment at init
            assert 0.85 < out[:, sl].pow(2).mean().item() < 1.15
        x, out = x[:20], out[:20]
        for R in (rand_matrix(), -rand_matrix(), -torch.eye(3)):
            out_r = gate({'x': x @ gate.irreps_in.D_from_matrix(R).T})['x']
            assert torch.allclose(
                out_r, out @ gate.irreps_out.D_from_matrix(R).T, atol=1e-11
            )
        with pytest.raises(ValueError):   # pseudo-tensors are not allowed
            S2InvGate(Irreps('4x0e+2x1e'), act, act)
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'ffn_type': 's2_inv'},
    {**_metaformer, 'ffn_type': 's2_inv', 'ffn_s2_channels': 2,
     'lmax': 3, 'irreps_manual': ['4x0e', '4x0e+4x1o+4x2e+4x3o',
                                  '4x0e+4x1o+4x2e+4x3o', '4x0e']},
    {**_metaformer, 'ffn_type': 's2_inv', 'ffn_film': True,
     'layer_scale_init': 0.1, 'radial_pair_embedding': True,
     'readout_activation': 'silu'},
    {**_metaformer, 'ffn_type': 's2_inv', 'parallel_block': [False, True, True]},
])
def test_s2_inv_ffn(cf):
    from e3nn.util.jit import script

    from sevenn.nn.equivariant_gate import EquivariantGate
    from sevenn.nn.ffn_variants import ElementFiLM, S2InvGate
    from sevenn.train.muon import build_muon_param_groups

    model = get_model(cf)
    mods = dict(model.named_children())
    gate = mods['1_ffn_gate']
    assert isinstance(gate, S2InvGate)
    assert gate.C == cf.get('ffn_s2_channels', 4)   # hidden 8x1o // 2
    assert Irreps(mods['1_ffn_linear_1'].irreps_out) == gate.get_gate_irreps_in()
    assert Irreps(mods['1_ffn_linear_2'].irreps_in) == gate.irreps_out
    assert isinstance(mods['2_ffn_gate'], EquivariantGate)   # scalar-only FFN
    # same FFN output side as s2_bilinear
    ref = get_model({**cf, 'ffn_type': 's2_bilinear'})
    ref_mods = dict(ref.named_children())
    assert ref_mods['1_ffn_gate'].irreps_out == gate.irreps_out
    assert (ref_mods['1_ffn_linear_2'].linear.weight.shape
            == mods['1_ffn_linear_2'].linear.weight.shape)
    if cf.get('ffn_film'):
        with torch.no_grad():
            for m in mods.values():
                if isinstance(m, ElementFiLM):
                    m.gamma.normal_(0, 0.3)
                    m.beta.normal_(0, 0.3)
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    assert torch.isfinite(f).all() and f.abs().sum() > 0
    from tests.unit_tests.test_model import _graph_list

    g = _graph_list[0].clone()   # inversion: E invariant, F odd
    g_inv = g.clone()
    g_inv[KEY.POS] = -g[KEY.POS]
    if KEY.CELL in g:
        g_inv[KEY.CELL] = -g[KEY.CELL]
    if KEY.EDGE_VEC in g:
        g_inv[KEY.EDGE_VEC] = -g[KEY.EDGE_VEC]
    o_i = model(g_inv)
    assert torch.allclose(o_i[KEY.PRED_TOTAL_ENERGY], e, atol=1e-4)
    assert torch.allclose(o_i[KEY.PRED_FORCE], -f, atol=1e-4)
    groups = build_muon_param_groups(model)   # FFN linears stay on Muon
    name = {id(p): n for n, p in model.named_parameters()}
    muon = {name[id(p)] for p in groups[0]['params']}
    assert any('1_ffn_linear_1' in n for n in muon)
    assert not any('1_ffn_gate' in n for n in name.values())   # no parameters
    model.set_is_batch_data(True)
    out = model(get_graphs(batched=True))
    assert torch.isfinite(out[KEY.PRED_FORCE]).all()
    model.set_is_batch_data(False)   # deploy: script + freeze (LAMMPS / ML-IAP)
    g = _graph_list[0].clone()
    ref = model.eval()(g.clone())   # before scripting (e3nn scripts in place)
    inp = {k: v for k, v in g.to_dict().items() if isinstance(v, torch.Tensor)}
    inp[KEY.NODE_FEATURE] = model._atomic_numbers_to_onehot(inp[KEY.ATOMIC_NUMBERS])
    model.eval_type_map = False  # as sevenn deploy
    out = torch.jit.freeze(script(model))(inp)
    assert torch.allclose(out[KEY.PRED_TOTAL_ENERGY], ref[KEY.PRED_TOTAL_ENERGY],
                          atol=1e-5)
    assert torch.allclose(out[KEY.PRED_FORCE], ref[KEY.PRED_FORCE], atol=1e-5)


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'metaformer_norm': False},
    {**_metaformer, 'metaformer_norm': False, 'radial_pair_embedding': True,
     'readout_activation': 'silu', 'use_ffn': [True, False, True]},
])
def test_metaformer_without_norm(cf):
    """metaformer_norm: false drops norm_1 / norm_2 and nothing else"""
    from e3nn.util.jit import script

    from sevenn.nn.normalization import EquivariantRMSNorm

    torch.manual_seed(0)
    ref = get_model({**cf, 'metaformer_norm': True})
    torch.manual_seed(0)
    model = get_model(cf)
    names = list(dict(model.named_children()))
    assert not any(isinstance(m, EquivariantRMSNorm) for m in model.modules())
    ref_names = [k for k in dict(ref.named_children())
                 if not k.endswith(('_norm_1', '_norm_2'))]
    assert names == ref_names
    sd, sd_ref = model.state_dict(), ref.state_dict()
    assert set(sd) == {k for k in sd_ref if '_norm_' not in k}
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    assert torch.isfinite(f).all() and f.abs().sum() > 0
    script(model.eval())


_modal_cf = {
    'use_modality': True,
    '_number_of_modalities': 2,
    '_modal_map': {'x1': 0, 'x2': 1},
    'use_modal_node_embedding': False,
    'use_modal_self_inter_intro': True,
    'use_modal_self_inter_outro': True,
    'use_modal_output_block': True,
    'use_modal_wise_shift': False,
    'use_modal_wise_scale': False,
}


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'readout_block': {}},
    {**_metaformer, 'readout_block': {'type': 'swiglu', 'hidden_scale': 2},
     'radial_pair_embedding': True, **_modal_cf},
])
def test_readout_block(cf):
    """norm -> FFN -> identity residual -> modal-aware linear readout"""
    from e3nn.util.jit import script

    from sevenn.nn.equivariant_gate import EquivariantGate
    from sevenn.nn.ffn_variants import SwiGLUGate
    from sevenn.nn.normalization import EquivariantRMSNorm
    from sevenn.train.muon import build_muon_param_groups

    model = get_model(cf)
    mods = dict(model.named_children())
    names = list(mods)
    order = ['readout_ffn_intro', 'readout_norm', 'readout_ffn_linear_1',
             'readout_ffn_gate', 'readout_ffn_linear_2', 'readout_ffn_outro',
             'reduce_input_to_hidden', 'reduce_hidden_to_energy',
             'rescale_atomic_energy']
    i = names.index('readout_ffn_intro')
    assert names[i: i + len(order)] == order
    assert 'reduce_hidden_activation' not in names
    assert isinstance(mods['readout_norm'], EquivariantRMSNorm)
    opts = cf['readout_block']
    scale = opts.get('hidden_scale', 4)
    gate = mods['readout_ffn_gate']
    assert isinstance(
        gate, SwiGLUGate if opts.get('type') == 'swiglu' else EquivariantGate
    )
    assert Irreps(mods['readout_ffn_linear_2'].irreps_in) == Irreps(
        f'{4 * scale}x0e')
    if cf.get('use_modality'):
        assert mods['reduce_input_to_hidden'].num_modalities == 2
        assert mods['readout_ffn_linear_1'].num_modalities == 0
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    muon = {name[id(p)] for p in groups[0]['params']}
    assert {'readout_ffn_linear_1.linear.weight',
            'readout_ffn_linear_2.linear.weight'} <= muon
    assert 'readout_norm.weight' not in muon
    if cf.get('use_modality'):
        return  # modal batches: covered by the parallel / LunaTP tests
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    assert torch.isfinite(f).all() and f.abs().sum() > 0
    script(model.eval())


def test_readout_block_replaces_activation():
    with pytest.raises(ValueError):
        get_model({**_metaformer, 'readout_block': {},
                   'readout_activation': 'silu'})


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'per_layer_readout': True, 'radial_pair_embedding': True},
    {**_partial, 'per_layer_readout': True},
    {**_metaformer, 'per_layer_readout': True, 'use_ffn': [True, False, True],
     'readout_block': {}},
])
def test_per_layer_readout(cf):
    """zero-initialized linear energy heads after every layer but the last"""
    from e3nn.util.jit import script

    from sevenn.model_build import build_E3_equivariant_model
    from sevenn.nn.layer_readout import (
        LayerEnergyPack,
        LayerEnergyToReadout,
        LayerEnergyUnpack,
        ZeroInitIrrepsLinear,
    )
    from sevenn.train.muon import build_muon_param_groups
    from tests.unit_tests.test_model import get_model_config

    torch.manual_seed(0)
    ref = get_model({**cf, 'per_layer_readout': False})
    model = get_model(cf)
    mods = dict(model.named_children())
    names = list(mods)
    for t in (0, 1):
        head = mods[f'{t}_layer_readout']
        assert isinstance(head, ZeroInitIrrepsLinear)
        assert head.linear.weight.abs().sum() == 0
        assert Irreps(head.irreps_out) == Irreps('1x0e')
        assert names.index(f'{t}_layer_energy_acc') == names.index(
            f'{t}_layer_readout') + 1
        last = 'ffn_outro' if f'{t}_ffn_outro' in mods else (
            'equivariant_gate' if f'{t}_equivariant_gate' in mods
            and cf.get('interaction_type') != 'metaformer'
            else 'self_connection_outro')
        assert names.index(f'{t}_layer_readout') == names.index(f'{t}_{last}') + 1
    assert '2_layer_readout' not in names
    assert isinstance(mods['layer_energy_to_readout'], LayerEnergyToReadout)
    assert names.index('layer_energy_to_readout') == names.index(
        'rescale_atomic_energy') - 1
    # zero heads: identical to the model without them
    missing, unexpected = model.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and all('layer_readout' in k for k in missing)
    ref.set_is_batch_data(True)
    model.set_is_batch_data(True)
    o_ref = ref(get_graphs(batched=True))
    o = model(get_graphs(batched=True))
    for k in (KEY.PRED_TOTAL_ENERGY, KEY.PRED_FORCE, KEY.PRED_STRESS):
        assert torch.allclose(o_ref[k], o[k], atol=1e-7, rtol=0), k
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    adam = {name[id(p)] for p in groups[1]['params']}
    assert '0_layer_readout.linear.weight' in adam
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'layer_readout' in n:
                p.normal_(0, 0.3)
    o2 = model(get_graphs(batched=True))
    assert (o2[KEY.PRED_TOTAL_ENERGY] - o[KEY.PRED_TOTAL_ENERGY]).abs().max() > 1e-3
    out, out_rot, R = _rotated_pair(model)
    e, e_r = out[KEY.PRED_TOTAL_ENERGY], out_rot[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = out[KEY.PRED_FORCE], out_rot[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    script(model.eval())
    # e3gnn/parallel segments: the energy sum crosses inside self_cont_tmp
    config = get_model_config()
    config.update(cf)
    segs = build_E3_equivariant_model(config, parallel=True)
    for i, seg in enumerate(segs):
        sm = list(seg._modules.values())
        assert isinstance(sm[-1], LayerEnergyPack) == (i < len(segs) - 1)
        assert any(isinstance(m, LayerEnergyUnpack) for m in sm) == (i > 0)


@pytest.mark.parametrize('block', [
    {},
    {'type': 'mlp', 'norm': 'layer', 'num_blocks': 2, 'hidden_scale': 1},
    {'final_norm': False},
])
def test_radial_mlp_block(block):
    """pre-norm residual hidden stack of the pair radial MLP"""
    from e3nn.util.jit import script

    from sevenn.nn.radial_pair import PairRadialMLP
    from sevenn.train.muon import build_muon_param_groups

    cf = {**_metaformer, 'radial_pair_embedding': True,
          'radial_mlp_block': block}
    torch.manual_seed(0)
    model = get_model(cf)
    convs = _convs(model)
    w = convs['1_convolution'].weight_nn
    assert isinstance(w, PairRadialMLP) and w.use_block
    assert len(w.hidden) == 0
    assert len(w.blocks) == block.get('num_blocks', 1)  # hs = [8, 64, 64, .]
    hidden = 64 * block.get('hidden_scale', 2)
    swiglu = block.get('type', 'swiglu') == 'swiglu'
    up = w.blocks[0].up.weight
    assert tuple(up.shape) == (64, 2 * hidden if swiglu else hidden)
    assert (w.blocks[0].norm.bias is not None) == (block.get('norm') == 'layer')
    with torch.no_grad():  # fp32 rotation noise grows with the table scale
        for c in convs.values():
            c.weight_nn.table_src.normal_(0, 0.3)
            c.weight_nn.table_dst.normal_(0, 0.3)
    # envelope zero -> weights exactly zero; smooth (finite 2nd derivative)
    # also where the hidden vector vanishes (rbf = 0, zero tables)
    nb = w.num_basis
    inp = torch.randn(6, nb + 3)
    inp[:, nb] = 0.0
    inp[:, nb + 1:] = torch.randint(0, w.table_src.shape[0], (6, 2)).float()
    assert torch.equal(w(inp), torch.zeros_like(w(inp)))
    w0 = PairRadialMLP([nb, 64, 64, 10], torch.nn.functional.silu, 3,
                       block=block)
    x = torch.zeros(2, nb + 3, dtype=torch.float64)
    x[:, nb] = 0.5
    w0 = w0.double()
    xr = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(w0(xr).square().sum(), xr, create_graph=True)
    (h,) = torch.autograd.grad(g[:, :nb].sum(), xr)
    assert torch.isfinite(g).all() and torch.isfinite(h).all()
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    assert torch.isfinite(f).all() and f.abs().sum() > 0
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    muon = {name[id(p)] for p in groups[0]['params']}
    adam = {name[id(p)] for p in groups[1]['params']}
    assert {'1_convolution.weight_nn.blocks.0.up.weight',
            '1_convolution.weight_nn.blocks.0.down.weight'} <= muon
    assert '1_convolution.weight_nn.blocks.0.norm.weight' in adam
    script(model.eval())


def test_radial_mlp_block_needs_pair_radial():
    with pytest.raises(ValueError):
        get_model({**_metaformer, 'radial_mlp_block': {}})


def _inversion_check(model):
    """E invariant and F odd under inversion (parity of the 0o/1e/2o paths)"""
    from tests.unit_tests.test_model import _graph_list

    model.set_is_batch_data(False)
    g = _graph_list[0].clone()
    g_inv = g.clone()
    g_inv[KEY.POS] = -g[KEY.POS]
    if KEY.CELL in g:
        g_inv[KEY.CELL] = -g[KEY.CELL]
    if KEY.EDGE_VEC in g:
        g_inv[KEY.EDGE_VEC] = -g[KEY.EDGE_VEC]
    o, o_i = model(g.clone()), model(g_inv)
    assert torch.allclose(o_i[KEY.PRED_TOTAL_ENERGY], o[KEY.PRED_TOTAL_ENERGY],
                          atol=1e-4)
    assert torch.allclose(o_i[KEY.PRED_FORCE], -o[KEY.PRED_FORCE], atol=1e-4)


_full_parity = ['4x0e', '4x0e+2x0o+4x1o+2x1e+4x2e+2x2o',
                '4x0e+2x0o+4x1o+2x1e+4x2e+2x2o', '4x0e']


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'conv_gate': False},
    {**_metaformer, 'conv_gate': False, 'irreps_manual': _full_parity},
    {**_metaformer, 'parallel_block': True},
    {**_metaformer, 'conv_gate': False, 'parallel_block': True,
     'radial_pair_embedding': True, 'irreps_manual': _full_parity},
    {**_metaformer, 'parallel_block': [False, True, True], 'ffn_type': 'swiglu',
     'layer_scale_init': 0.1, 'ffn_film': True, 'per_layer_readout': True},
    {**_metaformer, 'parallel_block': True, 'ffn_type': 's2_bilinear',
     'lmax': 3, 'irreps_manual': ['4x0e', '4x0e+4x1o+4x2e+4x3o',
                                  '4x0e+4x1o+4x2e+4x3o', '4x0e']},
])
def test_light_conv(cf):
    """conv_gate: false (TP -> linear) and parallel_block (conv + FFN branches
    from the same pre-norm input)"""
    from e3nn.util.jit import script

    from sevenn.model_build import build_E3_equivariant_model
    from sevenn.nn.ffn_variants import AddToResidual, ElementFiLM
    from sevenn.train.muon import build_muon_param_groups
    from tests.unit_tests.test_model import get_model_config

    model = get_model(cf)
    mods = dict(model.named_children())
    names = list(mods)
    n_layer = 3
    pb = cf.get('parallel_block', False)
    pb = pb if isinstance(pb, list) else [pb] * n_layer
    irreps = cf['irreps_manual']
    for t in range(n_layer):
        si2 = mods[f'{t}_self_interaction_2']
        if not cf.get('conv_gate', True):
            assert f'{t}_equivariant_gate' not in mods
            assert f'{t}_self_interaction_3' not in mods
            assert Irreps(si2.irreps_out) == Irreps(irreps[t + 1])
        else:
            assert f'{t}_self_interaction_3' in mods
        if pb[t]:
            assert f'{t}_norm_2' not in mods and f'{t}_ffn_intro' not in mods
            add = mods[f'{t}_ffn_to_residual']
            assert isinstance(add, AddToResidual)
            i_si1 = names.index(f'{t}_self_interaction_1')
            assert names.index(f'{t}_norm_1') < names.index(f'{t}_ffn_linear_1')
            assert names.index(f'{t}_ffn_to_residual') == i_si1 - 1
            lin1 = mods[f'{t}_ffn_linear_1']
            assert lin1.key_input == KEY.NODE_FEATURE
            assert lin1.key_output == add.key_in != KEY.NODE_FEATURE
            assert Irreps(lin1.irreps_in) == Irreps(irreps[t])
            assert Irreps(mods[f'{t}_ffn_linear_2'].irreps_out) == Irreps(
                irreps[t + 1])
        else:
            assert f'{t}_ffn_outro' in mods and f'{t}_norm_2' in mods
    with torch.no_grad():
        for m in mods.values():
            if isinstance(m, ElementFiLM):
                m.gamma.normal_(0, 0.3)
                m.beta.normal_(0, 0.3)
        for n, p in model.named_parameters():
            if 'layer_readout' in n:
                p.normal_(0, 0.3)
    o, o_r, R = _rotated_pair(model)
    e, e_r = o[KEY.PRED_TOTAL_ENERGY], o_r[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = o[KEY.PRED_FORCE], o_r[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    assert torch.isfinite(f).all() and f.abs().sum() > 0
    _inversion_check(model)
    build_muon_param_groups(model)
    script(model.eval())
    # e3gnn/parallel: layer-0 FFN branch needs no ghost twin
    config = get_model_config()
    config.update(cf)
    segs = build_E3_equivariant_model(config, parallel=True)
    names0 = list(segs[0]._modules)
    assert not any(k.startswith('ghost_0_ffn') for k in names0)


def test_light_conv_errors():
    with pytest.raises(ValueError):
        get_model({**_partial, 'conv_gate': False})
    with pytest.raises(ValueError):
        get_model({**_partial, 'parallel_block': True})
    with pytest.raises(ValueError):
        get_model({**_metaformer, 'conv_gate': False, 'gate_type': 'bilinear'})
    with pytest.raises(NotImplementedError):
        get_model({**_metaformer, 'parallel_block': True, 'ffn_moe': True})


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'radial_pair_embedding': True, 'radial_short_basis': {}},
    {**_metaformer, 'radial_pair_embedding': True, 'radial_node_cond': True,
     'radial_mlp_block': {}, 'radial_short_basis': {'cutoff': 2.5,
                                                    'num_basis': 4}},
])
def test_short_radial_basis(cf):
    """second, short-cutoff basis in the pair radial MLP input"""
    from e3nn.util.jit import script

    from sevenn.nn.radial_pair import EdgePairInput, ShortRangeBasis
    from sevenn.train.muon import build_muon_param_groups

    opts = cf['radial_short_basis']
    rs, n = opts.get('cutoff', 3.0), opts.get('num_basis', 6)
    torch.manual_seed(0)
    ref = get_model({**cf, 'radial_short_basis': None})
    model = get_model(cf)
    pair = dict(model.named_children())['edge_pair_input']
    assert isinstance(pair, EdgePairInput) and pair.use_short
    w = _convs(model)['1_convolution'].weight_nn
    assert w.num_short == n and w.w_short.abs().sum() == 0
    # zero-initialized: identical to the model without the short basis
    missing, unexpected = model.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and all(k.endswith('w_short') for k in missing)
    ref.set_is_batch_data(True)
    model.set_is_batch_data(True)
    o_ref = ref(get_graphs(batched=True))
    o = model(get_graphs(batched=True))
    for k in (KEY.PRED_TOTAL_ENERGY, KEY.PRED_FORCE, KEY.PRED_STRESS):
        assert torch.allclose(o_ref[k], o[k], atol=1e-7, rtol=0), k
    # basis: zero beyond r_s, value / 1st / 2nd derivative -> 0 at r_s
    basis = ShortRangeBasis(rs, n).double()
    r = torch.tensor([0.5, rs - 1e-3, rs, rs + 1e-3, 3.9],
                     dtype=torch.float64, requires_grad=True)
    f = basis(r)
    assert f.shape == (5, n) and f[0].abs().max() > 0.1
    assert torch.equal(f[2:], torch.zeros_like(f[2:]))
    (d1,) = torch.autograd.grad(f.sum(), r, create_graph=True)
    (d2,) = torch.autograd.grad(d1.sum(), r)
    assert f[1].abs().max() < 1e-6 and d1[1].abs() < 1e-3 and d2[1].abs() < 1.0
    assert torch.equal(d1[2:], torch.zeros_like(d1[2:]))
    with torch.no_grad():
        for c in _convs(model).values():
            c.weight_nn.w_short.normal_(0, 0.3)
    o2 = model(get_graphs(batched=True))
    assert (o2[KEY.PRED_FORCE] - o[KEY.PRED_FORCE]).abs().max() > 1e-4
    out, out_rot, R = _rotated_pair(model)
    e, e_r = out[KEY.PRED_TOTAL_ENERGY], out_rot[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = out[KEY.PRED_FORCE], out_rot[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    muon = {name[id(p)] for p in groups[0]['params']}
    assert '1_convolution.weight_nn.w_short' in muon
    script(model.eval())


def test_short_radial_basis_errors():
    with pytest.raises(ValueError):
        get_model({**_metaformer, 'radial_short_basis': {}})
    with pytest.raises(ValueError):
        get_model({**_metaformer, 'radial_pair_embedding': True,
                   'radial_short_basis': {'cutoff': 4.5}})


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'radial_pair_embedding': True,
     'pair_repulsion': {'cutoff_scale': 1.4}},
    {**_partial, 'pair_repulsion': {'num_terms': 3, 'max_cutoff': 3.0,
                                    'decay_init': [1.0, 3.0, 9.0],
                                    'param_scale': 1.0}},
])
def test_pair_repulsion(cf):
    """learned pair repulsion added to the atomic energy after the rescale"""
    import math

    from e3nn.util.jit import script

    from sevenn.logger import Logger
    from sevenn.model_build import build_E3_equivariant_model
    from sevenn.nn.pair_repulsion import PairRepulsion, PairRepulsionToEnergy
    from sevenn.train.muon import build_muon_param_groups
    from tests.unit_tests.test_model import get_model_config

    opts = cf['pair_repulsion']
    torch.manual_seed(0)
    ref = get_model({**cf, 'pair_repulsion': None})
    model = get_model(cf)
    mods = dict(model.named_children())
    names = list(mods)
    pair = mods['pair_repulsion']
    assert isinstance(pair, PairRepulsion)
    # computed after the layer-1 convolution, added after the rescale
    assert names.index('pair_repulsion') == names.index('1_convolution') + 1
    add = 'pair_repulsion_to_energy'
    assert isinstance(mods[add], PairRepulsionToEnergy)
    assert names.index(add) == names.index('rescale_atomic_energy') + 1
    assert names.index('reduce_total_enegy') == names.index(add) + 1
    k = opts.get('num_terms', 2)
    assert pair.log_amp.shape == pair.decay_raw.shape == (pair.radius.numel(), k)
    # init: A_ij = amp_init (1e-4 eV A), pair decay constants = decay_init;
    # parameters stored / param_scale (default 8, exact: power of two)
    c = opts.get('param_scale', 8.0)
    assert pair.param_scale == c
    kap = torch.nn.functional.softplus(c * pair.decay_raw[0]) * 2
    want = opts.get('decay_init', [2.0, 5.0])
    assert torch.allclose(kap, torch.tensor(want), rtol=1e-5)
    lam0 = 0.5 * math.log(1e-4)
    assert torch.equal(c * pair.log_amp, torch.full_like(pair.log_amp, lam0))
    log_a = pair.pair_table()[:, 1:1 + k]  # pair log-amplitudes = log(1e-4)
    assert torch.allclose(log_a, torch.full_like(log_a, math.log(1e-4)))
    # tiny init: the model without the term plus < 1e-4 eV
    missing, unexpected = model.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and all(m.startswith('pair_repulsion.') for m in missing)
    ref.set_is_batch_data(True)
    model.set_is_batch_data(True)
    o_ref = ref(get_graphs(batched=True))
    o = model(get_graphs(batched=True))
    de = (o[KEY.PRED_TOTAL_ENERGY] - o_ref[KEY.PRED_TOTAL_ENERGY]).abs().max()
    assert de < 1e-4
    assert (o[KEY.PAIR_REPULSION_ENERGY] >= 0).all()
    assert torch.allclose(  # the per-atom term is in the atomic energy
        o[KEY.ATOMIC_ENERGY].squeeze(-1) - o_ref[KEY.ATOMIC_ENERGY].squeeze(-1),
        o[KEY.PAIR_REPULSION_ENERGY].to(o[KEY.ATOMIC_ENERGY].dtype), atol=1e-9)
    with torch.no_grad():  # effective lambda ~ N(1, 0.5^2), theta ~ N(0, 0.5^2)
        pair.log_amp.normal_(1.0 / c, 0.5 / c)
        pair.decay_raw.normal_(0.0, 0.5 / c)
    o2 = model(get_graphs(batched=True))
    assert (o2[KEY.PRED_FORCE] - o[KEY.PRED_FORCE]).abs().max() > 1e-3
    out, out_rot, R = _rotated_pair(model)
    e, e_r = out[KEY.PRED_TOTAL_ENERGY], out_rot[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = out[KEY.PRED_FORCE], out_rot[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    # element embeddings are Adam parameters, not Muon matrices
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    adam = {name[id(p)] for p in groups[1]['params']}
    assert {'pair_repulsion.log_amp', 'pair_repulsion.decay_raw'} <= adam
    assert 'pair_repulsion.radius' in model.state_dict()  # buffer, saved
    script(model.eval())
    config = get_model_config()
    config.update(cf)
    Logger().print_model_info(model, config)
    # e3gnn/parallel: last segment, ghost types (segment 0 builds them)
    segs = build_E3_equivariant_model(config, parallel=True)
    for i, seg in enumerate(segs):
        sm = seg._modules
        assert ('pair_repulsion' in sm) == (i == len(segs) - 1)
        assert ('atom_type_all' in sm) == (i == 0)
    last = list(segs[-1]._modules)  # right after the segment's edge inputs
    head = 2 if 'edge_pair_input' in last else 1
    assert last.index('pair_repulsion') == head
    assert segs[-1]._modules['pair_repulsion'].key_atom_type == KEY.ATOM_TYPE_ALL


def test_pair_repulsion_errors():
    import sevenn._const as _const

    with pytest.raises(ValueError):
        get_model({**_metaformer, 'pair_repulsion': {'max_cutoff': 4.5}})
    cond = _const.MODEL_CONFIG_CONDITION[KEY.PAIR_REPULSION]
    assert cond(None) and cond({}) and cond({'num_terms': 3, 'param_scale': 2})
    assert not cond({'cutoff': 3.0}) and not cond({'amp_init': 0.0})
    assert not cond({'param_scale': 0.0})


def test_norm_magnitude_module():
    """
    the log-RMS feature: zero weights = the plain norm (bitwise); scaling the
    input by c shifts the 0e outputs by 2 log(c) v and leaves the rest
    unchanged (the magnitude the norm removes stays readable); rotation
    commutes; 0o channels untouched
    """
    import math

    from e3nn.o3 import rand_matrix

    from sevenn.nn.normalization import EquivariantRMSNorm

    irreps = Irreps('6x0e+2x0o+4x1o+2x2e')
    torch.manual_seed(0)
    plain = EquivariantRMSNorm(irreps, 'x')
    mag = EquivariantRMSNorm(irreps, 'x', magnitude=True)
    with torch.no_grad():
        plain.weight.uniform_(0.5, 1.5)
        plain.bias.normal_()
    missing, unexpected = mag.load_state_dict(plain.state_dict(), strict=False)
    assert missing == ['mag_weight'] and not unexpected
    x = torch.randn(5, irreps.dim, dtype=torch.float64) * 3.0
    plain.double()
    mag.double()
    assert torch.equal(plain({'x': x.clone()})['x'], mag({'x': x.clone()})['x'])
    with torch.no_grad():
        mag.mag_weight.normal_()
    y1 = mag({'x': x.clone()})['x']
    c = 50.0
    y2 = mag({'x': c * x})['x']
    scal = mag.scalar_mask.bool()
    v = mag.mag_weight[scal]
    shift = (y2 - y1)[:, scal]
    # s = log(mean square + eps): shift 2 log(c) v (up to eps)
    assert torch.allclose(shift, 2 * math.log(c) * v.expand_as(shift), atol=1e-5)
    assert torch.allclose(y2[:, ~scal], y1[:, ~scal], atol=1e-6)
    assert torch.equal(y1[:, 6:8], plain({'x': x.clone()})['x'][:, 6:8])  # 0o
    D = irreps.D_from_matrix(rand_matrix().double())
    assert torch.allclose(mag({'x': x @ D.T})['x'], y1 @ D.T, atol=1e-10)
    # zero-initialized but trainable from the start: non-zero gradient
    mag.mag_weight.data.zero_()
    mag({'x': x.clone().requires_grad_(True)})['x'].pow(2).sum().backward()
    assert mag.mag_weight.grad[scal].abs().min() > 0
    assert mag.mag_weight.grad[~scal].abs().max() == 0


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'radial_pair_embedding': True, 'norm_magnitude': True},
    {**_metaformer, 'norm_magnitude': True, 'use_final_norm': True,
     'readout_block': {}, 'parallel_block': [False, True, True],
     'ffn_type': 's2_bilinear'},
])
def test_norm_magnitude(cf):
    """every EquivariantRMSNorm gets the zero-initialized log-RMS feature"""
    from e3nn.util.jit import script

    from sevenn.logger import Logger
    from sevenn.model_build import build_E3_equivariant_model
    from sevenn.nn.normalization import EquivariantRMSNorm
    from sevenn.train.muon import build_muon_param_groups
    from tests.unit_tests.test_model import get_model_config

    torch.manual_seed(0)
    ref = get_model({**cf, 'norm_magnitude': False})
    model = get_model(cf)
    norms = {k: m for k, m in model.named_children()
             if isinstance(m, EquivariantRMSNorm)}
    # layer-0 Norm1 reads the element embedding (constant magnitude): plain
    assert norms.pop('0_norm_1').mag_weight is None
    assert len(norms) >= 4
    assert all(m.mag_weight is not None and m.mag_weight.abs().sum() == 0
               for m in norms.values())
    missing, unexpected = model.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == sorted(
        f'{k}.mag_weight' for k in norms)
    ref.set_is_batch_data(True)
    model.set_is_batch_data(True)
    o_ref = ref(get_graphs(batched=True))
    o = model(get_graphs(batched=True))
    for k in (KEY.PRED_TOTAL_ENERGY, KEY.PRED_FORCE, KEY.PRED_STRESS):
        assert torch.equal(o_ref[k], o[k]), k  # zero init: bitwise the base
    o[KEY.PRED_TOTAL_ENERGY].sum().backward()
    assert any(m.mag_weight.grad.abs().sum() > 0 for m in norms.values())
    with torch.no_grad():
        for m in norms.values():
            m.mag_weight.normal_(0, 0.3)
    o2 = model(get_graphs(batched=True))
    assert (o2[KEY.PRED_FORCE] - o[KEY.PRED_FORCE]).abs().max() > 1e-4
    out, out_rot, R = _rotated_pair(model)
    e, e_r = out[KEY.PRED_TOTAL_ENERGY], out_rot[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = out[KEY.PRED_FORCE], out_rot[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    adam = {name[id(p)] for p in groups[1]['params']}
    assert {f'{k}.mag_weight' for k in norms} <= adam
    script(model.eval())
    config = get_model_config()
    config.update(cf)
    Logger().print_model_info(model, config)
    segs = build_E3_equivariant_model(config, parallel=True)
    twin = segs[0]._modules['ghost_0_norm_1']
    assert twin.mag_weight is None and twin.key_x == KEY.NODE_FEATURE_GHOST


def test_norm_magnitude_errors():
    with pytest.raises(ValueError):  # no norm to take the magnitude from
        get_model({**_partial, 'norm_magnitude': True})
    with pytest.raises(ValueError):
        get_model({**_metaformer, 'metaformer_norm': False,
                   'norm_magnitude': True})
    get_model({**_metaformer, 'metaformer_norm': False, 'use_final_norm': True,
               'norm_magnitude': True})


def test_magnitude_residual_module():
    """
    MagnitudeResidualOutro: gamma = 0 is SelfConnectionOutro (bitwise);
    log rho is the log of the RMS EquivariantRMSNorm divides by; the gain is
    (c rho / rho_ref)^gamma for an input scaled by c; gamma is clipped to
    [0, gamma_max] with a straight-through gradient; the gain is invariant
    (rotation, parity: 0o channels are not centered)
    """
    import math

    from e3nn.o3 import rand_matrix

    from sevenn.nn.normalization import (
        EquivariantRMSNorm,
        MagnitudeResidualOutro,
    )
    from sevenn.nn.self_connection import SelfConnectionOutro

    irreps = Irreps('6x0e+2x0o+4x1o+2x2e')
    tmp = KEY.SELF_CONNECTION_TEMP
    torch.manual_seed(0)
    x_in = torch.randn(5, irreps.dim, dtype=torch.float64) * 3.0 + 1.0
    b = torch.randn(5, irreps.dim, dtype=torch.float64)
    mod = MagnitudeResidualOutro(irreps, irreps, 'x', rho_ref=2.0).double()
    assert list(mod.state_dict()) == ['gamma'] and mod.gamma.item() == 0.0
    plain = SelfConnectionOutro('x')
    out = mod({'x': b.clone(), tmp: x_in.clone()})
    assert torch.equal(out['x'], plain({'x': b.clone(), tmp: x_in.clone()})['x'])
    assert tmp not in out
    # rho = 1 / inv_rms of the norm (centered over 0e, balanced, eps)
    norm = EquivariantRMSNorm(irreps, 'x', affine=False).double()
    y = norm({'x': x_in.clone()})['x']
    mask = norm.scalar_mask
    xc = x_in - (x_in * mask).sum(-1, keepdim=True) / 6 * mask
    rho = torch.sqrt((xc ** 2 * norm.balance).sum(-1) + norm.eps)
    assert torch.allclose(mod.log_rho(x_in), rho.log(), atol=1e-12)
    assert torch.allclose(y, xc / rho.unsqueeze(-1), atol=1e-12)
    for theta, g_eff in ((0.6, 0.6), (-0.5, 0.0), (2.0, 1.0)):
        with torch.no_grad():
            mod.gamma.fill_(theta)
        assert mod.effective_gamma().item() == g_eff
        for c in (1.0, 7.0):
            out = mod({'x': b.clone(), tmp: c * x_in})['x']
            gain = ((c * c * (rho ** 2 - norm.eps) + norm.eps).sqrt() / 2.0) ** g_eff
            assert torch.allclose(out, c * x_in + gain.unsqueeze(-1) * b,
                                  atol=1e-10)
        # straight-through: theta gets d loss / d gamma also when clipped
        mod.gamma.grad = None
        mod({'x': b.clone(), tmp: x_in.clone()})['x'].sum().backward()
        assert mod.gamma.grad.abs().item() > 0
    # rotation / parity: the gain is invariant, the output equivariant
    with torch.no_grad():
        mod.gamma.fill_(0.7)
    out = mod({'x': b.clone(), tmp: x_in.clone()})['x']
    for R in (rand_matrix().double(), -rand_matrix().double()):
        D = irreps.D_from_matrix(R)
        out_r = mod({'x': b @ D.T, tmp: x_in @ D.T})['x']
        assert torch.allclose(out_r, out @ D.T, atol=1e-10)
    # projection (last layer: irreps_out = the 0e channels of the input)
    proj = MagnitudeResidualOutro(irreps, Irreps('6x0e'), 'x').double()
    with torch.no_grad():
        proj.gamma.fill_(1.0)
    out = proj({'x': b[:, :6].clone(), tmp: x_in.clone()})['x']
    assert torch.allclose(out, x_in[:, :6] + rho.unsqueeze(-1) * b[:, :6])
    assert math.isclose(proj.log_rho_ref, 0.0)
    assert math.isclose(mod.log_rho_ref, math.log(2.0))
    # several 0e blocks (centered together) and no 0e (no centering): rho of
    # the norm as well
    for irr in ('3x0e+2x1o+5x0e', '4x1o+2x2e'):
        irr = Irreps(irr)
        m2 = MagnitudeResidualOutro(irr, irr, 'x').double()
        n2 = EquivariantRMSNorm(irr, 'x', affine=False).double()
        z = torch.randn(5, irr.dim, dtype=torch.float64) * 2.0 + 0.5
        y2 = n2({'x': z.clone()})['x']
        r2 = torch.exp(m2.log_rho(z))
        zc = z
        if n2.centering:
            zc = z - (z * n2.scalar_mask).sum(-1, keepdim=True) / n2.num_scalars * (
                n2.scalar_mask)
        assert m2.centering == n2.centering
        assert torch.allclose(y2, zc / r2.unsqueeze(-1), atol=1e-12)


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'radial_pair_embedding': True, 'ffn_type': 's2_bilinear',
     'magnitude_residual': True},
    {**_metaformer, 'magnitude_residual': {'gamma_max': 0.8, 'rho_ref': 1.5},
     'readout_block': {}, 'per_layer_readout': True, 'conv_gate': False,
     'layer_scale_init': 0.1, 'norm_magnitude': True, 'use_final_norm': True},
    {**_metaformer, 'radial_pair_embedding': True,
     'magnitude_residual': {'branches': 'conv'}, 'readout_block': {}},
    {**_metaformer, 'magnitude_residual': {'branches': 'ffn'}},
])
def test_magnitude_residual(cf):
    """every pre-norm branch but the layer-0 conv gets the zero-init gain"""
    from e3nn.util.jit import script

    from sevenn.logger import Logger
    from sevenn.model_build import build_E3_equivariant_model
    from sevenn.nn.normalization import (
        MagnitudeResidualOutro,
        SelfConnectionResidualIntro,
    )
    from sevenn.train.muon import build_muon_param_groups
    from tests.unit_tests.test_model import get_model_config

    torch.manual_seed(0)
    ref = get_model({**cf, 'magnitude_residual': False})
    model = get_model(cf)
    outs = {k: m for k, m in model.named_children()
            if isinstance(m, MagnitudeResidualOutro)}
    n_layer = sum(k.endswith('_convolution') for k, _ in model.named_children())
    branches = cf['magnitude_residual']
    branches = branches.get('branches', 'all') if branches is not True else 'all'
    expect = set()
    if branches != 'conv':
        expect |= {f'{t}_ffn_outro' for t in range(n_layer)}
        if cf.get('readout_block') is not None:
            expect.add('readout_ffn_outro')
    if branches != 'ffn':
        expect |= {f'{t}_self_connection_outro' for t in range(1, n_layer)}
    assert set(outs) == expect
    assert all(m.gamma.item() == 0.0 for m in outs.values())
    last = f'{n_layer - 1}_self_connection'
    key_full = model._modules[f'{last}_intro'].key_full
    assert (key_full != '') == (branches != 'ffn')
    if branches != 'ffn':
        assert outs[f'{last}_outro'].key_full == key_full
    missing, unexpected = model.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == sorted(f'{k}.gamma' for k in outs)
    ref.set_is_batch_data(True)
    model.set_is_batch_data(True)
    o_ref = ref(get_graphs(batched=True))
    o = model(get_graphs(batched=True))
    for k in (KEY.PRED_TOTAL_ENERGY, KEY.PRED_FORCE, KEY.PRED_STRESS):
        assert torch.equal(o_ref[k], o[k]), k  # gamma = 0: bitwise the base
    assert not any('magnitude_residual' in k for k in o)
    o[KEY.PRED_TOTAL_ENERGY].sum().backward()
    assert all(m.gamma.grad.abs().item() > 0 for m in outs.values())
    with torch.no_grad():
        for m in outs.values():
            m.gamma.fill_(0.6)
    o2 = model(get_graphs(batched=True))
    assert (o2[KEY.PRED_FORCE] - o[KEY.PRED_FORCE]).abs().max() > 1e-4
    out, out_rot, R = _rotated_pair(model)
    e, e_r = out[KEY.PRED_TOTAL_ENERGY], out_rot[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = out[KEY.PRED_FORCE], out_rot[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    adam = {name[id(p)] for p in groups[1]['params']}
    assert {f'{k}.gamma' for k in outs} <= adam
    from tests.unit_tests.test_model import _graph_list

    model.eval()
    model.set_is_batch_data(False)
    g = _graph_list[0].clone()
    ref_out = model(g.clone())  # before scripting (e3nn scripts in place)
    inp = {k: v for k, v in g.to_dict().items() if isinstance(v, torch.Tensor)}
    inp[KEY.NODE_FEATURE] = model._atomic_numbers_to_onehot(inp[KEY.ATOMIC_NUMBERS])
    model.eval_type_map = False  # as sevenn deploy
    out_s = torch.jit.freeze(script(model))(inp)
    for k in (KEY.PRED_TOTAL_ENERGY, KEY.PRED_FORCE):
        assert torch.allclose(out_s[k], ref_out[k], atol=1e-5), k
    config = get_model_config()
    config.update(cf)
    Logger().print_model_info(model, config)
    segs = build_E3_equivariant_model(config, parallel=True)
    intros = [(k, m) for s in segs for k, m in s._modules.items()
              if isinstance(m, SelfConnectionResidualIntro)]
    stored = [f'{last}_intro'] if branches != 'ffn' else []
    assert [k for k, m in intros if m.store_input] == stored
    assert all(m.key_full == '' for _, m in intros)


def test_magnitude_residual_errors():
    import sevenn._const as _const

    with pytest.raises(ValueError):  # nequip block
        get_model({**_partial, 'magnitude_residual': True})
    with pytest.raises(ValueError):  # no pre-norm
        get_model({**_metaformer, 'metaformer_norm': False,
                   'magnitude_residual': True})
    with pytest.raises(ValueError):
        get_model({**_metaformer, 'parallel_block': [False, True, True],
                   'magnitude_residual': True})
    with pytest.raises(ValueError):  # linear residual
        get_model({**_metaformer, 'self_connection_type': 'linear',
                   'magnitude_residual': True})
    cond = _const.MODEL_CONFIG_CONDITION[KEY.MAGNITUDE_RESIDUAL]
    assert cond(False) and cond(True) and cond(None) and cond({})
    assert cond({'gamma_max': 0.5, 'rho_ref': 2.0})
    assert not cond({'gamma_max': 1.5}) and not cond({'gamma_max': 0.0})
    assert not cond({'rho_ref': 0.0}) and not cond({'gamma': 1.0})
    assert cond({'branches': 'conv'}) and not cond({'branches': 'readout'})


@pytest.mark.parametrize('cf', [
    {**_metaformer, 'radial_pair_embedding': True, 'radial_exp_basis': {}},
    {**_metaformer, 'radial_pair_embedding': True, 'radial_node_cond': True,
     'radial_mlp_block': {}, 'radial_short_basis': {'num_basis': 4},
     'radial_exp_basis': {'scales': [0.3, 0.9], 'cutoff': 2.5}},
])
def test_exp_radial_basis(cf):
    """bounded exponential features in the pair radial MLP input"""
    from e3nn.util.jit import script

    from sevenn.nn.radial_pair import (
        EdgePairInput,
        ExpRadialBasis,
        exp_basis_scales,
    )
    from sevenn.train.muon import build_muon_param_groups

    opts = cf['radial_exp_basis']
    rs, scales = opts.get('cutoff', 3.0), exp_basis_scales(opts)
    assert exp_basis_scales({}) == [0.2, 0.4, 0.8, 1.6]
    torch.manual_seed(0)
    ref = get_model({**cf, 'radial_exp_basis': None})
    model = get_model(cf)
    pair = dict(model.named_children())['edge_pair_input']
    assert isinstance(pair, EdgePairInput) and pair.use_exp
    short = cf.get('radial_short_basis') is not None
    assert pair.use_short == short
    w = _convs(model)['1_convolution'].weight_nn
    assert w.num_exp == len(scales) and w.w_exp.abs().sum() == 0
    # zero-initialized: bitwise the model without the exponential basis
    missing, unexpected = model.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and all(k.endswith('w_exp') for k in missing)
    ref.set_is_batch_data(True)
    model.set_is_batch_data(True)
    o_ref = ref(get_graphs(batched=True))
    o = model(get_graphs(batched=True))
    for k in (KEY.PRED_TOTAL_ENERGY, KEY.PRED_FORCE, KEY.PRED_STRESS):
        assert torch.equal(o_ref[k], o[k]), k
    # basis: f_k(0) = 1, f_k'(0) = -1/a_k, decreasing, 0 < f <= 1 below r_s,
    # exactly zero beyond, value / 1st / 2nd derivative -> 0 at r_s
    basis = ExpRadialBasis(scales, rs).double()
    r = torch.tensor([0.0, 0.5, rs - 1e-3, rs, rs + 1e-3, 3.9],
                     dtype=torch.float64, requires_grad=True)
    f = basis(r)
    assert f.shape == (6, len(scales))
    assert torch.allclose(f[0], torch.ones_like(f[0]))
    d1 = torch.stack([torch.autograd.grad(f[:, k].sum(), r, create_graph=True)[0]
                      for k in range(len(scales))], -1)
    d2 = torch.stack([torch.autograd.grad(d1[:, k].sum(), r, retain_graph=True)[0]
                      for k in range(len(scales))], -1)
    inv = torch.tensor([1.0 / a for a in scales], dtype=torch.float64)
    assert torch.allclose(d1[0], -inv, atol=1e-10)
    assert f[2].abs().max() < 1e-6 and d1[2].abs().max() < 1e-3
    assert d2[2].abs().max() < 1.0
    assert torch.equal(f[3:], torch.zeros_like(f[3:]))
    assert torch.equal(d1[3:], torch.zeros_like(d1[3:]))
    rr = torch.linspace(0.0, rs, 301, dtype=torch.float64)
    ff = basis(rr)
    assert (ff[1:-1] > 0).all() and (ff <= 1).all()
    assert (ff[1:] - ff[:-1] <= 0).all()  # monotonically decreasing
    with torch.no_grad():
        for c in _convs(model).values():
            c.weight_nn.w_exp.normal_(0, 0.3)
    o2 = model(get_graphs(batched=True))
    assert (o2[KEY.PRED_FORCE] - o[KEY.PRED_FORCE]).abs().max() > 1e-4
    out, out_rot, R = _rotated_pair(model)
    e, e_r = out[KEY.PRED_TOTAL_ENERGY], out_rot[KEY.PRED_TOTAL_ENERGY]
    assert torch.allclose(e, e_r, atol=1e-4)
    f, f_r = out[KEY.PRED_FORCE], out_rot[KEY.PRED_FORCE]
    assert torch.allclose(f @ R.T, f_r, atol=1e-4)
    groups = build_muon_param_groups(model)
    name = {id(p): n for n, p in model.named_parameters()}
    muon = {name[id(p)] for p in groups[0]['params']}
    assert '1_convolution.weight_nn.w_exp' in muon  # as w_short
    script(model.eval())


def test_exp_radial_basis_errors():
    import sevenn._const as _const

    with pytest.raises(ValueError):  # needs the pair radial MLP
        get_model({**_metaformer, 'radial_exp_basis': {}})
    with pytest.raises(ValueError):
        get_model({**_metaformer, 'radial_pair_embedding': True,
                   'radial_exp_basis': {'cutoff': 4.5}})
    cond = _const.MODEL_CONFIG_CONDITION[KEY.RADIAL_EXP_BASIS]
    assert cond(None) and cond({}) and cond({'num_basis': 2})
    assert cond({'num_basis': 2, 'scales': [0.3, 0.6], 'cutoff': 2.5,
                 'poly_p': 6})
    assert not cond({'num_basis': 3, 'scales': [0.3, 0.6]})
    assert not cond({'scales': [0.3, -0.6]}) and not cond({'scales': []})
    assert not cond({'cutoff': 0.0}) and not cond({'poly_p': 2.5})
    assert not cond({'decay': 1.0})
