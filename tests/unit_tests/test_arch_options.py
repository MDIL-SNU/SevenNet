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
