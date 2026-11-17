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
