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
