"""Tests for grouped-modality centroid regularization (SevenNet-MF).

Centroid-L2 pulls the task-specific (modal) weight rows of modalities in the
same functional group toward their group centroid, leaving the centroid free.

Convention note (this checkout): the 1/2 factor is folded into get_loss, so
ModalityGroupAlign.get_loss returns (1/2) * sum_g sum_m ||W[m] - Wbar_g||^2,
matching L2Regularization.get_loss = (1/2)||w||^2; the raw weight is stored.
The lc.csv monitor uses L2RegLogError (2 * get_loss), so the recorded column is
the raw penalty (sum of squared deviations / ||w||^2). _group_centroid_l2 itself
returns the raw (un-halved) penalty.
"""

import pytest
import torch
import torch.nn as nn
from e3nn.o3 import Irreps

import sevenn._keys as KEY
from sevenn._const import error_record_condition
from sevenn.error_recorder import ErrorRecorder
from sevenn.nn.linear import IrrepsLinear
from sevenn.train.loss import (
    ModalityGroupAlign,
    _group_centroid_l2,
    get_modal_regularization,
)


# --------------------------------------------------------------------------
# Pure math: _group_centroid_l2(weight_view, group_indices)  -- raw, no 1/2
# --------------------------------------------------------------------------
def test_centroid_penalty_zero_when_group_identical():
    row = torch.tensor([1.0, 2.0, 3.0, 4.0])
    W = torch.stack([row, row.clone(), row.clone()])  # (3, 4) all equal
    out = _group_centroid_l2(W, [[0, 1, 2]])
    assert torch.allclose(out, torch.zeros(()))


def test_centroid_penalty_matches_hand_computed():
    # group {0,1,2} on column 0: rows 0,2,4 -> centroid 2 -> dev^2 = 4+0+4 = 8
    W = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    out = _group_centroid_l2(W, [[0, 1, 2]])
    assert torch.allclose(out, torch.tensor(8.0))


def test_singleton_and_empty_group_contribute_zero():
    W = torch.tensor([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])
    assert torch.allclose(_group_centroid_l2(W, [[1]]), torch.zeros(()))
    assert torch.allclose(_group_centroid_l2(W, [[]]), torch.zeros(()))


def test_multi_group_sums_independently():
    # group A {0,1}: centroid 1 -> 1+1 = 2 ; group B {2,3}: centroid 12 -> 4+4 = 8
    W = torch.tensor([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [14.0, 0.0]])
    out = _group_centroid_l2(W, [[0, 1], [2, 3]])
    assert torch.allclose(out, torch.tensor(10.0))


def test_centroid_penalty_gradient_is_deviation_from_centroid():
    W = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], requires_grad=True
    )
    out = _group_centroid_l2(W, [[0, 1, 2]])
    out.backward()
    # d/dW[m] sum ||W[m]-mean||^2 = 2 (W[m]-mean); centroid coupling cancels.
    expected = torch.tensor([[-4.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
    assert torch.allclose(W.grad, expected)


# --------------------------------------------------------------------------
# Glue: ModalityGroupAlign.get_loss over real IrrepsLinear modules
# (returns the 1/2-folded penalty)
# --------------------------------------------------------------------------
def _make_irreps_linear(num_modalities, out_scalars=5, in_scalars=8):
    return IrrepsLinear(
        Irreps(f'{in_scalars}x0e'),
        Irreps(f'{out_scalars}x0e'),
        data_key_in='x',
        num_modalities=num_modalities,
        lazy_layer_instantiate=False,
    )


class _TinyModalModel(nn.Module):
    def __init__(self, modal_map, num_modalities):
        super().__init__()
        self.l0_self_interaction_1 = _make_irreps_linear(num_modalities)
        self.l0_self_interaction_2 = _make_irreps_linear(num_modalities)
        self.modal_map = modal_map


def _set_modal_rows(model, module_key, col0_values):
    """Zero the modal weight view, then set column 0 of each modal row."""
    with torch.no_grad():
        wv = list(model._modules[module_key]._modules['linear'].weight_views())[-1]
        wv.zero_()
        for m, v in enumerate(col0_values):
            wv[m, 0] = v


def test_get_loss_sums_over_target_modules():
    model = _TinyModalModel({'a': 0, 'b': 1, 'c': 2}, num_modalities=3)
    keys = ['l0_self_interaction_1', 'l0_self_interaction_2']
    for k in keys:  # each module: rows 0,2,4 -> raw penalty 8
        _set_modal_rows(model, k, [0.0, 2.0, 4.0])
    reg = ModalityGroupAlign('Align_modal', keys, {'g': ['a', 'b', 'c']})
    out = reg.get_loss({'x': torch.zeros(1)}, model)
    # raw 8 + 8 = 16, halved by the 1/2 convention -> 8.0
    assert torch.allclose(out, torch.tensor([8.0]))


def test_get_loss_resolves_group_names_via_modal_map():
    # only modals a,c are grouped; b (the large outlier) is excluded -> penalty 0
    model = _TinyModalModel({'a': 0, 'b': 1, 'c': 2}, num_modalities=3)
    _set_modal_rows(model, 'l0_self_interaction_1', [3.0, 99.0, 3.0])
    _set_modal_rows(model, 'l0_self_interaction_2', [3.0, 99.0, 3.0])
    reg = ModalityGroupAlign(
        'Align_modal', ['l0_self_interaction_1'], {'g': ['a', 'c']}
    )
    out = reg.get_loss({'x': torch.zeros(1)}, model)
    assert torch.allclose(out, torch.zeros(1))


def test_unknown_modal_name_raises():
    model = _TinyModalModel({'a': 0, 'b': 1}, num_modalities=2)
    reg = ModalityGroupAlign(
        'Align_modal', ['l0_self_interaction_1'], {'g': ['a', 'zzz']}
    )
    with pytest.raises(ValueError, match='zzz'):
        reg.get_loss({'x': torch.zeros(1)}, model)


# --------------------------------------------------------------------------
# Config: get_modal_regularization  (raw weights; 1/2 lives in get_loss)
# --------------------------------------------------------------------------
def _modal_config(reg_param):
    return {
        KEY.USE_MODALITY: True,
        KEY.USE_MODAL_NODE_EMBEDDING: False,
        KEY.USE_MODAL_SELF_INTER_INTRO: True,
        KEY.USE_MODAL_SELF_INTER_OUTRO: True,
        KEY.USE_MODAL_OUTPUT_BLOCK: True,
        KEY.REG_PARAM: reg_param,
    }


def test_config_builds_both_l2_and_alignment():
    config = _modal_config({
        'modal': {'regularization_weight': 1e-6},
        'modal_group_alignment': {
            'alignment_weight': 1e-4,
            'groups': {'pbe': ['a', 'b']},
        },
    })
    model = _TinyModalModel({'a': 0, 'b': 1}, num_modalities=2)
    regs = get_modal_regularization(config, model)
    by_name = {r.name: (r, w) for r, w in regs}
    assert set(by_name) == {'L2_modal', 'Align_modal_pbe'}
    assert by_name['L2_modal'][1] == 1e-6
    assert by_name['Align_modal_pbe'][1] == 1e-4
    align = by_name['Align_modal_pbe'][0]
    assert align.groups == {'pbe': ['a', 'b']}
    # alignment targets the same modal modules as L2 (output block excluded)
    assert align.module_keys == ['l0_self_interaction_1', 'l0_self_interaction_2']


def test_config_emits_one_alignment_per_group():
    config = _modal_config({
        'modal_group_alignment': {
            'alignment_weight': 1e-4,
            'groups': {'pbe': ['a', 'b'], 'r2scan': ['c', 'd']},
        },
    })
    model = _TinyModalModel({'a': 0, 'b': 1, 'c': 2, 'd': 3}, num_modalities=4)
    regs = get_modal_regularization(config, model)
    by_name = {r.name: (r, w) for r, w in regs}
    assert set(by_name) == {'Align_modal_pbe', 'Align_modal_r2scan'}
    assert by_name['Align_modal_pbe'][0].groups == {'pbe': ['a', 'b']}
    assert by_name['Align_modal_r2scan'][0].groups == {'r2scan': ['c', 'd']}
    assert by_name['Align_modal_pbe'][1] == 1e-4
    assert by_name['Align_modal_r2scan'][1] == 1e-4
    assert by_name['Align_modal_pbe'][0].module_keys == [
        'l0_self_interaction_1', 'l0_self_interaction_2'
    ]


def test_per_group_alignment_sums_to_combined():
    # loss-equivalence: per-group penalties sum to the single combined penalty
    model = _TinyModalModel({'a': 0, 'b': 1, 'c': 2, 'd': 3}, num_modalities=4)
    keys = ['l0_self_interaction_1', 'l0_self_interaction_2']
    for k in keys:
        _set_modal_rows(model, k, [0.0, 2.0, 10.0, 14.0])
    b = {'x': torch.zeros(1)}
    combined = ModalityGroupAlign(
        'Align_modal', keys, {'g1': ['a', 'b'], 'g2': ['c', 'd']}
    ).get_loss(b, model)
    g1 = ModalityGroupAlign('Align_modal_g1', keys, {'g1': ['a', 'b']}).get_loss(b, model)
    g2 = ModalityGroupAlign('Align_modal_g2', keys, {'g2': ['c', 'd']}).get_loss(b, model)
    assert torch.allclose(combined, g1 + g2)


def test_config_alignment_only():
    config = _modal_config({
        'modal_group_alignment': {
            'alignment_weight': 5e-5,
            'groups': {'pbe': ['a', 'b']},
        },
    })
    model = _TinyModalModel({'a': 0, 'b': 1}, num_modalities=2)
    regs = get_modal_regularization(config, model)
    assert [r.name for r, _ in regs] == ['Align_modal_pbe']


def test_config_no_alignment_when_absent():
    config = _modal_config({'modal': {'regularization_weight': 1e-6}})
    model = _TinyModalModel({'a': 0, 'b': 1}, num_modalities=2)
    regs = get_modal_regularization(config, model)
    assert [r.name for r, _ in regs] == ['L2_modal']


def test_no_modal_reg_when_modality_off():
    config = _modal_config({
        'modal': {'regularization_weight': 1e-6},
        'modal_group_alignment': {
            'alignment_weight': 1e-4,
            'groups': {'pbe': ['a', 'b']},
        },
    })
    config[KEY.USE_MODALITY] = False
    model = _TinyModalModel({'a': 0, 'b': 1}, num_modalities=2)
    assert get_modal_regularization(config, model) == []


# --------------------------------------------------------------------------
# Monitoring: Align_modal error metric (lc.csv) -- L2RegLogError logs 2*get_loss
# so the recorded value is the raw penalty (un-halved).
# --------------------------------------------------------------------------
def test_align_modal_is_supported_error_type():
    assert error_record_condition([['Align_modal', 'Loss']]) is True


def test_align_modal_error_metric_monitors_penalty():
    model = _TinyModalModel({'a': 0, 'b': 1, 'c': 2}, num_modalities=3)
    for k in ['l0_self_interaction_1', 'l0_self_interaction_2']:
        _set_modal_rows(model, k, [0.0, 2.0, 4.0])  # raw penalty 8 per module
    reg = ModalityGroupAlign(
        'Align_modal',
        ['l0_self_interaction_1', 'l0_self_interaction_2'],
        {'g': ['a', 'b', 'c']},
    )
    config = {KEY.ERROR_RECORD: [['Align_modal', 'Loss']]}
    recorder = ErrorRecorder.from_config(
        config, loss_functions=[(reg, 1e-4)]
    )
    assert 'Align_modal_Loss' in [m.name for m in recorder.metrics]
    recorder.update({'x': torch.zeros(1)}, model=model)
    metric = next(m for m in recorder.metrics if m.name == 'Align_modal_Loss')
    # L2RegLogError logs 2 * get_loss = raw penalty = 8 + 8 = 16
    assert abs(metric.value.get() - 16.0) < 1e-5


def test_per_group_error_type_is_supported():
    assert error_record_condition([['Align_modal_pbe_family', 'Loss']]) is True


def test_per_group_align_metric_monitors_penalty():
    model = _TinyModalModel({'a': 0, 'b': 1, 'c': 2}, num_modalities=3)
    for k in ['l0_self_interaction_1', 'l0_self_interaction_2']:
        _set_modal_rows(model, k, [0.0, 2.0, 4.0])
    reg = ModalityGroupAlign(
        'Align_modal_pbe',
        ['l0_self_interaction_1', 'l0_self_interaction_2'],
        {'pbe': ['a', 'b', 'c']},
    )
    config = {KEY.ERROR_RECORD: [['Align_modal_pbe', 'Loss']]}
    recorder = ErrorRecorder.from_config(
        config, loss_functions=[(reg, 1e-4)]
    )
    assert 'Align_modal_pbe_Loss' in [m.name for m in recorder.metrics]
    recorder.update({'x': torch.zeros(1)}, model=model)
    metric = next(m for m in recorder.metrics if m.name == 'Align_modal_pbe_Loss')
    assert abs(metric.value.get() - 16.0) < 1e-5
