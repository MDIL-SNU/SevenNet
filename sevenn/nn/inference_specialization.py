"""Specialize a model for the species and modality used during inference."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import torch
from torch import nn

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .linear import IrrepsLinear
from .self_connection import SelfConnectionIntro


def _self_connection_for_species(sc, width):
    return sc.fc_tensor_product_cls(
        sc.irreps_in1, f'{width}x0e', sc.irreps_out,
        shared_weights=True, internal_weights=None,
        **sc.fc_tensor_product_kwargs,
    )


class PrunedSelfConnectionIntro(SelfConnectionIntro):
    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        operand = data[self.key_operand].index_select(1, self.columns)
        data[KEY.SELF_CONNECTION_TEMP] = self.fc_tensor_product(
            data[self.key_x], operand
        )
        return data


@torch.no_grad()
def prune_self_connection(model: nn.Module, columns: Sequence[int]) -> list[str]:
    """Specialize e3nn-compatible self connections for the allowed type indices."""
    columns = sorted({int(c) for c in columns})
    if not columns:
        raise ValueError('Expected at least one allowed species')
    config = getattr(model, '_lunatp_precompile_config', None)
    changed = []
    for name, sc in model._modules.items():
        if type(sc) is PrunedSelfConnectionIntro:
            weight = sc.fc_tensor_product.weight
            old = sc._full.to(device=weight.device, dtype=weight.dtype)
        elif type(sc) is SelfConnectionIntro and sc.layer_instantiated:
            old = sc.fc_tensor_product
        else:
            continue
        in2 = sc.irreps_in2
        if len(in2) != 1 or not in2[0].ir.is_scalar():
            continue

        if len(columns) == in2[0].mul:
            if type(sc) is PrunedSelfConnectionIntro:
                sc.__class__ = SelfConnectionIntro
                sc.fc_tensor_product = old
                del sc.columns, sc._full
                changed.append(name)
            continue
        if config is not None:
            from .lunatp_helper import precompile_species_width

            precompile_species_width(config, len(columns))
            config = None

        ref = old.weight
        new = _self_connection_for_species(sc, len(columns))
        new = new.to(ref.device, ref.dtype)
        cols = torch.tensor(columns, device=ref.device)
        old_tp = getattr(old, 'tp', old)
        new_tp = getattr(new, 'tp', new)
        for i, ins, weights in old_tp.weight_views(ref, yield_instruction=True):
            target = new_tp.weight_view_for_instruction(i, new.weight)
            scale = ins.path_weight / new_tp.instructions[i].path_weight
            target.copy_(weights.index_select(1, cols) * scale)

        new.requires_grad_(False)
        sc.__class__ = PrunedSelfConnectionIntro
        object.__setattr__(
            sc, '_full', old
        )  # Retain the original if species change.
        sc.fc_tensor_product = new
        sc.register_buffer('columns', cols)
        changed.append(name)
    return changed


class ModalFoldedLinear(IrrepsLinear):
    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.linear(data[self.key_input]) + self.modal_bias
        return data


def _folded_modal_linear(lin):
    return lin.linear_cls(
        lin._irreps_in_wo_modal, lin.irreps_out,
        **{**lin.linear_kwargs, 'biases': False},
    )


@torch.no_grad()
def fold_modal(model: nn.Module, modal: str) -> list[str]:
    """Fold e3nn-compatible modal linear layers for inference at a fixed modality."""
    index = model.modal_map[modal]
    changed = []
    for name, lin in model._modules.items():
        if type(lin) is not IrrepsLinear or lin.num_modalities <= 1:
            continue
        if not lin.layer_instantiated:
            continue
        old = lin.linear
        ref = old.weight
        irreps_in = lin._irreps_in_wo_modal
        new = _folded_modal_linear(lin).to(ref.device, ref.dtype)

        paths, offset = {}, 0
        for ins in old.instructions:
            if ins.i_in < 0:
                continue
            size = prod(ins.path_shape)
            paths[ins.i_in, ins.i_out] = (
                ref[offset : offset + size],
                ins.path_weight,
            )
            offset += size
        for _, ins, weights in new.weight_views(yield_instruction=True):
            values, scale = paths[ins.i_in, ins.i_out]
            weights.copy_(values.view_as(weights) * (scale / ins.path_weight))

        zero = ref.new_zeros((1, irreps_in.dim + lin.num_modalities))
        zero[0, irreps_in.dim + index] = 1
        bias = old(zero).squeeze(0)
        new.requires_grad_(False)
        lin.__class__ = ModalFoldedLinear
        lin.linear = new
        lin.register_buffer('modal_bias', bias)
        changed.append(name)
    return changed
