"""Project an existing e3nn checkpoint onto the compact convolution layout."""

from math import prod

import torch


def _tp_paths(tp):
    offset = 0
    for ins in tp.instructions:
        if not ins.has_weight:
            raise ValueError('Expected an external radial weight for each TP path')
        size = prod(ins.path_shape)
        key = (ins.i_in1, ins.i_in2, tp.irreps_out[ins.i_out], ins.connection_mode)
        yield key, slice(offset, offset + size), ins.path_weight
        offset += size
    if offset != tp.weight_numel:
        raise ValueError('Unexpected tensor-product weight layout')


def _linear_paths(linear):
    return [
        (
            linear.irreps_in[ins.i_in] if ins.i_in >= 0 else None,
            ins.i_out,
            ins.path_shape,
            ins.path_weight,
        )
        for ins in linear.instructions
    ]


def compact_convolution_state(source, target):
    """Keep the radial columns for the TP paths present in the compact model."""
    projected = {}
    regenerated = []
    for name, conv in target._modules.items():
        if not name.endswith('_convolution'):
            continue
        old_conv = source._modules[name]
        old_tp, tp = old_conv.convolution, conv.convolution
        if tp.irreps_out == old_tp.irreps_out:
            continue

        linear_name = name.replace('convolution', 'self_interaction_2')
        old_linear = source._modules[linear_name].linear
        linear = target._modules[linear_name].linear
        if _linear_paths(old_linear) != _linear_paths(linear):
            raise ValueError(f'{linear_name}: changed surviving linear paths')

        paths = {key: (columns, scale) for key, columns, scale in _tp_paths(old_tp)}
        if len(paths) != len(old_tp.instructions):
            raise ValueError(f'{name}: duplicate tensor-product paths')
        old_last = old_conv.weight_nn[-1].weight.detach()
        weights = []
        for key, columns, scale in _tp_paths(tp):
            old_columns, old_scale = paths[key]
            if columns.stop - columns.start != old_columns.stop - old_columns.start:
                raise ValueError(f'{name}: changed surviving TP path shape')
            weights.append(old_last[:, old_columns] * (old_scale / scale))
        last = len(conv.weight_nn) - 1
        projected[f'{name}.weight_nn.layer{last}.weight'] = torch.cat(weights, dim=1)
        regenerated.append(f'{name}.convolution.')

    if not projected:
        return None
    original = source.state_dict()
    compact = target.state_dict()
    tp_prefixes = tuple(regenerated)
    for name, value in compact.items():
        # TP buffers are generated for its new irreps and instructions.
        if name.startswith(tp_prefixes):
            continue
        replacement = projected[name] if name in projected else original[name]
        if replacement.shape != value.shape:
            raise ValueError(f'{name}: inconsistent compact checkpoint shape')
        compact[name] = replacement
    return compact
