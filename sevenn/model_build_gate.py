"""
Build of the isolation-gate model (model config architecture: isolation_gate,
sevenn/nn/isolation_gate.py). A config without the key, or with 'sevennet',
never reaches this file: model_build.build_E3_equivariant_model builds the
original SevenNet model (every pretrained checkpoint).

The modules are those of model_build.assemble_layers (same names, same state
dict), with the gate added on top:
- every interaction block: a block whose residual is not the identity (nequip
  block; metaformer with 'nequip' / 'linear' self connection) becomes one
  update x <- P(x) + s * (block(x) - P(x)) ({t}_gate_ref, {t}_gate_blend)
- every residual outro (SelfConnectionOutro, AddToResidual,
  MagnitudeResidualOutro) outside such a blend region: its gated subclass
  (branch * s); every IrrepsLinear: GatedIrrepsLinear (modal one-hot * s)
- node_gate (s_i) after edge_embedding (which then saves the envelope) and
  coord_number,
  node_reference (x_0) after onehot_to_feature_x
- final_norm between final_gate_ref / final_gate_blend
- readout R(x_L - x_0) without bias (readout_reference), per-layer readouts
  on x_t - x_0 ({t}_layer_reference)
- e3gnn/parallel: num_convolution_layer + 1 segments (to_parallel_model_gate)
"""
import copy
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from e3nn.o3 import Irreps

import sevenn._const as _const
import sevenn._keys as KEY

from .model_build import _ghost_aware_modules, assemble_layers, patch_modules
from .nn.coord_scale import CoordNumber
from .nn.edge_embedding import EdgeEmbedding
from .nn.interaction_blocks import NequIP_interaction_block
from .nn.isolation_gate import (
    GATED_RESIDUAL_CLASSES,
    GateBlend,
    GatedIrrepsLinear,
    GateReference,
    IsolationGate,
    NodeReference,
    ReferenceDelta,
    SegmentEdgeLink,
    SegmentPack,
    SegmentUnpack,
    residual_copy_index,
)
from .nn.layer_readout import (
    LayerEnergyAccumulate,
    LayerEnergyPack,
    LayerEnergyUnpack,
)
from .nn.linear import IrrepsLinear
from .nn.normalization import MagnitudeResidualOutro, SelfConnectionResidualIntro
from .nn.pair_repulsion import PairRepulsion
from .nn.radial_pair import EdgePairInput
from .nn.sequential import AtomGraphSequential

_REDUCE_KEYS = (
    'reduce_input_to_hidden',
    'reduce_hidden_activation',
    'reduce_hidden_to_energy',
    'readout_FCN',
)


def carry_reference(ref_map, irreps_in: Irreps, irreps_out: Irreps):
    """
    ref_map (see ReferenceDelta) after one identity residual irreps_in ->
    irreps_out: the columns of x_0 follow the channel copies of the layer.
    """
    irreps_in, irreps_out = Irreps(irreps_in), Irreps(irreps_out)
    if irreps_in == irreps_out:
        return ref_map
    src, dst = residual_copy_index(irreps_in, irreps_out)
    out = ref_map.new_full((irreps_out.dim,), -1)
    out[dst] = ref_map[src]
    return out


def _check_readout_activation(name: str) -> None:
    """R(0) = 0 needs act(0) = 0 (sevenn/nn/isolation_gate.py)"""
    if float(_const.ACTIVATION[name](torch.zeros(1))) != 0.0:
        raise ValueError(
            f'readout activation {name} has act(0) != 0: an isolated atom would '
            'keep a constant energy (use silu, relu, tanh, abs or elu)'
        )


def _insert(layers: OrderedDict, key: str, item: Tuple[str, Any], after: bool):
    """insert item before / after layers[key] (in place)"""
    items = list(layers.items())
    idx = [k for k, _ in items].index(key)
    items.insert(idx + 1 if after else idx, item)
    layers.clear()
    layers.update(items)


def _gate_block(block: Dict[str, Any], kw: Dict[str, Any], nequip: bool):
    """
    Interaction block t without an identity residual: the whole block is the
    update, between {t}_gate_ref (P(x), before the block) and {t}_gate_blend
    (after the nequip gate / the metaformer conv outro, before its FFN).
    """
    t = kw['t']
    key_x = kw.get('data_key_x', KEY.NODE_FEATURE)
    intro = block[f'{t}_self_connection_intro']
    if nequip:
        end = f'{t}_equivariant_gate'
    elif not isinstance(intro, SelfConnectionResidualIntro):
        end = f'{t}_self_connection_outro'
    else:
        return block  # identity residual: the outros are gated
    key_ref = f'{t}_gate_ref'
    out = {key_ref: GateReference(kw['irreps_x'], kw['irreps_out'], key_x, key_ref)}
    for k, m in block.items():
        out[k] = m
        if k == end:
            out[f'{t}_gate_blend'] = GateBlend(key_ref, key_x)
    assert f'{t}_gate_blend' in out, end
    return out


def _swap_classes(layers: OrderedDict) -> None:
    """
    Gated subclasses (same state): residual outros outside a GateBlend region,
    every IrrepsLinear (only its modal input differs)
    """
    region = None  # data key of the open GateReference
    for k, m in layers.items():
        if isinstance(m, GateReference):
            assert region is None, k
            region = m.key_out
        elif isinstance(m, GateBlend):
            assert m.key_ref == region, k
            region = None
        elif type(m) in GATED_RESIDUAL_CLASSES:
            if isinstance(m, MagnitudeResidualOutro):
                assert region is None, f'{k}: magnitude residual under GateBlend'
            if region is None:
                m.__class__ = GATED_RESIDUAL_CLASSES[type(m)]
        elif type(m) is IrrepsLinear:
            m.__class__ = GatedIrrepsLinear
    assert region is None


def _apply_gate(
    layers: OrderedDict,
    config: Dict[str, Any],
    record: List[Tuple[Irreps, Irreps]],
) -> None:
    num_conv = config[KEY.NUM_CONVOLUTION]
    assert len(record) == num_conv, record

    # s_i from the cutoff function's envelope, after edge_embedding
    edge_embedding = layers['edge_embedding']
    assert isinstance(edge_embedding, EdgeEmbedding)
    edge_embedding.save_components = True
    after = 'coord_number' if 'coord_number' in layers else 'edge_embedding'
    _insert(layers, after, ('node_gate', IsolationGate()), True)
    _insert(layers, 'onehot_to_feature_x', ('node_reference', NodeReference()), True)

    # columns of x_0 in x after every layer
    ref_map = torch.arange(Irreps(layers['onehot_to_feature_x'].irreps_out).dim)
    ref_maps = []
    for irreps_x, irreps_out in record:
        ref_map = carry_reference(ref_map, irreps_x, irreps_out)
        ref_maps.append(ref_map)
    irreps_final = record[-1][1]

    for t in range(num_conv - 1):  # per-layer readout on x_t - x_0
        key = f'{t}_layer_readout'
        if key not in layers:
            continue
        key_delta = f'{t}_layer_delta'
        ref = ReferenceDelta(ref_maps[t], key_delta)
        _insert(layers, key, (f'{t}_layer_reference', ref), False)
        layers[key].key_input = key_delta

    if 'final_norm' in layers:
        # the norm's bias and scale act without neighbours: gated as an update
        ref = GateReference(
            irreps_final, irreps_final, data_key_out='final_gate_ref'
        )
        _insert(layers, 'final_norm', ('final_gate_ref', ref), False)
        blend = GateBlend('final_gate_ref')
        _insert(layers, 'final_norm', ('final_gate_blend', blend), True)

    # readout R(x_L - x_0), R without bias (readout_FCN: e3nn, never a bias)
    keys = [k for k in layers if k in _REDUCE_KEYS]
    for k in keys:
        m = layers[k]
        if isinstance(m, IrrepsLinear):
            assert not m.layer_instantiated, k
            m.linear_kwargs['biases'] = False
    _insert(layers, keys[0], ('readout_reference', ReferenceDelta(ref_map)), False)

    _swap_classes(layers)


def to_parallel_model_gate(
    layers: OrderedDict, config: Dict[str, Any]
) -> List[OrderedDict]:
    """
    Split the serial isolation-gate model into the segments of LAMMPS
    e3gnn/parallel (sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp). Segment 0 runs
    up to 0_self_interaction_1, segment t >= 1 from (t-1)_convolution up to
    t_self_interaction_1 (the last one to the energy): num_convolution_layer + 1
    segments. As model_build._to_parallel_model, except:
    - layer 0: the convolution input of a ghost atom depends on its s_i
      (gated modal one-hot of 0_self_interaction_1, of onehot_to_feature_x with
      use_modal_node_embedding), and s_i needs the edges of that atom, which
      only its owner has: x is exchanged after 0_self_interaction_1 as in every
      other layer (no ghost twins). Segment 0 holds no convolution, so x and
      self_cont_tmp are linked to edge_vec by a zero (SegmentEdgeLink), LAMMPS
      needs their gradient w.r.t. it
    - node_gate is rebuilt at the start of every segment, with edge_embedding
      (and edge_pair_input, coord_number)
    - x_0 and the reference of a GateBlend cross every boundary inside
      self_cont_tmp (SegmentPack / SegmentUnpack)
    - pair repulsion: head of the last segment, for any number of layers
    Communication: one forward and one reverse exchange of the convolution
    input per layer t >= 0.
    """
    num_convolution_layer = config[KEY.NUM_CONVOLUTION]
    _layers = list(layers.items())
    layers_list = []

    def slice_until_this(module_name, layers):
        idx = [key for key, _ in layers].index(module_name)
        return layers[: idx + 1], layers[idx + 1 :]

    _layers = _ghost_aware_modules(_layers)

    edge_embedding = layers['edge_embedding']
    pair_input = [m for _, m in _layers if isinstance(m, EdgePairInput)]
    coord = [m for _, m in _layers if isinstance(m, CoordNumber)]
    layer_energy = any(isinstance(m, LayerEnergyAccumulate) for _, m in _layers)
    pair_rep = [(k, m) for k, m in _layers if isinstance(m, PairRepulsion)]
    if pair_rep:  # -> head of the last segment
        _layers = [(k, m) for k, m in _layers if not isinstance(m, PairRepulsion)]
    ref_dim = layers['onehot_to_feature_x'].irreps_out.dim
    for i in range(num_convolution_layer):
        sliced, _layers = slice_until_this(f'{i}_self_interaction_1', _layers)
        keys, dims = [KEY.NODE_REF], [ref_dim]
        for key, module in sliced:
            if key == f'{i}_gate_ref':  # read by {i}_gate_blend after the cut
                keys.append(module.key_out)
                dims.append(module.dim_out)
        sliced.append(('reference_pack', SegmentPack(keys)))
        if layer_energy and i > 0:  # per-layer energies inside self_cont_tmp
            sliced.append(('layer_energy_pack', LayerEnergyPack()))
        if i == 0:
            sliced.append(('edge_link', SegmentEdgeLink()))
        layers_list.append(OrderedDict(sliced))
        # rebuild position-dependent edge inputs from this segment's edge_vec
        head = [('edge_embedding', copy.deepcopy(edge_embedding))]
        if pair_input:
            head.append(('edge_pair_input', copy.deepcopy(pair_input[0])))
        head.append(('node_gate', IsolationGate()))
        if coord:
            head.append(('coord_number', copy.deepcopy(coord[0])))
        if layer_energy and i > 0:
            head.append(('layer_energy_unpack', LayerEnergyUnpack()))
        head.append(('reference_unpack', SegmentUnpack(keys, dims)))
        if pair_rep and i == num_convolution_layer - 1:
            head.append(pair_rep[0])
        _layers = head + _layers

    layers_list.append(OrderedDict(_layers))
    del layers_list[-1]['force_output']  # done in LAMMPS
    return layers_list


def build_isolation_gate_model(
    config: Dict[str, Any], parallel: bool = False
) -> Union[AtomGraphSequential, List[AtomGraphSequential]]:
    """model_build.build_E3_equivariant_model for architecture: isolation_gate"""
    if config[KEY.READOUT_AS_FCN] is False:
        if config.get(KEY.READOUT_ACTIVATION, None) is not None:
            _check_readout_activation(config[KEY.READOUT_ACTIVATION])
    else:
        _check_readout_activation(config[KEY.READOUT_FCN_ACTIVATION])

    record: List[Tuple[Irreps, Irreps]] = []  # (irreps_x, irreps_out) per layer

    def wrap_block(builder: Callable) -> Callable:
        nequip = builder is NequIP_interaction_block

        def gated_builder(**kw):
            block = builder(**kw)
            record.append((Irreps(kw['irreps_x']), Irreps(kw['irreps_out'])))
            return _gate_block(block, kw, nequip)

        return gated_builder

    layers, common_args = assemble_layers(config, parallel, wrap_block)
    _apply_gate(layers, config, record)

    if parallel:
        return [
            AtomGraphSequential(patch_modules(seg, config), **common_args)
            for seg in to_parallel_model_gate(layers, config)
        ]
    return AtomGraphSequential(patch_modules(layers, config), **common_args)
