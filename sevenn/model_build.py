import copy
import os
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Tuple, Type, Union, overload

from e3nn.o3 import Irreps

import sevenn._const as _const
import sevenn._keys as KEY
import sevenn.util as util

from .nn.convolution import IrrepsConvolution
from .nn.edge_embedding import (
    AgnesiTransform,
    BesselBasis,
    EdgeEmbedding,
    PolynomialCutoff,
    SphericalEncoding,
    XPLORCutoff,
)
from .nn.bilinear_gate import BilinearGate
from .nn.equivariant_gate import EquivariantGate
from .nn.force_output import ForceStressOutputFromEdge
from .nn.interaction_blocks import (
    Metaformer_interaction_block,
    NequIP_interaction_block,
)
from .nn.linear import AtomReduce, FCN_e3nn, IrrepsLinear, ScalarActivation
from .nn.node_embedding import GhostAtomTypes, OnehotEmbedding
from .nn.normalization import EquivariantRMSNorm, SelfConnectionResidualIntro
from .nn.radial_pair import EdgePairInput, NodeCondRadialInput, PairRadialMLP
from .nn.edge_attention import AttentionOutGate, EdgeAttention
from .nn.moe_ffn import type_to_expert_table
from .nn.scale import ModalWiseRescale, Rescale, SpeciesWiseRescale
from .nn.self_connection import (
    SelfConnectionIntro,
    SelfConnectionLinearIntro,
    SelfConnectionOutro,
)
from .nn.sequential import AtomGraphSequential

# warning from PyTorch, about e3nn type annotations
warnings.filterwarnings(
    'ignore',
    message=(
        "The TorchScript type system doesn't support instance-level annotations"
    ),
)


def _insert_after(module_name_after, key_module_pair, layers):
    idx = -1
    for i, (key, _) in enumerate(layers):
        if key == module_name_after:
            idx = i
            break
    if idx == -1:
        return layers  # do nothing if not found
    layers.insert(idx + 1, key_module_pair)
    return layers


def init_self_connection(config: Dict[str, Any]) -> List[Tuple[Type, Type]]:
    self_connection_type_list = config[KEY.SELF_CONNECTION_TYPE]
    num_conv = config[KEY.NUM_CONVOLUTION]
    if isinstance(self_connection_type_list, str):
        self_connection_type_list = [self_connection_type_list] * num_conv

    io_pair_list = []
    for sc_type in self_connection_type_list:
        if sc_type == 'none':
            io_pair = None
        elif sc_type == 'nequip':
            io_pair = SelfConnectionIntro, SelfConnectionOutro
        elif sc_type == 'linear':
            io_pair = SelfConnectionLinearIntro, SelfConnectionOutro
        elif sc_type == 'identity':
            if config.get(KEY.INTERACTION_TYPE, 'nequip') != 'metaformer':
                raise ValueError('identity self connection needs metaformer')
            io_pair = SelfConnectionResidualIntro, SelfConnectionOutro
        else:
            raise ValueError(f'Unknown self_connection_type found: {sc_type}')
        io_pair_list.append(io_pair)
    return io_pair_list


def _patch_pair_radial(layers: OrderedDict, num_species: int) -> None:
    """Element-pair radial weights for every convolution (in place)."""
    _layers = _insert_after(
        'onehot_idx_to_onehot',
        ('edge_pair_input', EdgePairInput()),
        list(layers.items()),
    )
    layers.clear()
    layers.update(_layers)
    for k, module in layers.items():
        if isinstance(module, IrrepsConvolution):
            module.key_weight_input = KEY.EDGE_PAIR_INPUT
            module.weight_nn_cls = PairRadialMLP
            module.weight_nn_kwargs['num_species'] = num_species


def _patch_radial_node_cond(layers: OrderedDict) -> None:
    """Node-state terms in every (element-pair) radial MLP (in place)."""
    items = list(layers.items())
    for k, conv in [(k, m) for k, m in items if isinstance(m, IrrepsConvolution)]:
        t = int(k.split('_')[0])
        hidden = conv.weight_nn_kwargs['hs'][1]
        mod = NodeCondRadialInput(conv.convolution_kwargs['irreps_in1'], hidden, t)
        items = _insert_after(
            f'{t}_self_interaction_1', (f'{t}_radial_node_cond', mod), items
        )
        conv.key_weight_input = mod.key_out
        conv.weight_nn_kwargs['node_cond'] = hidden
    layers.clear()
    layers.update(items)


def _patch_edge_attention(layers: OrderedDict, config: Dict[str, Any]) -> None:
    """Envelope-gated neighbour attention for every convolution (in place)."""
    opts = dict(config[KEY.CONV_ATTENTION])
    skip_first = opts.pop('skip_first', True)   # layer 0 keeps the plain sum
    num_radial = config[KEY.RADIAL_BASIS].get(KEY.BESSEL_BASIS_NUM, 8)
    items = list(layers.items())
    convs = [(k, m) for k, m in items if isinstance(m, IrrepsConvolution)]
    for k, conv in convs:
        t = int(k.split('_')[0])
        if skip_first and t == 0:
            continue
        irreps_x = conv.convolution_kwargs['irreps_in1']
        irreps_msg = conv.convolution_kwargs['irreps_out']
        attn = EdgeAttention(
            irreps_x, irreps_msg, num_radial, config[KEY.CUTOFF],
            config[KEY.NUM_SPECIES], t, **opts,
        )
        items = _insert_after(
            f'{t}_self_interaction_1', (f'{t}_edge_attention', attn), items
        )
        if attn.use_gate:
            items = _insert_after(
                k, (f'{t}_attn_out_gate', AttentionOutGate(irreps_msg, t)), items
            )
        conv.key_filter = attn.key_filter_out
        conv.denominator.data.fill_(1.0)   # attention weights already normalize
    layers.clear()
    layers.update(items)


def init_edge_embedding(config: Dict[str, Any]) -> EdgeEmbedding:
    _cutoff_param = {'cutoff_length': config[KEY.CUTOFF]}
    rbf, env, sph = None, None, None

    rbf_dct = copy.deepcopy(config[KEY.RADIAL_BASIS])
    rbf_dct.update(_cutoff_param)
    rbf_name = rbf_dct.pop(KEY.RADIAL_BASIS_NAME)
    if rbf_name == 'bessel':
        rbf = BesselBasis(**rbf_dct)

    envelop_dct = copy.deepcopy(config[KEY.CUTOFF_FUNCTION])
    envelop_dct.update(_cutoff_param)
    envelop_name = envelop_dct.pop(KEY.CUTOFF_FUNCTION_NAME)
    if envelop_name == 'poly_cut':
        env = PolynomialCutoff(**envelop_dct)
    elif envelop_name == 'XPLOR':
        env = XPLORCutoff(**envelop_dct)

    lmax_edge = config[KEY.LMAX]
    if config[KEY.LMAX_EDGE] > 0:
        lmax_edge = config[KEY.LMAX_EDGE]
    parity = -1 if config[KEY.IS_PARITY] else 1
    _normalize_sph = config[KEY._NORMALIZE_SPH]
    sph = SphericalEncoding(lmax_edge, parity, normalize=_normalize_sph)

    transform = None
    if config.get(KEY.DISTANCE_TRANSFORM, None) == 'agnesi':
        transform = AgnesiTransform(config[KEY.TYPE_MAP])
    return EdgeEmbedding(
        basis_module=rbf,
        cutoff_module=env,
        spherical_module=sph,
        save_components=bool(
            config.get(KEY.RADIAL_PAIR_EMBEDDING, False)
            or config.get(KEY.CONV_ATTENTION, None) is not None
        ),
        distance_transform=transform,
    )


def init_feature_reduce(config: Dict[str, Any], irreps_x: Irreps) -> OrderedDict:
    # features per node to scalar per node
    layers = OrderedDict()
    if config[KEY.READOUT_AS_FCN] is False:
        hidden_irreps = Irreps([(irreps_x.dim // 2, (0, 1))])
        layers.update(
            {
                'reduce_input_to_hidden': IrrepsLinear(
                    irreps_x,
                    hidden_irreps,
                    data_key_in=KEY.NODE_FEATURE,
                    biases=config[KEY.USE_BIAS_IN_LINEAR],
                ),
            }
        )
        # optional nonlinearity keeps the modal-aware reduce_input_to_hidden
        readout_act = config.get(KEY.READOUT_ACTIVATION, None)
        if readout_act is not None:
            layers['reduce_hidden_activation'] = ScalarActivation(
                _const.ACTIVATION[readout_act]
            )
        layers.update(
            {
                'reduce_hidden_to_energy': IrrepsLinear(
                    hidden_irreps,
                    Irreps([(1, (0, 1))]),
                    data_key_in=KEY.NODE_FEATURE,
                    data_key_out=KEY.SCALED_ATOMIC_ENERGY,
                    biases=config[KEY.USE_BIAS_IN_LINEAR],
                ),
            }
        )
    else:
        act = _const.ACTIVATION[config[KEY.READOUT_FCN_ACTIVATION]]
        hidden_neurons = config[KEY.READOUT_FCN_HIDDEN_NEURONS]
        layers.update(
            {
                'readout_FCN': FCN_e3nn(
                    dim_out=1,
                    hidden_neurons=hidden_neurons,
                    activation=act,
                    data_key_in=KEY.NODE_FEATURE,
                    data_key_out=KEY.SCALED_ATOMIC_ENERGY,
                    irreps_in=irreps_x,
                )
            }
        )
    return layers


def init_shift_scale(
    config: Dict[str, Any],
) -> Union[Rescale, SpeciesWiseRescale, ModalWiseRescale]:
    # for mm, ex, shift: modal_idx -> shifts
    shift_scale = []
    train_shift = config.get(KEY.TRAIN_SHIFT, False)
    train_scale = config.get(KEY.TRAIN_SCALE, False)

    # Legacy: train_shift_scale overrides both
    # TODO: log this as legacy warning
    train_shift_scale = config.get(KEY.TRAIN_SHIFT_SCALE, False)
    if train_shift_scale:
        train_shift = True
        train_scale = True
    type_map = config[KEY.TYPE_MAP]

    # in case of modal, shift or scale has more dims [][]
    # correct typing (I really want static python)
    for s in (config[KEY.SHIFT], config[KEY.SCALE]):
        if hasattr(s, 'tolist'):  # numpy or torch
            s = s.tolist()
        if isinstance(s, dict):
            s = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in s.items()}
        if isinstance(s, list) and len(s) == 1:
            s = s[0]
        shift_scale.append(s)
    shift, scale = shift_scale

    ss_kwargs = {'train_shift': train_shift, 'train_scale': train_scale}
    rescale_module = None
    if config.get(KEY.USE_MODALITY, False):
        rescale_module = ModalWiseRescale.from_mappers(  # type: ignore
            shift,
            scale,
            config[KEY.USE_MODAL_WISE_SHIFT],
            config[KEY.USE_MODAL_WISE_SCALE],
            type_map=type_map,
            modal_map=config[KEY.MODAL_MAP],
            **ss_kwargs,
        )
    elif all([isinstance(s, float) for s in shift_scale]):
        rescale_module = Rescale(shift, scale, **ss_kwargs)
    elif any([isinstance(s, list) for s in shift_scale]):
        rescale_module = SpeciesWiseRescale.from_mappers(  # type: ignore
            shift, scale, type_map=type_map, **ss_kwargs
        )
    else:
        raise ValueError('shift, scale should be list of float or float')

    return rescale_module


def patch_modality(layers: OrderedDict, config: Dict[str, Any]) -> OrderedDict:
    """
    Postprocess 7net-model to multimodal model.
    1. prepend modality one-hot embedding layer
    2. patch modalities of IrrepsLinear layers
    Modality aware shift scale is handled by init_shift_scale, not here
    """
    cfg = config
    if not cfg.get(KEY.USE_MODALITY, False):
        return layers

    _layers = list(layers.items())
    _layers = _insert_after(
        'onehot_idx_to_onehot',
        (
            'one_hot_modality',
            OnehotEmbedding(
                num_classes=config[KEY.NUM_MODALITIES],
                data_key_x=KEY.MODAL_TYPE,
                data_key_out=KEY.MODAL_ATTR,
                data_key_save=None,
                data_key_additional=None,
            ),
        ),
        _layers,
    )
    layers = OrderedDict(_layers)

    num_modal = config[KEY.NUM_MODALITIES]
    for k, module in layers.items():
        if not isinstance(module, IrrepsLinear):
            continue
        if (
            (cfg[KEY.USE_MODAL_NODE_EMBEDDING] and k.endswith('onehot_to_feature_x'))
            or (
                cfg[KEY.USE_MODAL_SELF_INTER_INTRO]
                and k.endswith('self_interaction_1')
            )
            or (
                cfg[KEY.USE_MODAL_SELF_INTER_OUTRO]
                and k.endswith('self_interaction_2')
            )
            or (cfg[KEY.USE_MODAL_OUTPUT_BLOCK] and k == 'reduce_input_to_hidden')
        ):
            module.set_num_modalities(num_modal)
    return layers


def patch_cue(layers: OrderedDict, config: Dict[str, Any]) -> OrderedDict:
    cue_cfg = copy.deepcopy(config.get(KEY.CUEQUIVARIANCE_CONFIG, {}))
    if not cue_cfg.pop('use', False):
        return layers

    import sevenn.nn.cue_helper as cue_helper

    if not cue_helper.is_cue_available():
        warnings.warn(
            (
                'cuEquivariance is requested, but the package is not installed. '
                + 'Fallback to e3nn.'
            )
        )
        return layers

    # TODO: merge into cue_ava func
    if not cue_helper.is_cue_cuda_available_model(config):
        return layers

    use_scatter_fusion = (
        os.environ.get('CUEQ_USE_SCATTER_FUSION')
        or cue_cfg.pop('use_scatter_fusion', True)
    )
    if isinstance(use_scatter_fusion, str):
        use_scatter_fusion = use_scatter_fusion.lower() in ('1', 'true', 'yes')

    tp_method = (
        os.environ.get('CUEQ_TP_METHOD')
        or cue_cfg.pop('cueq_tp_method', 'uniform_1d')
    )
    assert tp_method in ('uniform_1d', 'naive', 'fused_tp', 'indexed_linear')

    group = 'O3' if config[KEY.IS_PARITY] else 'SO3'
    cueq_patch_kwargs = dict(layout='mul_ir')
    cueq_patch_kwargs.update(cue_cfg)
    updates = {}
    for k, module in layers.items():
        # TODO: based on benchmark on A100 GPU & cuEq 0.4.0. (250307)
        if isinstance(module, (IrrepsLinear, SelfConnectionLinearIntro)):
            continue
            """
            if k == 'reduce_hidden_to_energy':  # TODO: has bug with 0 shape
                continue
            module_patched = cue_helper.patch_linear(
                module, group, **cueq_patch_kwargs
            )
            updates[k] = module_patched
            """
        elif isinstance(module, SelfConnectionIntro):
            continue
            """
            module_patched = cue_helper.patch_fully_connected(
                module, group, **cueq_patch_kwargs
            )
            updates[k] = module_patched
            """
        elif isinstance(module, IrrepsConvolution):
            module_patched = cue_helper.patch_convolution(
                module,
                group,
                use_scatter_fusion=use_scatter_fusion,
                tp_method=tp_method,
                **cueq_patch_kwargs,
            )
            updates[k] = module_patched

    layers.update(updates)
    return layers


def patch_flash_tp(layers: OrderedDict, config: Dict[str, Any]) -> OrderedDict:
    import sevenn.nn.flash_helper as flash_helper

    if not config.get('use_flash_tp', False):
        return layers

    if not flash_helper.is_flash_available():
        warnings.warn(
            (
                'FlashTP is requested, but the package is not installed '
                + 'or GPU not available. Fallback to e3nn.'
            )
        )
        return layers

    # sevenn/checkpoint.py::build_model
    _flash_lammps = config.get('_flash_lammps', False)
    updates = {}
    for k, module in layers.items():
        if isinstance(module, IrrepsConvolution):
            updates[k] = flash_helper.patch_convolution(module, _flash_lammps)

    layers.update(updates)
    return layers


def patch_oeq(layers: OrderedDict, config: Dict[str, Any]) -> OrderedDict:
    import sevenn.nn.oeq_helper as oeq_helper

    if not config.get(KEY.USE_OEQ, False):
        return layers

    if not oeq_helper.is_oeq_available():
        warnings.warn(
            (
                'OpenEquivariance (oeq) is requested, but the package is not '
                'installed or GPU not available. Fallback to e3nn.'
            )
        )
        return layers

    updates = {}
    for k, module in layers.items():
        if isinstance(module, IrrepsConvolution):
            updates[k] = oeq_helper.patch_convolution(module)

    layers.update(updates)
    return layers


def patch_lunatp(layers: OrderedDict, config: Dict[str, Any]) -> OrderedDict:
    import sevenn.nn.lunatp_helper as lunatp_helper

    if not config.get(KEY.USE_LUNATP, False):
        return layers

    # raises ImportError if LunaTP is not available, no fallback to e3nn
    # edge attention: one LunaTP graph and one attention envelope per forward
    # (same function, see lunatp_helper); False: per-layer graph and envelope
    fast_attn = config.get(KEY.LUNATP_FAST_ATTENTION, True)
    updates = {}
    for k, module in layers.items():
        if isinstance(module, IrrepsConvolution):
            updates[k] = lunatp_helper.patch_convolution(module, fast_attn)
        elif isinstance(module, EdgeAttention) and fast_attn:
            updates[k] = lunatp_helper.patch_edge_attention(module)
        elif isinstance(module, (IrrepsLinear, SelfConnectionLinearIntro)):
            updates[k] = lunatp_helper.patch_linear(module)
        elif isinstance(module, SelfConnectionIntro):
            updates[k] = lunatp_helper.patch_fully_connected(module)
        elif isinstance(module, EquivariantGate):
            updates[k] = lunatp_helper.patch_gate(module)
        elif isinstance(module, EquivariantRMSNorm) and config.get(
            KEY.LUNATP_FUSED_NORM, True
        ):
            # same function as the PyTorch norm (fp32 rounding differs)
            updates[k] = lunatp_helper.patch_rms_norm(module)
        elif isinstance(module, BilinearGate):
            updates[k] = lunatp_helper.patch_bilinear_gate(module)

    layers.update(updates)
    return layers


def patch_modules(layers: OrderedDict, config: Dict[str, Any]) -> OrderedDict:
    layers = patch_modality(layers, config)
    layers = patch_cue(layers, config)
    layers = patch_flash_tp(layers, config)
    layers = patch_oeq(layers, config)
    layers = patch_lunatp(layers, config)
    return layers


_SELF_CONNECTION_INTROS = (
    SelfConnectionIntro,
    SelfConnectionLinearIntro,
    SelfConnectionResidualIntro,
)


def _ghost_twin(name: str, module):
    """
    Copy of a node-wise layer-0 module acting on the ghost features
    (KEY.NODE_FEATURE_GHOST). deploy copies the weights of `name` into it
    (every 'ghost_<name>' entry, sevenn/scripts/deploy.py).
    """
    twin = copy.deepcopy(module)
    if isinstance(module, IrrepsLinear):
        assert module.key_input == module.key_output == KEY.NODE_FEATURE, name
        twin.key_input = twin.key_output = KEY.NODE_FEATURE_GHOST
    elif isinstance(module, EquivariantRMSNorm):
        assert module.key_x == KEY.NODE_FEATURE, name
        twin.key_x = KEY.NODE_FEATURE_GHOST
    else:
        raise NotImplementedError(
            f'e3gnn/parallel: {name} ({type(module).__name__}) transforms the '
            'layer-0 node features before the first convolution, but has no '
            'ghost twin (add it to model_build._ghost_twin)'
        )
    return twin


def _to_parallel_model(
    layers: OrderedDict, config: Dict[str, Any]
) -> List[OrderedDict]:
    """
    Split the serial model into the segments of LAMMPS e3gnn/parallel
    (sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp). Segment 0 runs up to
    1_self_interaction_1, segment t >= 1 from t_convolution up to
    (t+1)_self_interaction_1 (the last one to the energy). Between segments
    LAMMPS fills x_ghost with the owners' x (forward_comm), and back-propagates
    every segment on its own w.r.t. (edge_vec, x, self_cont_tmp, x_ghost),
    sending the ghost gradients back (reverse_comm). The same math as the
    serial model therefore needs:
    - layer 0: the ghost features come from the ghost types (no exchange)
      through 'ghost_*' twins of every node-wise module from
      onehot_to_feature_x to 0_self_interaction_1 (e.g. metaformer 0_norm_1)
    - every position-dependent input of a segment is rebuilt from that
      segment's edge_vec: edge_embedding and (pair radial) edge_pair_input
      are inserted at the start of every segment; a tensor cached from an
      earlier segment would lose its force contribution
    - per-edge type lookups (pair radial, attention bias, Agnesi) of ghost
      neighbours read KEY.ATOM_TYPE_ALL (local + ghost types, segment 0)
    - edge modules gathering source-node features (attention keys,
      node-conditioned radial MLP) also read x_ghost
    - self_cont_tmp must not be an ancestor of x at a segment boundary
      (identity residual: store a copy)
    Communication is unchanged: one forward and one reverse exchange of the
    convolution input per layer t >= 1.
    """
    num_classes = layers['onehot_idx_to_onehot'].num_classes
    num_convolution_layer = config[KEY.NUM_CONVOLUTION]
    _layers = list(layers.items())
    layers_list = []

    def slice_until_this(module_name, layers):
        idx = [key for key, _ in layers].index(module_name)
        return layers[: idx + 1], layers[idx + 1 :]

    # layer 0: ghost twins of the node-wise modules before the convolution
    names = [key for key, _ in _layers]
    i0, i1 = names.index('onehot_to_feature_x'), names.index('0_self_interaction_1')
    for key, module in list(_layers[i0 : i1 + 1]):
        if isinstance(module, _SELF_CONNECTION_INTROS):
            continue  # residual of local atoms only
        _layers = _insert_after(
            key, (f'ghost_{key}', _ghost_twin(key, module)), _layers
        )
    _layers = _insert_after(
        'onehot_to_feature_x',
        (
            'one_hot_ghost',
            OnehotEmbedding(
                data_key_x=KEY.NODE_FEATURE_GHOST,
                num_classes=num_classes,
                data_key_save=None,
                data_key_additional=None,
            ),
        ),
        _layers,
    )

    # ghost-aware edge modules
    need_types = False
    for key, module in _layers:
        if isinstance(module, EdgeEmbedding):
            if module.distance_transform is not None:
                module.key_node_type = KEY.ATOM_TYPE_ALL
                need_types = True
        elif isinstance(module, EdgePairInput):
            module.key_atom_type = KEY.ATOM_TYPE_ALL
            need_types = True
        elif isinstance(module, EdgeAttention):
            module.key_atom_type = KEY.ATOM_TYPE_ALL
            module.is_parallel = True
            need_types = need_types or module.use_rb
        elif isinstance(module, NodeCondRadialInput):
            module.is_parallel = True
        elif isinstance(module, SelfConnectionResidualIntro):
            if key.endswith('_self_connection_intro'):
                module.clone_identity = True
    if need_types:
        _layers.insert(0, ('atom_type_all', GhostAtomTypes()))

    edge_embedding = layers['edge_embedding']
    pair_input = [m for _, m in _layers if isinstance(m, EdgePairInput)]
    for i in range(1, num_convolution_layer):
        sliced, _layers = slice_until_this(f'{i}_self_interaction_1', _layers)
        layers_list.append(OrderedDict(sliced))
        # rebuild position-dependent edge inputs from this segment's edge_vec
        head = [('edge_embedding', copy.deepcopy(edge_embedding))]
        if pair_input:
            head.append(('edge_pair_input', copy.deepcopy(pair_input[0])))
        _layers = head + _layers

    layers_list.append(OrderedDict(_layers))
    del layers_list[-1]['force_output']  # done in LAMMPS
    return layers_list


@overload
def build_E3_equivariant_model(
    config: dict, parallel: Literal[False] = False
) -> AtomGraphSequential:  # noqa
    ...


@overload
def build_E3_equivariant_model(
    config: dict, parallel: Literal[True]
) -> List[AtomGraphSequential]:  # noqa
    ...


def build_E3_equivariant_model(
    config: dict, parallel: bool = False
) -> Union[AtomGraphSequential, List[AtomGraphSequential]]:
    """
    output shapes (w/o batch)

    PRED_TOTAL_ENERGY: (),
    ATOMIC_ENERGY: (natoms, 1),  # intended
    PRED_FORCE: (natoms, 3),
    PRED_STRESS: (6,),

    for data w/o cell volume, pred_stress has garbage values
    """
    layers = OrderedDict()

    cutoff = config[KEY.CUTOFF]
    num_species = config[KEY.NUM_SPECIES]
    feature_multiplicity = config[KEY.NODE_FEATURE_MULTIPLICITY]
    num_convolution_layer = config[KEY.NUM_CONVOLUTION]
    interaction_type = config[KEY.INTERACTION_TYPE]
    use_bias_in_linear = config[KEY.USE_BIAS_IN_LINEAR]

    lmax_node = config[KEY.LMAX]  # ignore second (lmax_edge)
    # if config[KEY.LMAX_EDGE] > 0:  # not yet used
    #     _ = config[KEY.LMAX_EDGE]
    if config[KEY.LMAX_NODE] > 0:
        lmax_node = config[KEY.LMAX_NODE]

    act_radial = _const.ACTIVATION[config[KEY.ACTIVATION_RADIAL]]
    self_connection_pair_list = init_self_connection(config)

    irreps_manual = None
    if config[KEY.IRREPS_MANUAL] is not False:
        irreps_manual = config[KEY.IRREPS_MANUAL]
        try:
            irreps_manual = [Irreps(irr) for irr in irreps_manual]
            assert len(irreps_manual) == num_convolution_layer + 1
        except Exception:
            raise RuntimeError('invalid irreps_manual input given')

    conv_denominator = config[KEY.CONV_DENOMINATOR]
    if not isinstance(conv_denominator, list):
        conv_denominator = [conv_denominator] * num_convolution_layer
    train_conv_denominator = config[KEY.TRAIN_DENOMINTAOR]

    edge_embedding = init_edge_embedding(config)
    irreps_filter = edge_embedding.spherical.irreps_out
    radial_basis_num = edge_embedding.basis_function.num_basis
    layers.update({'edge_embedding': edge_embedding})

    one_hot_irreps = Irreps(f'{num_species}x0e')
    irreps_x = (
        Irreps(f'{feature_multiplicity}x0e')
        if irreps_manual is None
        else irreps_manual[0]
    )

    layers.update(
        {
            'onehot_idx_to_onehot': OnehotEmbedding(
                num_classes=num_species,
                data_key_x=KEY.NODE_FEATURE,
                data_key_out=KEY.NODE_FEATURE,
                data_key_save=KEY.ATOM_TYPE,  # atomic numbers
                data_key_additional=KEY.NODE_ATTR,  # one-hot embeddings
            ),
            'onehot_to_feature_x': IrrepsLinear(
                irreps_in=one_hot_irreps,
                irreps_out=irreps_x,
                data_key_in=KEY.NODE_FEATURE,
                biases=use_bias_in_linear,
            ),
        }
    )

    weight_nn_hidden = config[KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS]
    weight_nn_layers = [radial_basis_num] + weight_nn_hidden

    param_interaction_block = {
        'gate_type': config.get(KEY.GATE_TYPE, 'gate'),
        'bilinear_paths': config.get(KEY.BILINEAR_PATHS, 'all'),
        'bilinear_b': config.get(KEY.BILINEAR_B, 'own'),
        'irreps_filter': irreps_filter,
        'weight_nn_layers': weight_nn_layers,
        'train_conv_denominator': train_conv_denominator,
        'act_radial': act_radial,
        'bias_in_linear': use_bias_in_linear,
        'num_species': num_species,
        'parallel': parallel,
    }

    interaction_builder = None

    if interaction_type in ['nequip', 'metaformer']:
        act_scalar = {}
        act_gate = {}
        for k, v in config[KEY.ACTIVATION_SCARLAR].items():
            act_scalar[k] = _const.ACTIVATION_DICT[k][v]
        for k, v in config[KEY.ACTIVATION_GATE].items():
            act_gate[k] = _const.ACTIVATION_DICT[k][v]
        param_interaction_block.update(
            {
                'act_scalar': act_scalar,
                'act_gate': act_gate,
            }
        )

    if interaction_type == 'nequip':
        interaction_builder = NequIP_interaction_block
    elif interaction_type == 'metaformer':
        interaction_builder = Metaformer_interaction_block
        param_interaction_block.update(
            {
                'use_ffn': True,  # set per layer below
                'ffn_hidden_scale': config.get(KEY.FFN_HIDDEN_SCALE, 2),
                'norm_eps': config.get(KEY.NORM_EPS, 1e-5),
                'use_norm': config.get(KEY.METAFORMER_NORM, True),
                'ffn_type': config.get(KEY.FFN_TYPE, 'gate'),
                'ffn_s2_channels': config.get(KEY.FFN_S2_CHANNELS, None),
                'layer_scale_init': config.get(KEY.LAYER_SCALE_INIT, None),
                'ffn_film': config.get(KEY.FFN_FILM, False),
                'ffn_moe_table': (
                    type_to_expert_table(config[KEY.TYPE_MAP], num_species)
                    if config.get(KEY.FFN_MOE, False) else None
                ),
            }
        )
    else:
        raise ValueError(f'Unknown interaction type: {interaction_type}')

    irreps_conv_list = config.get(KEY.IRREPS_CONV, None)  # None or per layer
    if irreps_conv_list:
        assert len(irreps_conv_list) == num_convolution_layer
    use_ffn = config.get(KEY.USE_FFN, True)  # bool or one bool per layer
    if not isinstance(use_ffn, list):
        use_ffn = [use_ffn] * num_convolution_layer
    assert len(use_ffn) == num_convolution_layer, 'use_ffn: one entry per layer'
    for t in range(num_convolution_layer):
        param_interaction_block.update(
            {
                'irreps_x': irreps_x,
                't': t,
                'conv_denominator': conv_denominator[t],
                'self_connection_pair': self_connection_pair_list[t],
            }
        )
        if interaction_type == 'metaformer':
            param_interaction_block['use_ffn'] = bool(use_ffn[t])
        if interaction_type in ['nequip', 'metaformer']:
            parity_mode = 'full'
            fix_multiplicity = False
            if t == num_convolution_layer - 1:
                lmax_node = 0
                parity_mode = 'even'
            # TODO: irreps_manual is applicable to both irreps_out_tp and irreps_out
            irreps_out = (
                util.infer_irreps_out(
                    irreps_x,  # type: ignore
                    irreps_filter,
                    lmax_node,  # type: ignore
                    parity_mode,
                    fix_multiplicity=feature_multiplicity,
                )
                if irreps_manual is None
                else irreps_manual[t + 1]
            )
            irreps_conv_t = (
                Irreps(irreps_conv_list[t]) if irreps_conv_list else irreps_x
            )
            irreps_out_tp = util.infer_irreps_out(
                irreps_conv_t,  # type: ignore
                irreps_filter,
                irreps_out.lmax,  # type: ignore
                parity_mode,
                fix_multiplicity,
            )
            if config.get(KEY.DROP_UNUSED_TP_IRREPS, False):
                # self_interaction_2 cannot map irreps absent in irreps_out
                used = {ir for _, ir in Irreps(irreps_out)}
                irreps_out_tp = Irreps(
                    [(mul, ir) for mul, ir in irreps_out_tp if ir in used]
                )
        else:
            raise ValueError(f'Unknown interaction type: {interaction_type}')
        param_interaction_block.update(
            {
                'irreps_out_tp': irreps_out_tp,
                'irreps_out': irreps_out,
                'irreps_conv': irreps_conv_t if irreps_conv_list else None,
            }
        )
        layers.update(interaction_builder(**param_interaction_block))
        irreps_x = irreps_out

    if config.get(KEY.RADIAL_PAIR_EMBEDDING, False):
        _patch_pair_radial(layers, num_species)
        if config.get(KEY.RADIAL_NODE_COND, False):
            _patch_radial_node_cond(layers)

    if config.get(KEY.CONV_ATTENTION, None) is not None:
        _patch_edge_attention(layers, config)

    if config.get(KEY.USE_FINAL_NORM, False):
        layers['final_norm'] = EquivariantRMSNorm(
            irreps_x, eps=config.get(KEY.NORM_EPS, 1e-5)
        )
    layers.update(init_feature_reduce(config, irreps_x))  # type: ignore

    layers.update(
        {
            'rescale_atomic_energy': init_shift_scale(config),
            'reduce_total_enegy': AtomReduce(
                data_key_in=KEY.ATOMIC_ENERGY,
                data_key_out=KEY.PRED_TOTAL_ENERGY,
            ),
        }
    )

    gradient_module = ForceStressOutputFromEdge()
    grad_key = gradient_module.get_grad_key()
    layers.update({'force_output': gradient_module})

    common_args = {
        'cutoff': cutoff,
        'type_map': config[KEY.TYPE_MAP],
        'modal_map': config.get(KEY.MODAL_MAP, None),
        'eval_type_map': False if parallel else True,
        'eval_modal_map': False
        if not config.get(KEY.USE_MODALITY, False) or parallel
        else True,
        'data_key_grad': grad_key,
    }

    if parallel:
        layers_list = _to_parallel_model(layers, config)
        return [
            AtomGraphSequential(patch_modules(layers, config), **common_args)
            for layers in layers_list
        ]
    else:
        return AtomGraphSequential(patch_modules(layers, config), **common_args)
