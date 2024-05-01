from typing import List

from e3nn.o3 import Irreps

import sevenn._keys as KEY
import sevenn._const as _const
from .convolution import IrrepsConvolution
from .equivariant_gate import EquivariantGate
from .linear import IrrepsLinear
from .self_connection import (
    SelfConnectionIntro,
    SelfConnectionLinearIntro,
    SelfConnectionOutro,
)


def init_self_connection(config):
    self_connection_type = config[KEY.SELF_CONNECTION_TYPE]
    intro, outro = None, None
    if self_connection_type == 'none':
        pass
    elif self_connection_type == 'nequip':
        intro, outro = SelfConnectionIntro, SelfConnectionOutro
        return SelfConnectionIntro, SelfConnectionOutro
    elif self_connection_type == 'linear':
        intro, outro = SelfConnectionLinearIntro, SelfConnectionOutro
    else:
        raise ValueError('something went wrong...')
    return intro, outro


def _NequIP_interaction_block(
    config, 
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_tp_out: Irreps,
    irreps_out: Irreps,
    weight_nn_layers: List[int],
    conv_denominator: float,
    t: int, 
    parallel=False
):
    if not hasattr(_NequIP_interaction_block, "act_scalar"):
        # TODO: Not beautiful
        act_scalar = {}
        act_gate = {}
        for k, v in config[KEY.ACTIVATION_SCARLAR].items():
            act_scalar[k] = _const.ACTIVATION_DICT[k][v]
        for k, v in config[KEY.ACTIVATION_GATE].items():
            act_gate[k] = _const.ACTIVATION_DICT[k][v]
        act_radial = _const.ACTIVATION[config[KEY.ACTIVATION_RADIAL]]
        _NequIP_interaction_block.act_scalar = act_scalar
        _NequIP_interaction_block.act_gate = act_gate
        _NequIP_interaction_block.act_radial = act_radial
    act_scalar = _NequIP_interaction_block.act_scalar
    act_gate = _NequIP_interaction_block.act_gate
    act_radial = _NequIP_interaction_block.act_radial

    block = {}
    num_species = config[KEY.NUM_SPECIES]
    irreps_node_attr = Irreps(f'{num_species}x0e')
    use_bias_in_linear = config[KEY.USE_BIAS_IN_LINEAR]
    train_avg_num_neigh = config[KEY.TRAIN_AVG_NUM_NEIGH]

    sc_intro, sc_outro = init_self_connection(config)

    gate_layer = EquivariantGate(irreps_out, act_scalar, act_gate)
    irreps_for_gate_in = gate_layer.get_gate_irreps_in()

    block[f'{t}_self_connection_intro'] = sc_intro(
        irreps_x=irreps_x,
        irreps_operand=irreps_node_attr,
        irreps_out=irreps_for_gate_in,
    )

    block[f'{t}_self_interaction_1'] = IrrepsLinear(
        irreps_x,
        irreps_x,
        data_key_in=KEY.NODE_FEATURE,
        biases=use_bias_in_linear,
    )

    # convolution part, l>lmax is droped as defined in irreps_out
    block[f'{t}_convolution'] = IrrepsConvolution(
        irreps_x=irreps_x,
        irreps_filter=irreps_filter,
        irreps_out=irreps_tp_out,
        data_key_weight_input=KEY.EDGE_EMBEDDING,
        weight_layer_input_to_hidden=weight_nn_layers,
        weight_layer_act=act_radial,
        denominator=conv_denominator,
        train_denominator=train_avg_num_neigh,
        is_parallel=parallel,
    )

    # irreps of x increase to gate_irreps_in
    block[f'{t}_self_interaction_2'] = IrrepsLinear(
        irreps_tp_out,
        irreps_for_gate_in,
        data_key_in=KEY.NODE_FEATURE,
        biases=use_bias_in_linear,
    )

    block[f'{t}_self_connection_outro'] = sc_outro()
    block[f'{t}_equivariant_gate'] = gate_layer

    return block
