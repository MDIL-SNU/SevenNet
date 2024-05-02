from typing import List, Callable, Tuple

from e3nn.o3 import Irreps

import sevenn._keys as KEY
from .convolution import IrrepsConvolution
from .equivariant_gate import EquivariantGate
from .linear import IrrepsLinear
from .equivariant_product_basis import EquivariantProductBasis


def NequIP_interaction_block(
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_out_tp: Irreps,
    irreps_out: Irreps,
    weight_nn_layers: List[int],
    conv_denominator: float,
    train_conv_denominator: bool,
    self_connection_pair: Tuple[Callable, Callable],
    act_scalar: Callable,
    act_gate: Callable,
    act_radial: Callable,
    bias_in_linear: bool,
    num_species: int,
    t: int,   # interaction layer index
    data_key_x = KEY.NODE_FEATURE,
    data_key_weight_input = KEY.EDGE_EMBEDDING,
    parallel=False,
):
    block = {}
    irreps_node_attr = Irreps(f'{num_species}x0e')
    sc_intro, sc_outro = self_connection_pair

    gate_layer = EquivariantGate(irreps_out, act_scalar, act_gate)
    irreps_for_gate_in = gate_layer.get_gate_irreps_in()

    block[f'{t}_self_connection_intro'] = sc_intro(
        irreps_x=irreps_x,
        irreps_operand=irreps_node_attr,
        irreps_out=irreps_for_gate_in,
    )

    block[f'{t}_self_interaction_1'] = IrrepsLinear(
        irreps_x, irreps_x,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    # convolution part, l>lmax is droped as defined in irreps_out
    block[f'{t}_convolution'] = IrrepsConvolution(
        irreps_x=irreps_x,
        irreps_filter=irreps_filter,
        irreps_out=irreps_out_tp,
        data_key_weight_input=data_key_weight_input,
        weight_layer_input_to_hidden=weight_nn_layers,
        weight_layer_act=act_radial,
        denominator=conv_denominator,
        train_denominator=train_conv_denominator,
        is_parallel=parallel,
    )

    # irreps of x increase to gate_irreps_in
    block[f'{t}_self_interaction_2'] = IrrepsLinear(
        irreps_out_tp,
        irreps_for_gate_in,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    block[f'{t}_self_connection_outro'] = sc_outro()
    block[f'{t}_equivariant_gate'] = gate_layer

    return block


def MACE_interaction_block(
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_out_tp: Irreps,
    irreps_out: Irreps,
    correlation: int,
    weight_nn_layers: List[int],
    conv_denominator: float,
    train_conv_denominator: bool,
    self_connection_pair: Tuple[Callable, Callable],
    act_radial: Callable,
    bias_in_linear: bool,
    num_species: int,
    t: int,   # interaction layer index
    data_key_x = KEY.NODE_FEATURE,
    data_key_weight_input = KEY.EDGE_EMBEDDING,
    parallel=False,
):
    # parity shold be sph like
    assert(all([p == (-1)**l for _, (l, p) in irreps_out]))
    block = {}
    sc_intro, sc_outro = self_connection_pair

    feature_mul = irreps_out[0].mul
    # multiplicity should be all same
    assert(all([m == feature_mul for m, _ in irreps_out]))
    irreps_out_si2 = Irreps()
    for _, ir in irreps_out_tp:
        irreps_out_si2 += Irreps(f'{feature_mul}x{str(ir)}')

    irreps_node_attr = Irreps(f'{num_species}x0e')

    block[f'{t}_self_connection_intro'] = sc_intro(
        irreps_x=irreps_x,
        irreps_operand=irreps_node_attr,
        irreps_out=irreps_out,
    )
    block[f'{t}_self_interaction_1'] = IrrepsLinear(
        irreps_x, irreps_x,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )
    # convolution part, l>lmax is droped as defined in irreps_out
    block[f'{t}_convolution'] = IrrepsConvolution(
        irreps_x=irreps_x,
        irreps_filter=irreps_filter,
        irreps_out=irreps_out_tp,
        data_key_weight_input=data_key_weight_input,
        weight_layer_input_to_hidden=weight_nn_layers,
        weight_layer_act=act_radial,
        denominator=conv_denominator,
        train_denominator=train_conv_denominator,
        is_parallel=parallel,
    )
    block[f'{t}_self_interaction_2'] = IrrepsLinear(
        irreps_out_tp, irreps_out_si2,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )
    block[f'{t}_equivariant_product_basis'] = EquivariantProductBasis(
        irreps_x=irreps_out_si2,
        irreps_out=irreps_out,
        correlation=correlation,
        num_elements=num_species,
    )
    block[f'{t}_self_interaction_3'] = IrrepsLinear(
        irreps_out, irreps_out,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )
    block[f'{t}_self_connection_outro'] = sc_outro()
    return block

