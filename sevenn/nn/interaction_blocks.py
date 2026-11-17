import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from e3nn.o3 import Irreps
from torch.nn import Module

import sevenn._keys as KEY

from .convolution import IrrepsConvolution
from .bilinear_gate import BilinearGate
from .equivariant_gate import EquivariantGate
from .ffn_variants import ElementFiLM, LayerScale, S2BilinearGate, SwiGLUGate
from .linear import IrrepsLinear


def _gate(
    irreps_out: Irreps, act_scalar, act_gate, gate_type: str,
    paths: str = 'all', b: str = 'own',
):
    if gate_type == 'bilinear' and any(ir.l > 0 for _, ir in Irreps(irreps_out)):
        return BilinearGate(irreps_out, act_scalar, act_gate, paths=paths, b=b)
    return EquivariantGate(irreps_out, act_scalar, act_gate)


def NequIP_interaction_block(
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_out_tp: Irreps,
    irreps_out: Irreps,
    weight_nn_layers: List[int],
    conv_denominator: float,
    train_conv_denominator: bool,
    self_connection_pair: Tuple[Callable, Callable],
    act_scalar: Dict[str, Callable],
    act_gate: Dict[str, Callable],
    act_radial: Callable,
    bias_in_linear: bool,
    num_species: int,
    t: int,   # interaction layer index
    gate_type: str = 'gate',
    irreps_conv: Optional[Irreps] = None,
    bilinear_paths: str = 'all',
    bilinear_b: str = 'own',
    data_key_x: str = KEY.NODE_FEATURE,
    data_key_weight_input: str = KEY.EDGE_EMBEDDING,
    parallel: bool = False,
    **conv_kwargs,
) -> Dict[str, Module]:
    block = {}
    irreps_node_attr = Irreps(f'{num_species}x0e')
    sc_intro, sc_outro = self_connection_pair

    gate_layer = _gate(
        irreps_out, act_scalar, act_gate, gate_type, bilinear_paths, bilinear_b
    )
    irreps_for_gate_in = gate_layer.get_gate_irreps_in()

    block[f'{t}_self_connection_intro'] = sc_intro(
        irreps_x,
        irreps_operand=irreps_node_attr,
        irreps_out=irreps_for_gate_in,
    )

    # convolution may run on narrower irreps than the node features (MACE-MH-1)
    irreps_conv = irreps_x if irreps_conv is None else Irreps(irreps_conv)
    block[f'{t}_self_interaction_1'] = IrrepsLinear(
        irreps_x, irreps_conv,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )

    # convolution part, l>lmax is dropped as defined in irreps_out
    block[f'{t}_convolution'] = IrrepsConvolution(
        irreps_x=irreps_conv,
        irreps_filter=irreps_filter,
        irreps_out=irreps_out_tp,
        data_key_weight_input=data_key_weight_input,
        weight_layer_input_to_hidden=weight_nn_layers,
        weight_layer_act=act_radial,
        denominator=conv_denominator,
        train_denominator=train_conv_denominator,
        is_parallel=parallel,
        **conv_kwargs,
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


def Metaformer_interaction_block(
    irreps_x: Irreps,
    irreps_filter: Irreps,
    irreps_out_tp: Irreps,
    irreps_out: Irreps,
    weight_nn_layers: List[int],
    conv_denominator: float,
    train_conv_denominator: bool,
    self_connection_pair: Tuple[Callable, Callable],
    act_scalar: Dict[str, Callable],
    act_gate: Dict[str, Callable],
    act_radial: Callable,
    bias_in_linear: bool,
    num_species: int,
    t: int,   # interaction layer index
    use_ffn: bool = True,
    ffn_hidden_scale: int = 2,
    norm_eps: float = 1e-5,
    gate_type: str = 'gate',
    ffn_type: str = 'gate',
    layer_scale_init: Optional[float] = None,
    ffn_film: bool = False,
    irreps_conv: Optional[Irreps] = None,
    bilinear_paths: str = 'all',
    bilinear_b: str = 'own',
    ffn_moe_table: Optional[torch.Tensor] = None,
    ffn_s2_channels: Optional[int] = None,
    use_norm: bool = True,
    data_key_x: str = KEY.NODE_FEATURE,
    data_key_weight_input: str = KEY.EDGE_EMBEDDING,
    parallel: bool = False,
    **conv_kwargs,
) -> Dict[str, Module]:
    """
    Pre-norm residual block (eSEN, Fu et al. arXiv:2502.12147; layout as in
    EquFlashV2, github.com/SamsungDS/GGNN, idea only; pre-norm residuals per
    Xiong et al. arXiv:2002.04745):
        x = R(x) + Lin3(Gate(Lin2(Conv(Lin1(Norm1(x))))))
        x = x + FFN(Norm2(x)),  FFN = Lin(Gate(Lin(.)))  (if use_ffn)
    R is the residual of self_connection_pair (identity or linear).
    Module names self_interaction_1/2 are kept for modality patching.
    use_norm=False (config metaformer_norm: false, ablation): Norm1 and Norm2
    are left out (identity), everything else is unchanged.
    """
    from .normalization import EquivariantRMSNorm

    block = {}
    irreps_node_attr = Irreps(f'{num_species}x0e')
    sc_intro, sc_outro = self_connection_pair

    gate_layer = _gate(
        irreps_out, act_scalar, act_gate, gate_type, bilinear_paths, bilinear_b
    )
    irreps_for_gate_in = gate_layer.get_gate_irreps_in()

    block[f'{t}_self_connection_intro'] = sc_intro(
        irreps_x,
        irreps_operand=irreps_node_attr,
        irreps_out=irreps_out,
    )
    if use_norm:
        block[f'{t}_norm_1'] = EquivariantRMSNorm(
            irreps_x, data_key_x, eps=norm_eps
        )
    # convolution may run on narrower irreps than the node features (MACE-MH-1)
    irreps_conv = irreps_x if irreps_conv is None else Irreps(irreps_conv)
    block[f'{t}_self_interaction_1'] = IrrepsLinear(
        irreps_x, irreps_conv,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )
    block[f'{t}_convolution'] = IrrepsConvolution(
        irreps_x=irreps_conv,
        irreps_filter=irreps_filter,
        irreps_out=irreps_out_tp,
        data_key_weight_input=data_key_weight_input,
        weight_layer_input_to_hidden=weight_nn_layers,
        weight_layer_act=act_radial,
        denominator=conv_denominator,
        train_denominator=train_conv_denominator,
        is_parallel=parallel,
        **conv_kwargs,
    )
    block[f'{t}_self_interaction_2'] = IrrepsLinear(
        irreps_out_tp,
        irreps_for_gate_in,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )
    block[f'{t}_equivariant_gate'] = gate_layer
    block[f'{t}_self_interaction_3'] = IrrepsLinear(
        irreps_out, irreps_out,
        data_key_in=data_key_x,
        biases=bias_in_linear,
    )
    if layer_scale_init is not None:
        block[f'{t}_layer_scale_1'] = LayerScale(
            irreps_out, layer_scale_init, data_key_x
        )
    block[f'{t}_self_connection_outro'] = sc_outro()

    if use_ffn:
        # ffn_moe_table (type -> expert): shared FFN + one element-group expert,
        # each with half of the dense hidden width (sevenn/nn/moe_ffn.py)
        moe = ffn_moe_table is not None
        scale = ffn_hidden_scale / 2 if moe else ffn_hidden_scale
        irreps_hidden = Irreps(
            [(int(mul * scale), ir) for mul, ir in irreps_out]
        )
        irreps_gate_out = irreps_hidden
        if ffn_type == 'swiglu' and all(
            ir.p == 1 for _, ir in irreps_hidden if ir.l == 0
        ):
            ffn_gate = SwiGLUGate(irreps_hidden, act_scalar, act_gate)
        elif ffn_type == 's2_bilinear' and any(
            ir.l > 0 for _, ir in irreps_hidden
        ):
            # scalar-only FFNs (last layer) keep the gate
            ffn_gate = S2BilinearGate(irreps_hidden, act_scalar, ffn_s2_channels)
            irreps_gate_out = ffn_gate.irreps_out
        else:
            ffn_gate = EquivariantGate(irreps_hidden, act_scalar, act_gate)
        from .normalization import SelfConnectionResidualIntro
        from .self_connection import SelfConnectionOutro

        block[f'{t}_ffn_intro'] = SelfConnectionResidualIntro(
            irreps_out, irreps_out, data_key_x
        )
        if use_norm:
            block[f'{t}_norm_2'] = EquivariantRMSNorm(
                irreps_out, data_key_x, eps=norm_eps
            )
        if moe:
            from .moe_ffn import MoEDispatch
            block[f'{t}_moe_dispatch'] = MoEDispatch(ffn_moe_table, t, data_key_x)
        block[f'{t}_ffn_linear_1'] = IrrepsLinear(
            irreps_out, ffn_gate.get_gate_irreps_in(),
            data_key_in=data_key_x,
            biases=bias_in_linear,
        )
        block[f'{t}_ffn_gate'] = ffn_gate
        if ffn_film:
            block[f'{t}_ffn_film'] = ElementFiLM(
                irreps_gate_out, num_species, data_key_x
            )
        block[f'{t}_ffn_linear_2'] = IrrepsLinear(
            irreps_gate_out, irreps_out,
            data_key_in=data_key_x,
            biases=bias_in_linear,
        )
        if moe:
            from .moe_ffn import MoECombine
            n_exp = int(ffn_moe_table.max().item()) + 1
            for k in range(n_exp):
                key = f'{t}_moe_x{k}'
                gate_k = EquivariantGate(irreps_hidden, act_scalar, act_gate, key)
                block[f'{t}_ffn_linear_1_e{k}'] = IrrepsLinear(
                    irreps_out, gate_k.get_gate_irreps_in(),
                    data_key_in=key, biases=bias_in_linear,
                )
                block[f'{t}_ffn_gate_e{k}'] = gate_k
                block[f'{t}_ffn_linear_2_e{k}'] = IrrepsLinear(
                    irreps_hidden, irreps_out,
                    data_key_in=key, biases=bias_in_linear,
                )
            block[f'{t}_moe_combine'] = MoECombine(n_exp, t, data_key_x)
        if layer_scale_init is not None:
            block[f'{t}_layer_scale_2'] = LayerScale(
                irreps_out, layer_scale_init, data_key_x
            )
        block[f'{t}_ffn_outro'] = SelfConnectionOutro(data_key_x)

    return block
