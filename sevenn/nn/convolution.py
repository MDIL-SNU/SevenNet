from typing import List

import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, Linear, TensorProduct
from e3nn.util.jit import compile_mode
from torch_scatter import scatter

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType
from sevenn.nn.activation import ShiftedSoftPlus


@compile_mode('script')
class IrrepsConvolution(nn.Module):
    """
    same as nequips convolution part (fig 1.d)
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denumerator: float = 1.0,
        train_denumerator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        is_parallel: bool = False,
    ):
        super().__init__()
        self.denumerator = nn.Parameter(
            torch.FloatTensor([denumerator]), requires_grad=train_denumerator
        )
        self.KEY_X = data_key_x
        self.KEY_FILTER = data_key_filter
        self.KEY_WEIGHT_INPUT = data_key_weight_input
        self.KEY_EDGE_IDX = data_key_edge_idx
        self.is_parallel = is_parallel

        instructions = []
        irreps_mid = []
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (mul_filter, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, 'uvu', True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]
        self.convolution = TensorProduct(
            irreps_x,
            irreps_filter,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_nn = FullyConnectedNet(
            weight_layer_input_to_hidden + [self.convolution.weight_numel],
            weight_layer_act,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        weight = self.weight_nn(data[self.KEY_WEIGHT_INPUT])
        x = data[self.KEY_X]
        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        # note that 1 -> src 0 -> dst
        edge_src = data[self.KEY_EDGE_IDX][1]
        edge_dst = data[self.KEY_EDGE_IDX][0]

        message = self.convolution(x[edge_src], data[self.KEY_FILTER], weight)

        x = scatter(message, edge_dst, dim=0, dim_size=len(x))
        x = x.div(self.denumerator)
        if self.is_parallel:
            # NLOCAL is # of atoms in system at 'CPU'
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.KEY_X] = x
        return data


@compile_mode('script')
class ElementDependentRadialWeights(nn.Module):
    """
    Implement of elem dependent raidal weight of MACE on M3GNet
    J. Chem. Phys. 159, 044118 (2023)
    """

    def __init__(
        self,
        irreps_x: Irreps,
        scalar_dim=None,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_radial_weights_prev: str = KEY.EDGE_EMBEDDING,
        data_key_radial_weights_new: str = 'radial_weights',
        data_key_edge_idx: str = KEY.EDGE_IDX,
    ):
        super().__init__()
        self.key_x = data_key_x
        self.key_radial_weights_prev = data_key_radial_weights_prev
        self.key_radial_weights_new = data_key_radial_weights_new
        self.key_edge_idx = data_key_edge_idx

        if scalar_dim is None:
            scalar_dim = irreps_x.sort().irreps.simplify()[0].mul
        irreps_scalar = Irreps(f'{scalar_dim}x0e')
        self.linear = Linear(irreps_x, irreps_scalar)
        self.additional_weights_dim = scalar_dim * 2

    def get_additional_weights_dim(self):
        return self.additional_weights_dim

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        scalar_node_features = self.linear(data[self.key_x])
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]
        data[self.key_radial_weights_new] = torch.cat(
            [
                data[self.key_radial_weights_prev],
                scalar_node_features[edge_src],
                scalar_node_features[edge_dst],
            ],
            dim=-1,
        )
        return data
