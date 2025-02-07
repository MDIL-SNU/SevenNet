from typing import List

import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, TensorProduct
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .activation import ShiftedSoftPlus
from .util import broadcast


def message_gather(
    node_features: torch.Tensor,
    edge_dst: torch.Tensor,
    message: torch.Tensor
):
    index = broadcast(edge_dst, message, 0)
    out_shape = [len(node_features)] + list(message.shape[1:])
    out = torch.zeros(
        out_shape,
        dtype=node_features.dtype,
        device=node_features.device
    )
    out.scatter_reduce_(0, index, message, reduce='sum')
    return out


@compile_mode('script')
class IrrepsConvolution(nn.Module):
    """
    convolution of (fig 2.b), comm. in LAMMPS
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_filter: str = KEY.EDGE_ATTR,
        data_key_weight_input: str = KEY.EDGE_EMBEDDING,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        lazy_layer_instantiate: bool = True,
        is_parallel: bool = False,
    ):
        super().__init__()
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )
        self.key_x = data_key_x
        self.key_filter = data_key_filter
        self.key_weight_input = data_key_weight_input
        self.key_edge_idx = data_key_edge_idx
        self.is_parallel = is_parallel

        instructions = []
        irreps_mid = []
        weight_numel = 0
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        weight_numel += mul_x * 1  # path shape
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, 'uvu', True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()  # type: ignore
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        # From v0.11.x, to compatible with cuEquivariance
        self._instructions_before_sort = instructions
        instructions = sorted(instructions, key=lambda x: x[2])

        self.convolution_kwargs = dict(
            irreps_in1=irreps_x,
            irreps_in2=irreps_filter,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_nn_kwargs = dict(
            hs=weight_layer_input_to_hidden + [weight_numel],
            act=weight_layer_act
        )

        self.convolution = None
        self.weight_nn = None
        self.layer_instantiated = False
        self.convolution_cls = TensorProduct
        self.weight_nn_cls = FullyConnectedNet

        if not lazy_layer_instantiate:
            self.instantiate()

        self._comm_size = irreps_x.dim  # used in parallel

    def instantiate(self):
        if self.convolution is not None:
            raise ValueError('Convolution layer already exists')
        if self.weight_nn is not None:
            raise ValueError('Weight_nn layer already exists')

        self.convolution = self.convolution_cls(**self.convolution_kwargs)
        self.weight_nn = self.weight_nn_cls(**self.weight_nn_kwargs)
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.convolution is not None, 'Convolution is not instantiated'
        assert self.weight_nn is not None, 'Weight_nn is not instantiated'
        weight = self.weight_nn(data[self.key_weight_input])
        x = data[self.key_x]
        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        # note that 1 -> src 0 -> dst
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]

        message = self.convolution(x[edge_src], data[self.key_filter], weight)

        x = message_gather(x, edge_dst, message)
        x = x.div(self.denominator)
        if self.is_parallel:
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data
