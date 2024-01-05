import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, Linear
from e3nn.util.jit import compile_mode
from torch_scatter import scatter

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class IrrepsLinear(nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,
        data_key_out: str = None,
        **e3nn_linear_params,
    ):
        super().__init__()
        self.KEY_INPUT = data_key_in
        if data_key_out is None:
            self.KEY_OUTPUT = data_key_in
        else:
            self.KEY_OUTPUT = data_key_out

        self.linear = Linear(irreps_in, irreps_out, **e3nn_linear_params)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.KEY_OUTPUT] = self.linear(data[self.KEY_INPUT])
        return data


@compile_mode('script')
class AtomReduce(nn.Module):
    """
    atomic energy -> total energy
    constant is multiplied to data
    """

    def __init__(
        self,
        data_key_in: str,
        data_key_out: str,
        reduce='sum',
        constant: float = 1.0,
    ):
        super().__init__()

        self.KEY_INPUT = data_key_in
        self.KEY_OUTPUT = data_key_out
        self.constant = constant
        self.reduce = reduce

        # controlled by the upper most wrapper 'AtomGraphSequential'
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            data[self.KEY_OUTPUT] = (
                scatter(
                    data[self.KEY_INPUT],
                    data[KEY.BATCH],
                    dim=0,
                    reduce=self.reduce,
                )
                * self.constant
            )
            data[self.KEY_OUTPUT] = data[self.KEY_OUTPUT].squeeze(1)
        else:
            data[self.KEY_OUTPUT] = (
                torch.sum(data[self.KEY_INPUT]) * self.constant
            )

        return data


@compile_mode('script')
class FCN_e3nn(nn.Module):
    """
    wrapper class of e3nn FullyConnectedNet
    doesn't necessarily have irrpes since it is only
    applicable scalar but for consistency(?_?)
    """

    def __init__(
        self,
        irreps_in: Irreps,  # confirm it is scalar & input size
        dim_out: int,
        hidden_neurons,
        activation,
        data_key_in: str,
        data_key_out: str = None,
        **e3nn_params,
    ):
        super().__init__()
        self.KEY_INPUT = data_key_in
        self.irreps_in = irreps_in
        if data_key_out is None:
            self.KEY_OUTPUT = data_key_in
        else:
            self.KEY_OUTPUT = data_key_out

        for mul, irrep in irreps_in:
            assert irrep.is_scalar()
        inp_dim = irreps_in.dim

        self.fcn = FullyConnectedNet(
            [inp_dim] + hidden_neurons + [dim_out],
            activation,
            **e3nn_params,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.KEY_OUTPUT] = self.fcn(data[self.KEY_INPUT])
        return data
