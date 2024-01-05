import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class SelfConnectionIntro(nn.Module):
    """
    do TensorProduct of x and some data(here attribute of x)
    and save it (to concatenate updated x at SelfConnectionOutro)
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_operand: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_operand: str = KEY.NODE_ATTR,
        **kwargs,
    ):
        super().__init__()

        self.fc_tensor_product = FullyConnectedTensorProduct(
            irreps_x, irreps_operand, irreps_out
        )
        self.KEY_X = data_key_x
        self.KEY_OPERAND = data_key_operand

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.SELF_CONNECTION_TEMP] = self.fc_tensor_product(
            data[self.KEY_X], data[self.KEY_OPERAND]
        )
        return data


@compile_mode('script')
class SelfConnectionMACEIntro(nn.Module):
    """
    MACE style self connection update
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        **kwargs,
    ):
        super().__init__()
        self.linear = Linear(irreps_x, irreps_out)
        self.KEY_X = data_key_x

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.SELF_CONNECTION_TEMP] = self.linear(data[self.KEY_X])
        return data


@compile_mode('script')
class SelfConnectionOutro(nn.Module):
    """
    do TensorProduct of x and some data(here attribute of x)
    and save it (to concatenate updated x at SelfConnectionOutro)
    """

    def __init__(
        self,
        data_key_x: str = KEY.NODE_FEATURE,
        **kwargs,
    ):
        super().__init__()
        self.KEY_X = data_key_x

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.KEY_X] = data[self.KEY_X] + data[KEY.SELF_CONNECTION_TEMP]
        del data[KEY.SELF_CONNECTION_TEMP]
        return data
