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
        irreps_in: Irreps,
        irreps_operand: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_operand: str = KEY.NODE_ATTR,
        lazy_layer_instantiate: bool = True,
        **kwargs,  # for compatibility
    ):
        super().__init__()

        self.fc_tensor_product = FullyConnectedTensorProduct(
            irreps_in, irreps_operand, irreps_out
        )
        self.irreps_in1 = irreps_in
        self.irreps_in2 = irreps_operand
        self.irreps_out = irreps_out

        self.key_x = data_key_x
        self.key_operand = data_key_operand

        self.fc_tensor_product = None
        self.layer_instantiated = False
        self.fc_tensor_product_cls = FullyConnectedTensorProduct
        self.fc_tensor_product_kwargs = kwargs

        if not lazy_layer_instantiate:
            self.instantiate()

    def instantiate(self):
        if self.fc_tensor_product is not None:
            raise ValueError('fc_tensor_product layer already exists')
        self.fc_tensor_product = self.fc_tensor_product_cls(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            shared_weights=True,
            internal_weights=None,  # same as True
            **self.fc_tensor_product_kwargs,
        )
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.fc_tensor_product is not None, 'Layer is not instantiated'
        data[KEY.SELF_CONNECTION_TEMP] = self.fc_tensor_product(
            data[self.key_x], data[self.key_operand]
        )
        return data


@compile_mode('script')
class SelfConnectionLinearIntro(nn.Module):
    """
    Linear style self connection update
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        lazy_layer_instantiate: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.key_x = data_key_x

        self.linear = None
        self.layer_instantiated = False
        self.linear_cls = Linear

        # TODO: better to have SelfConnectionIntro super class
        kwargs.pop('irreps_operand')
        self.linear_kwargs = kwargs

        if not lazy_layer_instantiate:
            self.instantiate()

    def instantiate(self):
        if self.linear is not None:
            raise ValueError('Linear layer already exists')
        self.linear = self.linear_cls(
            self.irreps_in, self.irreps_out, **self.linear_kwargs
        )
        self.layer_instantiated = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.linear is not None, 'Layer is not instantiated'
        data[KEY.SELF_CONNECTION_TEMP] = self.linear(data[self.key_x])
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
    ):
        super().__init__()
        self.key_x = data_key_x

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_x] = data[self.key_x] + data[KEY.SELF_CONNECTION_TEMP]
        del data[KEY.SELF_CONNECTION_TEMP]
        return data
