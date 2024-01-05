import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class GradsCalc(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        data_key_of: str,
        data_key_wrt: list,
        data_key_out: list,
    ):
        super().__init__()
        self.model = model
        self.KEY_OF = data_key_of
        self.KEY_WRT_LIST = data_key_wrt
        self.KEY_OUT_LIST = data_key_out

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        tensors = []
        for wrt_key in self.KEY_WRT_LIST:
            data[wrt_key].requires_grad_(True)
            tensors.append(data[wrt_key])

        data = self.model.forward(data)
        grads = torch.autograd.grad(data[self.KEY_OF], tensors)
        for tensor, grad, out_key in zip(tensors, grads, self.KEY_OUT_LIST):
            data[out_key] = grad
            # tensor.grad = None
            tensor.requires_grad_(False)
