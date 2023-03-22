from typing import Dict
from collections import OrderedDict

import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class AtomGraphSequential(nn.Sequential):
    """
    same as nn.Sequential but with type notation
    see
    https://github.com/pytorch/pytorch/issues/52588
    """
    def __init__(self, modules: Dict[str, nn.Module]):
        if type(modules) != OrderedDict:
            modules = OrderedDict(modules)
        super().__init__(modules)

    def set_is_batch_data(self, flag: bool):
        # whether given data is batched or not some module have to change
        # is behavior. checking whether data is batched or not inside
        # forward function make problem harder when make it into torchscript
        for module in self:
            try:  # Easier to ask for forgiveness than permission.
                module._is_batch_data = flag
            except AttributeError:
                pass

    def prepand_module(self, key: str, module: nn.Module):
        self._modules.update({key: module})
        self._modules.move_to_end(key, last=False)
    
    def replace_module(self, key: str, module: nn.Module):
        self._modules.update({key: module})
    
    def delete_module_by_key(self, key: str):
        if key in self._modules.keys():
            del self._modules[key]

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        for module in self:
            data = module(data)
        return data
