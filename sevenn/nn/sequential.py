import warnings
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class AtomGraphSequential(nn.Sequential):
    """
    Wrapper of SevenNet model

    Args:
        modules: OrderedDict of nn.Modules
        cutoff: not used internally, but makes sense to have
        type_map: atomic_numbers => onehot index (see nn/node_embedding.py)
        eval_type_map: perform index mapping using type_map defaults to True
        data_key_atomic_numbers: used when eval_type_map is True
        data_key_node_feature: used when eval_type_map is True
        data_key_grad: if given, sets its requires grad True before pred
    """

    def __init__(
        self,
        modules: Dict[str, nn.Module],
        cutoff: float = 0.0,
        type_map: Optional[Dict[int, int]] = None,
        eval_type_map: bool = True,
        data_key_atomic_numbers: str = KEY.ATOMIC_NUMBERS,
        data_key_node_feature: str = KEY.NODE_FEATURE,
        data_key_grad: Optional[str] = None,
    ):
        if not isinstance(modules, OrderedDict):
            modules = OrderedDict(modules)
        self.cutoff = cutoff
        self.type_map = type_map
        self.eval_type_map = eval_type_map

        if cutoff == 0.0:
            warnings.warn('cutoff is 0.0 or not given', UserWarning)

        if self.type_map is None:
            warnings.warn('type_map is not given', UserWarning)
            self.eval_type_map = False
        else:
            z_to_onehot_tensor = torch.neg(torch.ones(120, dtype=torch.long))
            for z, onehot in self.type_map.items():
                z_to_onehot_tensor[z] = onehot
            self.z_to_onehot_tensor = z_to_onehot_tensor

        self.key_atomic_numbers = data_key_atomic_numbers
        self.key_node_feature = data_key_node_feature
        self.key_grad = data_key_grad

        super().__init__(modules)

    def set_is_batch_data(self, flag: bool):
        # whether given data is batched or not some module have to change
        # its behavior. checking whether data is batched or not inside
        # forward function make problem harder when make it into torchscript
        for module in self:
            try:  # Easier to ask for forgiveness than permission.
                module._is_batch_data = flag  # type: ignore
            except AttributeError:
                pass

    def get_irreps_in(self, modlue_name: str, attr_key: str = 'irreps_in'):
        tg_module = self._modules[modlue_name]
        for m in tg_module.modules():
            try:
                return repr(m.__getattribute__(attr_key))
            except AttributeError:
                pass
        return None

    def prepand_module(self, key: str, module: nn.Module):
        self._modules.update({key: module})
        self._modules.move_to_end(key, last=False)  # type: ignore

    def replace_module(self, key: str, module: nn.Module):
        self._modules.update({key: module})

    def delete_module_by_key(self, key: str):
        if key in self._modules.keys():
            del self._modules[key]

    def _atomic_numbers_to_onehot(self, atomic_numbers: torch.Tensor):
        assert atomic_numbers.dtype == torch.int64
        device = atomic_numbers.device
        z_to_onehot_tensor = self.z_to_onehot_tensor.to(device)
        return torch.index_select(
            input=z_to_onehot_tensor, dim=0, index=atomic_numbers
        )

    def forward(self, input: AtomGraphDataType) -> AtomGraphDataType:
        data = input
        if self.eval_type_map:
            atomic_numbers = data[self.key_atomic_numbers]
            onehot = self._atomic_numbers_to_onehot(atomic_numbers)
            data[self.key_node_feature] = onehot

        if self.key_grad is not None:
            data[self.key_grad].requires_grad_(True)

        for module in self:
            data = module(data)
        return data
