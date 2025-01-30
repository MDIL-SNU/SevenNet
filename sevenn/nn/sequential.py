import warnings
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


def _instantiate_modules(modules):
    # see IrrepsLinear of linear.py
    for module in modules.values():
        if not getattr(module, 'layer_instantiated', True):
            module.instantiate()


@compile_mode('script')
class _ModalInputPrepare(nn.Module):

    def __init__(
        self,
        modal_idx: int
    ):
        super().__init__()
        self.modal_idx = modal_idx

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.MODAL_TYPE] = torch.tensor(
            self.modal_idx,
            dtype=torch.int64,
            device=data['x'].device,
        )
        return data


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
        modal_map: Optional[Dict[str, int]] = None,
        eval_type_map: bool = True,
        eval_modal_map: bool = False,
        data_key_atomic_numbers: str = KEY.ATOMIC_NUMBERS,
        data_key_node_feature: str = KEY.NODE_FEATURE,
        data_key_grad: Optional[str] = None,
    ):
        if not isinstance(modules, OrderedDict):  # backward compat
            modules = OrderedDict(modules)
        self.cutoff = cutoff
        self.type_map = type_map
        self.eval_type_map = eval_type_map
        self.is_batch_data = True

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

        if eval_modal_map and modal_map is None:
            raise ValueError('eval_modal_map is True but modal_map is None')
        self.eval_modal_map = eval_modal_map
        self.modal_map = modal_map

        self.key_atomic_numbers = data_key_atomic_numbers
        self.key_node_feature = data_key_node_feature
        self.key_grad = data_key_grad

        _instantiate_modules(modules)
        super().__init__(modules)
        if not isinstance(self._modules, OrderedDict):  # backward compat
            self._modules = OrderedDict(self._modules)

    def set_is_batch_data(self, flag: bool):
        # whether given data is batched or not some module have to change
        # its behavior. checking whether data is batched or not inside
        # forward function make problem harder when make it into torchscript
        for module in self:
            try:  # Easier to ask for forgiveness than permission.
                module._is_batch_data = flag  # type: ignore
            except AttributeError:
                pass
        self.is_batch_data = flag

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

    @torch.jit.unused
    def _atomic_numbers_to_onehot(self, atomic_numbers: torch.Tensor):
        assert atomic_numbers.dtype == torch.int64
        device = atomic_numbers.device
        z_to_onehot_tensor = self.z_to_onehot_tensor.to(device)
        return torch.index_select(
            input=z_to_onehot_tensor, dim=0, index=atomic_numbers
        )

    @torch.jit.unused
    def _eval_modal_map(self, data: AtomGraphDataType):
        assert self.modal_map is not None
        # modal_map: dict[str, int]
        if not self.is_batch_data:
            modal_idx = self.modal_map[data[KEY.DATA_MODALITY]]  # type: ignore
        else:
            modal_idx = [
                self.modal_map[ii]  # type: ignore
                for ii in data[KEY.DATA_MODALITY]
            ]
        modal_idx = torch.tensor(
            modal_idx,
            dtype=torch.int64,
            device=data.x.device,  # type: ignore
        )
        data[KEY.MODAL_TYPE] = modal_idx

    def _preprocess(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self.eval_type_map:
            atomic_numbers = data[self.key_atomic_numbers]
            onehot = self._atomic_numbers_to_onehot(atomic_numbers)
            data[self.key_node_feature] = onehot

        if self.eval_modal_map:
            self._eval_modal_map(data)

        if self.key_grad is not None:
            data[self.key_grad].requires_grad_(True)

        return data

    def prepare_modal_deploy(self, modal: str):
        if self.modal_map is None:
            return
        self.eval_modal_map = False
        self.set_is_batch_data(False)
        modal_idx = self.modal_map[modal]  # type: ignore
        self.prepand_module('modal_input_prepare', _ModalInputPrepare(modal_idx))

    def forward(self, input: AtomGraphDataType) -> AtomGraphDataType:
        data = self._preprocess(input)
        for module in self:
            data = module(data)
        return data
