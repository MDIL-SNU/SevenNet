from typing import List

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class Rescale(nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    """
    def __init__(
        self,
        shift: float,
        scale: float,
        data_key_in = KEY.SCALED_ATOMIC_ENERGY,
        data_key_out = KEY.ATOMIC_ENERGY,
        train_shift_scale: bool = False,
    ):
        super().__init__()
        self.shift = \
            nn.Parameter(torch.FloatTensor([shift]),
                         requires_grad=train_shift_scale)
        self.scale = \
            nn.Parameter(torch.FloatTensor([scale]),
                         requires_grad=train_shift_scale)
        self.KEY_INPUT = data_key_in
        self.KEY_OUTPUT = data_key_out

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.KEY_OUTPUT] =\
            data[self.KEY_INPUT] * self.scale + self.shift

        return data

@compile_mode('script')
class SpeciesWiseRescale(nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    """
    def __init__(
        self,
        shift: List[float],
        scale: List[float],
        data_key_in = KEY.SCALED_ATOMIC_ENERGY,
        data_key_out = KEY.ATOMIC_ENERGY,
        data_key_indicies = KEY.ATOM_TYPE,
        train_shift_scale: bool = False,
    ):
        super().__init__()
        self.shift = \
            nn.Parameter(torch.FloatTensor(shift),
                         requires_grad=train_shift_scale)
        self.scale = \
            nn.Parameter(torch.FloatTensor(scale),
                         requires_grad=train_shift_scale)
        self.KEY_INPUT = data_key_in
        self.KEY_OUTPUT = data_key_out
        self.KEY_INDICIES = data_key_indicies

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        indicies = data[self.KEY_INDICIES]
        data[self.KEY_OUTPUT] =\
            data[self.KEY_INPUT] * self.scale[indicies].view(-1, 1)\
            + self.shift[indicies].view(-1, 1)

        return data

@compile_mode('script')
class ModalWiseRescale(nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    """
    def __init__(
        self,
        shift: List[List[float]],
        scale: List[List[float]],
        data_key_in = KEY.SCALED_ATOMIC_ENERGY,
        data_key_out = KEY.ATOMIC_ENERGY,
        data_key_modal_indices = KEY.MODAL_TYPE,
        data_key_atom_indices = KEY.ATOM_TYPE,
        train_shift_scale: bool = False,
    ):
        super().__init__()
        self.shift = \
            nn.Parameter(torch.FloatTensor(shift),
                         requires_grad=train_shift_scale)
        self.scale = \
            nn.Parameter(torch.FloatTensor(scale),
                         requires_grad=train_shift_scale)
        self.KEY_INPUT = data_key_in
        self.KEY_OUTPUT = data_key_out
        self.KEY_ATOM_INDICES = data_key_atom_indices
        self.KEY_MODAL_INDICES = data_key_modal_indices

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        batch = data[KEY.BATCH]
        modal_indices = data[self.KEY_MODAL_INDICES][batch]
        atom_indices = data[self.KEY_ATOM_INDICES]
        data[self.KEY_OUTPUT] =\
            data[self.KEY_INPUT] * self.scale[modal_indices, atom_indices].view(-1, 1)\
            + self.shift[modal_indices, atom_indices].view(-1, 1)

        return data