from typing import List, Union

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
        data_key_in: str = KEY.SCALED_ATOMIC_ENERGY,
        data_key_out: str = KEY.ATOMIC_ENERGY,
        train_shift_scale: bool = False,
        **kwargs
    ):
        assert isinstance(shift, float) and isinstance(scale, float)
        super().__init__()
        self.shift = nn.Parameter(
            torch.FloatTensor([shift]), requires_grad=train_shift_scale
        )
        self.scale = nn.Parameter(
            torch.FloatTensor([scale]), requires_grad=train_shift_scale
        )
        self.key_input = data_key_in
        self.key_output = data_key_out

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = data[self.key_input] * self.scale + self.shift

        return data


@compile_mode('script')
class SpeciesWiseRescale(nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    Use as it is if given list, expand to list if one of them is float
    If two lists are given and length is not the same, raise error
    """

    def __init__(
        self,
        shift: Union[List[float], float],
        scale: Union[List[float], float],
        data_key_in: str = KEY.SCALED_ATOMIC_ENERGY,
        data_key_out: str = KEY.ATOMIC_ENERGY,
        data_key_indices: str = KEY.ATOM_TYPE,
        train_shift_scale: bool = False,
        **kwargs
    ):
        super().__init__()
        assert isinstance(shift, float) or isinstance(shift, list)
        assert isinstance(scale, float) or isinstance(scale, list)

        if (
            isinstance(shift, list)
            and isinstance(scale, list)
            and len(shift) != len(scale)
        ):
            raise ValueError('List length should be same')

        if isinstance(shift, list):
            num_species = len(shift)
        elif isinstance(scale, list):
            num_species = len(scale)
        else:
            raise ValueError('Both shift and scale is not a list')

        shift = [shift] * num_species if isinstance(shift, float) else shift
        scale = [scale] * num_species if isinstance(scale, float) else scale

        self.shift = nn.Parameter(
            torch.FloatTensor(shift), requires_grad=train_shift_scale
        )
        self.scale = nn.Parameter(
            torch.FloatTensor(scale), requires_grad=train_shift_scale
        )
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.key_indices = data_key_indices

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        indices = data[self.key_indices]
        data[self.key_output] = data[self.key_input] * self.scale[indices].view(
            -1, 1
        ) + self.shift[indices].view(-1, 1)

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
        data_key_in=KEY.SCALED_ATOMIC_ENERGY,
        data_key_out=KEY.ATOMIC_ENERGY,
        data_key_modal_indices=KEY.MODAL_TYPE,
        data_key_atom_indices=KEY.ATOM_TYPE,
        use_modal_wise_shift: bool = False,
        use_modal_wise_scale: bool = False,
        train_shift_scale: bool = False,
    ):
        super().__init__()
        self.shift = nn.Parameter(
            torch.FloatTensor(shift), requires_grad=train_shift_scale
        )
        self.scale = nn.Parameter(
            torch.FloatTensor(scale), requires_grad=train_shift_scale
        )
        self.KEY_INPUT = data_key_in
        self.KEY_OUTPUT = data_key_out
        self.KEY_ATOM_INDICES = data_key_atom_indices
        self.KEY_MODAL_INDICES = data_key_modal_indices
        self.use_modal_wise_shift = use_modal_wise_shift
        self.use_modal_wise_scale = use_modal_wise_scale

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        batch = data[KEY.BATCH]
        modal_indices = data[self.KEY_MODAL_INDICES][batch]
        atom_indices = data[self.KEY_ATOM_INDICES]
        shift = (
            self.shift[modal_indices, atom_indices]
            if self.use_modal_wise_shift
            else self.shift[atom_indices]
        )
        scale = (
            self.scale[modal_indices, atom_indices]
            if self.use_modal_wise_scale
            else self.scale[atom_indices]
        )
        data[self.KEY_OUTPUT] = data[self.KEY_INPUT] * scale.view(
            -1, 1
        ) + shift.view(-1, 1)

        return data
