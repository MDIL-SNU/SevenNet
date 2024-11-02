from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import NUM_UNIV_ELEMENT, AtomGraphDataType


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
        **kwargs,
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

    @staticmethod
    def from_mappers(
        shift: Union[float, List[float]],
        scale: Union[float, List[float]],
        type_map: Dict[int, int],
        **kwargs,
    ):
        """
        Fit dimensions or mapping raw shift scale values to that is valid under
        the given type_map: (atomic_numbers -> type_indices)
        """
        shift_scale = []
        n_atom_types = len(type_map)
        for s in (shift, scale):
            if isinstance(s, list) and len(s) > n_atom_types:
                if len(s) != NUM_UNIV_ELEMENT:
                    raise ValueError('given shift or scale is strange')
                s = [s[z] for z in sorted(type_map, key=lambda x: type_map[x])]
                # s = [s[z] for z in sorted(type_map, key=type_map.get)]
            elif isinstance(s, float):
                s = [s] * n_atom_types
            elif isinstance(s, list) and len(s) == 1:
                s = s * n_atom_types
            shift_scale.append(s)
        assert all([len(s) == n_atom_types for s in shift_scale])
        return SpeciesWiseRescale(shift, scale, **kwargs)

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
    Given shift or scale is either modal-wise and atom-wise or
    not modal-wise but atom-wise. It is always atom-wise.
    If given scalar, it tries to expand to atom-wise
    """

    def __init__(
        self,
        shift: List[List[float]],
        scale: List[List[float]],
        data_key_in: str = KEY.SCALED_ATOMIC_ENERGY,
        data_key_out: str = KEY.ATOMIC_ENERGY,
        data_key_modal_indices: str = KEY.MODAL_TYPE,
        data_key_atom_indices: str = KEY.ATOM_TYPE,
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
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.key_atom_indices = data_key_atom_indices
        self.key_modal_indices = data_key_modal_indices
        self.use_modal_wise_shift = use_modal_wise_shift
        self.use_modal_wise_scale = use_modal_wise_scale
        self._is_batch_data = True

    @staticmethod
    def from_mappers(
        shift: Union[float, List[float], Dict[str, Any]],
        scale: Union[float, List[float], Dict[str, Any]],
        use_modal_wise_shift: bool,
        use_modal_wise_scale: bool,
        type_map: Dict[int, int],
        modal_map: Dict[str, int],
        **kwargs,
    ):
        """
        Fit dimensions or mapping raw shift scale values to that is valid under
        the given type_map: (atomic_numbers -> type_indices)
        If given List[float] and its length matches length of _const.NUM_UNIV_ELEMENT
        , assume it is element-wise list
        otherwise, it is modal-wise list
        """

        def solve_mapper(arr, map):
            # value is attr index and never overlap, key is either 'z' or modal str
            return [arr[z] for z in sorted(map, key=lambda x: map[x])]

        shift_scale = []
        n_atom_types = len(type_map)
        n_modals = len(modal_map)

        for s, use_mw in (
            (shift, use_modal_wise_shift),
            (scale, use_modal_wise_scale),
        ):
            # solve elemewise, or broadcast
            if isinstance(s, float):
                # given, modal-wise: no, elem-wise: no => broadcast
                shape = (n_modals, n_atom_types) if use_mw else (n_atom_types,)
                res = torch.full(shape, s).tolist()  # TODO: w/o torch
            elif isinstance(s, list) and len(s) == NUM_UNIV_ELEMENT:
                # given, modal-wise: no, elem-wise: yes(univ) => solve elem map
                s = solve_mapper(s, type_map)
                res = [s] * n_modals if use_mw else s
            elif (  # given, modal-wise: yes, elem-wise: no => broadcast to elemwise
                isinstance(s, list)
                and isinstance(s[0], float)
                and len(s) == n_modals
                and use_mw
            ):
                res = [[v] * n_atom_types for v in s]
            elif (  # given, modal-wise: no, elem-wise: yes => as it is
                isinstance(s, list)
                and isinstance(s[0], float)
                and len(s) == n_atom_types
                and not use_mw
            ):
                res = s
            elif (  # given, modal-wise: yes, elem-wise: yes => as it is
                isinstance(s, list)
                and isinstance(s[0], list)
                and len(s) == n_modals
                and len(s[0]) == n_atom_types
                and use_mw
            ):
                res = s
            elif isinstance(s, dict) and use_mw:
                # solve modal dict, modal-wise: yes
                s = solve_mapper(s, modal_map)
                if isinstance(s[0], list) and len(s[0]) == NUM_UNIV_ELEMENT:
                    # elem-wise: yes(univ) => solve elem map
                    res = [solve_mapper(v, type_map) for v in s]
                elif isinstance(s[0], float):
                    # elem-wise: no => broadcast to elemwise
                    res = [[v] * n_atom_types for v in s]
                elif isinstance(s[0], list) and len(s[0]) == n_atom_types:
                    # elem-wise: yes => not univ, already mapped => as it is
                    res = s
                else:
                    raise ValueError(f'Invalid shift or scale {s}')
            else:
                raise ValueError(f'Invalid shift or scale {s}')

            if use_mw:
                assert (
                    isinstance(res, list)
                    and isinstance(res[0], list)
                    and len(res) == n_modals
                    and len(res[0]) == n_atom_types
                )
            else:
                assert (
                    isinstance(res, list)
                    and isinstance(res[0], float)
                    and len(res) == n_atom_types
                )
            shift_scale.append(res)
        shift, scale = shift_scale

        return ModalWiseRescale(
            shift,
            scale,
            use_modal_wise_shift=use_modal_wise_shift,
            use_modal_wise_scale=use_modal_wise_scale,
            **kwargs,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            batch = data[KEY.BATCH]
            modal_indices = data[self.key_modal_indices][batch]
        else:
            modal_indices = data[self.key_modal_indices]
        atom_indices = data[self.key_atom_indices]
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
        data[self.key_output] = data[self.key_input] * scale.view(
            -1, 1
        ) + shift.view(-1, 1)

        return data
