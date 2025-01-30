from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import NUM_UNIV_ELEMENT, AtomGraphDataType


def _as_univ(
    ss: List[float], type_map: Dict[int, int], default: float
) -> List[float]:
    assert len(ss) <= NUM_UNIV_ELEMENT, 'shift scale is too long'
    return [
        ss[type_map[z]] if z in type_map else default
        for z in range(NUM_UNIV_ELEMENT)
    ]


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

    def get_shift(self) -> float:
        return self.shift.detach().cpu().tolist()[0]

    def get_scale(self) -> float:
        return self.scale.detach().cpu().tolist()[0]

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

    def get_shift(self, type_map: Optional[Dict[int, int]] = None) -> List[float]:
        """
        Return shift in list of float. If type_map is given, return type_map reversed
        shift, which index equals atomic_number. 0.0 is assigned for atomis not found
        """
        shift = self.shift.detach().cpu().tolist()
        if type_map:
            shift = _as_univ(shift, type_map, 0.0)
        return shift

    def get_scale(self, type_map: Optional[Dict[int, int]] = None) -> List[float]:
        """
        Return scale in list of float. If type_map is given, return type_map reversed
        scale, which index equals atomic_number. 1.0 is assigned for atomis not found
        """
        scale = self.scale.detach().cpu().tolist()
        if type_map:
            scale = _as_univ(scale, type_map, 1.0)
        return scale

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
        shift, scale = shift_scale
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
    not modal-wise but atom-wise. It is always interpreted as atom-wise.
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

    def get_shift(
        self,
        type_map: Optional[Dict[int, int]] = None,
        modal_map: Optional[Dict[str, int]] = None,
    ) -> Union[List[float], Dict[str, List[float]]]:
        """
        Nothing is given: return as it is
        type_map is given but not modal wise shift: return univ shift
        both type_map and modal_map is given and modal wise shift: return fully
            resolved modalwise univ shift
        """
        shift = self.shift.detach().cpu().tolist()
        if type_map and not self.use_modal_wise_shift:
            shift = _as_univ(shift, type_map, 0.0)
        elif self.use_modal_wise_shift and modal_map and type_map:
            shift = [_as_univ(s, type_map, 0.0) for s in shift]
            shift = {modal: shift[idx] for modal, idx in modal_map.items()}

        return shift

    def get_scale(
        self,
        type_map: Optional[Dict[int, int]] = None,
        modal_map: Optional[Dict[str, int]] = None,
    ) -> Union[List[float], Dict[str, List[float]]]:
        """
        Nothing is given: return as it is
        type_map is given but not modal wise scale: return univ scale
        both type_map and modal_map is given and modal wise scale: return fully
            resolved modalwise univ scale
        """
        scale = self.scale.detach().cpu().tolist()
        if type_map and not self.use_modal_wise_scale:
            scale = _as_univ(scale, type_map, 0.0)
        elif self.use_modal_wise_scale and modal_map and type_map:
            scale = [_as_univ(s, type_map, 0.0) for s in scale]
            scale = {modal: scale[idx] for modal, idx in modal_map.items()}
        return scale

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
                res = []
                for v in s:
                    if isinstance(v, list) and len(v) == NUM_UNIV_ELEMENT:
                        # elem-wise: yes(univ) => solve elem map
                        v = solve_mapper(v, type_map)
                    elif isinstance(v, float):
                        # elem-wise: no => broadcast to elemwise
                        v = [v] * n_atom_types
                    else:
                        raise ValueError(f'Invalid shift or scale {s}')
                    res.append(v)
            else:
                raise ValueError(f'Invalid shift or scale {s}')

            if use_mw:
                assert (
                    isinstance(res, list)
                    and isinstance(res[0], list)
                    and len(res) == n_modals
                )
                assert all([len(r) == n_atom_types for r in res])  # type: ignore
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


def get_resolved_shift_scale(
    module: Union[Rescale, SpeciesWiseRescale, ModalWiseRescale],
    type_map: Optional[Dict[int, int]] = None,
    modal_map: Optional[Dict[str, int]] = None,
):
    """
    Return resolved shift and scale from scale modules. For element wise case,
    convert to list of floats where idx is atomic number. For modal wise case, return
    dictionary of shift scale where key is modal name given in modal_map

    Return:
        Tuple of solved shift and scale
    """

    if isinstance(module, Rescale):
        return (module.get_shift(), module.get_scale())
    elif isinstance(module, SpeciesWiseRescale):
        return (module.get_shift(type_map), module.get_scale(type_map))
    elif isinstance(module, ModalWiseRescale):
        return (
            module.get_shift(type_map, modal_map),
            module.get_scale(type_map, modal_map),
        )
    raise ValueError('Not scale module')
