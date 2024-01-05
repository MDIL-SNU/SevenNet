import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class GhostControlSplit(nn.Module):
    def __init__(
        self,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_ghost: str = KEY.NODE_FEATURE_GHOST,
        data_key_num_atoms: str = KEY.NUM_ATOMS,
        data_key_num_ghost: str = KEY.NUM_GHOSTS,
    ):
        super().__init__()
        self.KEY_X = data_key_x
        self.KEY_GHOST = data_key_ghost
        self.KEY_NUM_ATOMS = data_key_num_atoms
        self.KEY_NUM_GHOST = data_key_num_ghost

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x, ghost = torch.tensor_split(
            data[self.KEY_X], data[self.KEY_NUM_ATOMS]
        )
        print(len(x))
        print(len(ghost))
        data[self.KEY_X] = x
        data[self.KEY_GHOST] = ghost
        return data


@compile_mode('script')
class GhostControlCat(nn.Module):
    def __init__(
        self,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_ghost: str = KEY.NODE_FEATURE_GHOST,
        # data_key_num_ghost: str = KEY.NUM_GHOSTS,
    ):
        super().__init__()
        self.KEY_X = data_key_x
        self.KEY_GHOST = data_key_ghost
        # self.KEY_NUM_GHOT = data_key_num_ghost

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.KEY_X] = torch.cat([data[self.KEY_X], data[self.KEY_GHOST]])
        print(len(data[self.KEY_X]))
        return data
