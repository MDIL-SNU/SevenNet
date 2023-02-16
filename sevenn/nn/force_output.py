import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class ForceOutputFromEdge(nn.Module):
    """
    works when edge_vec.requires_grad_ is True
    """
    def __init__(
        self,
        data_key_edge_vec: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.SCALED_ENERGY,
        data_key_force: str = KEY.SCALED_FORCE,
    ):
        super().__init__()
        self.KEY_EDGE_VEC = data_key_edge_vec
        self.KEY_ENERGY = data_key_energy
        self.KEY_FORCE = data_key_force
        self.KEY_EDGE_IDX = data_key_edge_idx

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        #TODO: way to avoid tot_num from 'len'
        tot_num = len(data[KEY.NODE_FEATURE])
        edge_idx = data[self.KEY_EDGE_IDX]

        edge_vec_tensor = [data[self.KEY_EDGE_VEC]]
        energy = [(data[self.KEY_ENERGY]).sum()]

        dE_dr = torch.autograd.grad(energy, edge_vec_tensor,
                                    create_graph=self.training)[0]

        if dE_dr is not None:
            force = torch.zeros(tot_num, 3)
            force = scatter(dE_dr, edge_idx[0], dim=0, dim_size=tot_num)
            force -= scatter(dE_dr, edge_idx[1], dim=0, dim_size=tot_num)

            data[self.KEY_FORCE] = force

        return data


@compile_mode('script')
class ForceOutput(nn.Module):
    """
    works when pos.requires_grad_ is True
    """
    def __init__(
        self,
        data_key_pos: str = KEY.POS,
        data_key_energy: str = KEY.PRED_TOTAL_ENERGY,
        data_key_force: str = KEY.PRED_FORCE,
    ):
        super().__init__()
        self.KEY_POS = data_key_pos
        self.KEY_ENERGY = data_key_energy
        self.KEY_FORCE = data_key_force

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        pos_tensor = [data[self.KEY_POS]]
        energy = [(data[self.KEY_ENERGY]).sum()]

        grad = torch.autograd.grad(energy, pos_tensor,
                                   create_graph=self.training)[0]

        # without this 'if', type(grad) is 'Optional[Tensor]' which result in error
        if grad is not None:
            data[self.KEY_FORCE] = torch.neg(grad)
        return data

