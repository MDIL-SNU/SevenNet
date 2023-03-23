import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class ForceOutputFromEdgeParallel(nn.Module):
    """
    works when edge_vec.requires_grad_ is True
    """
    def __init__(
        self,
        data_key_edge_vec: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.SCALED_ENERGY,
        data_key_force: str = KEY.SCALED_FORCE,
        data_key_num_atoms: str = KEY.NUM_ATOMS,
        data_key_num_ghosts: str = KEY.NUM_GHOSTS,
    ):
        super().__init__()
        self.KEY_EDGE_VEC = data_key_edge_vec
        self.KEY_ENERGY = data_key_energy
        self.KEY_FORCE = data_key_force
        self.KEY_EDGE_IDX = data_key_edge_idx
        self.KEY_NUM_ATOMS = data_key_num_atoms
        self.KEY_NUM_GHOSTS = data_key_num_ghosts

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        tot_num = len(data[KEY.NODE_FEATURE]) + len(data[KEY.NODE_FEATURE_GHOST])
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
        #tot_num = len(data[KEY.NODE_FEATURE])
        tot_num = torch.sum(data[KEY.NUM_ATOMS])
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
        data_key_energy: str = KEY.SCALED_ENERGY,
        data_key_force: str = KEY.SCALED_FORCE,
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


@compile_mode('script')
class ForceStressOutput(nn.Module):

    def __init__(
        self,
        data_key_pos: str = KEY.POS,
        data_key_energy: str = KEY.SCALED_ENERGY,
        data_key_force: str = KEY.SCALED_FORCE,
        data_key_stress: str = KEY.SCALED_STRESS
    ):

        super().__init__()
        self.KEY_POS = data_key_pos
        self.KEY_ENERGY = data_key_energy
        self.KEY_FORCE = data_key_force
        self.KEY_STRESS = data_key_stress
    
    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        pos_tensor = data[self.KEY_POS]
        energy = [(data[self.KEY_ENERGY]).sum()]

        grad = torch.autograd.grad(energy, [pos_tensor, data["_strain"]],
                            create_graph=self.training)

        force = torch.neg(grad[0])

        data[self.KEY_FORCE] = force

        volume = data[KEY.CELL_VOLUME]

        stress = grad[1] / volume.view(-1, 1, 1)
        stress = torch.neg(stress)

        voigt_stress = torch.vstack((stress[:,0,0], stress[:,1,1], stress[:,2,2], stress[:,0,1], stress[:,1,2], stress[:,0,2]))
        data[self.KEY_STRESS] = voigt_stress.transpose(0, 1)
        
        return data