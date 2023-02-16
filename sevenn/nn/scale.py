import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


#TODO: rename these confuising scale, total, peratom things
#      SCALED_PER_ATOM_ENERGY is not per atom energy (its total energy, see AtomReduce)
@compile_mode('script')
class Scale(nn.Module):
    """
    Scaling and shifting energy and force.
    """
    def __init__(
        self,
        shift: float,
        scale: float,
        scaled_energy_key: str = KEY.SCALED_PER_ATOM_ENERGY,
        scaled_force_key: str = KEY.SCALED_FORCE,
        n_atoms_key: str = KEY.NUM_ATOMS,
        scale_per_atom: bool = False
    ):
        super().__init__()
        self.shift = shift
        self.scale = scale
        self.scaled_energy_key = scaled_energy_key
        self.scaled_force_key = scaled_force_key
        self.n_atoms_key = n_atoms_key
        self.scale_per_atom = scale_per_atom

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self.scale_per_atom:  # 'un'scale atomic energy
            data[KEY.ATOMIC_ENERGY] = \
                data[KEY.ATOMIC_ENERGY] * self.scale + self.shift
        data[KEY.PRED_TOTAL_ENERGY] = \
            (data[KEY.SCALED_ENERGY] * self.scale) +\
            (self.shift * data[self.n_atoms_key].unsqueeze(-1))

        data[self.scaled_energy_key] = \
            torch.div(data[KEY.SCALED_ENERGY], data[self.n_atoms_key].unsqueeze(-1))
        data[KEY.PRED_FORCE] = data[self.scaled_force_key] * self.scale
        return data
