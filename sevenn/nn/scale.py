import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


#TODO: rename these confuising scale, total, peratom things
#      SCALED_PER_ATOM_ENERGY is not per atom energy (its total energy, see
@compile_mode('script')
class Rescale(nn.Module):
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
        scale_per_atom: bool = False,
        train_shift_scale: bool = False,
        is_stress: bool = False,
    ):
        super().__init__()
        self.shift = torch.FloatTensor([shift])
        self.scale = torch.FloatTensor([scale])
        self.scaled_energy_key = scaled_energy_key
        self.scaled_force_key = scaled_force_key
        self.n_atoms_key = n_atoms_key
        self.scale_per_atom = scale_per_atom
        self.is_stress = is_stress
        if train_shift_scale:
            self.shift = nn.Parameter(self.shift)
            self.scale = nn.Parameter(self.scale)
        #self.scale_only_energy = scale_only_energy

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self.scale_per_atom:  # 'un'scale atomic energy
            data[KEY.ATOMIC_ENERGY] = \
                data[KEY.ATOMIC_ENERGY] * self.scale + self.shift

        data[KEY.PRED_TOTAL_ENERGY] = \
            (data[KEY.SCALED_ENERGY] * self.scale) +\
            (self.shift * data[self.n_atoms_key].unsqueeze(-1))

        data[KEY.PRED_PER_ATOM_ENERGY] = \
            torch.div(data[KEY.PRED_TOTAL_ENERGY], data[self.n_atoms_key].unsqueeze(-1))

        data[self.scaled_energy_key] = \
            torch.div(data[KEY.SCALED_ENERGY], data[self.n_atoms_key].unsqueeze(-1))
        data[KEY.PRED_FORCE] = data[self.scaled_force_key] * self.scale
        if self.is_stress:
            data[KEY.PRED_STRESS] = data[KEY.SCALED_STRESS] * self.scale
        return data
