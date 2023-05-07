import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class Rescale(nn.Module):
    """
    Scaling and shifting energy and force.
    """
    def __init__(
        self,
        shift: float,
        scale: float,
        train_shift_scale: bool = False,
        is_stress: bool = False,
    ):
        super().__init__()
        self.shift = \
            nn.Parameter(torch.FloatTensor([shift]),
                         requires_grad=train_shift_scale)
        self.scale = \
            nn.Parameter(torch.FloatTensor([scale]),
                         requires_grad=train_shift_scale)
        self.is_stress = is_stress
        #self.scale_only_energy = scale_only_energy

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        # scaled total energy to total energy
        data[KEY.PRED_TOTAL_ENERGY] = \
            (data[KEY.SCALED_ENERGY] * self.scale) +\
            (self.shift * data[KEY.NUM_ATOMS].unsqueeze(-1))

        # TODO: replace trainer per atom energy to total energy
        #       (divide in trainer not here) # for trainer (line 36)
        data[KEY.PRED_PER_ATOM_ENERGY] = \
            torch.div(data[KEY.PRED_TOTAL_ENERGY], data[KEY.NUM_ATOMS].unsqueeze(-1))

        """
        data[self.scaled_energy_key] =
            torch.div(data[KEY.SCALED_ENERGY], data[KEY.NUM_ATOMS].unsqueeze(-1))
        """

        data[KEY.PRED_FORCE] = data[KEY.SCALED_FORCE] * self.scale
        if self.is_stress:
            data[KEY.PRED_STRESS] = data[KEY.SCALED_STRESS] * self.scale
        return data
