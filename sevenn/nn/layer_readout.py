"""
Per-layer readout (config per_layer_readout): every interaction layer but
the last adds an atomic-energy term read out linearly from its output
features, E_i = sum_t E_i^(t) (I. Batatia et al., MACE, arXiv:2206.07697:
site energies read out after every layer, linear readouts for the
intermediate layers and the model's own readout for the last one; code
github.com/ACEsuit/mace, mace/modules/blocks.py LinearReadoutBlock; idea
only). The heads are e3nn linears to 1x0e (only the 0e
channels contribute), zero-initialized so that the model starts as the
model without them, and their sum is added to the scaled atomic energy
before the shift/scale.

Parallel MD (LAMMPS e3gnn/parallel back-propagates every segment on its own
w.r.t. (edge_vec, x, self_cont_tmp, x_ghost)): the running sum of the
per-layer energies (KEY.LAYER_ENERGY) computed in an earlier segment would
reach the energy as a separate tensor and lose its force contribution. In
the parallel segments it therefore travels inside self_cont_tmp, the local
residual that already crosses every segment boundary with its gradient:
LayerEnergyPack appends it as one extra column at the end of a segment,
LayerEnergyUnpack splits it off at the start of the next. self_cont_tmp is
not communicated, so the exchange is unchanged; the serial model never
packs (model_build._to_parallel_model inserts the pair).
"""
import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .linear import IrrepsLinear


@compile_mode('script')
class ZeroInitIrrepsLinear(IrrepsLinear):
    """IrrepsLinear whose weights are zero when instantiated"""

    def instantiate(self) -> None:
        super().instantiate()
        assert self.linear is not None
        with torch.no_grad():
            for p in self.linear.parameters():
                p.zero_()


@compile_mode('script')
class LayerEnergyAccumulate(nn.Module):
    """KEY.LAYER_ENERGY (+)= data[key_in] ([N, 1] energy of one layer)"""

    def __init__(self, data_key_in: str, first: bool) -> None:
        super().__init__()
        self.key_in = data_key_in
        self.first = first

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        e = data[self.key_in]
        if not self.first:
            e = data[KEY.LAYER_ENERGY] + e
        data[KEY.LAYER_ENERGY] = e
        return data


@compile_mode('script')
class LayerEnergyToReadout(nn.Module):
    """scaled atomic energy += sum of the per-layer energies"""

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.SCALED_ATOMIC_ENERGY] = (
            data[KEY.SCALED_ATOMIC_ENERGY] + data[KEY.LAYER_ENERGY]
        )
        return data


@compile_mode('script')
class LayerEnergyPack(nn.Module):
    """parallel segments: self_cont_tmp <- [self_cont_tmp | layer energy]"""

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.SELF_CONNECTION_TEMP] = torch.cat(
            [data[KEY.SELF_CONNECTION_TEMP], data[KEY.LAYER_ENERGY]], dim=-1
        )
        return data


@compile_mode('script')
class LayerEnergyUnpack(nn.Module):
    """parallel segments: inverse of LayerEnergyPack (start of a segment)"""

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        packed = data[KEY.SELF_CONNECTION_TEMP]
        dim = packed.shape[1] - 1
        sc, e = torch.split(packed, [dim, 1], dim=1)
        data[KEY.SELF_CONNECTION_TEMP] = sc
        data[KEY.LAYER_ENERGY] = e
        return data
