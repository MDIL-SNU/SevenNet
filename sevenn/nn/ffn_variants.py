"""
Optional pieces of the metaformer FFN / residual branches (config switches,
all off by default):
- SwiGLUGate: equivariant GLU nonlinearity. Scalars: silu(b) * a (Shazeer,
  GLU Variants Improve Transformer, arXiv:2002.05202); l>0 channels keep the
  gated form v * silu(g) of the e3nn Gate.
"""

from typing import Callable, Dict

import torch
import torch.nn as nn
from e3nn.math import normalize2mom
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


def _channel_index(irreps: Irreps) -> torch.Tensor:
    """component -> channel index for e3nn's [mul, 2l+1] flattened layout"""
    idx, c0 = [], 0
    for mul, ir in irreps:
        idx.append(torch.arange(c0, c0 + mul).repeat_interleave(ir.dim))
        c0 += mul
    return torch.cat(idx) if idx else torch.zeros(0, dtype=torch.long)


@compile_mode('script')
class SwiGLUGate(nn.Module):
    """
    Input  [a: n0 x 0e | b: n0 x 0e | g: one 0e per l>0 channel | v: l>0]
    Output [silu(b) * a | v * silu(g)]   (irreps_x = [n0 x 0e | v irreps])
    """

    def __init__(
        self,
        irreps_x: Irreps,
        act_scalar_dict: Dict[str, Callable],
        act_gate_dict: Dict[str, Callable],
        data_key_x: str = KEY.NODE_FEATURE,
    ) -> None:
        super().__init__()
        self.key_x = data_key_x
        irreps_x = Irreps(irreps_x)
        scalars = Irreps([(m, ir) for m, ir in irreps_x if ir.l == 0])
        gated = Irreps([(m, ir) for m, ir in irreps_x if ir.l > 0])
        assert all(ir.p == 1 for _, ir in scalars), 'SwiGLU: even scalars only'
        self.n0 = scalars.dim
        self.act = normalize2mom(act_scalar_dict['e'])
        # l>0 part: v * act(g) per channel, as e3nn Gate (normalize2mom gate
        # activation, 'component' elementwise product == plain multiply), but
        # as one gather + multiply so that it is cheap on every backend
        self.gate_act = normalize2mom(act_gate_dict['e'])
        self.n_gate = gated.num_irreps
        self.register_buffer('gated_channel_idx', _channel_index(gated))
        gates = Irreps([(m, (0, 1)) for m, _ in gated])
        self.irreps_in = scalars + scalars + gates + gated

    def get_gate_irreps_in(self) -> Irreps:
        return self.irreps_in

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        n0, n1 = self.n0, 2 * self.n0 + self.n_gate
        s = x[:, :n0] * self.act(x[:, n0: 2 * n0])
        if self.n_gate > 0:
            g = self.gate_act(x[:, 2 * n0: n1])
            g = g.index_select(1, self.gated_channel_idx)   # index_add backward
            s = torch.cat([s, x[:, n1:] * g], dim=-1)
        data[self.key_x] = s
        return data
