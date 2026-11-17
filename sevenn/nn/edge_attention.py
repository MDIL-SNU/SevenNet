"""
Envelope-gated neighbour attention for the convolution (config `conv_attention`).

Scalar logits from destination/source node scalars with QK-norm, a radial and
species bias, an envelope^2 prior and a learnable null logit (attention sink):
    l_ij = cap * tanh((<rms(q_i), rms(k_j)> / sqrt(d) + b_ij) / cap)
    b_ij = u . phi(r_ij) + t_src[Z_j] + t_dst[Z_i]   (phi: radial basis, no envelope)
    w_ij = s5_ij^2 exp(l_ij) / (softplus(zeta) + sum_k s5_ik^2 exp(l_ik))
s5(x) = (1-x)^4 (1 + 4x + 10x^2 + 20x^3 + 35x^4), x = r / r_c: a C3 envelope over the
whole range (DPA4 Eq. 53), independent of the model's cutoff function (XPLOR is only
C1 at r_on). By default the first layer keeps the plain normalized sum (as DPA4's
embedding stage), so the neighbour count still enters the residual stream.
w_ij scales the spherical harmonics of edge ij for this layer only; the tensor
product is linear in them, so the message becomes sum_j w_ij m_ij with unchanged
(fused) TP kernels and radial weights. A destination output gate 2*sigmoid(W s_i)
(identity at init) scales the aggregated message per channel.

Sources (ideas only, independent implementation): DPA4 (arXiv:2606.02419,
Eq. 20-24; deepmd-kit is LGPL-3.0, no code reused); QK-norm (Henry et al.,
arXiv:2010.04245); attention sink / null logit (Xiao et al., arXiv:2309.17453;
gpt-oss, arXiv:2508.10925); gated attention output (Qiu et al., arXiv:2505.06708);
logit soft-capping (Gemma 2, arXiv:2408.00118); envelope inside the softmax
(EquiformerV3, arXiv:2604.09130).

Smoothness: exp/softplus/tanh of bounded logits; env^2 makes the weight vanish with
the envelope (C2 for the polynomial envelope); the denominator is > softplus(zeta)
> 0, so an atom losing its last neighbour is smooth.

Parallel MD: the softmax runs over the incoming edges of a destination, which is
always a local atom, so it needs no extra communication. Keys of ghost sources
come from their exchanged convolution input: is_parallel (e3gnn/parallel)
concatenates KEY.NODE_FEATURE_GHOST to the local x; ML-IAP exchanges x before
this module, so x already holds ghost rows. Species biases of ghost sources read
the types from key_atom_type = KEY.ATOM_TYPE_ALL (local + ghost) in both cases.

LunaTP backend (lunatp_fast_attention, default True): the attention layers reuse
the forward's LunaTP graph (LunaTPSharedGraphConvolution) and one envelope s5^2
(LunaTPEdgeAttention), see sevenn/nn/lunatp_helper.py; same function.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .ffn_variants import _channel_index


@compile_mode('script')
class EdgeAttention(nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        irreps_msg: Irreps,
        num_radial: int,
        cutoff: float,
        num_species: int,
        layer: int,
        qk_dim: int = 32,
        logit_cap: float = 10.0,
        radial_bias: bool = True,
        null_logit: bool = True,
        out_gate: bool = True,
        eps: float = 1e-6,
        data_key_x: str = KEY.NODE_FEATURE,
    ) -> None:
        super().__init__()
        irreps_x = Irreps(irreps_x)
        sl = [s for (_, ir), s in zip(irreps_x, irreps_x.slices())
              if ir.l == 0 and ir.p == 1]
        assert len(sl) == 1, 'edge attention needs exactly one 0e block'
        self.s0, self.s1 = sl[0].start, sl[0].stop
        n0 = self.s1 - self.s0
        self.key_x = data_key_x
        self.key_filter_out = f'{layer}_edge_filter_attn'
        self.key_gate = f'{layer}_attn_gate'
        self.q = nn.Linear(n0, qk_dim, bias=False)
        self.k = nn.Linear(n0, qk_dim, bias=False)
        self.gq = nn.Parameter(torch.ones(qk_dim))
        self.gk = nn.Parameter(torch.ones(qk_dim))
        self.scale = 1.0 / math.sqrt(qk_dim)
        self.cap = float(logit_cap)
        self.eps = eps
        self.use_rb = radial_bias
        self.rb = nn.Linear(num_radial, 1, bias=False)
        nn.init.zeros_(self.rb.weight)
        self.t_src = nn.Parameter(torch.zeros(num_species, 1))
        self.t_dst = nn.Parameter(torch.zeros(num_species, 1))
        self.use_null = null_logit
        self.zeta = nn.Parameter(torch.full((1,), 0.5413))   # softplus = 1.0
        self.use_gate = out_gate
        n_ch = Irreps(irreps_msg).num_irreps
        self.og = nn.Linear(n0, n_ch)
        nn.init.zeros_(self.og.weight)
        nn.init.zeros_(self.og.bias)                # 2 * sigmoid(0) = 1
        self.inv_cutoff = 1.0 / float(cutoff)
        self.key_atom_type = KEY.ATOM_TYPE
        self.is_parallel = False

    def _rms(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(v, (v.shape[-1],), weight=g, eps=self.eps)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        s = data[self.key_x][:, self.s0: self.s1]
        s_src = s
        if self.is_parallel:  # sources may be ghosts (rows after the local ones)
            g = data[KEY.NODE_FEATURE_GHOST][:, self.s0: self.s1]
            s_src = torch.cat([s, g])
        q = self._rms(self.q(s), self.gq)
        k = self._rms(self.k(s_src), self.gk)
        dst = data[KEY.EDGE_IDX][0]   # messages flow edge_idx[1] -> edge_idx[0]
        src = data[KEY.EDGE_IDX][1]
        # index_select: its backward is index_add (fast); q[dst] would use a
        # sort-based index_put backward (dominated the step time)
        qk = q.index_select(0, dst) * k.index_select(0, src)
        logit = qk.sum(-1) * self.scale
        if self.use_rb:
            z = data[self.key_atom_type]
            logit = logit + self.rb(data[KEY.EDGE_RBF]).squeeze(-1)
            t_src = F.embedding(z, self.t_src).squeeze(-1)
            t_dst = F.embedding(z, self.t_dst).squeeze(-1)
            logit = logit + t_src.index_select(0, src) + t_dst.index_select(0, dst)
        logit = self.cap * torch.tanh(logit / self.cap)
        x = data[KEY.EDGE_LENGTH] * self.inv_cutoff
        poly = 1.0 + x * (4.0 + x * (10.0 + x * (20.0 + 35.0 * x)))
        s5 = (1.0 - x).pow(4) * poly
        a = s5 * s5 * torch.exp(logit)
        den = torch.zeros(s.shape[0], dtype=a.dtype, device=a.device)
        den = den.index_add(0, dst, a)
        if self.use_null:
            den = den + F.softplus(self.zeta)
        else:
            den = den + self.eps
        w = a / den.index_select(0, dst)
        data[self.key_filter_out] = data[KEY.EDGE_ATTR] * w.unsqueeze(-1)
        if self.use_gate:
            data[self.key_gate] = 2.0 * torch.sigmoid(self.og(s))
        return data


@compile_mode('script')
class AttentionOutGate(nn.Module):
    """x <- x * gate[channel], gate computed by EdgeAttention of the same layer"""

    def __init__(self, irreps_msg: Irreps, layer: int,
                 data_key_x: str = KEY.NODE_FEATURE) -> None:
        super().__init__()
        self.key_x = data_key_x
        self.key_gate = f'{layer}_attn_gate'
        self.register_buffer('channel_idx', _channel_index(Irreps(irreps_msg)))

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        gate = data[self.key_gate].index_select(1, self.channel_idx)
        data[self.key_x] = data[self.key_x] * gate
        return data
