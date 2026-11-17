import math

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .util import frozen_product, frozen_tanh


@compile_mode('script')
class EdgeGate(nn.Module):
    """
    Bounded positive edge gate (config edge_gate): every tensor-product
    weight of the edge j -> i (all paths, from the element-pair radial MLP)
    is multiplied by one factor
        b_ij = 1 + alpha * tanh(<P u_i, Q u_j> / sqrt(d)),  u = RMSNorm(s),
        alpha = alpha_max * tanh(theta),  theta = 0 at init (b_ij = 1),
    where s are the scalar (0e) channels of the layer's convolution input
    (the features the convolution already gathers from the neighbours),
    u = s / sqrt(mean(s^2) + eps) per node (no weights) and P, Q: scalars -> d
    are node-wise linear maps (gathered per edge). The norm keeps the logit
    ~N(0, 1) at init whatever the scale of s (at a08's random init RMS(s) ~ 2
    gives |logit| ~ 5-9, a saturated, almost binary gate), as QK-norm does for
    attention logits (A. Henry et al., arXiv:2010.04245; M. Dehghani et al.,
    ViT-22B, arXiv:2302.05442); it is the P(RMSNorm h) of the bond-order gate
    sketch in dev/ai-arch-transfer (graph note C9).
    b_ij lies in (1 - alpha_max, 1 + alpha_max), alpha_max < 1: positive and
    bounded for any parameters and features, so it can neither flip the sign
    of an edge's contribution nor switch an edge off, and at most rescales
    one edge relative to another by (1 + alpha_max) / (1 - alpha_max). No
    softmax, no normalisation over neighbours: the aggregation stays the plain
    sum with the fixed denominator (extensive), and db_ij/dr_k vanishes for
    k not in {i, j} except through the node features (no neighbour
    competition terms in the force constants).
    theta = 0 gives exactly the model without the gate (bitwise; P and Q are
    random so that theta receives a gradient at init, they themselves get
    one once alpha != 0). No distance bias: a monotone bias would have to be
    tied to the sign of alpha to strengthen walls, and b is bounded anyway.
    Physics and ideas: Tersoff bond order, bond strength as a bounded
    function of the environment (J. Tersoff, PRB 37, 6991 (1988)); sigmoid
    instead of softmax attention (bounded per-edge weights, no competition;
    J. Ramapuram et al., arXiv:2409.04431); output gating (Z. Qiu et al.,
    Qwen gated attention, arXiv:2505.06708) - here kept dense around 1
    instead of sparse, so it cannot suppress short-range walls; Orb-v3
    sigmoid edge gating (B. Rhodes et al., arXiv:2504.06231). Bounded,
    identity-initialized version of the node-conditioned radial MLP
    (NodeCondRadialInput) and of a scalar, unnormalised edge attention.
    Implementation: the factor is appended as the last column of the pair
    radial MLP input (key {layer}_edge_pair_input_eg); PairRadialMLP
    (edge_gate=True) multiplies its envelope by it, i.e. scales the last
    hidden layer (= all output weights, the last layer is linear without
    bias) before any (fused) tensor-product kernel. One GEMM for P and Q,
    gathers on the transposed projection, one addcmul: the model is launch-
    bound at ~2k atoms, so the cost is the kernel count, not the width d.
    Layer 0 is skipped by default (model_build, skip_first): its convolution
    input depends on the element only, so its gate would be a constant
    element-pair factor, redundant with the element-pair radial MLP.
    is_parallel (e3gnn/parallel): x holds the local nodes only; the source
    scalars of ghost neighbours come from KEY.NODE_FEATURE_GHOST (the
    exchanged convolution input), as in NodeCondRadialInput. ML-IAP: x holds
    local + ghost rows (exchanged after self_interaction_1). No extra
    communication.
    """

    def __init__(
        self,
        irreps_x,
        layer: int,
        dim: int = 16,
        alpha_max: float = 0.9,
        eps: float = 1e-5,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_pair: str = KEY.EDGE_PAIR_INPUT,
    ) -> None:
        super().__init__()
        irreps_x = Irreps(irreps_x)
        sl = [s for (_, ir), s in zip(irreps_x, irreps_x.slices())
              if ir.l == 0 and ir.p == 1]
        assert len(sl) == 1, 'edge_gate needs exactly one 0e block'
        assert 0.0 < alpha_max < 1.0, 'edge_gate alpha_max must be in (0, 1)'
        self.s0, self.s1 = sl[0].start, sl[0].stop
        n = self.s1 - self.s0
        self.n = n
        self.dim = int(dim)
        self.alpha_max = float(alpha_max)
        self.eps = float(eps)
        # [P | Q] stored N(0, 1) (e3nn convention), applied as u @ W / sqrt(n);
        # the logit's 1/sqrt(d) is folded in as d^-1/4 on both projections
        self.weight = nn.Parameter(torch.randn(n, 2 * self.dim))
        self.scale = 1.0 / math.sqrt(n) / self.dim ** 0.25
        self.theta = nn.Parameter(torch.zeros(1))
        self.register_buffer('one', torch.ones(1), persistent=False)
        self.key_x = data_key_x
        self.key_pair = data_key_pair
        self.key_out = f'{layer}_edge_pair_input_eg'
        self.is_parallel = False

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        s = data[self.key_x][:, self.s0: self.s1]
        if self.is_parallel:  # sources can be ghosts: local rows first
            s = torch.cat([s, data[KEY.NODE_FEATURE_GHOST][:, self.s0: self.s1]])
        u = torch.rms_norm(s, [self.n], None, self.eps)
        # [2d, rows]: gathering columns of the transposed projection is ~5x
        # faster than gathering rows of [rows, d] (index_select, d >= 4)
        if torch.jit.is_scripting():
            w = self.weight * self.scale
        else:
            w = frozen_product(self.weight, self.scale)
        pq = w.t() @ u.t()
        # split, not slices: one cat in the backward (same values)
        p, q = torch.split(pq, self.dim)
        dst = data[KEY.EDGE_IDX][0]
        src = data[KEY.EDGE_IDX][1]
        logit = (p.index_select(1, dst) * q.index_select(1, src)).sum(0)
        # 1 + alpha_max tanh(theta) tanh(logit)
        if torch.jit.is_scripting():
            tanh_theta = torch.tanh(self.theta)
        else:
            tanh_theta = frozen_tanh(self.theta)
        gate = torch.addcmul(
            self.one, tanh_theta, torch.tanh(logit), value=self.alpha_max
        )
        pair = data[self.key_pair]
        data[self.key_out] = torch.cat(
            [pair, gate.to(pair.dtype).unsqueeze(-1)], dim=-1
        )
        return data
