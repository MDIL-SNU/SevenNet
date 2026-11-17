import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class EquivariantRMSNorm(nn.Module):
    """
    Equivariant RMS normalization of node features (EquiformerV2 / eSEN style;
    Liao et al., arXiv:2306.12059; Fu et al., arXiv:2502.12147). Independent
    implementation.
    One RMS over all irreps of a node, where every irrep block contributes
    equally regardless of its multiplicity and dimension (component
    normalization). Scalars (0e) are centered and get a bias; every channel
    gets an affine weight shared over its m components.

    magnitude=True (config norm_magnitude): the magnitude of the input,
    which the norm removes, is passed on as an invariant scalar feature,
        s = log(sum_i balance_i x_i^2 + eps)   (per node, raw input),
        y_0e <- y_0e + s * v,   v zero-initialized (one weight per 0e channel)
    i.e. twice the log of the balanced RMS of the input (the RMS the norm
    divides by, before the 0e mean is taken out; the factor 2 is absorbed by
    v), as a log-magnitude-dependent bias
    of the 0e outputs; the following linear (self_interaction_1 /
    ffn_linear_1) sees s as if it were one more 0e input channel
    (Lin(y + s v) = Lin(y) + s Lin(v)). Smooth (eps > 0, no abs/max; s >=
    log eps), invariant (a norm, added to 0e channels only), zero at
    init (the module is the plain norm). The uncentered RMS is one GEMV of
    x^2 (no second centering pass next to the fused LunaTP kernel, whose
    statistics are not differentiable outputs).
    Purpose: a pre-norm block sees a unit-RMS input whatever the residual-
    stream magnitude, so growth of the stream under compression (dimer stream
    RMS 4 -> 161 between 2.6 and 0.15 A, dev/diatomics-eval/
    short_range_coverage.txt section 4) is invisible to every block; s keeps
    it visible, as one log-scale number per node. The
    stream itself, the normalized directions and the training stability of
    pre-norm (R. Xiong et al., arXiv:2002.04745) are unchanged. Related:
    Peri-LN on the magnitude growth of pre-norm residual streams (J. Kim et
    al., arXiv:2502.02732), magnitude-preserving layers (T. Karras et al.,
    EDM2, arXiv:2312.02696), FiLM-style scalar conditioning (E. Perez et al.,
    arXiv:1709.07871); own design.
    """

    def __init__(
        self,
        irreps: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        eps: float = 1e-5,
        centering: bool = True,
        affine: bool = True,
        magnitude: bool = False,
    ) -> None:
        super().__init__()
        self.irreps = Irreps(irreps)
        self.key_x = data_key_x
        self.eps = eps
        self.centering = centering
        self.affine = affine

        dim = self.irreps.dim
        n_blocks = len(self.irreps)
        balance = torch.zeros(dim)
        channel_idx = torch.zeros(dim, dtype=torch.long)
        scalar_mask = torch.zeros(dim)
        start, channel = 0, 0
        for mul, ir in self.irreps:
            end = start + mul * ir.dim
            balance[start:end] = 1.0 / (mul * ir.dim * n_blocks)
            # e3nn layout: [mul, 2l+1] flattened per block
            channel_idx[start:end] = (
                torch.arange(channel, channel + mul).repeat_interleave(ir.dim)
            )
            if ir.l == 0 and ir.p == 1:
                scalar_mask[start:end] = 1.0
            start, channel = end, channel + mul
        self.register_buffer('balance', balance)
        self.register_buffer('channel_idx', channel_idx)
        self.register_buffer('scalar_mask', scalar_mask)
        self.num_scalars = int(scalar_mask.sum().item())
        if self.num_scalars == 0:
            self.centering = False

        if affine:
            self.weight = nn.Parameter(torch.ones(self.irreps.num_irreps))
            self.bias = nn.Parameter(torch.zeros(dim))  # used on scalars only
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if magnitude:
            if self.num_scalars == 0:
                raise ValueError('norm magnitude feature needs 0e channels')
            # one entry per feature (as bias), only those of 0e channels act
            self.mag_weight = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('mag_weight', None)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        if self.centering:
            mean = (x * self.scalar_mask).sum(-1, keepdim=True) / self.num_scalars
            x = x - mean * self.scalar_mask
        inv_rms = torch.rsqrt(
            (x.pow(2) * self.balance).sum(-1, keepdim=True) + self.eps
        )
        x = x * inv_rms
        if self.weight is not None and self.bias is not None:
            x = x * self.weight[self.channel_idx]
            if self.centering:
                x = x + self.bias * self.scalar_mask
        mag = self.mag_weight
        if mag is not None:  # log RMS of the input, on the 0e channels
            x_in = data[self.key_x]
            s = torch.log((x_in * x_in) @ self.balance + self.eps)
            x = torch.addcmul(x, s.unsqueeze(-1), mag * self.scalar_mask)
        data[self.key_x] = x
        return data


@compile_mode('script')
class SelfConnectionResidualIntro(nn.Module):
    """
    Identity residual for pre-norm blocks: save x for SelfConnectionOutro.
    If irreps_in != irreps_out, channels of matching irreps are copied
    (zero padded or truncated); other irreps of irreps_out start from zero.
    clone_identity (set for the e3gnn/parallel segments): store a copy instead
    of x itself. LAMMPS e3gnn/parallel back-propagates every segment w.r.t. its
    inputs x and self_cont_tmp separately (autograd.grad per segment); if
    self_cont_tmp were x_in itself, it would be an ancestor of the next x
    (= Linear(Norm(x_in))) and that path would be counted twice (or its graph
    freed by the first backward). The copy has the same value.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        **kwargs,  # compatibility with other self connection intros
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.key_x = data_key_x
        self.is_identity = self.irreps_in == self.irreps_out
        self.dim_out = self.irreps_out.dim
        self.clone_identity = False

        src, dst = [], []
        if not self.is_identity:
            in_slices = self.irreps_in.slices()
            out_slices = self.irreps_out.slices()
            used = set()
            for i_out, (mul_o, ir_o) in enumerate(self.irreps_out):
                for i_in, (mul_i, ir_i) in enumerate(self.irreps_in):
                    if ir_i != ir_o or i_in in used:
                        continue
                    mul = min(mul_i, mul_o)
                    s_in, s_out = in_slices[i_in].start, out_slices[i_out].start
                    src.append(torch.arange(s_in, s_in + mul * ir_i.dim))
                    dst.append(torch.arange(s_out, s_out + mul * ir_o.dim))
                    used.add(i_in)
                    break
        empty = torch.zeros(0, dtype=torch.long)
        self.register_buffer('src_idx', torch.cat(src) if src else empty)
        self.register_buffer('dst_idx', torch.cat(dst) if dst else empty)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        if self.is_identity:
            data[KEY.SELF_CONNECTION_TEMP] = x.clone() if self.clone_identity else x
        else:
            out = x.new_zeros(x.shape[0], self.dim_out)
            out[:, self.dst_idx] = x[:, self.src_idx]
            data[KEY.SELF_CONNECTION_TEMP] = out
        return data
