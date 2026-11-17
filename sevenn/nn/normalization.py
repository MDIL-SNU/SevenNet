import math
from typing import Tuple

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


def _rms_layout(irreps: Irreps) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (balance, channel_idx, scalar_mask) of EquivariantRMSNorm: per feature,
    1 / (mul * dim * n_blocks) (every irrep block weighs the same in the mean
    square), its channel index (e3nn layout [mul, 2l+1] per block) and 1 on
    the 0e features
    """
    dim = irreps.dim
    n_blocks = len(irreps)
    balance = torch.zeros(dim)
    channel_idx = torch.zeros(dim, dtype=torch.long)
    scalar_mask = torch.zeros(dim)
    start, channel = 0, 0
    for mul, ir in irreps:
        end = start + mul * ir.dim
        balance[start:end] = 1.0 / (mul * ir.dim * n_blocks)
        # e3nn layout: [mul, 2l+1] flattened per block
        channel_idx[start:end] = (
            torch.arange(channel, channel + mul).repeat_interleave(ir.dim)
        )
        if ir.l == 0 and ir.p == 1:
            scalar_mask[start:end] = 1.0
        start, channel = end, channel + mul
    return balance, channel_idx, scalar_mask


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
        balance, channel_idx, scalar_mask = _rms_layout(self.irreps)
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
    key_full (set for magnitude_residual if irreps_in != irreps_out):
    MagnitudeResidualOutro needs the full input (the RMS its norm divides
    out), so the intro also leaves x under data[key_full] (a reference, no
    copy; the residual copy stays where it was, so the serial graph and its
    gradients are those of the plain block). store_input (e3gnn/parallel
    segments, set by model_build._to_parallel_model): a tensor under a second
    key would cross the segment cut without its gradient, so the intro stores
    x itself in self_cont_tmp and the outro applies the channel copy.
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
        self.key_full = ''
        self.store_input = False

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
        if self.is_identity or self.store_input:
            data[KEY.SELF_CONNECTION_TEMP] = x.clone() if self.clone_identity else x
        else:
            out = x.new_zeros(x.shape[0], self.dim_out)
            out[:, self.dst_idx] = x[:, self.src_idx]
            data[KEY.SELF_CONNECTION_TEMP] = out
            if self.key_full != '':
                data[self.key_full] = x
        return data


@compile_mode('script')
class MagnitudeResidualOutro(nn.Module):
    """
    Magnitude-restoring residual of a pre-norm branch (config
    magnitude_residual), the outro of SelfConnectionResidualIntro:
        x <- P(x_in) + g * B(Norm(x_in)),
        g = (rho / rho_ref)^gamma = exp(gamma (log rho - log rho_ref)),
        rho = sqrt(sum_i balance_i (x_in - mean_0e(x_in))_i^2 + eps),
    rho the per-node statistic of EquivariantRMSNorm (centered over the 0e
    features, every irrep block weighted equally, the same eps), i.e. the
    magnitude the pre-norm divides out of the branch input, B the existing
    branch (conv: Lin1 -> TP -> Lin2 -> Gate -> Lin3; FFN), P the residual
    projection irreps_in -> irreps_out (identity, or the channel copy of
    SelfConnectionResidualIntro, which then also passes the full input on,
    see its key_full / store_input).
    gamma: one learnable scalar per branch, effective value clip(theta, 0,
    gamma_max), gamma_max <= 1, theta = 0 at init, so g = exp(0) = 1 exactly
    and the module is SelfConnectionOutro (bitwise: 1.0 * B + x). Gradient of
    theta = d loss / d gamma always (straight-through clip,
    theta + (clip(theta) - theta).detach()): a plain clamp passes the
    gradient at theta = 0 (PyTorch's clamp backward is inclusive) but one
    Adam step of the wrong sign leaves theta < 0 with zero gradient forever;
    with the straight-through form theta can return. The model only ever
    uses clip(theta), so g lies between 1 and (rho / rho_ref)^gamma_max.
    rho_ref: a fixed constant (default 1.0), not a data statistic. Any
    constant rho_ref is absorbed by the last linear of B (its weights and
    bias times rho_ref^-gamma), so it does not change the function class; it
    only sets the state where g = 1 (trained a08 bulk: rho ~ 1.3-2.2 for the
    blocks of layers 1-3, dev/arch-k2k3/notes.txt).
    gamma = 0 is the plain pre-norm block (increment scale-invariant in x_in);
    gamma = 1 makes the block positively 1-homogeneous in x_in
    (x + rho B(x / rho)): its increment scales with the residual stream it
    is added to, as in an unnormalized residual network, while B still sees a
    unit-RMS input (pre-norm stability kept; Xiong et al., arXiv:2002.04745).
    gamma_max <= 1 keeps every block at most linear in the stream magnitude,
    so five blocks cannot amplify it super-linearly.
    Smooth in positions (eps > 0: rho > 0, log rho finite; no abs/max/where
    on data), invariant (rho is a norm of the whole node vector; centering
    and balance act per irrep block), node-local (rho and g of a node from its
    own residual stream). In LAMMPS e3gnn/parallel the conv outro runs in the
    segment after the cut and reads rho from self_cont_tmp (a local tensor
    that crosses the cut with its gradient), so no extra tensor is exchanged
    or cached across segments; the fused LunaTP norm is not touched (its
    statistics are not differentiable outputs, so rho is recomputed here in
    PyTorch, as for norm_magnitude). Cost: ~13 small kernels forward and
    ~20 in the force backward per branch (the model is launch-bound at ~2k
    atoms: ~0.5 ms per branch on a GH200; variance / pow / addcmul
    formulations were not faster, dev/arch-k2k3/scripts/micro_outro.py).
    Ideas (no code taken): de-normalization by the removed scale, RevIN
    (T. Kim et al., Reversible Instance Normalization for Accurate
    Time-Series Forecasting against Distribution Shift, ICLR 2022) as used
    in multiple-physics pretraining (M. McCabe et al., arXiv:2310.02994);
    magnitude-preserving design (T. Karras et al., EDM2, arXiv:2312.02696).
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        eps: float = 1e-5,
        gamma_max: float = 1.0,
        rho_ref: float = 1.0,
        centering: bool = True,
        key_full: str = '',
    ) -> None:
        super().__init__()
        assert 0.0 < gamma_max <= 1.0 and rho_ref > 0.0
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.key_x = data_key_x
        self.key_full = key_full  # full input of a projecting intro (serial)
        self.eps = float(eps)
        self.gamma_max = float(gamma_max)
        self.log_rho_ref = math.log(float(rho_ref))
        balance, _, scalar_mask = _rms_layout(self.irreps_in)
        num_scalars = int(scalar_mask.sum().item())
        self.centering = centering and num_scalars > 0
        self.register_buffer('balance', balance, persistent=False)
        self.register_buffer('scalar_mask', scalar_mask, persistent=False)
        self.register_buffer(
            'scalar_mean', scalar_mask / max(num_scalars, 1), persistent=False
        )
        # residual projection as SelfConnectionResidualIntro(irreps_in, out)
        proj = SelfConnectionResidualIntro(self.irreps_in, self.irreps_out)
        self.is_identity = proj.is_identity
        self.dim_out = proj.dim_out
        self.register_buffer('src_idx', proj.src_idx.clone(), persistent=False)
        self.register_buffer('dst_idx', proj.dst_idx.clone(), persistent=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def effective_gamma(self) -> torch.Tensor:
        """clip(theta, 0, gamma_max); training: straight-through gradient"""
        theta = self.gamma
        gamma = torch.clamp(theta, 0.0, self.gamma_max)
        if self.training:
            gamma = theta + (gamma - theta).detach()
        return gamma

    def log_rho(self, x: torch.Tensor) -> torch.Tensor:
        """log of the RMS EquivariantRMSNorm divides x by (per node)"""
        if self.centering:
            x = x - (x @ self.scalar_mean).unsqueeze(-1) * self.scalar_mask
        return 0.5 * torch.log((x * x) @ self.balance + self.eps)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        res = data[KEY.SELF_CONNECTION_TEMP]
        if self.key_full != '' and self.key_full in data:
            x_in = data[self.key_full]  # serial: res is already projected
            del data[self.key_full]
        else:
            x_in = res
            if not self.is_identity:  # parallel segments: intro stored x_in
                out = res.new_zeros(res.shape[0], self.dim_out)
                out[:, self.dst_idx] = res[:, self.src_idx]
                res = out
        gain = torch.exp(
            self.effective_gamma() * (self.log_rho(x_in) - self.log_rho_ref)
        )
        data[self.key_x] = gain.unsqueeze(-1) * x + res
        del data[KEY.SELF_CONNECTION_TEMP]
        return data
