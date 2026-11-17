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
    """

    def __init__(
        self,
        irreps: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        eps: float = 1e-5,
        centering: bool = True,
        affine: bool = True,
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
