"""
Optional pieces of the metaformer FFN / residual branches (config switches,
all off by default):
- SwiGLUGate: equivariant GLU nonlinearity. Scalars: silu(b) * a (Shazeer,
  GLU Variants Improve Transformer, arXiv:2002.05202); l>0 channels keep the
  gated form v * silu(g) of the e3nn Gate.
- LayerScale: learnable per-channel scale of a residual branch (Touvron et
  al., Going deeper with Image Transformers, arXiv:2103.17239).
- ElementFiLM: element-dependent per-channel scale (all irreps) and shift
  (scalars) of hidden features (Perez et al., FiLM, arXiv:1709.07871).
- S2BilinearGate: exactly equivariant bilinear product of two hidden
  features on a spherical quadrature grid (Gaunt tensor product), plus a
  SwiGLU scalar branch (ffn_type 's2_bilinear', see the class docstring).
- AddToResidual: adds a branch output to the saved residual self_cont_tmp
  (parallel_block: conv and FFN branches from the same pre-norm input).
"""

import math
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


@compile_mode('script')
class LayerScale(nn.Module):
    """x <- gamma[channel] * x, gamma initialized to `init`"""

    def __init__(
        self, irreps: Irreps, init: float, data_key_x: str = KEY.NODE_FEATURE
    ) -> None:
        super().__init__()
        irreps = Irreps(irreps)
        self.key_x = data_key_x
        self.gamma = nn.Parameter(torch.full((irreps.num_irreps,), float(init)))
        self.register_buffer('channel_idx', _channel_index(irreps))

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_x] = data[self.key_x] * self.gamma[self.channel_idx]
        return data


@compile_mode('script')
class ElementFiLM(nn.Module):
    """
    x <- x * (1 + gamma[Z, channel]) + beta[Z] (beta on 0e components only).
    Tables start at zero (identity). Node types from KEY.ATOM_TYPE.
    """

    def __init__(
        self, irreps: Irreps, num_species: int, data_key_x: str = KEY.NODE_FEATURE
    ) -> None:
        super().__init__()
        irreps = Irreps(irreps)
        self.key_x = data_key_x
        self.gamma = nn.Parameter(torch.zeros(num_species, irreps.num_irreps))
        self.register_buffer('channel_idx', _channel_index(irreps))
        sidx, start = [], 0
        for mul, ir in irreps:
            end = start + mul * ir.dim
            if ir.l == 0 and ir.p == 1:
                sidx.append(torch.arange(start, end))
            start = end
        scalar_idx = torch.cat(sidx) if sidx else torch.zeros(0, dtype=torch.long)
        self.register_buffer('scalar_idx', scalar_idx)
        self.beta = nn.Parameter(torch.zeros(num_species, len(scalar_idx)))

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        z = data[KEY.ATOM_TYPE]
        x = data[self.key_x]
        # expand the small [species, .] tables first, then one gather per table
        scale = 1.0 + self.gamma[:, self.channel_idx]
        shift = torch.zeros_like(scale).index_copy(1, self.scalar_idx, self.beta)
        data[self.key_x] = torch.addcmul(
            F.embedding(z, shift), x, F.embedding(z, scale)
        )
        return data


def lebedev_38() -> np.ndarray:
    """
    Lebedev rule of algebraic order 9 (38 points), weights summing to 1:
    6 octahedron vertices (1/105), 8 cube vertices (9/280) and 24 points
    (+-p, +-q, 0) and permutations with p, q = sqrt((1 +- 1/sqrt(3)) / 2)
    (1/35). Integrates every polynomial of degree <= 9 on S2 exactly
    (V. I. Lebedev, Zh. Vychisl. Mat. Mat. Fiz. 15, 48 (1975); same rule as
    scipy.integrate.lebedev_rule(9)). Returns [38, 4] = (x, y, z, w).
    """
    pts: List[List[float]] = []
    for ax in range(3):
        for s in (1.0, -1.0):
            v = [0.0, 0.0, 0.0]
            v[ax] = s
            pts.append(v + [1.0 / 105.0])
    c = 1.0 / math.sqrt(3.0)
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                pts.append([sx * c, sy * c, sz * c, 9.0 / 280.0])
    p = math.sqrt((1.0 + c) / 2.0)
    q = math.sqrt((1.0 - c) / 2.0)
    for i, j in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)):
        for si in (1.0, -1.0):
            for sj in (1.0, -1.0):
                v = [0.0, 0.0, 0.0]
                v[i], v[j] = si * p, sj * q
                pts.append(v + [1.0 / 35.0])
    return np.asarray(pts, dtype=np.float64)


def s2_grid_matrices(lmax: int):
    """
    Real spherical harmonics (e3nn, 'component' normalization: mean of Y^2
    over the sphere = 1) on the 38-point Lebedev grid.
    Returns (to_grid [K, P], from_grid [P, K]) in float64 with
    K = (lmax+1)^2: f(grid) = c @ to_grid and c = f(grid) @ from_grid, so
    to_grid @ from_grid = I (exact for lmax <= 3: degree 2*lmax <= 9).
    """
    from e3nn.o3 import spherical_harmonics

    if lmax > 3:
        raise NotImplementedError('S2 grid: Lebedev 38 is exact for lmax <= 3')
    grid = torch.from_numpy(lebedev_38())
    y = spherical_harmonics(
        list(range(lmax + 1)), grid[:, :3], normalize=True,
        normalization='component',
    )  # [P, K]
    return y.T.contiguous(), grid[:, 3:] * y


@compile_mode('script')
class S2BilinearGate(nn.Module):
    """
    FFN nonlinearity 'ffn_type: s2_bilinear' (metaformer block): bilinear
    product of two equivariant features on the sphere; scalars additionally
    pass through the scalar activation as in the e3nn Gate.

    Input  [s: n0 x 0e | A|B: 2C x l for every degree l = 0..L]
           (A = first C channels of each degree block, B = the next C)
    Output [act(s): n0 x 0e | Proj(Grid(A) * Grid(B)): C x l, l = 0..L]

    Grid(A)(r_p) = sum_lm A_lm Y_lm(r_p) on a quadrature grid r_p and Proj
    is the weighted transpose (Proj Grid = I). With k_max = 0 (plain S2
    functions) and natural-parity irreps (0e, 1o, 2e, 3o) the result is the
    Gaunt tensor product truncated to l <= L: exactly O(3)-equivariant as
    long as the grid integrates degree 3L exactly (Lebedev 38 = degree 9 for
    L = 3), and a polynomial (smooth) map of the input. Its l = 0 output
    holds the per-degree invariants sum_m A_lm B_lm, its l > 0 outputs
    include scalar-gated vectors A_00 B_lm. Each output degree is scaled so
    that unit-variance independent inputs give unit second moment (as e3nn
    normalize2mom does for the gate activations).
    C defaults to half the widest l > 0 multiplicity of irreps_x (the FFN
    hidden irreps), i.e. A|B together are as wide as that block.
    Layout changes are one index_select each way; grid transforms are plain
    GEMMs with fixed buffers.

    Sources (ideas only, own implementation):
      - Gaunt tensor product: S. Luo, T. Chen, A. S. Krishnapriyan,
        "Enabling Efficient Equivariant Operations in the Fourier Basis via
        Gaunt Tensor Products", ICLR 2024, arXiv:2401.10216.
      - DPA4 "WBN" bilinear FFN on an SO(3)/S2 grid, arXiv:2606.02419
        Eq. 28-32, Table S-8 (deepmd-kit LGPL-3.0: no code reused).
      - EquiformerV3 SwiGLU-S2 FFN, arXiv:2604.09130.
    """

    def __init__(
        self,
        irreps_x: Irreps,
        act_scalar_dict: Dict[str, Callable],
        channels: Optional[int] = None,
        data_key_x: str = KEY.NODE_FEATURE,
    ) -> None:
        super().__init__()
        self.key_x = data_key_x
        irreps_x = Irreps(irreps_x)
        for _, ir in irreps_x:
            if ir.p != (-1) ** ir.l:
                raise ValueError(
                    f's2_bilinear needs natural-parity irreps, got {irreps_x}'
                )
        lmax = max(ir.l for _, ir in irreps_x)
        if lmax < 1:
            raise ValueError('s2_bilinear needs l > 0 irreps')
        n0 = sum(mul for mul, ir in irreps_x if ir.l == 0)
        if channels is None:
            channels = max(mul for mul, ir in irreps_x if ir.l > 0) // 2
        C = int(channels)
        assert C > 0
        K = (lmax + 1) ** 2
        self.n0, self.C, self.K = n0, C, K
        self.n_ab = 2 * C * K
        self.act = normalize2mom(act_scalar_dict['e'])

        degrees = [(1, (l, (-1) ** l)) for l in range(lmax + 1)]
        self.irreps_in = Irreps(
            [(n0, (0, 1))] + [(2 * C, ir) for _, ir in degrees]
        )
        self.irreps_out = Irreps(
            [(n0, (0, 1))] + [(C, ir) for _, ir in degrees]
        )

        # e3nn layout of the A|B input region ([2C, 2l+1] per degree) ->
        # [2, C, K]; and [C, K] -> e3nn layout of the output ([C, 2l+1])
        perm_in = torch.empty(self.n_ab, dtype=torch.long)
        perm_out: List[int] = []
        off = 0
        for l in range(lmax + 1):
            d = 2 * l + 1
            for h in range(2):
                for c in range(C):
                    for m in range(d):
                        perm_in[(h * C + c) * K + l * l + m] = off
                        off += 1
            for c in range(C):
                for m in range(d):
                    perm_out.append(c * K + l * l + m)
        self.register_buffer('perm_in', perm_in)
        self.register_buffer('perm_out', torch.tensor(perm_out, dtype=torch.long))

        to_grid, from_grid = s2_grid_matrices(lmax)
        # Gaunt coefficients G[i, j, k] = mean_S2(Y_i Y_j Y_k) (exact on the
        # grid); E[out_k^2] = sum_ij G_ijk^2 for unit-variance inputs
        gaunt = torch.einsum('ip,jp,pk->ijk', to_grid, to_grid, from_grid)
        var = gaunt.pow(2).sum((0, 1))
        scale = torch.empty(K, dtype=torch.float64)
        for l in range(lmax + 1):
            sl = slice(l * l, (l + 1) ** 2)
            scale[sl] = var[sl].mean().rsqrt()
        dtype = torch.get_default_dtype()
        self.register_buffer('to_grid', to_grid.to(dtype))
        self.register_buffer('from_grid', (from_grid * scale).to(dtype))

    def get_gate_irreps_in(self) -> Irreps:
        return self.irreps_in

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        n = x.shape[0]
        # split (not slicing): one cat in the backward instead of zero-filled
        # full-width gradients that are added up
        s, ab = torch.split(x, [self.n0, self.n_ab], dim=1)
        ab = ab.index_select(1, self.perm_in).view(n, 2 * self.C, self.K)
        g = torch.matmul(ab, self.to_grid)                    # [N, 2C, P]
        ga, gb = torch.split(g, self.C, dim=1)                # cat backward
        z = torch.matmul(ga * gb, self.from_grid).view(n, -1)  # [N, C*K]
        data[self.key_x] = torch.cat(
            [self.act(s), z.index_select(1, self.perm_out)], dim=-1
        )
        return data


@compile_mode('script')
class AddToResidual(nn.Module):
    """
    self_cont_tmp <- self_cont_tmp + data[key] (metaformer parallel_block:
    the FFN branch output joins the residual before self_interaction_1, so
    that it crosses the LAMMPS e3gnn/parallel segment cut with its gradient)
    """

    def __init__(self, data_key_in: str) -> None:
        super().__init__()
        self.key_in = data_key_in

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.SELF_CONNECTION_TEMP] = (
            data[KEY.SELF_CONNECTION_TEMP] + data[self.key_in]
        )
        return data
