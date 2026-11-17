import math
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class ShortRangeBasis(nn.Module):
    """
    Second, short-range radial basis (config radial_short_basis):
        f_n(r) = (2 / r_s) sin(n pi r / r_s) / r * u(r / r_s),  n = 1..N
    for r < r_s and exactly 0 beyond, with frozen Bessel frequencies
    (DimeNet, J. Gasteiger et al., arXiv:2003.03123) and the DimeNet
    polynomial envelope u (p = 6: u, u', u'' vanish at r_s, so every f_n is
    C2 at r_s; the switch at r_s is a branch between two functions that
    agree there up to the second derivative).
    """

    def __init__(
        self, cutoff: float = 3.0, num_basis: int = 6, poly_p: int = 6
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.num_basis = int(num_basis)
        self.prefactor = 2.0 / self.cutoff
        p = float(poly_p)
        self.p = p
        self.c0 = (p + 1.0) * (p + 2.0) / 2.0
        self.c1 = p * (p + 2.0)
        self.c2 = p * (p + 1.0) / 2.0
        freq = torch.tensor(
            [n * math.pi / self.cutoff for n in range(1, self.num_basis + 1)]
        )
        self.register_buffer('freq', freq, persistent=False)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = r / self.cutoff
        # beyond r_s: x = 1, where u(1) = 0 exactly (integer coefficients) and
        # no gradient flows (where), so f and its derivatives are exactly 0
        x = torch.where(x < 1.0, x, torch.ones_like(x))
        env = 1.0 - torch.pow(x, self.p) * (self.c0 - x * (self.c1 - self.c2 * x))
        ur = r.unsqueeze(-1)
        f = torch.sin(self.freq.to(r.dtype) * ur) * (self.prefactor / ur)
        return f * env.unsqueeze(-1)


@compile_mode('script')
class EdgePairInput(nn.Module):
    """
    Pack the input of PairRadialMLP per edge:
        [radial basis (no envelope) (| short-range basis, if short_basis) |
         envelope | type of src | type of dst]
    Needs EdgeEmbedding(save_components=True) and node types in KEY.ATOM_TYPE,
    so it must come after onehot_idx_to_onehot.
    Parallel MD (e3gnn/parallel): edge sources can be ghosts, so the types are
    read from KEY.ATOM_TYPE_ALL (local + ghost), and every segment rebuilds the
    input after its own edge_embedding (the rbf/envelope must depend on that
    segment's edge vectors; see model_build._to_parallel_model). The
    short-range basis is computed here from that segment's KEY.EDGE_LENGTH.
    """

    def __init__(
        self,
        data_key_out: str = KEY.EDGE_PAIR_INPUT,
        data_key_atom_type: str = KEY.ATOM_TYPE,
        short_basis: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.key_out = data_key_out
        self.key_atom_type = data_key_atom_type
        self.use_short = short_basis is not None
        self.short = ShortRangeBasis(**(short_basis or {}))

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        rbf = data[KEY.EDGE_RBF]
        env = data[KEY.EDGE_ENVELOPE].unsqueeze(-1)
        atom_type = data[self.key_atom_type].to(rbf.dtype)
        edge_src = data[KEY.EDGE_IDX][1]
        edge_dst = data[KEY.EDGE_IDX][0]
        parts = [rbf]
        if self.use_short:  # next to rbf: one matmul in PairRadialMLP
            parts.append(self.short(data[KEY.EDGE_LENGTH]).to(rbf.dtype))
        parts += [
            env,
            atom_type[edge_src].unsqueeze(-1),
            atom_type[edge_dst].unsqueeze(-1),
        ]
        data[self.key_out] = torch.cat(parts, dim=-1)
        return data


@compile_mode('script')
class _ScaledLinear(nn.Module):
    """x @ W / sqrt(fan_in), W ~ N(0, 1), no bias (as e3nn FullyConnectedNet)."""

    def __init__(self, h_in: int, h_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(h_in, h_out))
        self.norm = 1.0 / math.sqrt(h_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ (self.weight * self.norm)  # scale the small matrix


@compile_mode('script')
class _EdgeRMSNorm(nn.Module):
    """
    RMS norm of the hidden vector of every edge with a per-channel weight
    (centering=True: layer norm, mean removed first, plus a bias):
    h / sqrt(mean(h^2) + eps) * w. PyTorch's fused rms_norm / layer_norm
    (twice differentiable, also the fused CUDA kernels: gradgradcheck);
    eps keeps it smooth at h = 0.
    """

    def __init__(self, dim: int, centering: bool, eps: float) -> None:
        super().__init__()
        self.dim = dim
        self.centering = centering
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        if centering:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.centering:
            return F.layer_norm(h, [self.dim], self.weight, self.bias, self.eps)
        return torch.rms_norm(h, [self.dim], self.weight, self.eps)


@compile_mode('script')
class _RadialMLPBlock(nn.Module):
    """h + Down(Act(Up(Norm(h)))), Act = SwiGLU (a * act(b)) or act"""

    def __init__(
        self, dim: int, hidden: int, swiglu: bool, act: Callable,
        centering: bool, eps: float,
    ) -> None:
        super().__init__()
        self.norm = _EdgeRMSNorm(dim, centering, eps)
        self.swiglu = swiglu
        self.hidden = hidden
        self.up = _ScaledLinear(dim, 2 * hidden if swiglu else hidden)
        self.down = _ScaledLinear(hidden, dim)
        self.act = act

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = self.up(self.norm(h))
        if self.swiglu:
            a, b = torch.split(u, [self.hidden, self.hidden], dim=-1)
            u = a * self.act(b)
        else:
            u = self.act(u)
        return h + self.down(u)


@compile_mode('script')
class PairRadialMLP(nn.Module):
    """
    Radial weights depending on distance and on the element pair (eSEN,
    Fu et al. arXiv:2502.12147; MACE-MH-1, Batatia et al. arXiv:2510.25380,
    mace/modules/blocks.py). Concatenating [rbf, emb(src), emb(dst)] before the
    first layer is factorised as rbf @ W1 + T_src[z_src] + T_dst[z_dst].
    The pair tables act like biases, so the envelope multiplies the MLP
    (not its input) to keep weights -> 0 smoothly at the cutoff; it is applied
    to the last hidden layer, which equals scaling the output (the last
    layer is linear without bias) but touches hidden-size instead of
    weight_numel-size tensors.
    Tables start at zero: at init this is a distance-only radial MLP.
    Drop-in for FullyConnectedNet(hs, act) as IrrepsConvolution.weight_nn.

    num_short (config radial_short_basis): N short-range basis functions
    (ShortRangeBasis, right after rbf in the input) enter the first hidden
    layer as + f_short @ W_short (one matmul [rbf | f_short] @ [W1; W_short]),
    W_short zero-initialized (the MLP above at
    init); they vanish C2 at r_s < r_c, so they only refine the weights of
    near neighbours (multi-resolution radial input; motivation: spectral
    bias / phonon softening of the scalar path, arXiv:2606.15892; nested
    short/long cutoffs as in Prophet/MCSD, Kairos Materials, idea only).

    block (config radial_mlp_block, default None = the MLP above): the hidden
    stack becomes pre-norm residual MLP blocks (MetaFormer, W. Yu et al.,
    arXiv:2111.11418; pre-LN, R. Xiong et al., arXiv:2002.04745):
        h = rbf @ W1 + T_src[z_src] + T_dst[z_dst] (+ node terms)
        h = h + Down(Act(Up(Norm(h))))       (num_blocks times)
        weights = Out(Norm_f(h) * envelope)
    Act = SwiGLU, a * act(b) (N. Shazeer, arXiv:2002.05202), or act (the
    radial activation, SiLU in our runs); Norm = RMS norm or layer norm of
    the hidden vector of every edge (eps-guarded, so smooth also where h -> 0)
    with a per-channel weight. Keys: num_blocks (default len(hs) - 3, at
    least 1), hidden_scale (2: Up width = 2 x hidden), type ('swiglu' |
    'mlp'), norm ('rms' | 'layer'), final_norm (True), eps (1e-5). The
    element-pair tables, node terms and the envelope on the last hidden layer
    are unchanged, all hidden widths must be equal.
    """

    def __init__(
        self,
        hs: List[int],
        act: Callable,
        num_species: int,
        node_cond: int = 0,
        block: Optional[Dict[str, Any]] = None,
        num_short: int = 0,
    ) -> None:
        super().__init__()
        assert len(hs) >= 3, 'needs at least one hidden layer'
        assert node_cond in (0, hs[1]), 'node_cond must equal the first hidden size'
        self.num_basis = hs[0]
        self.node_cond = node_cond   # > 0: input carries per-edge node terms
        self.w1 = nn.Parameter(torch.randn(hs[0], hs[1]))
        self.norm_rbf = 1.0 / math.sqrt(hs[0])
        self.table_src = nn.Parameter(torch.zeros(num_species, hs[1]))
        self.table_dst = nn.Parameter(torch.zeros(num_species, hs[1]))
        self.num_short = num_short   # > 0: short-range basis after rbf
        if num_short > 0:
            self.w_short = nn.Parameter(torch.zeros(num_short, hs[1]))
            self.norm_short = 1.0 / math.sqrt(num_short)
        else:
            self.register_parameter('w_short', None)
            self.norm_short = 1.0
        self.act = normalize2mom(act)
        self.use_block = block is not None
        self.blocks = nn.ModuleList()
        self.final_norm = nn.Identity()
        if block is None:
            self.hidden = nn.ModuleList(
                [_ScaledLinear(hs[i], hs[i + 1]) for i in range(1, len(hs) - 2)]
            )
        else:
            dim = hs[1]
            assert all(h == dim for h in hs[1:-1]), 'radial_mlp_block: equal widths'
            unknown = set(block) - {
                'num_blocks', 'hidden_scale', 'type', 'norm', 'final_norm', 'eps'
            }
            assert not unknown, f'radial_mlp_block: unknown keys {unknown}'
            n_blocks = int(block.get('num_blocks', max(1, len(hs) - 3)))
            hidden = int(dim * block.get('hidden_scale', 2))
            kind = block.get('type', 'swiglu')
            norm = block.get('norm', 'rms')
            eps = float(block.get('eps', 1e-5))
            assert kind in ('swiglu', 'mlp') and norm in ('rms', 'layer')
            centering = norm == 'layer'
            self.hidden = nn.ModuleList()
            self.blocks = nn.ModuleList([
                _RadialMLPBlock(
                    dim, hidden, kind == 'swiglu', self.act, centering, eps
                )
                for _ in range(n_blocks)
            ])
            if block.get('final_norm', True):
                self.final_norm = _EdgeRMSNorm(dim, centering, eps)
        self.out = _ScaledLinear(hs[-2], hs[-1])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        nb = self.num_basis
        nr = nb + self.num_short  # [rbf | short basis] columns
        env = inp[:, nr : nr + 1]
        z_src = inp[:, nr + 1].to(torch.long)
        z_dst = inp[:, nr + 2].to(torch.long)
        w_short = self.w_short
        if w_short is None:
            h = inp[:, :nb] @ (self.w1 * self.norm_rbf)
        else:   # short-range basis (EdgePairInput) in the same matmul
            w = torch.cat([self.w1 * self.norm_rbf, w_short * self.norm_short])
            h = inp[:, :nr] @ w
        # F.embedding: sort-based backward; table[z] backward scatters millions
        # of edges into ~100 rows with atomics (6x slower in training)
        h = h + F.embedding(z_src, self.table_src)
        h = h + F.embedding(z_dst, self.table_dst)
        if self.node_cond > 0:   # node-state terms (NodeCondRadialInput)
            h = h + inp[:, nr + 3: nr + 3 + self.node_cond]
        if self.use_block:
            for blk in self.blocks:
                h = blk(h)
            h = self.final_norm(h)
        else:
            h = self.act(h)
            for layer in self.hidden:
                h = self.act(layer(h))
        return self.out(h * env)


@compile_mode('script')
class NodeCondRadialInput(nn.Module):
    """
    Per-layer input of a node-conditioned PairRadialMLP (config
    `radial_node_cond`): [EDGE_PAIR_INPUT | P_src s_src + P_dst s_dst], where s
    are the current 0e node features of the convolution input. The node terms
    are added to the first hidden layer next to the element-pair tables, so the
    radial weights depend on the states of both atoms (Orb-v3 edge MLP of
    sender/receiver features, Rhodes et al. arXiv:2504.06231; eSEN source/
    destination concatenation, Fu et al. arXiv:2502.12147; MACE-SOYO
    model_block.py, github.com/PrincessHakuryu/MACE-SOYO). Projections start at
    zero (identical to the element-pair MLP at init); the envelope still
    multiplies the MLP, so the weights vanish smoothly at the cutoff.
    is_parallel (e3gnn/parallel): x holds local nodes only and the source
    states of ghost neighbours come from KEY.NODE_FEATURE_GHOST (the exchanged
    convolution input), as in IrrepsConvolution.
    """

    def __init__(
        self, irreps_x, hidden: int, layer: int, data_key_x: str = KEY.NODE_FEATURE
    ) -> None:
        super().__init__()
        from e3nn.o3 import Irreps

        irreps_x = Irreps(irreps_x)
        sl = [s for (_, ir), s in zip(irreps_x, irreps_x.slices())
              if ir.l == 0 and ir.p == 1]
        assert len(sl) == 1, 'radial_node_cond needs exactly one 0e block'
        self.s0, self.s1 = sl[0].start, sl[0].stop
        self.key_x = data_key_x
        self.key_out = f'{layer}_edge_pair_input_nc'
        self.p_src = nn.Linear(self.s1 - self.s0, hidden, bias=False)
        self.p_dst = nn.Linear(self.s1 - self.s0, hidden, bias=False)
        nn.init.zeros_(self.p_src.weight)
        nn.init.zeros_(self.p_dst.weight)
        self.is_parallel = False

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        s = data[self.key_x][:, self.s0: self.s1]
        s_src = s
        if self.is_parallel:
            g = data[KEY.NODE_FEATURE_GHOST][:, self.s0: self.s1]
            s_src = torch.cat([s, g])
        src = data[KEY.EDGE_IDX][1]
        dst = data[KEY.EDGE_IDX][0]
        e = self.p_src(s_src).index_select(0, src) + self.p_dst(s).index_select(
            0, dst
        )
        data[self.key_out] = torch.cat([data[KEY.EDGE_PAIR_INPUT], e], dim=-1)
        return data
