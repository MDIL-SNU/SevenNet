import math
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class EdgePairInput(nn.Module):
    """
    Pack the input of PairRadialMLP per edge:
        [radial basis (no envelope) | envelope | type of src | type of dst]
    Needs EdgeEmbedding(save_components=True) and node types in KEY.ATOM_TYPE,
    so it must come after onehot_idx_to_onehot.
    Parallel MD (e3gnn/parallel): edge sources can be ghosts, so the types are
    read from KEY.ATOM_TYPE_ALL (local + ghost), and every segment rebuilds the
    input after its own edge_embedding (the rbf/envelope must depend on that
    segment's edge vectors; see model_build._to_parallel_model).
    """

    def __init__(
        self,
        data_key_out: str = KEY.EDGE_PAIR_INPUT,
        data_key_atom_type: str = KEY.ATOM_TYPE,
    ) -> None:
        super().__init__()
        self.key_out = data_key_out
        self.key_atom_type = data_key_atom_type

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        rbf = data[KEY.EDGE_RBF]
        env = data[KEY.EDGE_ENVELOPE].unsqueeze(-1)
        atom_type = data[self.key_atom_type].to(rbf.dtype)
        edge_src = data[KEY.EDGE_IDX][1]
        edge_dst = data[KEY.EDGE_IDX][0]
        data[self.key_out] = torch.cat(
            [
                rbf,
                env,
                atom_type[edge_src].unsqueeze(-1),
                atom_type[edge_dst].unsqueeze(-1),
            ],
            dim=-1,
        )
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
    """

    def __init__(
        self,
        hs: List[int],
        act: Callable,
        num_species: int,
        node_cond: int = 0,
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
        self.act = normalize2mom(act)
        self.hidden = nn.ModuleList(
            [_ScaledLinear(hs[i], hs[i + 1]) for i in range(1, len(hs) - 2)]
        )
        self.out = _ScaledLinear(hs[-2], hs[-1])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        nb = self.num_basis
        rbf = inp[:, :nb]
        env = inp[:, nb : nb + 1]
        z_src = inp[:, nb + 1].to(torch.long)
        z_dst = inp[:, nb + 2].to(torch.long)
        h = rbf @ (self.w1 * self.norm_rbf)
        # F.embedding: sort-based backward; table[z] backward scatters millions
        # of edges into ~100 rows with atomics (6x slower in training)
        h = h + F.embedding(z_src, self.table_src)
        h = h + F.embedding(z_dst, self.table_dst)
        if self.node_cond > 0:   # node-state terms (NodeCondRadialInput)
            h = h + inp[:, nb + 3: nb + 3 + self.node_cond]
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
