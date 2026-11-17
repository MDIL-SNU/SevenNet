import math

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class CoordNumber(nn.Module):
    """
    log(1 + n_i / c), n_i = sum_j f_cut(r_ij) over the edges with receiver i
    (KEY.EDGE_ENVELOPE, the model's cutoff function), for every node, once per
    forward right after edge_embedding (config coord_scale; read by every
    CoordScale). Rows: KEY.NODE_FEATURE at that point (serial / batched: all
    nodes; ML-IAP: local + ghost atoms; e3gnn/parallel: the local atoms of
    the segment). Receivers (edge_index[0]) are always among the leading
    rows; ghost rows get n = 0 and are never read. Position-dependent:
    e3gnn/parallel rebuilds it at the head of every segment from that
    segment's envelope (model_build._to_parallel_model), as edge_pair_input.
    """

    def __init__(
        self, c: float = 1.0, data_key_out: str = KEY.NODE_COORD_LOG1P
    ) -> None:
        super().__init__()
        assert c > 0.0, 'coord_scale c must be positive'
        self.c = float(c)
        self.inv_c = 1.0 / self.c
        self.key_out = data_key_out

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        env = data[KEY.EDGE_ENVELOPE]
        n = torch.zeros(
            data[KEY.NODE_FEATURE].shape[0], dtype=env.dtype, device=env.device
        )
        n = n.index_add(0, data[KEY.EDGE_IDX][0], env, alpha=self.inv_c)
        data[self.key_out] = torch.log1p(n)
        return data


@compile_mode('script')
class CoordScale(nn.Module):
    """
    Sublinear coordination scaling of the aggregated message (config
    coord_scale), right after the convolution:
        m_i,c <- m_i,c * ((c0 + N) / (c0 + n_i))^p_c,
        n_i = sum_j f_cut(r_ij)   (smooth coordination of atom i),
        p_c = clamp(theta_c, 0, p_max),  theta_c = 0 at init (exactly the sum)
    with one exponent per multiplicity channel c of the convolution output
    (the same factor for the 2l+1 components of a channel, so it is
    equivariant), f_cut the model's cutoff function (KEY.EDGE_ENVELOPE, so n_i
    is as smooth as the envelope: C2 for poly_cut, C1 at r_on and r_c for
    XPLOR), N the convolution's fixed denominator (the training-set average
    neighbour count: typical atoms keep the scale of the plain sum, the
    factor only acts on deviations from the average coordination) and
    c0 = 1 (keeps the factor finite for n_i -> 0).
    Physics: for p in [0, 1/2] the message grows as n^(1 - p) under
    compression, never saturates (p = 1 would be the neighbour mean, the
    failure mode of softmax edge attention); p = 1/2 is the Tersoff /
    second-moment bond order (bond strength ~ coordination^(-1/2),
    J. Tersoff, PRB 37, 6991 (1988); D. G. Pettifor, PRL 63, 2480 (1989)),
    an embedding-like many-body term as in EAM (M. S. Daw and M. I. Baskes,
    PRB 29, 6443 (1984)). For a dimer n_i <= 1 (= 1 inside r_on for XPLOR),
    so its factor is constant at short range: the dimer curve's shape is
    not touched (only a constant rescaling of its messages relative to the
    bulk). ML idea: degree scalers of PNA (G. Corso et al., NeurIPS 2020,
    arXiv:2004.05718), here smooth, sum-preserving at init and capped.
    The box constraint is applied with a straight-through gradient
    (p = clamp(theta) + (theta - theta.detach()): the forward value is
    exactly the clamped exponent, the gradient that of p), i.e. lazy
    projected gradient: a channel pushed below 0 is not stuck there.
    Node-local: n_i needs only the atom's own edges (edge_index[0] is the
    receiver, always a local atom in e3gnn/parallel and ML-IAP); it is
    computed once per forward by CoordNumber (log(1 + n_i / c)), which every
    e3gnn/parallel segment rebuilds from its own edge_vec; no extra
    communication. Acts after any (fused) convolution backend. Per layer:
    one exp on the [nodes, channels] factor, expanded to the feature columns
    and multiplied (the model is launch-bound at ~2k atoms: few kernels).
    """

    def __init__(
        self,
        irreps_x,
        avg_num_neigh: float,
        c: float = 1.0,
        p_max: float = 0.5,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_coord: str = KEY.NODE_COORD_LOG1P,
    ) -> None:
        super().__init__()
        assert c > 0.0, 'coord_scale c must be positive'
        assert 0.0 < p_max < 1.0, 'coord_scale p_max must be in (0, 1)'
        irreps_x = Irreps(irreps_x)
        index = []
        ch = 0
        for mul, ir in irreps_x:
            for u in range(mul):
                index += [ch + u] * ir.dim
            ch += mul
        # feature column -> channel (mul_ir layout)
        self.register_buffer(
            'channel_index', torch.tensor(index, dtype=torch.long),
            persistent=False,
        )
        self.num_channels = ch
        self.theta = nn.Parameter(torch.zeros(ch))
        self.c = float(c)
        self.p_max = float(p_max)
        self.avg_num_neigh = float(avg_num_neigh)
        # log((c + N) / (c + n)) = log((c + N) / c) - log(1 + n / c)
        self.log_ref = math.log((self.c + self.avg_num_neigh) / self.c)
        self.key_x = data_key_x
        self.key_coord = data_key_coord

    def exponents(self) -> torch.Tensor:
        th = self.theta
        p = th.clamp(0.0, self.p_max)
        if th.requires_grad:  # straight-through: value exactly the clamp
            p = p + (th - th.detach())
        return p

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        log_ratio = self.log_ref - data[self.key_coord][: x.shape[0]].to(x.dtype)
        p = self.exponents().to(x.dtype)
        f = torch.exp(log_ratio.unsqueeze(-1) * p.unsqueeze(0))  # [N, channels]
        data[self.key_x] = x * f.index_select(1, self.channel_index)
        return data
