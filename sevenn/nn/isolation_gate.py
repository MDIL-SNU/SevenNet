"""
Isolated-atom gate (model config architecture: isolation_gate, built by
sevenn/model_build_gate.py; a config without the key, e.g. every pretrained
or older checkpoint, builds the original model with none of this): the
network only learns the deviation from the isolated atom, so an atom without
neighbours gets E = shift exactly, for any weights.

    s_i = 1 - exp(-c n_i^2),  n_i = sum_j w(r_ij)   (w: the model's cutoff function)
    x <- x + s_i * Delta(x)                       (every residual update)
    modal one-hot m -> s_i * m                    (every modal linear)
    E_i = shift + scale * R(x_L - x_0)            (x_0: right after the embedding)

Why E_iso = shift is exact: an isolated atom has no edge, so n_i = 0 and s_i = 0
are exact zeros (a sum over nothing), every update adds 0 * finite = 0 and x_L =
x_0 bit for bit; the residual only copies channels, ReferenceDelta applies the
same copy to x_0. Then x_L - x_0 = 0, the modal input is 0 and R is linear(s)
without bias around activations with act(0) = 0, so R(0) = 0, also exactly.
Near the cutoff s = O(w^2): w and w' vanish at r_c (XPLOR), so s, s' and s''
do, and the gate adds no kink in the Hessian where a first neighbour enters (s =
O(w) would make E only C1 there, as XPLOR itself).

c = ISOLATION_C = 1.0: s > 0.999 needs n > 2.6 (n^2 > 6.9 / c). Sampled at r_c = 6 A (XPLOR
5.5, 150 structures each of OMat24 val, MPtrj, sAlex, WBM): median n 26-57, 1st
percentile 9-20, at most 0.25 % of the atoms below 6.9, lowest n 4.0 (s = 0.98).
A larger c only sharpens the turn-on where a first neighbour enters and has no
bulk benefit. Dimer: s = 0.632 where w = 1. Force of s: dE/dr contains dE/ds_i *
2 c n_i exp(-c n_i^2) * dw/dr; exp(-n^2) is 4e-44 at n = 10, and dw/dr is nonzero
only between r_on and r_c, so it only matters for atoms with few neighbours.

A residual that is not the identity (nequip block, 'nequip' / 'linear' self
connection) has no fixed point at s = 0: GateBlend makes the whole layer the
update, x <- P(x) + s * (layer(x) - P(x)), P the identity residual.

Node-local: s_i needs the atom's own edges only, so ML-IAP and e3gnn/parallel
rebuild it from their edge lists. Ghost atoms have no edge list:
e3gnn/parallel exchanges the output of 0_self_interaction_1 instead of
computing ghost features (model_build_gate.to_parallel_model_gate), and x_0
crosses the segments inside self_cont_tmp (SegmentPack).

The shared modules (residual outros, AddToResidual, modal IrrepsLinear, the
identity residual intro) are unchanged; the gated behaviour is in the
subclasses below, which model_build_gate swaps in (same state dict).

Sources (ideas, no code copied): per-element reference E0 + interaction energy,
I. Batatia et al., MACE, arXiv:2206.07697; DeePMD-kit atom_ener, J. Zeng et al.,
J. Chem. Phys. 159, 054801 (2023), arXiv:2304.09409.
"""
from typing import Any, Dict, List

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .ffn_variants import AddToResidual
from .linear import IrrepsLinear
from .normalization import MagnitudeResidualOutro, SelfConnectionResidualIntro
from .self_connection import SelfConnectionOutro

ARCH_DEFAULT = 'sevennet'  # the original SevenNet model
ARCH_ISOLATION_GATE = 'isolation_gate'
ARCHITECTURES = (ARCH_DEFAULT, ARCH_ISOLATION_GATE)


def is_isolation_gate(config: Dict[str, Any]) -> bool:
    """config architecture: isolation_gate (missing key: original model)"""
    arch = config.get(KEY.ARCHITECTURE, None) or ARCH_DEFAULT
    if arch not in ARCHITECTURES:
        raise ValueError(f'unknown {KEY.ARCHITECTURE}: {arch} ({ARCHITECTURES})')
    return arch == ARCH_ISOLATION_GATE


def coordination(data: AtomGraphDataType, alpha: float = 1.0) -> torch.Tensor:
    """
    alpha * n_i per node, n_i = sum_j f_cut(r_ij) over the edges with receiver i
    (KEY.EDGE_ENVELOPE). Rows: KEY.NODE_FEATURE at the call. Exactly 0 for a
    node without edges. (The same sum as coord_scale.CoordNumber.)
    """
    env = data[KEY.EDGE_ENVELOPE]
    n = torch.zeros(
        data[KEY.NODE_FEATURE].shape[0], dtype=env.dtype, device=env.device
    )
    return n.index_add(0, data[KEY.EDGE_IDX][0], env, alpha=alpha)


def residual_copy_index(irreps_in: Irreps, irreps_out: Irreps):
    """
    (src, dst) feature indices of the channel copy irreps_in -> irreps_out of
    the identity residual (SelfConnectionResidualIntro): out[:, dst] = x[:, src]
    """
    proj = SelfConnectionResidualIntro(Irreps(irreps_in), Irreps(irreps_out))
    return proj.src_idx.clone(), proj.dst_idx.clone()


ISOLATION_C = 1.0


@compile_mode('script')
class IsolationGate(nn.Module):
    """
    KEY.NODE_GATE: s_i per node, right after edge_embedding. Rows as
    KEY.NODE_FEATURE at that point (ML-IAP: narrowed to the local atoms by
    MLIAPWrappedIrrepsLinear; ghost rows would be 0 and are never read).
    """

    def __init__(self, c: float = ISOLATION_C) -> None:
        super().__init__()
        self.c = float(c)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        n = coordination(data)
        data[KEY.NODE_GATE] = -torch.expm1(-self.c * n * n)
        return data


@compile_mode('script')
class NodeReference(nn.Module):
    """KEY.NODE_REF: x_0, the node feature right after the embedding"""

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.NODE_REF] = data[KEY.NODE_FEATURE]
        return data


@compile_mode('script')
class ReferenceDelta(nn.Module):
    """
    data[key_out] = x - x_ref. x_ref: x_0 carried to the irreps of x by the same
    channel copies as the identity residual; ref_map[k] = column of x_0 in
    column k of x, -1 where the residual gives zero
    (model_build_gate.carry_reference).
    """

    def __init__(
        self,
        ref_map: torch.Tensor,
        data_key_out: str = KEY.NODE_FEATURE,
        data_key_x: str = KEY.NODE_FEATURE,
    ) -> None:
        super().__init__()
        self.key_x = data_key_x
        self.key_out = data_key_out
        self.register_buffer('ref_idx', ref_map.clamp(min=0), persistent=False)
        self.register_buffer(
            'ref_mask', (ref_map >= 0).to(torch.get_default_dtype()),
            persistent=False,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        ref = data[KEY.NODE_REF].index_select(1, self.ref_idx)
        data[self.key_out] = x - ref * self.ref_mask.type(x.dtype)
        return data


@compile_mode('script')
class GateBlend(nn.Module):
    """
    x <- ref + s_i * (x - ref), ref = data[key_ref]: a block or norm whose
    output is not ref + branch (the reference is saved by GateReference,
    data_key_out=key_ref)
    """

    def __init__(self, key_ref: str, data_key_x: str = KEY.NODE_FEATURE) -> None:
        super().__init__()
        self.key_ref = key_ref
        self.key_x = data_key_x

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        ref = data[self.key_ref]
        gate = data[KEY.NODE_GATE].unsqueeze(-1).type(x.dtype)
        data[self.key_x] = ref + gate * (x - ref)
        del data[self.key_ref]
        return data


@compile_mode('script')
class SegmentPack(nn.Module):
    """
    e3gnn/parallel: self_cont_tmp <- [self_cont_tmp | data[k] for k in keys], at
    the end of a segment: LAMMPS back-propagates a segment only w.r.t. (edge_vec,
    x, self_cont_tmp, x_ghost), a second tensor would lose its gradient. Not
    communicated, as self_cont_tmp itself (see LayerEnergyPack).
    """

    def __init__(self, keys: List[str]) -> None:
        super().__init__()
        self.keys_pack: List[str] = list(keys)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        tensors = [data[KEY.SELF_CONNECTION_TEMP]]
        for k in self.keys_pack:
            tensors.append(data[k])
        data[KEY.SELF_CONNECTION_TEMP] = torch.cat(tensors, dim=1)
        return data


@compile_mode('script')
class SegmentUnpack(nn.Module):
    """e3gnn/parallel: inverse of SegmentPack, dims: columns of each key"""

    def __init__(self, keys: List[str], dims: List[int]) -> None:
        super().__init__()
        self.keys_pack: List[str] = list(keys)
        self.dims: List[int] = [int(d) for d in dims]

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        packed = data[KEY.SELF_CONNECTION_TEMP]
        sizes = [packed.shape[1] - sum(self.dims)] + self.dims
        parts = torch.split(packed, sizes, dim=1)
        data[KEY.SELF_CONNECTION_TEMP] = parts[0]
        for i, k in enumerate(self.keys_pack):
            data[k] = parts[i + 1]
        return data


@compile_mode('script')
class SegmentEdgeLink(nn.Module):
    """
    End of e3gnn/parallel segment 0, which has no convolution: LAMMPS asks for
    d(x, self_cont_tmp) / d(edge_vec) there, which needs both tensors on the
    autograd graph of edge_vec. Adds a zero, values and gradients are unchanged.
    """

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        zero = data[KEY.NODE_GATE].sum() * 0.0
        x = data[KEY.NODE_FEATURE]
        data[KEY.NODE_FEATURE] = x + zero.type(x.dtype)
        sc = data[KEY.SELF_CONNECTION_TEMP]
        data[KEY.SELF_CONNECTION_TEMP] = sc + zero.type(sc.dtype)
        return data


@compile_mode('script')
class GateReference(SelfConnectionResidualIntro):
    """
    data[key_out] = P(x), P the identity residual irreps_in -> irreps_out
    (channel copy of SelfConnectionResidualIntro): the reference of a
    GateBlend. The copy indices are not saved (a derived constant: the module
    adds nothing to the state dict).
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_out: str = 'gate_ref',
    ) -> None:
        super().__init__(irreps_in, irreps_out, data_key_x)
        self.key_out = data_key_out
        src, dst = self.src_idx, self.dst_idx
        del self.src_idx
        del self.dst_idx
        self.register_buffer('src_idx', src, persistent=False)
        self.register_buffer('dst_idx', dst, persistent=False)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        if self.is_identity:
            data[self.key_out] = x
        else:
            out = x.new_zeros(x.shape[0], self.dim_out)
            out[:, self.dst_idx] = x[:, self.src_idx]
            data[self.key_out] = out
        return data


def _gate(data: AtomGraphDataType, x: torch.Tensor) -> torch.Tensor:
    return data[KEY.NODE_GATE].unsqueeze(-1).type(x.dtype)


@compile_mode('script')
class GatedSelfConnectionOutro(SelfConnectionOutro):
    """residual update x <- self_cont_tmp + s_i * x (x: the branch output)"""

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        sc_temp = data[KEY.SELF_CONNECTION_TEMP]
        data[self.key_x] = x * _gate(data, x) + sc_temp
        del data[KEY.SELF_CONNECTION_TEMP]
        return data


@compile_mode('script')
class GatedAddToResidual(AddToResidual):
    """self_cont_tmp <- self_cont_tmp + s_i * data[key] (parallel_block FFN)"""

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        branch = data[self.key_in]
        data[KEY.SELF_CONNECTION_TEMP] = (
            data[KEY.SELF_CONNECTION_TEMP] + branch * _gate(data, branch)
        )
        return data


@compile_mode('script')
class GatedMagnitudeResidualOutro(MagnitudeResidualOutro):
    """x <- P(x_in) + s_i * g * B(Norm(x_in)) (see MagnitudeResidualOutro)"""

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
        gate = data[KEY.NODE_GATE].type(x.dtype)
        data[self.key_x] = (gate * gain).unsqueeze(-1) * x + res
        del data[KEY.SELF_CONNECTION_TEMP]
        return data


@compile_mode('script')
class GatedIrrepsLinear(IrrepsLinear):
    """
    IrrepsLinear whose modal input is gated: x <- [x | s_i * modal one-hot],
    so no modal constant without neighbours. Every backend's linear (e3nn,
    LunaEqLinear) gets its input here; without modalities it is IrrepsLinear.
    """

    def _patch_modal_to_data(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_input]
        if self._is_batch_data:
            batch_modality_onehot = data[self.key_modal_attr].reshape(
                -1, self.num_modalities
            )
            batch_modality_onehot = batch_modality_onehot.type(x.dtype)
            modality_onehot = batch_modality_onehot[data[KEY.BATCH]]
        else:
            modality_onehot = data[self.key_modal_attr].expand(len(x), -1)
            modality_onehot = modality_onehot.type(x.dtype)
        data[self.key_input] = torch.cat(
            [x, modality_onehot * _gate(data, x)], dim=1
        )
        return data


# shared class -> gated subclass (model_build_gate swaps the class of every
# instance outside a GateBlend region; the subclasses add no state)
GATED_RESIDUAL_CLASSES = {
    SelfConnectionOutro: GatedSelfConnectionOutro,
    AddToResidual: GatedAddToResidual,
    MagnitudeResidualOutro: GatedMagnitudeResidualOutro,
}
