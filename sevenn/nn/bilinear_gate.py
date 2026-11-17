from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.nn import Gate
from e3nn.o3 import Irreps, Linear, TensorProduct
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

BILINEAR_PATHS = ('all', 'same_l', 'lmax2', 'scalar')
BILINEAR_B = ('own', 'shared')


def _keep_path(paths: str, i: int, j: int, l_out: int) -> bool:
    if paths == 'all':
        return True
    if paths == 'same_l':
        return i == j
    if paths == 'lmax2':
        return l_out <= 2
    if paths == 'scalar':
        return l_out == 0
    raise ValueError(f'unknown bilinear paths {paths}, expected {BILINEAR_PATHS}')


def _sorted_layout(groups: List[Irreps]) -> Tuple[Irreps, List[int]]:
    """
    Memory layout of an e3nn Gate input built from groups (scalars, gates,
    gated; as e3nn _Sortcut: each group simplified, then all blocks stably
    sorted by irrep). The last group is kept unsimplified so that each of its
    blocks has its own entry (same memory layout). Returns the unsimplified
    sorted irreps and the position of every block of the last group in it.
    """
    blocks = []
    for g in groups[:-1]:
        blocks += list(g.simplify())
    n_head = len(blocks)
    blocks += list(groups[-1])
    srt = Irreps(blocks).sort()
    return srt.irreps, [srt.p[n_head + i] for i in range(len(groups[-1]))]


@compile_mode('script')
class BilinearGate(nn.Module):
    """
    Gate nonlinearity plus a node-wise quadratic (bilinear) term:
        out = Gate(s, g, A) + Linear( A (x)_uuu (B * silu(w)) )
    A are the gated l>0 features of the usual gate input, B is an extra copy
    with the same irreps and w one extra scalar per channel of B. The
    channel-wise ('uuu') CG products use every pair of A/B irrep blocks with
    equal multiplicity whose product reaches an irrep of the output, so
    cross-l couplings (e.g. 2e x 3o -> 1o) are included where widths match.
    The output linear starts at zero: at init this equals the plain gate.

    Idea: the node-wise bilinear gate of EquFlashV2 (Samsung SDS GGNN,
    github.com/SamsungDS/GGNN, GGNN/model/EquFlashV2/nn/nonlinears.py,
    CC BY-NC-SA 4.0; idea only, independent implementation), related to the
    higher body-order node products of MACE (Batatia et al., arXiv:2206.07697).

    Implementation (b='own' is the function of the first version, 8d623b7):
    * One e3nn Gate evaluates both gates: Gate(s; gates g+w; gated A+B), so
      y = [act(s), act_g(g)*A, c*silu(w)*B] with c the e3nn normalize2mom
      constant of silu; 1/c is folded into the TP path weights.
    * The gate input layout is that of this e3nn Gate (sorted, simplified,
      e.g. 768x0e+256x1o+128x2e+128x3o: s, g, w, then the A and B blocks of
      each irrep side by side); self_interaction_2 outputs this layout.
    * The 'uuu' TP reads A from the gate input x and B' from the gate output
      y through instructions on the full buffers (other blocks are never
      read): forward is Gate -> TP -> Linear -> add without slicing copies,
      and each op maps 1:1 to a fused kernel when LunaTP patches the module
      (LunaEqGate, LunaTP 'uuu' TensorProduct, LunaEqLinear; see
      lunatp_helper.patch_bilinear_gate). The e3nn ops are the pure-PyTorch
      fallback and the TorchScript (deploy) path.
    * TP outputs are grouped by output irrep (faster output linear).

    Function changes (defaults paths='all', b='own'):
    paths: 'same_l' keeps A_l x B_l pairs only, 'lmax2' outputs with l <= 2,
    'scalar' only 0e outputs.
    b='shared': B = A, i.e. Linear(A (x)_uuu (A * silu(w))). The gate input is
    the plain gate input plus w (self_interaction_2 does not output B), and
    silu(w) enters as per-node TP path weights.
    """

    def __init__(
        self,
        irreps_x: Irreps,
        act_scalar_dict: Dict[str, Callable],
        act_gate_dict: Dict[str, Callable],
        data_key_x: str = KEY.NODE_FEATURE,
        paths: str = 'all',
        b: str = 'own',
    ) -> None:
        super().__init__()
        if b not in BILINEAR_B:
            raise ValueError(f'unknown bilinear b {b}, expected {BILINEAR_B}')
        self.key_x = data_key_x
        irreps_x = Irreps(irreps_x)
        parity_map = {1: 'e', -1: 'o'}
        irreps_scalars = Irreps([(m, ir) for m, ir in irreps_x if ir.l == 0])
        irreps_a = Irreps([(m, ir) for m, ir in irreps_x if ir.l > 0])
        if len(irreps_a) == 0:
            raise ValueError(f'no l>0 irreps in {irreps_x}; use the plain gate')
        # gate parity rule and activations as EquivariantGate
        p_gate = 1 if '0e' in irreps_scalars else -1
        gates_a = Irreps([(m, (0, p_gate)) for m, _ in irreps_a])
        act_s = [act_scalar_dict[parity_map[ir.p]] for _, ir in irreps_scalars]
        act_g = [act_gate_dict[parity_map[p_gate]]] * len(irreps_a)
        self.irreps_gated = irreps_a
        self.paths = paths
        self.shared = b == 'shared'
        n_ch = irreps_a.num_irreps

        if self.shared:
            self.gate = Gate(irreps_scalars, act_s, gates_a, act_g, irreps_a)
            layout, pos_a = _sorted_layout([irreps_scalars, gates_a, irreps_a])
            self.dim_gate_in = layout.dim
            irreps_in1 = layout + Irreps(f'{n_ch}x0e')  # w at the end
            irreps_in2, pos_b = irreps_in1, pos_a
            path_weight = 1.0
        else:
            gates_b = Irreps([(m, (0, 1)) for m, _ in irreps_a])
            self.gate = Gate(
                irreps_scalars, act_s, gates_a + gates_b,
                act_g + [F.silu] * len(irreps_a), irreps_a + irreps_a,
            )
            layout, pos_ab = _sorted_layout(
                [irreps_scalars, gates_a + gates_b, irreps_a + irreps_a]
            )
            pos_a = pos_ab[: len(irreps_a)]
            self.dim_gate_in = layout.dim
            irreps_in1 = layout
            # gate output y = [act(s), A', B'] (B gates are 0e: B' ~ irreps_a)
            dim_head = self.gate.irreps_out.dim - irreps_a.dim
            irreps_in2 = Irreps(f'{dim_head}x0e') + irreps_a  # head never read
            pos_b = [1 + j for j in range(len(irreps_a))]
            # 1/c for e3nn's normalize2mom(silu) of the B gates; e3nn multiplies
            # alpha by path_weight before the square root
            c_w = self.gate.act_gates.acts[len(irreps_a)].cst  # type: ignore
            path_weight = 1.0 / c_w**2
        if layout.simplify() != self.gate.irreps_in:
            raise AssertionError('unexpected e3nn Gate input layout')
        n_head_out = len(self.gate.act_scalars.irreps_out) + len(irreps_a)
        irreps_out = self.gate.irreps_out[:n_head_out]
        self.dim_out = irreps_out.dim

        out_order: List = []  # output irreps in order of first appearance
        for _, ir in irreps_out:
            if ir not in out_order:
                out_order.append(ir)
        found: List[Tuple[int, int, int, int]] = []  # (out rank, k, i, j)
        found_ir = []
        for i, (mul_a, ir_a) in enumerate(irreps_a):
            for j, (mul_b, ir_b) in enumerate(irreps_a):
                if mul_a != mul_b:
                    continue
                for ir_out in ir_a * ir_b:
                    if ir_out in out_order and _keep_path(paths, i, j, ir_out.l):
                        found.append((out_order.index(ir_out), len(found), i, j))
                        found_ir.append(ir_out)
        if not found:
            raise ValueError(f'no bilinear path for {irreps_x}; use the plain gate')
        instructions = []
        irreps_mid = []
        path_ij = []  # (A block, B block) of TP output k
        b_start = [sum(m for m, _ in irreps_a[:j]) for j in range(len(irreps_a))]
        w_index: List[int] = []
        # TP outputs grouped by output irrep (stable): faster output linear
        for _, k, i, j in sorted(found):
            instructions.append(
                (pos_a[i], pos_b[j], len(irreps_mid), 'uuu', self.shared,
                 path_weight)
            )
            irreps_mid.append((irreps_a[i].mul, found_ir[k]))
            path_ij.append((i, j))
            w_index += list(range(b_start[j], b_start[j] + irreps_a[j].mul))
        self.path_ij = path_ij
        tp_kw = {}
        if self.shared:  # per-node path weights: silu(w) of the B channel
            tp_kw = {'shared_weights': False, 'internal_weights': False}
        self.tp = TensorProduct(
            irreps_in1, irreps_in2, Irreps(irreps_mid), instructions, **tp_kw
        )
        self.register_buffer(
            'w_index', torch.tensor(w_index, dtype=torch.long), persistent=False
        )
        self.linear = Linear(self.tp.irreps_out, irreps_out)
        with torch.no_grad():
            self.linear.weight.zero_()

    def get_gate_irreps_in(self) -> Irreps:
        """own: [s, g, w] scalars, then the A and B blocks per irrep (e3nn Gate)
        shared: the plain gate input [s, g, A] (sorted) + w"""
        if self.shared:
            n_ch = self.irreps_gated.num_irreps
            return self.gate.irreps_in + Irreps(f'{n_ch}x0e')
        return self.gate.irreps_in

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        if self.shared:
            y = self.gate(x[:, : self.dim_gate_in].contiguous())
            w = torch.index_select(
                F.silu(x[:, self.dim_gate_in:]), 1, self.w_index
            )
            mid = self.tp(x, x, w)
        else:
            y = self.gate(x)
            mid = self.tp(x, y)
        data[self.key_x] = y[:, : self.dim_out] + self.linear(mid)
        return data
