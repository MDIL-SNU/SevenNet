"""
LunaTP convolution with its radial MLP, for inference with frozen weights
(forces by one backward): the per-edge weight array (E x W floats, W up to
3520 in 7net-omni) is what the convolution's time goes to, in five passes
over it per layer (written by the MLP, read by the forward kernel, read by
the backward kernel, its gradient written by the backward kernel and read by
the MLP's backward). Here

- the weights are computed once per distinct edge length (edges i-j and j-i
  of a full neighbor list share a row: U ~ E / 2 rows), and the LunaTP
  kernels read row rid[e] instead of row e;
- next to w the same GEMM gives dw/dr (forward mode through the MLP's two
  hidden layers), and the backward kernel contracts the weight gradient with
  it in registers: dE/dr_e = sum_w dE/dw[e, w] * dw/dr[rid[e], w]. The weight
  gradient array is never written and the MLP has no backward.

The kernels are LunaEquivariance's generated uvu_conv forward and first-order
backward sources (HPC-AI-Team/LunaEquivariance, BSD-3-Clause, commit a045a88,
lunatp/uvu_conv/kernels/{forward,backward}.py), patched as text: weight row
index, and the weight-gradient stores replaced by the contraction. Same
function as LunaTPSharedGraphConvolution up to round-off.
"""
import re
from typing import List, Optional

import torch
import torch.nn as nn

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .lunatp_helper import LunaTPConvolution, LunaTPSharedGraphConvolution
from .radial_table import FoldedRadialMLP, _convolutions

_KEY_R = 'fused_radial_r'  # the KEY.EDGE_LENGTH tensor the entries belong to
_KEY_RID = 'fused_radial_rid'
_KEY_RU = 'fused_radial_r_unique'
_KEY_EMB = 'fused_radial_embedding'
_KEY_EMB_R = 'fused_radial_embedding_r'
KEY_N_UNIQUE = 'fused_radial_n_unique'


def _sub(pattern, repl, text, count=None, regex=True):
    if regex:
        text, n = re.subn(pattern, repl, text)
    else:
        n = text.count(pattern)
        text = text.replace(pattern, repl)
    if (count is None and n == 0) or (count is not None and n != count):
        raise RuntimeError(f'LunaTP source patch: {n} matches of {pattern!r}')
    return text


def _closing(text: str, start: int) -> int:
    """index of the brace closing the one at text[start]"""
    depth, pos = 1, start + 1
    while depth:
        c = text[pos]
        depth += (c == '{') - (c == '}')
        pos += 1
    return pos - 1


def forward_source(op) -> str:
    from LunaEquivariance.lunatp.uvu_conv.kernels import forward as F

    uniform = len({int(s['channels']) for s in op.generated_specs}) == 1
    src = F.make_uvu_conv_fwd_source(
        op.generated_specs,
        in1_dim=op.in1_dim,
        in2_dim=op.in2_dim,
        out_dim=op.out_dim,
        weight_numel=op.weight_numel,
        max_channels=op.max_channels,
        signed_pair_signatures=True,
        weight_on_x_paths=None,
        return_l0=False,
        with_env=False,
        merge_channel_chunks=(uniform or op.in2_dim >= 25),
        variable_channels=not uniform,
        weight_index_type='WeightIndex',
        resident_threads=1024,
        degree_tiers=op.forward_degree_tiers,
    )  # as load_forward (fp32)
    src = _sub(
        r'weight \+ (?:static_cast<uint64_t>\(static_cast<WeightIndex>\(e\)\)'
        r'|static_cast<WeightIndex>\(e\))',
        'weight + static_cast<uint64_t>(rid[e])', src,
    )
    if len(re.findall(r'weight \+', src)) != len(
        re.findall(r'weight \+ static_cast<uint64_t>\(rid\[e\]\)', src)
    ):
        raise RuntimeError('LunaTP source patch: unknown weight access')
    sig = 'int64_t num_nodes) {'
    parts = src.split(sig)
    if len(parts) != 5:  # two kernels, then their two launchers
        raise RuntimeError('LunaTP source patch: forward signatures')
    kernel = 'int64_t num_nodes, const int32_t* __restrict__ rid) {'
    host = 'int64_t num_nodes, torch::Tensor rid) {'
    src = (
        parts[0] + kernel + parts[1] + kernel + parts[2] + host + parts[3]
        + host + parts[4]
    )
    return _sub(
        r'num_nodes\);', 'num_nodes, rid.data_ptr<int32_t>());', src, count=2
    )


def backward_source(op):
    """(source, segmented, partials, vectorized, zero_x, zero_y)"""
    from LunaEquivariance.lunatp.uvu_conv.kernels import backward as B

    src, *flags = B.make_uvu_conv_bwd_source(op)
    src = _sub(
        'float* __restrict__ grad_weight, int64_t edges) {',
        'const float* __restrict__ grad_weight, int64_t edges, '
        'const int32_t* __restrict__ rid, float* __restrict__ gr_partial) {',
        src, count=1, regex=False,
    )
    src = _sub(
        'const INDEX wb = INDEX(e) * W_DIM;',
        'const INDEX wb = INDEX(__ldg(rid + e)) * W_DIM; float gr_e = 0.0f;',
        src, regex=False,
    )
    # dE/dw of one (edge, weight) is complete where LunaTP stores it
    src, n_scalar = re.subn(
        r'store_cs\(grad_weight \+ wb \+ (\d+) \+ u, (gw\d+)\);',
        r'gr_e = fmaf(\2, __ldg(grad_weight + wb + \1 + u), gr_e);',
        src,
    )
    src, n_vector = re.subn(
        r'asm volatile\("st\.global\.cs\.v2\.f32 \[%0\], \{%1, %2\};" :: '
        r'"l"\(grad_weight \+ wb \+ (\d+) \+ u\), "f"\((gw\d+)\), '
        r'"f"\((gw\d+)\) : "memory"\);',
        r'gr_e = fmaf(\2, __ldg(grad_weight + wb + \1 + u), gr_e); '
        r'gr_e = fmaf(\3, __ldg(grad_weight + wb + \1 + u + 1), gr_e);',
        src,
    )
    if n_scalar + n_vector == 0 or re.search(
        r'grad_weight(?! \+ wb \+ \d+ \+ u(?: \+ 1)?\), gr_e\)|, int64_t edges, )',
        src,
    ):
        raise RuntimeError('LunaTP source patch: weight gradient stores')
    # one partial sum per (part, edge) at the end of each edge loop
    loop = 'for (Index p = begin; p < end; ++p) {'
    out, pos = [], 0
    while True:
        at = src.find(loop, pos)
        if at < 0:
            break
        end = _closing(src, at + len(loop) - 1)
        body = src[at:end]
        base = re.search(
            r'const int part = (\d+) \+ tile \* \(blockDim\.x / 32\) \+ warp;',
            body,
        )
        if base is None:
            raise RuntimeError('LunaTP source patch: partial index')
        out.append(src[pos:end])
        out.append(
            '{ const float sr = warp_sum(gr_e); if (lane == 0) gr_partial['
            f'int64_t({base.group(1)} + tile * (blockDim.x / 32) + warp)'
            ' * edges + e] = sr; }\n'
        )
        pos = end
    out.append(src[pos:])
    src = ''.join(out)
    src = _sub(
        'gw.data_ptr<float>(), edges);',
        'gw.data_ptr<float>(), edges, rid.data_ptr<int32_t>(), '
        'grp.data_ptr<float>());',
        src, regex=False,
    )
    src = _sub(
        'std::optional<torch::Tensor> grad_l0) {',
        'std::optional<torch::Tensor> grad_l0, torch::Tensor rid, '
        'torch::Tensor grp) {',
        src, count=1, regex=False,
    )
    return (src, *flags)


class _FusedConv(torch.autograd.Function):
    """out(x, y; w[rid]); r only receives dE/dr (through dw/dr)"""

    @staticmethod
    def forward(ctx, mod, x, y, r, rid, wbuf, tables):
        ctx.mod, ctx.tables = mod, tables
        ctx.save_for_backward(x, y, rid, wbuf)
        op = mod.convolution.tp_conv.op
        out = x.new_empty((x.shape[0], op.out_dim))
        if op.forward_output_needs_zero:
            out.zero_()
        mod.fwd_ext.run_uvu_conv_fwd_ordered(
            x, y, wbuf, None, None, tables.src, tables.dst_rowptr,
            tables.node_order, tables.node_tiers, out, x.shape[0], rid,
        )
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, go):
        from LunaEquivariance.lunatp.utils.runtime import get_overlap_state
        from LunaEquivariance.lunatp.uvu_conv.graph import SourceSegmentTables

        x, y, rid, wbuf = ctx.saved_tensors
        mod, g = ctx.mod, ctx.tables
        op = mod.convolution.tp_conv.op
        ext = mod.bwd_ext
        segmented, partials, _, zero_x, zero_y = mod.bwd_flags
        go = go.contiguous()
        n_edge = y.shape[0]
        if segmented:
            seg = g.source_segments
        else:
            seg = SourceSegmentTables(
                nodes=g.source_node_order, begin=g.src_rowptr, end=g.src_rowptr,
                order=g.source_node_order, rowptr=g.src_rowptr,
            )
        gx, gy = torch.empty_like(x), torch.empty_like(y)
        if segmented:
            gx_partial = (torch.zeros if zero_x else torch.empty)(
                (seg.nodes.numel(), op.in1_dim), device=x.device, dtype=x.dtype
            )
        else:
            gx_partial = gx
            if zero_x:
                gx.zero_()
        gy_partial = (torch.zeros if zero_y else torch.empty)(
            (partials, n_edge, op.in2_dim), device=x.device, dtype=x.dtype
        )
        gr_partial = x.new_zeros((partials, n_edge))
        half = wbuf.shape[0] // 2
        state = (
            (get_overlap_state(op, x.device, ext._Streams),)
            if hasattr(ext, '_Streams') else ()
        )
        ext.run_uvu_conv_bwd(
            *state, x, y, wbuf, go, g.dst, g.edge_ids_by_src, seg.nodes,
            seg.begin, seg.end, seg.order, seg.rowptr, gx_partial, gy_partial,
            gx, gy, wbuf[half:], x.shape[0], n_edge, seg.nodes.numel(),
            None, None, None, None, rid, gr_partial,
        )
        needs = ctx.needs_input_grad
        return (
            None,
            gx if needs[1] else None,
            gy if needs[2] else None,
            gr_partial.sum(0) if needs[3] else None,
            None, None, None,
        )


class LunaTPFusedRadialConvolution(LunaTPSharedGraphConvolution):
    """
    see the module docstring. unique_cap: None = rows for exactly the distinct
    edge lengths of the call (torch.unique: a host sync); an int = that many
    rows whatever the edges (fixed shapes: CUDA graph), and
    data[KEY_N_UNIQUE] holds the number needed (results are wrong if it is
    larger: the caller must check it).
    """

    unique_cap: Optional[int] = None

    def _setup(self, emb) -> None:
        folded = FoldedRadialMLP(
            self.weight_nn, 1.0 / float(self.denominator.detach().double())
        )
        self.denominator_folded = True
        assert folded.num_layers == 3
        self.register_buffer('rad_w0', folded.w0)
        self.register_buffer('rad_w1', folded.w1)
        self.register_buffer('rad_w2', folded.w2.contiguous())
        self._acts = folded.acts
        object.__setattr__(self, '_emb', emb)  # not a submodule of this one
        self.weight_nn = nn.Identity()
        self._kernels()

    def _kernels(self) -> None:
        from LunaEquivariance.utils.extension import load_tp_extension

        op = self.convolution.tp_conv.op
        self.fwd_ext = load_tp_extension(
            'sevenn_uvuconv_fused_fwd_fp32', forward_source(op),
            max_channels=op.max_channels, index_bits=32,
        )
        src, *self.bwd_flags = backward_source(op)
        self.bwd_ext = load_tp_extension(
            'sevenn_uvuconv_fused_bwd_fp32', src,
            max_channels=op.max_channels, index_bits=32,
        )

    def _embedding(self, data: AtomGraphDataType, ru: torch.Tensor):
        """[basis * cutoff ; its r-derivative] at ru, (2U, n): once per call"""
        if _KEY_EMB in data and data[_KEY_EMB_R] is ru:
            return data[_KEY_EMB]
        emb = self._emb

        def f(r):
            return emb.basis_function(r) * emb.cutoff_function(r).unsqueeze(-1)

        with torch.enable_grad():  # forward mode needs grad mode on
            e, de = torch.func.jvp(f, (ru,), (torch.ones_like(ru),))
        data[_KEY_EMB] = torch.cat([e, de]).detach()
        data[_KEY_EMB_R] = ru
        return data[_KEY_EMB]

    def _weights(self, data: AtomGraphDataType, ru: torch.Tensor):
        """[w ; dw/dr] (2U, W): the MLP in forward mode, rows [value ; d/dr]"""
        x = self._embedding(data, ru)
        n = ru.shape[0]
        for w, act in zip((self.rad_w0, self.rad_w1), self._acts):
            z = x @ w
            v, d = z[:n], z[n:]
            if act is torch.nn.functional.silu:
                s = torch.sigmoid(v)
                x = torch.cat([v * s, d * (s * (1.0 + v * (1.0 - s)))])
            else:
                with torch.enable_grad():
                    x = torch.cat(torch.func.jvp(act, (v,), (d,))).detach()
        return torch.mm(x, self.rad_w2)

    @staticmethod
    def _unique(data: AtomGraphDataType, cap: Optional[int]):
        r = data[KEY.EDGE_LENGTH]
        if _KEY_R in data and data[_KEY_R] is r:
            return data[_KEY_RID], data[_KEY_RU]
        rd = r.detach()
        if cap is None:
            ru, rid = torch.unique(rd, return_inverse=True)
            if ru.shape[0] % 2:  # dw at an even row offset
                ru = torch.cat([ru, ru[-1:]])
            n_unique = torch.tensor(ru.shape[0], device=r.device)
        else:
            rs, perm = torch.sort(rd)
            new = torch.ones_like(rs, dtype=torch.int64)
            new[1:] = rs[1:] != rs[:-1]
            ids = torch.cumsum(new, 0) - 1
            n_unique = ids[-1] + 1
            ids = ids.clamp(max=cap - 1)
            rid = torch.empty_like(ids).scatter_(0, perm, ids)
            ru = torch.ones(cap, dtype=r.dtype, device=r.device)
            ru = ru.scatter(0, ids, rs)
        rid = rid.to(torch.int32)
        data[_KEY_R], data[_KEY_RID], data[_KEY_RU] = r, rid, ru
        data[KEY_N_UNIQUE] = n_unique
        return rid, ru

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]
        y = data[self.key_filter]
        if edge_src.numel() == 0:
            data[self.key_x] = (
                x.new_zeros(x.shape[0], self._out_dim) + y.sum() * 0
            )
            return data
        r = data[KEY.EDGE_LENGTH]
        rid, ru = self._unique(data, self.unique_cap)
        if ru.shape[0] * self.rad_w2.shape[1] >= 2**31:
            raise RuntimeError('too many edges for the fused radial kernels')
        wbuf = self._weights(data, ru)
        order, graph = LunaTPConvolution.get_graph(
            data[self.key_edge_idx], edge_src, edge_dst, x.shape[0]
        )
        if order is not None:
            y, r, rid = y[order], r[order], rid[order]
        tables = graph._tables[('uvu', 32)]
        data[self.key_x] = _FusedConv.apply(
            self, x.contiguous(), y.contiguous(), r, rid.contiguous(), wbuf,
            tables,
        )
        return data


def fuse_radial_conv(model: nn.Module) -> List[str]:
    """
    In place: every LunaTP convolution of a serial model whose radial MLP is
    an e3nn FullyConnectedNet [n, h, h, W] of the edge embedding becomes a
    LunaTPFusedRadialConvolution. Inference only. Returns the names changed.
    """
    from e3nn.nn import FullyConnectedNet

    emb = model._modules['edge_embedding']
    if getattr(emb, 'distance_transform', None) is not None:
        return []
    changed = []
    for name, conv in _convolutions(model):
        if (
            type(conv) is not LunaTPSharedGraphConvolution
            or conv.is_parallel
            or conv.key_weight_input != KEY.EDGE_EMBEDDING
            or conv.key_filter != KEY.EDGE_ATTR
            or not isinstance(conv.weight_nn, FullyConnectedNet)
            or len(conv.weight_nn) != 3
            or getattr(conv, 'denominator_folded', True)
        ):
            continue
        conv.__class__ = LunaTPFusedRadialConvolution
        conv._setup(emb)
        changed.append(name)
    return changed


def set_unique_cap(model: nn.Module, cap: Optional[int]) -> None:
    for module in model.modules():
        if isinstance(module, LunaTPFusedRadialConvolution):
            module.unique_cap = cap
