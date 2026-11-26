"""Fused Metal kernels for SevenNet on the Apple GPU. The first part is the 'uvu'
convolution:

    out[n, ou] = sum_{e: dst(e)=n} sum_terms w[e, p, u] * g * x[src(e), iu] * y[e, j]

is evaluated in one compute kernel, gathering included, so no per-edge message tensor
exists. The forward pass reads x, y, w and writes a node tensor; the backward pass
also writes the per-edge gradients of y and w, and a temporary of y per channel. That
is the point on this hardware, where the work is limited by memory traffic and not by
arithmetic. MSL source is generated per convolution with the coefficients inline.
Derivatives are hand written. The product is linear in each of x, y, w and the
incoming gradient, so every derivative of the backward pass is one of the same three
kernels with an argument swapped; each is an autograd function whose backward calls
the others, so derivatives exist to any order (force training needs the second) and
each keeps only its inputs. The linear and gate kernels further down have first
derivatives only.

The coefficients g are read off the e3nn module itself by probing it (see probe_cg),
so every normalisation of e3nn is reproduced. Threads own their output row (node,
channel) or edge; nothing is written by two threads, so a kernel gives the same bits
for the same inputs and edge order, and needs no atomics. The rest of the model runs
PyTorch operations, which need not be bitwise reproducible on the GPU.

"""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
from e3nn import o3

_libs: Dict[str, object] = {}


def _check_int32(where: str, *sizes: int) -> None:
    """The kernels index buffers with 32-bit integers."""
    if max(sizes) >= 2**31:
        raise RuntimeError(
            f'{where}: graph too large for the fused kernels (32-bit indices); '
            f'SEVENN_DISABLE_MPS=1 runs the reference e3nn code'
        )


def _check_inputs(where: str, *checks) -> None:
    """Kernels read raw buffers with no bounds checks, so each (tensor, shape) must
    be float32 of that shape first."""
    for t, shape in checks:
        if t.dtype != torch.float32 or tuple(t.shape) != tuple(shape):
            raise ValueError(
                f'{where}: expected float32 of shape {tuple(shape)}, got {t.dtype} '
                f'of shape {tuple(t.shape)}'
            )


def _lib(src: str):
    if src not in _libs:
        _libs[src] = torch.mps.compile_shader(src)
    return _libs[src]


def probe_cg(tp: o3.TensorProduct) -> List[Dict]:
    """Per instruction: the tensor g[a, j, m] of channel 0, read off the module with
    one-hot inputs."""
    ir1, ir2, iro = tp.irreps_in1, tp.irreps_in2, tp.irreps_out
    sl1, sl2, slo = ir1.slices(), ir2.slices(), iro.slices()
    w_off = np.cumsum(
        [0] + [int(np.prod(ins.path_shape)) for ins in tp.instructions]
    )
    tp = copy.deepcopy(tp).to('cpu')
    out = []
    for p, ins in enumerate(tp.instructions):
        if (
            ins.connection_mode != 'uvu'
            or not ins.has_weight
            or ir2[ins.i_in2].mul != 1
        ):
            raise NotImplementedError(f'instruction {ins}')
        if ir1[ins.i_in1].mul != iro[ins.i_out].mul:
            raise NotImplementedError(
                'uvu path with different input and output multiplicity'
            )
        d1 = ir1[ins.i_in1].ir.dim
        d2 = ir2[ins.i_in2].ir.dim
        do = iro[ins.i_out].ir.dim
        B = d1 * d2
        x = torch.zeros(B, ir1.dim, dtype=torch.float32)
        y = torch.zeros(B, ir2.dim, dtype=torch.float32)
        w = torch.zeros(B, tp.weight_numel, dtype=torch.float32)
        for a in range(d1):
            for j in range(d2):
                x[a * d2 + j, sl1[ins.i_in1].start + a] = 1.0
                y[a * d2 + j, sl2[ins.i_in2].start + j] = 1.0
        w[:, int(w_off[p])] = 1.0  # this path only
        with torch.no_grad():
            r = tp(x, y, w)[:, slo[ins.i_out].start : slo[ins.i_out].start + do]
        out.append(
            dict(
                g=r.reshape(d1, d2, do).numpy(),
                xs=sl1[ins.i_in1].start,
                d1=d1,
                ys=sl2[ins.i_in2].start,
                os=slo[ins.i_out].start,
                do=do,
                w=int(w_off[p]),
                mul=ir1[ins.i_in1].mul,
            )
        )
    return out


def generate_msl(tp: o3.TensorProduct, tol: float = 1e-7) -> str:
    """MSL source of uvu_fwd, uvu_adj_x, uvu_adj_yw and uvu_sum_y for this tensor
    product.

    In the kernels x<k>, y<j>, o<k> are values at e3nn indices (channel u = 0 offset)
    of the input, the filter and the output (or its cotangent); accumulators reuse
    the name of the slot they fill, except b<j> for the gradient of y<j>. Threads run
    over channels up to the largest multiplicity; slots and paths of an irrep with
    fewer channels are guarded by if (u < mul).

    """
    ir1, iro = tp.irreps_in1, tp.irreps_out
    nx, ny, nw, nout = ir1.dim, tp.irreps_in2.dim, tp.weight_numel, iro.dim
    mul = max(m.mul for m in ir1)
    if max(m.mul for m in iro) > mul:
        raise NotImplementedError(
            'an output irrep has more channels than any input irrep, the threads '
            'would not fill it'
        )
    paths = probe_cg(tp)
    for P in paths:
        P['terms'] = [
            (
                P['xs'] + int(a),
                P['ys'] + int(j),
                P['os'] + int(m),
                f'{float(P["g"][a, j, m]):.9e}f',
            )
            for a, j, m in zip(*np.nonzero(np.abs(P['g']) > tol))
        ]
    # slot -> (irrep dim, multiplicity)
    all_x = {
        s.start + a: (m.ir.dim, m.mul)
        for m, s in zip(ir1, ir1.slices())
        for a in range(m.ir.dim)
    }
    all_o = {
        s.start + a: (m.ir.dim, m.mul)
        for m, s in zip(iro, iro.slices())
        for a in range(m.ir.dim)
    }
    used_x = sorted({t[0] for P in paths for t in P['terms']})
    used_y = sorted({t[1] for P in paths for t in P['terms']})
    used_o = sorted({t[2] for P in paths for t in P['terms']})

    def guarded(items, m_of):
        """Lines of code grouped by multiplicity, with a guard where a multiplicity
        is below the largest."""
        lines = []
        for m in sorted({m_of(i) for i in items}, reverse=True):
            body = (
                ' '.join(c for i, c in items.items() if m_of(i) == m)
                if isinstance(items, dict)
                else ''
            )
            lines.append(body if m == mul else f'if (u < {m}) {{ {body} }}')
        return '\n    '.join(lines)

    # Values of a channel that does not exist are 0, so the terms need no guards of
    # their own.
    def loads(name, ptr, slots, table):
        return guarded(
            {k: f'{name}{k} = {ptr}[{k} + u * {table[k][0]}];' for k in slots},
            lambda k: table[k][1],
        )

    decl_x = ' '.join(f'float x{k} = 0.0f;' for k in used_x)
    decl_o = ' '.join(f'float o{k} = 0.0f;' for k in used_o)
    load_y = ' '.join(f'const float y{j} = ye[{j}];' for j in used_y)
    load_x = loads('x', 'xs', used_x, all_x)
    load_o = loads('o', 'ob', used_o, all_o)
    init_b = ' '.join(f'float b{j} = 0.0f;' for j in used_y)
    store_x = guarded(
        {
            k: f'p[{k} + u * {d}] = {f"x{k}" if k in used_x else "0.0f"};'
            for k, (d, _) in all_x.items()
        },
        lambda k: all_x[k][1],
    )
    store_o = guarded(
        {
            k: f'p[{k} + u * {d}] = {f"o{k}" if k in used_o else "0.0f"};'
            for k, (d, _) in all_o.items()
        },
        lambda k: all_o[k][1],
    )
    store_b = ' '.join(
        f'yb[{j}] = {f"b{j}" if j in used_y else "0.0f"};' for j in range(ny)
    )

    fwd, adjx, adjyw, padw = [], [], [], []
    for P in paths:
        T = P['terms']
        guard = '' if P['mul'] == mul else f'if (u < {P["mul"]}) '
        padw.append(f'{guard}wb[{P["w"]} + u] = 0.0f;')
        fwd.append(f'{guard}{{ const float wv = we[{P["w"]} + u];')
        for o in sorted({t[2] for t in T}):
            s = ' + '.join(f'{g} * x{a} * y{j}' for a, j, m, g in T if m == o)
            fwd.append(f'o{o} += wv * ({s});')
        fwd.append('}')
        adjx.append(f'{guard}{{ const float wv = we[{P["w"]} + u];')
        for k in sorted({t[0] for t in T}):
            s = ' + '.join(f'{g} * y{j} * o{m}' for a, j, m, g in T if a == k)
            adjx.append(f'x{k} += wv * ({s});')
        adjx.append('}')
        adjyw.append(
            f'{guard}{{ const float wv = we[{P["w"]} + u]; float acc = 0.0f;'
        )
        for jj in sorted({t[1] for t in T}):
            s = ' + '.join(f'{g} * x{a} * o{m}' for a, j, m, g in T if j == jj)
            adjyw.append(
                f'{{ const float v = {s}; b{jj} += wv * v; acc += v * y{jj}; }}'
            )
        adjyw.append(f'wb[{P["w"]} + u] = acc; }}')
    nl = '\n    '
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void uvu_fwd(device float* out, device const float* x, device const float* y, device const float* w,
    device const int* src, device const int* perm, device const int* offs, uint2 pos [[thread_position_in_grid]]) {{
  const int u = pos.x, n = pos.y;
  {decl_o}
  for (int k = offs[n]; k < offs[n + 1]; ++k) {{
    const int e = perm[k];
    device const float* xs = x + src[e] * {nx};
    device const float* ye = y + e * {ny};
    device const float* we = w + e * {nw};
    {load_y}
    {decl_x}
    {load_x}
    {nl.join(fwd)}
  }}
  device float* p = out + n * {nout};
  {store_o}
}}
kernel void uvu_adj_x(device float* xbar, device const float* obar, device const float* y, device const float* w,
    device const int* dst, device const int* perm, device const int* offs, uint2 pos [[thread_position_in_grid]]) {{
  const int u = pos.x, n = pos.y;
  {decl_x}
  for (int k = offs[n]; k < offs[n + 1]; ++k) {{
    const int e = perm[k];
    device const float* ob = obar + dst[e] * {nout};
    device const float* ye = y + e * {ny};
    device const float* we = w + e * {nw};
    {load_y}
    {decl_o}
    {load_o}
    {nl.join(adjx)}
  }}
  device float* p = xbar + n * {nx};
  {store_x}
}}
kernel void uvu_adj_yw(device float* ytmp, device float* wbar, device const float* obar, device const float* x,
    device const float* y, device const float* w, device const int* src, device const int* dst,
    constant int& nreal, uint2 pos [[thread_position_in_grid]]) {{
  const int u = pos.x, e = pos.y;
  device float* wb = wbar + e * {nw};
  device float* yb = ytmp + (e * {mul} + u) * {ny};
  if (e >= nreal) {{   // pad edge
    {nl.join(padw)}
    for (int j = 0; j < {ny}; ++j) yb[j] = 0.0f;
    return;
  }}
  device const float* xs = x + src[e] * {nx};
  device const float* ob = obar + dst[e] * {nout};
  device const float* ye = y + e * {ny};
  device const float* we = w + e * {nw};
  {load_y}
  {init_b}
  {decl_x}
  {decl_o}
  {load_x}
  {load_o}
  {nl.join(adjyw)}
  {store_b}
}}
kernel void uvu_sum_y(device float* ybar, device const float* ytmp, uint2 pos [[thread_position_in_grid]]) {{
  const int j = pos.x, e = pos.y;
  float acc = 0.0f;
  device const float* t = ytmp + e * {mul} * {ny} + j;
  for (int u = 0; u < {mul}; ++u) acc += t[u * {ny}];
  ybar[e * {ny} + j] = acc;
}}
"""  # noqa: E501


def _first_order_only():
    if torch.is_grad_enabled():  # create_graph=True
        raise RuntimeError(
            'fused Metal kernels of linear and gate layers have first derivatives '
            'only in eval mode. For second derivatives use train mode, whose '
            'PyTorch path for those layers has derivatives to any order, or '
            'SEVENN_DISABLE_MPS=1.'
        )


class FusedTP:
    """Kernels of one tensor product. Holds only source and sizes (no Metal objects),
    so it survives deepcopy."""

    def __init__(self, tp: o3.TensorProduct):
        self.src = generate_msl(tp)
        _lib(self.src)
        self.mul = max(m.mul for m in tp.irreps_in1)
        self.nx, self.ny, self.nw = (
            tp.irreps_in1.dim,
            tp.irreps_in2.dim,
            tp.weight_numel,
        )
        self.nout = tp.irreps_out.dim


class EdgePlan:
    """CSR views of the edge list: edges per destination and per source, without pad
    edges at the end. edge_idx[0] is the destination of a message and edge_idx[1] its
    source, as in IrrepsConvolution. One plan per edge tensor: a new tensor (object
    identity) or an in-place change (version) gives a new plan.

    """

    def __init__(self):
        self.edge_idx = None
        self.version = -1
        self.num_nodes = -1
        self.expect_padded: Optional[tuple] = (
            # (real, total) edges of the next graph, set by the owner of padding
            None
        )

    def get(self, edge_idx: torch.Tensor, num_nodes: int) -> 'EdgePlan':
        if (
            edge_idx is not self.edge_idx
            or edge_idx._version != self.version
            or num_nodes != self.num_nodes
        ):
            self._build(edge_idx, num_nodes)
        return self

    def _build(self, edge_idx, num_nodes):
        idx = edge_idx.detach().cpu().numpy()
        if idx.size and (idx.min() < 0 or idx.max() >= num_nodes):
            raise ValueError(f'edge index out of range for {num_nodes} atoms')
        dst, src = idx[0].astype(np.int32), idx[1].astype(np.int32)
        device = edge_idx.device
        num_real = idx.shape[1]
        expected, self.expect_padded = self.expect_padded, None
        # a stale announcement, e.g. after a failed call, must not cut edges off
        # another graph
        if (
            expected is not None
            and expected[1] == idx.shape[1]
            and not idx[:, expected[0] :].any()
        ):
            num_real = expected[0]

        def csr(key):
            key = key[:num_real]
            perm = np.argsort(key, kind='stable').astype(np.int32)
            offs = np.bincount(key + 1, minlength=num_nodes + 1).astype(np.int32)
            return torch.from_numpy(perm).to(device), torch.from_numpy(
                np.cumsum(offs, dtype=np.int32)
            ).to(device)

        self.src = torch.from_numpy(src).to(device)
        self.dst = torch.from_numpy(dst).to(device)
        self.dperm, self.doffs = csr(dst)
        self.sperm, self.soffs = csr(src)
        self.num_edges, self.num_real, self.num_nodes = (
            int(idx.shape[1]),
            int(num_real),
            int(num_nodes),
        )
        self.edge_idx, self.version = edge_idx, edge_idx._version


def _uvu_fwd(x, y, w, K: FusedTP, p: Dict):
    out = x.new_empty((p['num_nodes'], K.nout))
    if p['num_nodes'] > 0:
        _lib(K.src).uvu_fwd(
            out,
            x.contiguous(),
            y.contiguous(),
            w.contiguous(),
            p['src'],
            p['dperm'],
            p['doffs'],
            threads=(K.mul, p['num_nodes']),
        )
    return out


def _uvu_adj_x(obar, y, w, K: FusedTP, p: Dict):
    xbar = obar.new_empty((p['num_nodes'], K.nx))
    if p['num_nodes'] > 0:
        _lib(K.src).uvu_adj_x(
            xbar,
            obar.contiguous(),
            y.contiguous(),
            w.contiguous(),
            p['dst'],
            p['sperm'],
            p['soffs'],
            threads=(K.mul, p['num_nodes']),
        )
    return xbar


def _uvu_adj_yw(obar, x, y, w, K: FusedTP, p: Dict):
    ybar, wbar = y.new_empty(y.shape), w.new_empty(w.shape)
    if p['num_edges'] > 0:
        ytmp = y.new_empty((y.shape[0] * K.mul, y.shape[1]))
        lib = _lib(K.src)
        lib.uvu_adj_yw(
            ytmp,
            wbar,
            obar.contiguous(),
            x.contiguous(),
            y.contiguous(),
            w.contiguous(),
            p['src'],
            p['dst'],
            p['num_real'],
            threads=(K.mul, p['num_edges']),
        )
        lib.uvu_sum_y(ybar, ytmp, threads=(y.shape[1], p['num_edges']))
    return ybar, wbar


class _UvuFwd(torch.autograd.Function):
    """F(x, y, w) -> o with the edge plan as a dict, so that later derivatives do not
    depend on a plan rebuilt since."""

    @staticmethod
    def forward(ctx, x, y, w, K: FusedTP, p: Dict):
        ctx.save_for_backward(x, y, w)
        ctx.K, ctx.p = K, p
        return _uvu_fwd(x, y, w, K, p)

    @staticmethod
    def backward(ctx, obar):
        x, y, w = ctx.saved_tensors
        K, p = ctx.K, ctx.p
        xbar = _UvuAdjX.apply(obar, y, w, K, p) if ctx.needs_input_grad[0] else None
        ybar = wbar = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            ybar, wbar = _UvuAdjYW.apply(obar, x, y, w, K, p)
        return xbar, ybar, wbar, None, None


class _UvuAdjX(torch.autograd.Function):
    """X(o, y, w) -> xbar. Its derivative to o is F(xd, y, w), to y and w the YW
    kernel with x = xd."""

    @staticmethod
    def forward(ctx, obar, y, w, K: FusedTP, p: Dict):
        ctx.save_for_backward(obar, y, w)
        ctx.K, ctx.p = K, p
        return _uvu_adj_x(obar, y, w, K, p)

    @staticmethod
    def backward(ctx, xd):
        obar, y, w = ctx.saved_tensors
        K, p = ctx.K, ctx.p
        g_o = _UvuFwd.apply(xd, y, w, K, p) if ctx.needs_input_grad[0] else None
        g_y = g_w = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            g_y, g_w = _UvuAdjYW.apply(obar, xd, y, w, K, p)
        return g_o, g_y, g_w, None, None


class _UvuAdjYW(torch.autograd.Function):
    """YW(o, x, y, w) -> (ybar, wbar), where ybar does not depend on y nor wbar on w.

    With cotangents yd, wd: <yd, ybar> + <wd, wbar> = <o, F(x, yd, w) + F(x, y, wd)>,
    as a function of o and of x; its derivative to y is the ybar part of YW(o, x, y,
    wd), to w the wbar part of YW(o, x, yd, w)."""

    @staticmethod
    def forward(ctx, obar, x, y, w, K: FusedTP, p: Dict):
        ctx.save_for_backward(obar, x, y, w)
        ctx.K, ctx.p = K, p
        return _uvu_adj_yw(obar, x, y, w, K, p)

    @staticmethod
    def backward(ctx, yd, wd):
        obar, x, y, w = ctx.saved_tensors
        K, p = ctx.K, ctx.p
        g_o = g_x = g_y = g_w = None
        if ctx.needs_input_grad[0]:
            g_o = _UvuFwd.apply(x, yd, w, K, p) + _UvuFwd.apply(x, y, wd, K, p)
        if ctx.needs_input_grad[1]:
            g_x = _UvuAdjX.apply(obar, yd, w, K, p) + _UvuAdjX.apply(
                obar, y, wd, K, p
            )
        if ctx.needs_input_grad[2]:
            g_y = _UvuAdjYW.apply(obar, x, y, wd, K, p)[0]
        if ctx.needs_input_grad[3]:
            g_w = _UvuAdjYW.apply(obar, x, yd, w, K, p)[1]
        return g_o, g_x, g_y, g_w, None, None


class FusedUvu(_UvuFwd):
    """x (N, nx), y (E, ny), w (E, nw) -> (N, nout), with hand-written derivatives to
    any order."""

    @staticmethod
    def forward(ctx, x, y, w, K: FusedTP, plan: EdgePlan):
        # the kernels index with 32-bit integers
        _check_int32(
            'convolution kernel',
            plan.num_edges * max(K.mul * K.ny, K.nw),
            plan.num_nodes * max(K.nx, K.nout),
        )
        _check_inputs(
            'uvu kernel',
            (x, (plan.num_nodes, K.nx)),
            (y, (plan.num_edges, K.ny)),
            (w, (plan.num_edges, K.nw)),
        )
        # the plan object may be rebuilt before a later backward. The inputs are
        # saved as given: a contiguous copy made here would not be connected to the
        # caller's graph, and higher derivatives through it would be lost; the
        # launches make their own contiguous copies.
        return _UvuFwd.forward(ctx, x, y, w, K, plan.__dict__.copy())


# ---- node level: a linear map with a block-sparse matrix, optionally one matrix per
# atom type --------------

_CSR_SHADER = """#include <metal_stdlib>
using namespace metal;
// out[n, o] = b[o] + sum_k val[k] * x[n, idx[k]], k in ptr[t * no + o] .. ptr[t * no + o + 1], t = type of node n
kernel void csr_mm(device float* out, device const float* x, device const int* ptr, device const int* idx,
    device const float* val, device const float* b, device const int* types, constant int* d,
    uint2 pos [[thread_position_in_grid]]) {
  const int o = pos.x, n = pos.y, ni = d[0], no = d[1], typed = d[2];
  const int r = (typed ? types[n] : 0) * no + o;
  device const float* xn = x + n * ni;
  float acc = b[o];
  for (int k = ptr[r]; k < ptr[r + 1]; ++k) acc += val[k] * xn[idx[k]];
  out[n * no + o] = acc;
}
"""  # noqa: E501


def _to_csr(rows, cols, vals, nrow, device):
    order = np.lexsort((cols, rows))
    ptr = np.bincount(rows + 1, minlength=nrow + 1)
    return (
        torch.tensor(np.cumsum(ptr), dtype=torch.int32, device=device),
        torch.tensor(cols[order], dtype=torch.int32, device=device),
        torch.tensor(vals[order], dtype=torch.float32, device=device),
    ), order.astype(np.int32)


class CsrMap:
    """x (N, ni) -> (N, no) by matrices S[t] (ni, no) given as numpy triplets per
    type, plus a constant bias."""

    def __init__(
        self,
        triplets: List,
        bias: np.ndarray,
        ni: int,
        no: int,
        device,
        keep_order: bool = False,
    ):
        nt = len(triplets)
        rows = np.concatenate(
            [t[1] + i * no for i, t in enumerate(triplets)]
        )  # (type, out)
        cols = np.concatenate([t[0] for t in triplets])
        vals = np.concatenate([t[2] for t in triplets])
        self.fwd, order_fwd = _to_csr(rows, cols, vals, nt * no, device)
        rows = np.concatenate(
            [t[0] + i * ni for i, t in enumerate(triplets)]
        )  # (type, in)
        cols = np.concatenate([t[1] for t in triplets])
        self.adj, order_adj = _to_csr(rows, cols, vals, nt * ni, device)
        # with the same positions, new values need no new sort (set_values)
        self.order = (order_fwd, order_adj) if keep_order else None
        self.bias = torch.tensor(bias, dtype=torch.float32, device=device)
        self.zero = torch.zeros(ni, dtype=torch.float32, device=device)
        self.dfwd = torch.tensor(
            [ni, no, int(nt > 1)], dtype=torch.int32, device=device
        )
        self.dadj = torch.tensor(
            [no, ni, int(nt > 1)], dtype=torch.int32, device=device
        )
        self.notype = torch.zeros(1, dtype=torch.int32, device=device)
        self.ni, self.no, self.nt = ni, no, nt

    def set_values(self, vals: np.ndarray, bias: np.ndarray) -> None:
        """New values for the same triplet positions, in the order given at
        construction."""
        for (_, _, v), order in zip((self.fwd, self.adj), self.order):
            v.copy_(torch.from_numpy(np.ascontiguousarray(vals[order])))
        self.bias.copy_(torch.from_numpy(np.asarray(bias, dtype=np.float32)))


class CsrApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, M: CsrMap, types: Optional[torch.Tensor]):
        _check_int32('linear kernel', x.shape[0] * max(M.ni, M.no))
        _check_inputs('linear kernel', (x, (x.shape[0], M.ni)))
        if M.nt > 1 and (
            types is None
            or types.shape != (x.shape[0],)
            or types.dtype != torch.int32
            or types.device != x.device
            or not types.is_contiguous()
        ):
            # the kernel reads 32-bit indices from a raw buffer
            raise ValueError(
                f'linear kernel: expected one int32 atom type for each of '
                f'{x.shape[0]} atoms, on {x.device}'
            )
        ctx.M, ctx.types = M, types if types is not None else M.notype
        x = x.contiguous()
        out = x.new_empty((x.shape[0], M.no))
        if x.shape[0] > 0:  # no launch on an empty grid
            _lib(_CSR_SHADER).csr_mm(
                out, x, *M.fwd, M.bias, ctx.types, M.dfwd, threads=(M.no, x.shape[0])
            )
        return out

    @staticmethod
    def backward(ctx, obar):
        _first_order_only()
        M = ctx.M
        obar = obar.contiguous()
        xbar = obar.new_empty((obar.shape[0], M.ni))
        if obar.shape[0] > 0:
            _lib(_CSR_SHADER).csr_mm(
                xbar,
                obar,
                *M.adj,
                M.zero,
                ctx.types,
                M.dadj,
                threads=(M.ni, obar.shape[0]),
            )
        return xbar, None, None


# ---- gate activation: slot tables, one kernel forward and one backward
# ------------------------------------

_GATE_SHADER = """#include <metal_stdlib>
using namespace metal;
inline float act(int c, float x) {
  return c == 0 ? x / (1.0f + exp(-x)) : tanh(x);
}
inline float dact(int c, float x) {
  if (c == 0) { const float s = 1.0f / (1.0f + exp(-x)); return s + x * s * (1.0f - s); }
  const float t = tanh(x);
  return 1.0f - t * t;
}
// out[o] = cst[o] * act(x[src[o]]) * (gated[o] >= 0 ? x[gated[o]] : 1), cst includes the normalisation of e3nn
kernel void gate_fwd(device float* out, device const float* x, device const int* src, device const int* gated,
    device const int* code, device const float* cst, constant int* d, uint2 pos [[thread_position_in_grid]]) {
  const int o = pos.x, n = pos.y;
  device const float* xn = x + n * d[0];
  float v = cst[o] * act(code[o], xn[src[o]]);
  if (gated[o] >= 0) v *= xn[gated[o]];
  out[n * d[1] + o] = v;
}
kernel void gate_adj(device float* xbar, device const float* obar, device const float* x, device const int* src,
    device const int* gated, device const int* code, device const float* cst, constant int* d,
    uint n [[thread_position_in_grid]]) {
  device const float* xn = x + n * d[0];
  device const float* ob = obar + n * d[1];
  device float* xb = xbar + n * d[0];
  for (int i = 0; i < d[0]; ++i) xb[i] = 0.0f;
  for (int o = 0; o < d[1]; ++o) {
    const int s = src[o], g = gated[o], c = code[o];
    const float dv = cst[o] * dact(c, xn[s]) * ob[o];
    if (g >= 0) {
      xb[s] += dv * xn[g];
      xb[g] += cst[o] * act(c, xn[s]) * ob[o];
    } else {
      xb[s] += dv;
    }
  }
}
"""  # noqa: E501


class GateTables:
    def __init__(self, src, gated, code, cst, n_in, nout, device):
        def ti(a):
            return torch.tensor(a, dtype=torch.int32, device=device)

        self.tables = (
            ti(src),
            ti(gated),
            ti(code),
            torch.tensor(cst, dtype=torch.float32, device=device),
        )
        self.dims = ti([n_in, nout])
        self.n_in, self.nout = n_in, nout


class GateApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T: GateTables):
        _check_int32('gate kernel', x.shape[0] * max(T.n_in, T.nout))
        _check_inputs('gate kernel', (x, (x.shape[0], T.n_in)))
        x = x.contiguous()
        ctx.save_for_backward(x)
        ctx.T = T
        out = x.new_empty((x.shape[0], T.nout))
        if x.shape[0] > 0:  # no launch on an empty grid
            _lib(_GATE_SHADER).gate_fwd(
                out, x, *T.tables, T.dims, threads=(T.nout, x.shape[0])
            )
        return out

    @staticmethod
    def backward(ctx, obar):
        _first_order_only()
        (x,) = ctx.saved_tensors
        T = ctx.T
        xbar = x.new_empty(x.shape)
        if x.shape[0] > 0:
            _lib(_GATE_SHADER).gate_adj(
                xbar, obar.contiguous(), x, *T.tables, T.dims, threads=x.shape[0]
            )
        return xbar, None
