"""Frozen LunaTP inference with radial weights fused into the convolution.

Weights and radial derivatives come from cubic Hermite tables or from one
MLP evaluation per distinct edge length. The force backward contracts the
weight gradient with the radial derivative in registers.

The CUDA source patches target LunaEquivariance 0.3.0's generated UVU
convolution kernels (HPC-AI-Team/LunaEquivariance, BSD-3-Clause).
"""
from __future__ import annotations

import dataclasses
import re
from typing import Sequence

import torch
from torch import nn

import sevenn._keys as KEY
from sevenn._const import DEFAULT_RADIAL_TABLE_R_MIN, AtomGraphDataType

from .inference_specialization import prune_self_connection
from .linear import IrrepsLinear
from .lunatp_helper import LunaTPSharedGraphConvolution, prepare_graph
from .radial_table import (
    FoldedRadialMLP,
    _convolutions,
    _copy_cpu_double,
    _cubics,
    _table,
)

# Shorter source segments expose more parallelism on small graphs.
SMALL_GRAPH_EDGES = 4096

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


# weights of one edge from a table (K - 1, W, 4): the cubic in t of each weight
# on each knot interval (the Hermite interpolant of w and dw/dr at the knots),
# one 16 byte load per weight. (k, t) of edge e are the int32 pair rid[2 e],
# bits of the float t in the second
_CUBIC = """
static __forceinline__ __device__ float sevenn_w(const float* p, float t) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p));
    return fmaf(t, fmaf(t, fmaf(t, a.w, a.z), a.y), a.x);
}
// h * dw/dr
static __forceinline__ __device__ float sevenn_dw(const float* p, float t) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p));
    return fmaf(t, fmaf(t, 3.0f * a.w, 2.0f * a.z), a.y);
}
"""
_KNOT = 'const int hk_ = {k}; const float ht_ = __int_as_float({t});'


def forward_source(op, hermite: bool = False, specs=None, tune=None) -> str:
    """Generate forward CUDA code for the selected TP paths.

    Pass tune to override LunaTP generator options used here.
    """
    from LunaEquivariance.lunatp.uvu_conv.kernels import forward as F

    specs = op.generated_specs if specs is None else specs
    uniform = len({int(s['channels']) for s in specs}) == 1
    args = {
        'merge_channel_chunks': (uniform or op.in2_dim >= 25),
        'resident_threads': 1024,
        'task_warps': 16,
    }
    args.update(tune or {})
    src = F.make_uvu_conv_fwd_source(
        specs,
        in1_dim=op.in1_dim,
        in2_dim=op.in2_dim,
        out_dim=op.out_dim,
        weight_numel=op.weight_numel,
        max_channels=op.max_channels,
        signed_pair_signatures=True,
        weight_on_x_paths=None,
        return_l0=False,
        with_env=False,
        variable_channels=not uniform,
        weight_index_type='WeightIndex',
        degree_tiers=op.forward_degree_tiers,
        **args,
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
    src = _sub(
        r'num_nodes\);', 'num_nodes, rid.data_ptr<int32_t>());', src, count=2
    )
    if hermite:
        src = _sub(
            r'const float\* weight_edge = weight \+ '
            r'static_cast<uint64_t>\(rid\[e\]\) \* \d+;',
            _KNOT.format(k='rid[2 * e]', t='rid[2 * e + 1]')
            + ' const float* weight_edge = weight + '
            f'static_cast<uint64_t>(hk_) * {4 * int(op.weight_numel)};', src,
        )
        src = _sub(
            r'= weight_edge\[(\d+) \+ u\];',
            r'= sevenn_w(weight_edge + 4 * (\1 + u), ht_);', src,
        )
        if 'weight_edge[' in src:
            raise RuntimeError('LunaTP source patch: unknown weight read')
        src = _CUBIC + src
    return src


def backward_source(op, hermite: bool = False, specs=None, tune=None):
    """Generate backward CUDA code and return it with the LunaTP kernel flags.

    In Hermite mode the partial sums are h * dE/dr for knot spacing h.
    """
    from types import SimpleNamespace

    from LunaEquivariance.lunatp.uvu_conv.kernels import backward as B

    if specs is not None:
        op = SimpleNamespace(
            compute_dtype=op.compute_dtype, generated_specs=specs,
            in1_dim=op.in1_dim, in2_dim=op.in2_dim, out_dim=op.out_dim,
            weight_numel=op.weight_numel,
        )
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
    if (tune or {}).get('streams'):
        # every source-group kernel on its own stream (LunaTP: two streams):
        # for graphs too small to fill the GPU with one kernel
        launch = re.compile(
            r'(lunatp_uvuconv_bwd_source_fp32<[^>]*><<<[^>]*?, 0, )'
            r'(state\.worker|stream)(>>>\()'
        )
        count = len(launch.findall(src))
        if count > 2:
            if count > 16:
                raise RuntimeError('LunaTP source patch: backward streams')
            index = iter(range(count))

            def own(m):
                n = next(index)
                where = f'sevenn_s[{n}]' if n < count - 1 else 'stream'
                return m.group(1) + where + m.group(3)

            src = launch.sub(own, src)
            src = _sub(
                'void run_uvu_conv_bwd(',
                'static cudaStream_t sevenn_s[16]; static cudaEvent_t sevenn_e[16];'
                ' static bool sevenn_init = false;\nvoid run_uvu_conv_bwd(',
                src, count=1, regex=False,
            )
            src = _sub(
                'C10_CUDA_CHECK(cudaStreamWaitEvent(state.worker, state.start, 0));',
                f'if (!sevenn_init) {{ for (int i = 0; i < {count - 1}; ++i) {{ '
                'C10_CUDA_CHECK(cudaStreamCreateWithFlags(&sevenn_s[i], '
                'cudaStreamNonBlocking)); C10_CUDA_CHECK(cudaEventCreateWithFlags('
                '&sevenn_e[i], cudaEventDisableTiming)); } sevenn_init = true; }\n'
                f'for (int i = 0; i < {count - 1}; ++i) C10_CUDA_CHECK('
                'cudaStreamWaitEvent(sevenn_s[i], state.start, 0));',
                src, count=1, regex=False,
            )
            src = _sub(
                'C10_CUDA_CHECK(cudaEventRecord(state.done, state.worker));', '',
                src, count=1, regex=False,
            )
            src = _sub(
                'C10_CUDA_CHECK(cudaStreamWaitEvent(stream, state.done, 0));',
                f'for (int i = 0; i < {count - 1}; ++i) {{ C10_CUDA_CHECK('
                'cudaEventRecord(sevenn_e[i], sevenn_s[i])); C10_CUDA_CHECK('
                'cudaStreamWaitEvent(stream, sevenn_e[i], 0)); }',
                src, count=1, regex=False,
            )
    min_blocks = (tune or {}).get('min_blocks')
    if min_blocks is not None:
        def launch(m):
            n = min_blocks.get(int(m.group(1)), m.group(4)) if isinstance(
                min_blocks, dict) else min_blocks
            return f'<{m.group(1)}, {m.group(2)}, {m.group(3)}, {n}, 32, Index>'

        src = _sub(r'<(\d+), (\d+), (\d+), (\d+), 32, Index>', launch, src)
    if hermite:

        def at(m, f, extra=''):
            return f'{f}(weight + 4 * (wb + {m.group(1)} + u{extra}), ht_)'

        src = _sub(
            'const INDEX wb = INDEX(__ldg(rid + e)) * W_DIM; float gr_e = 0.0f;',
            _KNOT.format(k='__ldg(rid + 2 * e)', t='__ldg(rid + 2 * e + 1)')
            + ' const INDEX wb = INDEX(hk_) * W_DIM; float gr_e = 0.0f;',
            src, regex=False,
        )
        src = re.sub(
            r'__ldg\(reinterpret_cast<const float2\*>'
            r'\(weight \+ wb \+ (\d+) \+ u\)\)',
            lambda m: f'make_float2({at(m, "sevenn_w")}, '
            f'{at(m, "sevenn_w", " + 1")})', src,
        )
        src = re.sub(
            r'__ldg\(weight \+ wb \+ (\d+) \+ u\)',
            lambda m: at(m, 'sevenn_w'), src,
        )
        src = _sub(
            r'__ldg\(grad_weight \+ wb \+ (\d+) \+ u( \+ 1)?\)',
            lambda m: at(m, 'sevenn_dw', m.group(2) or ''), src,
        )
        if 'grad_weight + wb' in src or re.search(r'[^(]weight \+ wb', src):
            raise RuntimeError('LunaTP source patch: table weight reads')
        src = _CUBIC + src
    return (src, *flags)


def _load_cuda(name: str, source: str):
    """a CUDA extension by LunaEquivariance's build cache"""
    from LunaEquivariance.utils.extension import BINDING_CFLAGS
    from LunaEquivariance.utils.jit import load_extension

    return load_extension(
        name, cuda_sources=source,
        extra_cflags=['-O3', *BINDING_CFLAGS],
        extra_cuda_cflags=['-O3', *BINDING_CFLAGS],
    )


_GEOMETRY_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
namespace {
__global__ void sevenn_edge_geometry_fwd_kernel(
    const float* __restrict__ rvec, float* __restrict__ r_out,
    float* __restrict__ y_out, int64_t n) {
    const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const double vx = rvec[3 * e], vy = rvec[3 * e + 1], vz = rvec[3 * e + 2];
    const double r = sqrt(vx * vx + vy * vy + vz * vz);
    const double ir = 1.0 / r;
    r_out[e] = float(r);
    float* Y = y_out + e * @DIM@;
@POWERS@
@VALUES@
}
__global__ void sevenn_edge_geometry_bwd_kernel(
    const float* __restrict__ rvec, const float* __restrict__ gr,
    const float* __restrict__ gy, float* __restrict__ out, int64_t n) {
    const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const double vx = rvec[3 * e], vy = rvec[3 * e + 1], vz = rvec[3 * e + 2];
    const double r = sqrt(vx * vx + vy * vy + vz * vz);
    const double ir = 1.0 / r;
    const float* G = gy + e * @DIM@;
@POWERS@
    double gx = 0.0, gyy = 0.0, gz = 0.0;
@GRADS@
    // through u = v / r, then dE/dr along u
    const double ug = gx * px1 + gyy * py1 + gz * pz1;
    const double s = gr[e];
    out[3 * e] = float((gx - px1 * ug) * ir + s * px1);
    out[3 * e + 1] = float((gyy - py1 * ug) * ir + s * py1);
    out[3 * e + 2] = float((gz - pz1 * ug) * ir + s * pz1);
}
}  // namespace
void sevenn_edge_geometry_fwd(torch::Tensor rvec, torch::Tensor r,
                              torch::Tensor y) {
    const int64_t n = r.numel();
    if (n == 0) return;
    sevenn_edge_geometry_fwd_kernel<<<(n + 127) / 128, 128, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        rvec.data_ptr<float>(), r.data_ptr<float>(), y.data_ptr<float>(), n);
}
void sevenn_edge_geometry_bwd(torch::Tensor rvec, torch::Tensor gr,
                              torch::Tensor gy, torch::Tensor out) {
    const int64_t n = gr.numel();
    if (n == 0) return;
    sevenn_edge_geometry_bwd_kernel<<<(n + 127) / 128, 128, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        rvec.data_ptr<float>(), gr.data_ptr<float>(), gy.data_ptr<float>(),
        out.data_ptr<float>(), n);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sevenn_edge_geometry_fwd", &sevenn_edge_geometry_fwd);
    m.def("sevenn_edge_geometry_bwd", &sevenn_edge_geometry_bwd);
}
"""


def _harmonics_polynomials(sph: nn.Module, lmax: int):
    """
    [(component, coefficient, (a, b, c))]: sph(u) of a unit vector u as
    polynomials sum coefficient x^a y^b z^c, fitted (fp64) to the module
    """
    sph = _copy_cpu_double(sph)
    g = torch.Generator().manual_seed(0)
    u = torch.randn(256, 3, dtype=torch.float64, generator=g)
    u = u / u.norm(dim=1, keepdim=True)
    y = sph(u)
    terms, at = [], 0
    for l in range(lmax + 1):
        mono = [(a, b, l - a - b) for a in range(l + 1) for b in range(l + 1 - a)]
        A = torch.stack(
            [u[:, 0] ** a * u[:, 1] ** b * u[:, 2] ** c for a, b, c in mono], 1
        )
        target = y[:, at:at + 2 * l + 1]
        coef = torch.linalg.lstsq(A, target).solution  # (monomials, 2l + 1)
        if float((A @ coef - target).abs().max()) > 1e-9:
            raise RuntimeError('spherical harmonics are not polynomials of u')
        for j, abc in enumerate(mono):
            for m in range(2 * l + 1):
                if abs(float(coef[j, m])) > 1e-11:
                    terms.append((at + m, float(coef[j, m]), abc))
        at += 2 * l + 1
    if at != y.shape[1]:
        raise RuntimeError('spherical harmonics: unexpected dimension')
    return terms, at


def _geometry_source(terms, dim: int, lmax: int) -> str:
    def mono(a, b, c):
        f = [f'p{n}{k}' for n, k in (('x', a), ('y', b), ('z', c)) if k > 0]
        return ' * '.join(f) if f else '1.0'

    # fp64 (a few hundred operations per edge)
    powers = ['    const double px1 = vx * ir, py1 = vy * ir, pz1 = vz * ir;']
    for k in range(2, lmax + 1):
        powers.append(
            f'    const double px{k} = px{k - 1} * px1, py{k} = py{k - 1} * py1, '
            f'pz{k} = pz{k - 1} * pz1;'
        )
    values = {m: [] for m in range(dim)}
    grads = {'gx': [], 'gyy': [], 'gz': []}
    for m, coef, (a, b, c) in terms:
        values[m].append(f'{coef!r} * {mono(a, b, c)}')
        for name, k, low in (
            ('gx', a, (a - 1, b, c)), ('gyy', b, (a, b - 1, c)),
            ('gz', c, (a, b, c - 1)),
        ):
            if k > 0:
                grads[name].append(f'G[{m}] * ({coef * k!r} * {mono(*low)})')
    v = '\n'.join(
        f'    Y[{m}] = float({" + ".join(t) if t else "0.0"});'
        for m, t in values.items()
    )
    g = '\n'.join(
        f'    {n} = {" + ".join(t) if t else "0.0"};' for n, t in grads.items()
    )
    src = _GEOMETRY_SOURCE.replace('@DIM@', str(dim))
    src = src.replace('@POWERS@', '\n'.join(powers))
    return src.replace('@VALUES@', v).replace('@GRADS@', g)


class _EdgeGeometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ext, dim, rvec):
        ctx.ext = ext
        ctx.save_for_backward(rvec)
        r = rvec.new_empty(rvec.shape[0])
        y = rvec.new_empty((rvec.shape[0], dim))
        ext.sevenn_edge_geometry_fwd(rvec, r, y)
        return r, y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gr, gy):
        (rvec,) = ctx.saved_tensors
        if gr is None:
            gr = rvec.new_zeros(rvec.shape[0])
        if gy is None:
            gy = rvec.new_zeros((rvec.shape[0], 1))
            raise RuntimeError('edge geometry: no gradient of the harmonics')
        out = torch.empty_like(rvec)
        ctx.ext.sevenn_edge_geometry_bwd(
            rvec, gr.contiguous(), gy.contiguous(), out
        )
        return None, None, out


class FusedEdgeGeometry(nn.Module):
    """Compute edge lengths and harmonics together when only they are read."""

    def __init__(self, emb: nn.Module) -> None:
        super().__init__()
        lmax = int(emb.spherical.lmax)
        terms, self.dim = _harmonics_polynomials(emb.spherical, lmax)
        self.ext = _load_cuda(
            'sevenn_edge_geometry', _geometry_source(terms, self.dim, lmax)
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        rvec = data[KEY.EDGE_VEC].contiguous()
        data[KEY.EDGE_LENGTH], data[KEY.EDGE_ATTR] = _EdgeGeometry.apply(
            self.ext, self.dim, rvec
        )
        return data


def _index_order(graph):
    key = ('sevenn_uvu_index_order', 32)
    cached = graph._tables.get(key)
    if cached is not None:
        return cached
    tables = graph._tables[('uvu', 32)]
    seg = tables.source_segments
    out = dataclasses.replace(
        tables,
        node_order=torch.arange(
            tables.node_order.numel(), device=tables.node_order.device,
            dtype=tables.node_order.dtype,
        ),
        source_segments=dataclasses.replace(
            seg, order=torch.arange(
                seg.order.numel(), device=seg.order.device, dtype=seg.order.dtype
            ),
        ),
    )
    graph._tables[key] = out
    return out


def index_coherence(edge_index: torch.Tensor, num_nodes: int) -> float:
    """
    fraction of edges between nodes closer than num_nodes / 4 in index: 7/16
    for a random node order, near 1 for a spatially sorted one
    """
    d = (edge_index[0] - edge_index[1]).abs()
    return float((d < max(1, num_nodes // 4)).float().mean())


class _FusedConv(torch.autograd.Function):
    """out(x, y; w[rid]); r only receives dE/dr (through dw/dr)"""

    @staticmethod
    def forward(ctx, mod, x, y, r, rid, wbuf, tables):
        ctx.mod, ctx.tables = mod, tables
        ctx.save_for_backward(x, y, rid, wbuf)
        op = mod.convolution.tp_conv.op
        # the fused kernels index with 32 bits (LunaTP's own switch to 64)
        widest = max(op.out_dim, op.in1_dim)
        if x.shape[0] * widest >= 2**31 or y.shape[0] * op.in2_dim >= 2**31:
            raise RuntimeError(
                f'{x.shape[0]} atoms / {y.shape[0]} edges are too many for the '
                f'fused LunaTP kernels (32-bit indices: at most '
                f'{(2**31 - 1) // widest} atoms for this model). Run without '
                'fused (SEVENN_MLIAP_FUSED=0, or fused=False).'
            )
        out = x.new_empty((x.shape[0], op.out_dim))
        # with paths, the columns of the skipped paths stay uninitialized: the
        # linear after has no weights for them (checked with NaN there)
        if op.forward_output_needs_zero:
            out.zero_()
        mod.fwd_ext.run_uvu_conv_fwd_ordered(
            x, y, wbuf, None, None, tables.src, tables.dst_rowptr,
            tables.node_order, tables.node_tiers, out, x.shape[0], rid.view(-1),
        )
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, go):
        from LunaEquivariance.lunatp.utils.runtime import get_overlap_state
        from LunaEquivariance.lunatp.uvu_conv.graph import SourceSegmentTables

        x, y, rid, wbuf = ctx.saved_tensors
        mod, g = ctx.mod, ctx.tables
        rid = rid.view(-1)
        op = mod.convolution.tp_conv.op
        n_edge = y.shape[0]
        ext = mod.bwd_small() if n_edge < SMALL_GRAPH_EDGES else mod.bwd_ext
        segmented, partials, _, zero_x, zero_y = mod.bwd_flags
        go = go.contiguous()
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
        zero_y = zero_y or mod.zero_padding_grad
        gy_partial = (torch.zeros if zero_y else torch.empty)(
            (partials, n_edge, op.in2_dim), device=x.device, dtype=x.dtype
        )
        gr_partial = x.new_zeros((partials, n_edge))
        n_rows = wbuf.shape[0] // 2
        state = (
            (get_overlap_state(op, x.device, ext._Streams),)
            if hasattr(ext, '_Streams') else ()
        )
        ext.run_uvu_conv_bwd(
            *state, x, y, wbuf, go, g.dst, g.edge_ids_by_src, seg.nodes,
            seg.begin, seg.end, seg.order, seg.rowptr, gx_partial, gy_partial,
            gx, gy, wbuf if mod.knots else wbuf[n_rows:], x.shape[0], n_edge,
            seg.nodes.numel(),
            None, None, None, None, rid, gr_partial,
        )
        needs = ctx.needs_input_grad
        gr = None
        if needs[3]:
            gr = gr_partial.sum(0)
            if mod.knots:
                gr = gr * mod.table_inv_h
        return (
            None,
            gx if needs[1] else None,
            gy if needs[2] else None,
            gr,
            None, None, None,
        )


class LunaTPFusedRadialConvolution(LunaTPSharedGraphConvolution):
    """Fuse radial weight lookup and radial gradients into the TP kernels.

    In exact mode, unique_cap fixes the number of distance rows for CUDA
    replay. The caller must check that data[KEY_N_UNIQUE] fits this capacity.
    """

    unique_cap: int | None = None
    zero_padding_grad = False
    # kernel blocks launched in node index order instead of LunaTP's order by
    # degree: with atoms in a spatially coherent order (LAMMPS atom sorting)
    # the rows that neighbors share (weight rows of i-j and j-i, the output
    # gradient rows) are still in the GPU cache when the next block reads them
    index_order: bool = True
    # > 0: the weights of an edge are cubic Hermite interpolated inside the
    # kernels from a table of that many knots on [table_r0, cutoff] (see
    # weight_table): no per-call weight array and GEMM.
    # The table is what the kernels read, mostly from the GPU cache.
    knots: int = 0
    table_r0: float = DEFAULT_RADIAL_TABLE_R_MIN
    table_inv_h: float = 0.0
    # indices of the paths (LunaTP specs) computed; None: all. See live_paths
    paths: list[int] | None = None

    def _setup(self, emb, knots, r_min, cutoff, paths) -> None:
        self.paths = paths
        self.knots = int(knots)
        ref = self.denominator
        folded = FoldedRadialMLP(
            _copy_cpu_double(self.weight_nn),
            1.0 / float(ref.detach().double()),
        )
        self.denominator_folded = True
        assert folded.num_layers == 3
        for n in range(3):
            w = getattr(folded, f'w{n}').to(ref.device, ref.dtype).contiguous()
            self.register_buffer(f'rad_w{n}', w)
        self._acts = folded.acts
        self._folded64 = folded  # CPU FP64 weights for fitting the radial table
        object.__setattr__(self, '_emb', emb)  # not a submodule of this one
        self.weight_nn = nn.Identity()
        if self.knots:
            table, h = weight_table(self, emb, r_min, cutoff, self.knots)
            self.table_r0, self.table_inv_h = float(r_min), 1.0 / h
            self.register_buffer(
                'rad_table', table.to(ref.device, ref.dtype).contiguous()
            )
        self._kernels()

    def _kernels(self) -> None:
        from LunaEquivariance.utils.extension import load_tp_extension

        op = self.convolution.tp_conv.op
        herm = self.knots > 0
        tag = 'herm' if herm else 'fp32'
        specs = None
        if self.paths is not None:
            specs = [op.generated_specs[n] for n in self.paths]
        self.fwd_ext = load_tp_extension(
            f'sevenn_uvuconv_fused_fwd_{tag}',
            forward_source(op, herm, specs),
            max_channels=op.max_channels, index_bits=32,
        )
        self._bwd_small = None
        src, *self.bwd_flags = backward_source(op, herm, specs)
        self.bwd_ext = load_tp_extension(
            f'sevenn_uvuconv_fused_bwd_{tag}', src,
            max_channels=op.max_channels, index_bits=32,
        )

    def bwd_small(self):
        """the backward kernels for a small graph: a stream per group kernel"""
        if self._bwd_small is None:
            from LunaEquivariance.utils.extension import load_tp_extension

            op = self.convolution.tp_conv.op
            herm = self.knots > 0
            tag = 'herm' if herm else 'fp32'
            specs = None
            if self.paths is not None:
                specs = [op.generated_specs[n] for n in self.paths]
            src, *flags = backward_source(
                op, herm, specs, {'streams': True}
            )
            assert flags == self.bwd_flags
            self._bwd_small = load_tp_extension(
                f'sevenn_uvuconv_fused_bwd_{tag}', src,
                max_channels=op.max_channels, index_bits=32,
            )
        return self._bwd_small

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
        return x @ self.rad_w2

    @staticmethod
    def _unique(data: AtomGraphDataType, cap: int | None):
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

    def _knots(self, data: AtomGraphDataType) -> torch.Tensor:
        """(E, 2) int32: knot interval of each edge, bits of the float t in it"""
        r = data[KEY.EDGE_LENGTH]
        key = (self.knots, self.table_r0)
        if _KEY_R in data and data[_KEY_R] is r and data[_KEY_RU] == key:
            return data[_KEY_RID]
        u = (r.detach().double() - self.table_r0) * self.table_inv_h
        k = u.floor().clamp(0, self.knots - 2)
        t = (u - k).to(torch.float32)
        rid = torch.stack([k.to(torch.int32), t.view(torch.int32)], 1)
        data[_KEY_R], data[_KEY_RID], data[_KEY_RU] = r, rid, key
        return rid

    def _prepare_graph(self, edge_index, num_nodes):
        segment_edges = 8 if edge_index.shape[1] < SMALL_GRAPH_EDGES else 0
        return prepare_graph(edge_index, num_nodes, segment_edges)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        self._begin_graph(data)
        x = data[self.key_x]
        edge_src = data[self.key_edge_idx][1]
        y = data[self.key_filter]
        if edge_src.numel() == 0:
            data[self.key_x] = (
                x.new_zeros(x.shape[0], self._out_dim) + y.sum() * 0
            )
            return data
        r = data[KEY.EDGE_LENGTH]
        if self.knots:
            if self.rad_table.numel() >= 2**31:
                raise RuntimeError('too many knots for the fused radial kernels')
            rid, wbuf = self._knots(data), self.rad_table
        else:
            rid, ru = self._unique(data, self.unique_cap)
            if ru.shape[0] * self.rad_w2.shape[1] >= 2**31:
                raise RuntimeError('too many edges for the fused radial kernels')
            wbuf = self._weights(data, ru)
        order, graph = self._take_graph(data, x.shape[0])
        tables = graph._tables[('uvu', 32)]
        if order is not None:
            y, r, rid = y[order], r[order], rid[order]
        if self.index_order and not self.convolution.tp_conv.op.forward_degree_tiers:
            tables = _index_order(graph)
        data[self.key_x] = _FusedConv.apply(
            self, x.contiguous(), y.contiguous(), r, rid.contiguous(), wbuf,
            tables,
        )
        return data


def weight_table(conv, emb, r_min: float, cutoff: float, knots: int):
    """Build FP64 Hermite coefficients for the folded radial MLP."""
    emb64 = _copy_cpu_double(emb)

    def radial(r):
        x = emb64.basis_function(r) * emb64.cutoff_function(r).unsqueeze(-1)
        return conv._folded64(x)

    return _table(radial, r_min, cutoff, knots)


_FIRST_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
namespace {
static __forceinline__ __device__ float4 cubic(const float* table, int64_t at) {
    return __ldg(reinterpret_cast<const float4*>(table + at * 4));
}
// out[i, block of c] = sum over the edges into i of G[species of the source,
// c](r) * Y[block of c]
__global__ void sevenn_first_fwd_kernel(
    const float* __restrict__ y, const int32_t* __restrict__ rid,
    const int32_t* __restrict__ src, const int32_t* __restrict__ slot,
    const int32_t* __restrict__ rowptr, const float* __restrict__ table,
    const int32_t* __restrict__ meta, float* __restrict__ out, int64_t n,
    int C, int K1, int YD, int OD) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * C) return;
    const int64_t i = idx / C;
    const int c = int(idx - i * C);
    const int yo = meta[3 * c], oo = meta[3 * c + 1], d = meta[3 * c + 2];
    float acc[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int e = rowptr[i]; e < rowptr[i + 1]; ++e) {
        const int k = rid[2 * e];
        const float t = __int_as_float(rid[2 * e + 1]);
        const float4 a = cubic(
            table, (int64_t(slot[src[e]]) * K1 + k) * C + c);
        const float g = fmaf(t, fmaf(t, fmaf(t, a.w, a.z), a.y), a.x);
        const float* ye = y + int64_t(e) * YD + yo;
        #pragma unroll
        for (int m = 0; m < 7; ++m)
            if (m < d) acc[m] = fmaf(g, ye[m], acc[m]);
    }
    float* o = out + i * OD + oo;
    #pragma unroll
    for (int m = 0; m < 7; ++m)
        if (m < d) o[m] = acc[m];
}
// one warp per (edge, chunk of up to 32 functions of one block), a lane per
// function: dE/dY of the block's components and the chunk's part of
// h * dE/dr, summed over the lanes, then added to the edge's
__global__ void sevenn_first_bwd_kernel(
    const float* __restrict__ y, const int32_t* __restrict__ rid,
    const int32_t* __restrict__ src, const int32_t* __restrict__ dst,
    const int32_t* __restrict__ slot, const float* __restrict__ table,
    const int32_t* __restrict__ chunks, const float* __restrict__ go,
    float* __restrict__ gy, float* __restrict__ gr, int64_t edges, int B,
    int C, int K1, int YD, int OD) {
    const int64_t warp = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= edges * B) return;
    const int64_t e = warp / B;
    const int b = int(warp - e * B);
    const int c0 = chunks[5 * b], count = chunks[5 * b + 1];
    const int yo = chunks[5 * b + 2], oo = chunks[5 * b + 3], d = chunks[5 * b + 4];
    const float* ye = y + e * YD + yo;
    float acc[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float r = 0.0f;
    if (lane < count) {
        const int k = rid[2 * e];
        const float t = __int_as_float(rid[2 * e + 1]);
        const float4 a = cubic(
            table, (int64_t(slot[src[e]]) * K1 + k) * C + c0 + lane);
        const float g = fmaf(t, fmaf(t, fmaf(t, a.w, a.z), a.y), a.x);
        const float dg = fmaf(t, fmaf(t, 3.0f * a.w, 2.0f * a.z), a.y);
        const float* q = go + int64_t(dst[e]) * OD + oo + lane * d;
        float s = 0.0f;
        #pragma unroll
        for (int m = 0; m < 7; ++m) {
            if (m < d) {
                const float qm = __ldg(q + m);
                acc[m] = qm * g;
                s = fmaf(qm, __ldg(ye + m), s);
            }
        }
        r = s * dg;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        #pragma unroll
        for (int m = 0; m < 7; ++m)
            if (m < d) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
        r += __shfl_xor_sync(0xffffffffu, r, o);
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < 7; ++m)
            if (m < d) atomicAdd(gy + e * YD + yo + m, acc[m]);
        atomicAdd(gr + e, r);
    }
}
}  // namespace
void sevenn_first_fwd(torch::Tensor y, torch::Tensor rid, torch::Tensor src,
                      torch::Tensor slot, torch::Tensor rowptr,
                      torch::Tensor table, torch::Tensor meta, torch::Tensor out) {
    const int64_t n = out.size(0);
    const int C = meta.size(0);
    if (n * C == 0) return;
    sevenn_first_fwd_kernel<<<(n * C + 127) / 128, 128, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<float>(), rid.data_ptr<int32_t>(), src.data_ptr<int32_t>(),
        slot.data_ptr<int32_t>(), rowptr.data_ptr<int32_t>(),
        table.data_ptr<float>(), meta.data_ptr<int32_t>(), out.data_ptr<float>(),
        n, C, int(table.size(1)), int(y.size(1)), int(out.size(1)));
}
void sevenn_first_bwd(torch::Tensor y, torch::Tensor rid, torch::Tensor src,
                      torch::Tensor dst, torch::Tensor slot, torch::Tensor table,
                      torch::Tensor blocks, torch::Tensor go, torch::Tensor gy,
                      torch::Tensor gr) {
    const int64_t edges = y.size(0);
    const int B = blocks.size(0);
    if (edges * B == 0) return;
    sevenn_first_bwd_kernel<<<(edges * B * 32 + 127) / 128, 128, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<float>(), rid.data_ptr<int32_t>(), src.data_ptr<int32_t>(),
        dst.data_ptr<int32_t>(), slot.data_ptr<int32_t>(), table.data_ptr<float>(),
        blocks.data_ptr<int32_t>(), go.data_ptr<float>(), gy.data_ptr<float>(),
        gr.data_ptr<float>(), edges, B, int(table.size(2)), int(table.size(1)),
        int(y.size(1)), int(go.size(1)));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sevenn_first_fwd", &sevenn_first_fwd);
    m.def("sevenn_first_bwd", &sevenn_first_bwd);
}
"""


class _FirstLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mod, y, r, rid, slot, tables):
        ctx.mod, ctx.tables = mod, tables
        ctx.save_for_backward(y, rid, slot)
        out = y.new_empty((slot.shape[0], mod.out_dim))
        mod.ext.sevenn_first_fwd(
            y, rid.view(-1), tables.src, slot, tables.dst_rowptr, mod.table,
            mod.meta, out,
        )
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, go):
        y, rid, slot = ctx.saved_tensors
        mod, g = ctx.mod, ctx.tables
        gy = torch.zeros_like(y)
        gr = y.new_zeros(y.shape[0])
        mod.ext.sevenn_first_bwd(
            y, rid.view(-1), g.src, g.dst, slot, mod.table, mod.chunks,
            go.contiguous(), gy, gr,
        )
        return None, gy, gr * mod.conv.table_inv_h, None, None, None


class FusedFirstLayer(nn.Module):
    """Combine the first scalar-input convolution and its two linear layers.

    Each message is G[species_j, l, v](r_ij) * Y_lm(unit_r_ij).
    The radial table G folds in the scalar species features and channel mixing.
    """

    def __init__(self, conv, reference, names) -> None:
        super().__init__()
        from e3nn.o3 import Irreps

        self.conv = conv  # the fused convolution: knots, graph tables
        object.__setattr__(self, '_ref', (reference, names))
        irreps_out = Irreps(reference._modules[names[2]].irreps_out)
        meta, chunks, c, at = [], [], 0, 0
        for mul, ir in irreps_out:
            yo = ir.l * ir.l  # offset of Y_l in the harmonics 0 .. lmax
            for v in range(mul):
                meta.append([yo, at + v * ir.dim, ir.dim])
            for v in range(0, mul, 32):  # (first function, count, Y, out, dim)
                chunks.append(
                    [c + v, min(32, mul - v), yo, at + v * ir.dim, ir.dim]
                )
            c += mul
            at += mul * ir.dim
        self.out_dim = at
        self._irreps_out = irreps_out
        dev = conv.rad_table.device
        self.register_buffer('meta', torch.tensor(meta, dtype=torch.int32).to(dev))
        self.register_buffer(
            'chunks', torch.tensor(chunks, dtype=torch.int32).to(dev)
        )
        self.register_buffer('table', torch.zeros((0, 1, 1, 4), device=dev))
        self.register_buffer('bias', torch.zeros(at, device=dev))
        self.register_buffer(
            'lut', torch.zeros(len(reference.type_map), dtype=torch.int32).to(dev)
        )
        self.columns: list[int] = []
        self.key_x, self.key_filter = conv.key_x, conv.key_filter
        self.ext = _load_cuda('sevenn_first_layer_fp32', _FIRST_SOURCE)

    def _reference(self, column: int, r: torch.Tensor, direction: torch.Tensor):
        """(x after the linear (K, out_dim), bias, Y (dim,)) of single edges
        of lengths r along direction from an atom of type index column"""
        ref, names = self._ref
        number = {v: k for k, v in ref.type_map.items()}[column]
        k = r.shape[0]
        data = {
            KEY.EDGE_IDX: torch.stack([
                torch.arange(1, k + 1), torch.zeros(k, dtype=torch.int64)
            ]),
            KEY.EDGE_VEC: r.unsqueeze(1) * direction.unsqueeze(0),
            KEY.ATOMIC_NUMBERS: torch.full((k + 2,), number, dtype=torch.int64),
            KEY.NUM_ATOMS: torch.tensor(k + 2, dtype=torch.int64),
        }
        with torch.no_grad():
            data = ref._preprocess(data)
            for name, module in ref._modules.items():
                data = module(data)
                for key, value in data.items():  # one-hot embeddings are fp32
                    if torch.is_tensor(value) and value.dtype == torch.float32:
                        data[key] = value.double()
                if name == names[2]:
                    break
        x = data[self.key_x]
        # the last atom has no neighbor: the constant part
        return x[1:k + 1] - x[k + 1], x[k + 1], data[self.key_filter][0]

    def set_species(self, columns: Sequence[int]) -> None:
        """the tables of the species of these type indices (all that occur)"""
        columns = sorted({int(c) for c in columns})
        if columns == self.columns:
            return
        conv = self.conv
        h = 1.0 / conv.table_inv_h
        r = conv.table_r0 + h * torch.arange(conv.knots, dtype=torch.float64)
        g = torch.Generator().manual_seed(0)
        direction = torch.randn(3, dtype=torch.float64, generator=g)
        direction = direction / direction.norm()
        other = torch.randn(3, dtype=torch.float64, generator=g)
        other = other / other.norm()
        eps = 1e-4
        tables, bias = [], None
        for column in columns:
            def radial(rr, direction=direction, column=column):
                x, b, y = self._reference(column, rr, direction)
                out = []
                for mul, ir in self._irreps_out:
                    yl = y[ir.l * ir.l:ir.l * ir.l + ir.dim]
                    m = int(yl.abs().argmax())
                    blk = x[:, :mul * ir.dim].reshape(-1, mul, ir.dim)
                    x = x[:, mul * ir.dim:]
                    out.append(blk[:, :, m] / yl[m])
                return torch.cat(out, 1), b

            v, bias = radial(r)
            # r-derivative: five-point stencil in fp64 (the reference is e3nn)
            d = (
                radial(r - 2 * eps)[0] - 8 * radial(r - eps)[0]
                + 8 * radial(r + eps)[0] - radial(r + 2 * eps)[0]
            ) / (12 * eps)
            # another direction: x = G(r) Y, component by component
            x2, _, y2 = self._reference(column, r[::97], other)
            expect = torch.cat([
                (v[::97, c:c + 1] * y2[yo:yo + dim]) for c, (yo, _, dim) in
                enumerate(self.meta.tolist())
            ], 1)
            if float((x2 - expect).abs().max()) > 1e-9 * max(
                1.0, float(x2.abs().max())
            ):
                raise RuntimeError('first layer is not radial functions times Y')
            tables.append(_cubics(v, d * h))
        dev = self.bias.device
        self.table = torch.stack(tables).to(dev, torch.float32).contiguous()
        self.bias = bias.to(dev, torch.float32)
        lut = torch.zeros_like(self.lut)
        lut[torch.tensor(columns)] = torch.arange(
            len(columns), dtype=torch.int32
        ).to(dev)
        self.lut = lut
        self.columns = columns

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if not self.columns:
            raise RuntimeError('FusedFirstLayer: set_species was not called')
        self.conv._begin_graph(data)
        y = data[self.key_filter]
        r = data[KEY.EDGE_LENGTH]
        types = data[KEY.ATOM_TYPE]
        if r.numel() == 0:
            data[self.key_x] = self.bias.expand(types.shape[0], -1) + r.sum() * 0
            return data
        rid = self.conv._knots(data)
        order, graph = self.conv._take_graph(data, types.shape[0])
        if order is not None:
            y, r, rid = y[order], r[order], rid[order]
        slot = self.lut[types]
        data[self.key_x] = _FirstLayer.apply(
            self, y.contiguous(), r, rid.contiguous(), slot,
            graph._tables[('uvu', 32)],
        ) + self.bias
        return data


def collapse_first_layer(
    model: nn.Module, reference: nn.Module, modal: str | None = None
) -> list[str]:
    """Fuse the first convolution with its adjacent linears in a tabulated model.

    Pass the corresponding e3nn model as reference with force_output removed.
    Call specialize_fused_species to build its species tables before inference.
    """
    names = list(model._modules)
    conv_names = [n for n in names if n.endswith('_convolution')]
    if not conv_names:
        return []
    at = names.index(conv_names[0])
    trio = names[at - 1:at + 2]
    if len(trio) != 3:
        return []
    conv = model._modules[conv_names[0]]
    lin1, lin2 = model._modules[trio[0]], model._modules[trio[2]]
    if (
        type(conv) is not LunaTPFusedRadialConvolution or not conv.knots
        or not isinstance(lin1, IrrepsLinear) or not isinstance(lin2, IrrepsLinear)
        or lin1.key_output != conv.key_x or lin2.key_input != conv.key_x
        or lin2.key_output != conv.key_x
        or any(ir.l != 0 or ir.p != 1 for _, ir in lin1.irreps_out)
        or any(ir.dim > 7 or ir.p != (-1) ** ir.l for _, ir in lin2.irreps_out)
        or 'onehot_to_feature_x' not in names[:at]
    ):
        return []
    # between the embedding and the linear nothing else may write the features
    for name in names[names.index('onehot_to_feature_x') + 1:at - 1]:
        if getattr(model._modules[name], 'key_output', None) == conv.key_x:
            return []
    reference = _copy_cpu_double(reference)
    reference.set_is_batch_data(False)
    reference.eval()
    if modal is not None:
        reference.prepare_modal_deploy(modal)
    first = FusedFirstLayer(conv, reference, trio)
    model._modules[trio[0]] = nn.Identity()
    model._modules[trio[1]] = first
    model._modules[trio[2]] = nn.Identity()
    return trio


def live_paths(conv, linear) -> list[int] | None:
    """Return the TP paths read by the next linear, or None if it reads all."""
    irreps_in = getattr(linear, '_irreps_in_wo_modal', None)
    irreps_out = getattr(linear, 'irreps_out', None)
    op = conv.convolution.tp_conv.op
    if irreps_in is None or irreps_out is None or irreps_in.dim != op.out_dim:
        return None
    used = {ir for _, ir in irreps_out}
    live, at = [], 0  # [begin, end) column ranges the linear reads
    for mul, ir in irreps_in:
        if ir in used:
            live.append((at, at + mul * ir.dim))
        at += mul * ir.dim
    keep = [
        n for n, spec in enumerate(op.generated_specs)
        if any(a <= int(spec['out_offset']) < b for a, b in live)
    ]
    return None if len(keep) == len(op.generated_specs) else keep


def fuse_radial_conv(
    model: nn.Module, knots: int = 0, r_min: float = DEFAULT_RADIAL_TABLE_R_MIN
) -> list[str]:
    """
    In place: every LunaTP convolution of a serial model whose radial MLP is
    an e3nn FullyConnectedNet [n, h, h, W] of the edge embedding becomes a
    LunaTPFusedRadialConvolution. Inference only. Returns the names changed.
    With positive knots, interpolate weights from the table on [r_min, cutoff].
    With zero knots, evaluate weights for each distinct distance.
    """
    from e3nn.nn import FullyConnectedNet

    if knots < 0 or knots == 1:
        raise ValueError('fused_knots must be 0 or at least 2')
    cutoff = float(model.cutoff)
    if knots and not 0 < r_min < cutoff:
        raise ValueError(f'fused_r_min must be between 0 and the cutoff ({cutoff})')
    emb = model._modules['edge_embedding']
    if getattr(emb, 'distance_transform', None) is not None:
        return []
    changed, fused = [], []
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
        paths = None
        linear = model._modules.get(
            name.replace('convolution', 'self_interaction_2')
        )
        if isinstance(linear, IrrepsLinear) and linear.key_input == conv.key_x:
            paths = live_paths(conv, linear)
        conv.__class__ = LunaTPFusedRadialConvolution
        conv._setup(emb, knots, r_min, cutoff, paths)
        changed.append(name if paths is None else f'{name}({len(paths)}paths)')
        fused.append(conv)
    if (
        fused and len(fused) == len(list(_convolutions(model)))
        and fused[0].rad_w2.is_cuda and not getattr(emb, 'save_components', False)
        and bool(getattr(emb.spherical.sph, 'normalize', False))
    ):
        model._modules['edge_embedding'] = FusedEdgeGeometry(emb)
        changed.append('edge_embedding')
    return changed


def prepare_fused_inference(model, build_reference, modal, knots, r_min, first=True):
    names = fuse_radial_conv(model, knots=knots, r_min=r_min)
    if names:
        from .inference_specialization import fold_modal

        if modal is not None and model.modal_map:
            names += fold_modal(model, modal)
        if knots and first:
            reference = build_reference()
            reference.delete_module_by_key('force_output')
            names += collapse_first_layer(model, reference, modal)
    return names


def specialize_fused_species(model: nn.Module, columns: Sequence[int]) -> list[str]:
    for module in model._modules.values():
        if isinstance(module, FusedFirstLayer):
            module.set_species(columns)
    return prune_self_connection(model, columns)


def set_index_order(model: nn.Module, flag: bool) -> None:
    for module in model.modules():
        if isinstance(module, LunaTPFusedRadialConvolution):
            module.index_order = flag


def set_unique_cap(model: nn.Module, cap: int | None) -> None:
    for module in model.modules():
        if isinstance(module, LunaTPFusedRadialConvolution):
            module.unique_cap = cap
