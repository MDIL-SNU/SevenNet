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

With knots > 0 (fuse_radial_conv) there is no per-call weight array at all:
the kernels interpolate each weight (cubic Hermite in r) from a table of the
MLP at fixed knots, (K - 1, W, 4) floats that stay mostly in the GPU cache,
and take dw/dr from the same load. The kernels are then bound by arithmetic, not
by memory traffic, and the last-layer GEMM is gone.

The kernels are LunaEquivariance's generated uvu_conv forward and first-order
backward sources (HPC-AI-Team/LunaEquivariance, BSD-3-Clause, commit a045a88,
lunatp/uvu_conv/kernels/{forward,backward}.py), patched as text: weight row
index, and the weight-gradient stores replaced by the contraction. Same
function as LunaTPSharedGraphConvolution up to round-off.
"""
import copy
import dataclasses
import re
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .linear import IrrepsLinear
from .lunatp_helper import LunaTPConvolution, LunaTPSharedGraphConvolution
from .radial_table import FoldedRadialMLP, _convolutions, _table
from .self_connection import SelfConnectionIntro

_KEY_R = 'fused_radial_r'  # the KEY.EDGE_LENGTH tensor the entries belong to
_KEY_RID = 'fused_radial_rid'
_KEY_RU = 'fused_radial_r_unique'
_KEY_EMB = 'fused_radial_embedding'
_KEY_EMB_R = 'fused_radial_embedding_r'
_KEY_HID = 'fused_radial_hidden'
_KEY_HID_R = 'fused_radial_hidden_r'
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


def forward_source(op, half: bool = False, hermite: bool = False) -> str:
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
    src = _sub(
        r'num_nodes\);', 'num_nodes, rid.data_ptr<int32_t>());', src, count=2
    )
    if hermite:
        assert not half
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
    if half:  # weight rows stored as fp16
        src = _sub(r'const float ?\*(?: ?__restrict__)? weight\b',
                   'const __half* __restrict__ weight', src, count=2)
        src = _sub('const float* weight_edge', 'const __half* weight_edge', src,
                   regex=False)
        src = _sub(r'= weight_edge\[(\d+) \+ u\];',
                   r'= __half2float(weight_edge[\1 + u]);', src)
        src = _sub('weight.data_ptr<float>()',
                   'reinterpret_cast<const __half*>(weight.data_ptr<at::Half>())',
                   src, count=2, regex=False)
        src = '#include <cuda_fp16.h>\n' + src
    return src


def backward_source(op, half: bool = False, hermite: bool = False):
    """
    (source, segmented, partials, vectorized, zero_x, zero_y). hermite: the
    partial sums are h * dE/dr, grad_weight is not read
    """
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
    if hermite:
        assert not half

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
    if half:  # weight rows (w and dw/dr) stored as fp16
        src = _sub('const float* __restrict__ weight,',
                   'const __half* __restrict__ weight,', src, count=1, regex=False)
        src = _sub('const float* __restrict__ grad_weight,',
                   'const __half* __restrict__ grad_weight,', src, count=1,
                   regex=False)
        src = re.sub(
            r'__ldg\(reinterpret_cast<const float2\*>\(weight \+ wb \+ (\d+) \+ u\)\)',
            r'make_float2(__half2float(weight[wb + \1 + u]), '
            r'__half2float(weight[wb + \1 + u + 1]))', src,
        )
        src = _sub(r'__ldg\(((?:grad_)?weight \+ wb \+ \d+ \+ u(?: \+ 1)?)\)',
                   r'__half2float(*(\1))', src)
        src = _sub('weight.data_ptr<float>()',
                   'reinterpret_cast<const __half*>(weight.data_ptr<at::Half>())',
                   src, regex=False)
        src = _sub('gw.data_ptr<float>()',
                   'reinterpret_cast<const __half*>(gw.data_ptr<at::Half>())',
                   src, regex=False)
        src = '#include <cuda_fp16.h>\n' + src
    return (src, *flags)


_HIDDEN_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
namespace {
__global__ void sevenn_radial_hidden_kernel(
    const float* __restrict__ r, const float* __restrict__ coef,
    float* __restrict__ out, int64_t n, int W, int H, int K1, double r0,
    double inv_h) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * W) return;
    const int64_t i = idx / W;
    const int c = int(idx - i * W);
    const double x = (double(r[i]) - r0) * inv_h;
    const int k = min(max(int(floor(x)), 0), K1 - 1);
    const float t = float(x - k);
    const float* a = coef + (int64_t(k) * W + c) * 4;
    const float a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    const int l = c / H, f = c - l * H;
    float* o = out + int64_t(l) * 2 * n * H;
    o[i * H + f] = a0 + t * (a1 + t * (a2 + t * a3));
    o[(n + i) * H + f] = (a1 + t * (2.0f * a2 + 3.0f * t * a3)) * float(inv_h);
}
}  // namespace
void sevenn_radial_hidden(torch::Tensor r, torch::Tensor coef, torch::Tensor out,
                          int64_t H, double r0, double inv_h) {
    const int64_t n = r.numel();
    const int W = coef.size(1);
    const int64_t total = n * W;
    if (total == 0) return;
    sevenn_radial_hidden_kernel<<<(total + 255) / 256, 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        r.data_ptr<float>(), coef.data_ptr<float>(), out.data_ptr<float>(), n, W,
        int(H), int(coef.size(0)), r0, inv_h);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sevenn_radial_hidden", &sevenn_radial_hidden);
}
"""


def _load_cuda(name: str, source: str):
    """a CUDA extension by LunaEquivariance's build cache"""
    from LunaEquivariance.utils.extension import BINDING_CFLAGS
    from LunaEquivariance.utils.jit import load_extension

    return load_extension(
        name, cuda_sources=source,
        extra_cflags=['-O3', *BINDING_CFLAGS],
        extra_cuda_cflags=['-O3', *BINDING_CFLAGS],
    )


class RadialHiddenTable(nn.Module):
    """
    The hidden activations of the radial MLPs of all fused convolutions (L
    layers of H) and their r-derivatives as one cubic Hermite table in r, read
    by one kernel: forward(r (U,)) -> (L, 2U, H), rows [value ; d/dr] of each
    layer. Built in fp64; num_knots on [r_min, cutoff] (shorter: the first
    interval's cubic).
    """

    def __init__(self, hidden64, num_layers: int, cutoff: float,
                 num_knots: int = 8192, r_min: float = 0.25):
        super().__init__()
        coef, h = _table(hidden64, r_min, cutoff, num_knots, torch.float32)
        # (K - 1, 4, W) -> (K - 1, W, 4)
        self.register_buffer('coef', coef.transpose(1, 2).contiguous())
        self.r0, self.inv_h = r_min, 1.0 / h
        self.num_layers = num_layers
        self.width = coef.shape[2] // num_layers
        self.ext = _load_cuda('sevenn_radial_hidden_fp32', _HIDDEN_SOURCE)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        out = r.new_empty((self.num_layers, 2 * r.shape[0], self.width))
        self.ext.sevenn_radial_hidden(
            r.contiguous(), self.coef, out, self.width, self.r0, self.inv_h
        )
        return out


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
    sph = copy.deepcopy(sph).to('cpu', torch.float64)
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
    """
    EdgeEmbedding of a model whose convolutions all take their radial weights
    from a table in r (fuse_radial_conv, knots > 0): the edge length and the
    spherical harmonics by one kernel, their backward to the edge vector by
    another (instead of ~60 + ~120 elementwise kernels). The radial basis,
    which no module reads, is not computed.
    """

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



def _index_order(tables):
    """tables with the launch orders in node / segment index order (cached)"""
    cached = getattr(tables.src, '_sevenn_index_order', None)
    if cached is not None and cached[0] is tables:
        return cached[1]
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
    tables.src._sevenn_index_order = (tables, out)
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
        out = x.new_empty((x.shape[0], op.out_dim))
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
            gx, gy, wbuf if mod.knots else wbuf[half:], x.shape[0], n_edge,
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
    """
    see the module docstring. unique_cap: None = rows for exactly the distinct
    edge lengths of the call (torch.unique: a host sync); an int = that many
    rows whatever the edges (fixed shapes: CUDA graph), and
    data[KEY_N_UNIQUE] holds the number needed (results are wrong if it is
    larger: the caller must check it).
    """

    unique_cap: Optional[int] = None
    # kernel blocks launched in node index order instead of LunaTP's order by
    # degree: with atoms in a spatially coherent order (LAMMPS atom sorting)
    # the rows that neighbors share (weight rows of i-j and j-i, the output
    # gradient rows) are still in the GPU cache when the next block reads them
    index_order: bool = True
    # weight rows (w, dw/dr) stored as fp16: half the bytes the kernels read,
    # forces change by ~1e-4 eV/A (not the same function to round-off)
    half_weights: bool = False
    # > 0: the weights of an edge are cubic Hermite interpolated inside the
    # kernels from a table of that many knots on [table_r0, cutoff] (see
    # weight_table): no per-call weight array and GEMM.
    # The table is what the kernels read, mostly from the GPU cache.
    knots: int = 0
    table_r0: float = 0.25
    table_inv_h: float = 0.0

    def _setup(self, emb, half_weights: bool = False, knots: int = 0,
               r_min: float = 0.25, cutoff: float = 0.0) -> None:
        self.half_weights = half_weights
        self.knots = int(knots)
        assert not (half_weights and knots)
        ref = self.denominator
        folded = FoldedRadialMLP(
            copy.deepcopy(self.weight_nn).to('cpu', torch.float64),
            1.0 / float(ref.detach().double()),
        )
        self.denominator_folded = True
        assert folded.num_layers == 3
        for n in range(3):
            w = getattr(folded, f'w{n}').to(ref.device, ref.dtype).contiguous()
            self.register_buffer(f'rad_w{n}', w)
        self._acts = folded.acts
        self._folded64 = folded  # fp64, cpu: for the table of the hidden layers
        self._hid_table, self._layer = None, 0
        if half_weights:
            wh = self.rad_w2.half()
            wl = (self.rad_w2 - wh.float()).half()
            self.register_buffer('rad_w2_half', torch.cat([wh, wh, wl]))
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
        tag = 'herm' if herm else 'w16' if self.half_weights else 'fp32'
        self.fwd_ext = load_tp_extension(
            f'sevenn_uvuconv_fused_fwd_{tag}',
            forward_source(op, self.half_weights, herm),
            max_channels=op.max_channels, index_bits=32,
        )
        src, *self.bwd_flags = backward_source(op, self.half_weights, herm)
        self.bwd_ext = load_tp_extension(
            f'sevenn_uvuconv_fused_bwd_{tag}', src,
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
        if self._hid_table is not None:
            if _KEY_HID not in data or data[_KEY_HID_R] is not ru:
                data[_KEY_HID] = self._hid_table(ru)
                data[_KEY_HID_R] = ru
            return self._last(data[_KEY_HID][self._layer])
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
        return self._last(x)

    def _last(self, x: torch.Tensor) -> torch.Tensor:
        """the last layer of the MLP, rows [value ; d/dr]"""
        if not self.half_weights:
            return torch.mm(x, self.rad_w2)
        # fp16 GEMM (fp32 accumulation) of inputs split in two fp16 parts:
        # x w = xh wh + xl wh + xh wl (+ xl wl, dropped); only the result is
        # rounded to fp16
        xh = x.half()
        xl = (x - xh.float()).half()
        return torch.mm(torch.cat([xh, xl, xh], 1), self.rad_w2_half)

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
        if KEY_N_UNIQUE not in data:
            data[KEY_N_UNIQUE] = torch.zeros((), dtype=torch.int64, device=r.device)
        return rid

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
        if self.knots:
            if self.rad_table.numel() >= 2**31:
                raise RuntimeError('too many knots for the fused radial kernels')
            rid, wbuf = self._knots(data), self.rad_table
        else:
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
        if self.index_order and not self.convolution.tp_conv.op.forward_degree_tiers:
            tables = _index_order(tables)
        data[self.key_x] = _FusedConv.apply(
            self, x.contiguous(), y.contiguous(), r, rid.contiguous(), wbuf,
            tables,
        )
        return data


def weight_table(conv, emb, r_min: float, cutoff: float, knots: int):
    """
    (table (K - 1, W, 4) fp64, h): on each of the intervals of K knots on
    [r_min, cutoff], the cubic in t = (r - r_k) / h of every weight of conv's
    radial MLP that has the MLP's value and r-derivative at both knots. conv
    after _setup (folded fp64 weights)
    """
    emb64 = copy.deepcopy(emb).to('cpu', torch.float64)
    f = conv._folded64

    def radial(r):
        x = emb64.basis_function(r) * emb64.cutoff_function(r).unsqueeze(-1)
        return conv._acts[1](conv._acts[0](x @ f.w0) @ f.w1) @ f.w2

    r = torch.linspace(r_min, cutoff, knots, dtype=torch.float64)
    h = (cutoff - r_min) / (knots - 1)
    v, dv = torch.autograd.functional.jvp(radial, r, torch.ones_like(r))
    v, d = v.detach(), dv.detach() * h
    dif = v[1:] - v[:-1]
    a2 = 3.0 * dif - 2.0 * d[:-1] - d[1:]
    a3 = -2.0 * dif + d[:-1] + d[1:]
    return torch.stack([v[:-1], d[:-1], a2, a3], 2), h


def fuse_radial_conv(
    model: nn.Module, hidden_table: bool = True, half_weights: bool = False,
    knots: int = 0, r_min: float = 0.25,
) -> List[str]:
    """
    In place: every LunaTP convolution of a serial model whose radial MLP is
    an e3nn FullyConnectedNet [n, h, h, W] of the edge embedding becomes a
    LunaTPFusedRadialConvolution. Inference only. Returns the names changed.
    knots > 0: weights interpolated in the kernels from a table of that many
    knots on [r_min, cutoff].
    """
    from e3nn.nn import FullyConnectedNet

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
        conv.__class__ = LunaTPFusedRadialConvolution
        conv._setup(emb, half_weights, knots, r_min, float(model.cutoff))
        changed.append(name)
        fused.append(conv)
    if (
        knots and fused and len(fused) == len(list(_convolutions(model)))
        and fused[0].rad_w2.is_cuda and not emb.save_components
        and bool(getattr(emb.spherical.sph, 'normalize', False))
    ):
        model._modules['edge_embedding'] = FusedEdgeGeometry(emb)
        changed.append('edge_embedding')
    widths = {c.rad_w2.shape[0] for c in fused}
    if (
        hidden_table and not knots and fused and len(widths) == 1
        and fused[0].rad_w2.is_cuda
    ):
        emb64 = copy.deepcopy(emb).to('cpu', torch.float64)

        def hidden64(r):
            x = emb64.basis_function(r) * emb64.cutoff_function(r).unsqueeze(-1)
            return torch.cat([
                c._acts[1](c._acts[0](x @ c._folded64.w0) @ c._folded64.w1)
                for c in fused
            ], 1)

        table = RadialHiddenTable(hidden64, len(fused), float(model.cutoff))
        table.to(fused[0].rad_w2.device)
        for n, c in enumerate(fused):
            object.__setattr__(c, '_hid_table', table)  # shared, not a submodule
            c._layer = n
    return changed


class PrunedSelfConnectionIntro(SelfConnectionIntro):
    """
    SelfConnectionIntro whose operand (species one-hot, 119 columns in 7net)
    is cut to the columns of the species that can occur: the same function
    for those species, a tensor product over len(columns) operand components
    instead of all.
    """

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        operand = data[self.key_operand].index_select(1, self.columns)
        data[KEY.SELF_CONNECTION_TEMP] = self.fc_tensor_product(
            data[self.key_x], operand
        )
        return data


def prune_self_connection(model: nn.Module, columns: Sequence[int]) -> List[str]:
    """
    In place: every SelfConnectionIntro of model (operand n x 0e) becomes a
    PrunedSelfConnectionIntro of the operand columns given (type indices of
    the species that can occur). Calls with other species are wrong. Calling
    it again prunes the original weights to the new columns.
    """
    from e3nn.o3 import Irreps

    columns = sorted(set(int(c) for c in columns))
    changed = []
    for name, sc in model._modules.items():
        if type(sc) is PrunedSelfConnectionIntro:  # again, other columns
            old = sc._full
        elif type(sc) is SelfConnectionIntro and sc.layer_instantiated:
            old = sc.fc_tensor_product
        else:
            continue
        in2 = Irreps(sc.irreps_in2)
        if len(in2) != 1 or in2[0].ir.l != 0 or in2[0].ir.p != 1:
            continue
        ref = old.weight
        new = sc.fc_tensor_product_cls(
            sc.irreps_in1, Irreps(f'{len(columns)}x0e'), sc.irreps_out,
            shared_weights=True, internal_weights=None,
            **sc.fc_tensor_product_kwargs,
        ).to(ref.device, ref.dtype)
        cols = torch.tensor(columns, device=ref.device)
        tp = getattr(old, 'tp', old)  # the e3nn module (instructions)
        pieces, offset = [], 0
        for ins in tp.instructions:
            if not ins.has_weight:
                continue
            m1, m2, m3 = ins.path_shape
            w = ref.detach()[offset: offset + m1 * m2 * m3].view(m1, m2, m3)
            pieces.append(w.index_select(1, cols).reshape(-1))
            offset += m1 * m2 * m3
        assert offset == ref.numel()
        # path normalization counts the operand multiplicity: one factor
        scale = (len(columns) / in2[0].mul) ** 0.5
        with torch.no_grad():
            new.weight.copy_(torch.cat(pieces) * scale)
            x = torch.randn(4, Irreps(sc.irreps_in1).dim, device=ref.device,
                            dtype=ref.dtype)
            op = torch.zeros(4, in2[0].mul, device=ref.device, dtype=ref.dtype)
            op[torch.arange(4), cols[torch.arange(4) % len(columns)]] = 1.0
            a, b = old(x, op), new(x, op.index_select(1, cols))
            if not torch.allclose(a, b, rtol=1e-4, atol=1e-5 * float(a.abs().max())):
                raise RuntimeError(f'{name}: pruned self connection differs')
        new.requires_grad_(False)
        sc.__class__ = PrunedSelfConnectionIntro
        object.__setattr__(sc, '_full', old)  # kept outside the module tree
        sc.fc_tensor_product = new
        sc.register_buffer('columns', cols)
        changed.append(name)
    return changed


def set_index_order(model: nn.Module, flag: bool) -> None:
    for module in model.modules():
        if isinstance(module, LunaTPFusedRadialConvolution):
            module.index_order = flag


def set_unique_cap(model: nn.Module, cap: Optional[int]) -> None:
    for module in model.modules():
        if isinstance(module, LunaTPFusedRadialConvolution):
            module.unique_cap = cap


class ModalFoldedLinear(IrrepsLinear):
    """
    IrrepsLinear of a multi-modal model for one fixed modality: the modal
    one-hot columns appended to the input (a copy of the whole input per
    call) only add a constant to the scalar outputs, kept as modal_bias.
    """

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.linear(data[self.key_input]) + self.modal_bias
        return data


def fold_modal(model: nn.Module, modal: str) -> List[str]:
    """
    In place: every IrrepsLinear with modal input becomes a ModalFoldedLinear
    of the modality given. The model then gives that modality whatever
    data[KEY.DATA_MODALITY] says for these layers.
    """
    from e3nn.o3 import Irreps, Linear

    index = model.modal_map[modal]
    changed = []
    for name, lin in model._modules.items():
        if type(lin) is not IrrepsLinear or lin.num_modalities <= 1:
            continue
        if not lin.layer_instantiated:
            continue
        old = lin.linear
        ref = old.weight
        kwargs = {k: v for k, v in lin.linear_kwargs.items()}
        ins_old = Linear(lin.irreps_in, lin.irreps_out, **kwargs).instructions
        ins_new = Linear(
            lin._irreps_in_wo_modal, lin.irreps_out, **kwargs
        ).instructions
        new = lin.linear_cls(
            lin._irreps_in_wo_modal, lin.irreps_out, **kwargs
        ).to(ref.device, ref.dtype)
        chunks, offset = {}, 0
        for ins in ins_old:
            n = ins.path_shape[0] * ins.path_shape[1]
            chunks[(ins.i_in, ins.i_out)] = (
                ref.detach()[offset: offset + n], ins.path_weight
            )
            offset += n
        assert offset == ref.numel()
        pieces = []
        for ins in ins_new:
            w, path_weight = chunks[(ins.i_in, ins.i_out)]
            pieces.append(w * (path_weight / ins.path_weight))
        in_dim = Irreps(lin._irreps_in_wo_modal).dim
        with torch.no_grad():
            new.weight.copy_(torch.cat(pieces))
            onehot = torch.zeros(
                4, lin.num_modalities, device=ref.device, dtype=ref.dtype
            )
            onehot[:, index] = 1.0
            zero = torch.zeros(4, in_dim, device=ref.device, dtype=ref.dtype)
            bias = old(torch.cat([zero, onehot], 1))[0].clone()
            x = torch.randn_like(zero)
            a, b = old(torch.cat([x, onehot], 1)), new(x) + bias
            if not torch.allclose(a, b, rtol=1e-4, atol=1e-5 * float(a.abs().max())):
                raise RuntimeError(f'{name}: modal-folded linear differs')
        new.requires_grad_(False)
        lin.__class__ = ModalFoldedLinear
        lin.linear = new
        lin.register_buffer('modal_bias', bias)
        changed.append(name)
    return changed
