"""
Tabulated radial MLP (inference). In the original SevenNet model the weights
of a convolution are weight_nn(basis(r) * cutoff(r)): a function of the edge
length only. tabulate_radial replaces it by a cubic Hermite table in r (values
and exact derivatives of the fp64 function at uniform knots, so the table is
C1 and its derivative is the derivative of what the forward returns), i.e.
E x weight_numel reads instead of the E x (64 x weight_numel) matrix products
of the forward and the backward.

Not applicable to (and refused for) a model whose radial input depends on
more than r: pair / node conditioned radial MLPs, distance transforms.
"""
import copy
from typing import List

import torch
import torch.nn as nn

import sevenn._keys as KEY


class _Hermite(torch.autograd.Function):
    """w[e] = sum_k coef[i_e, k] t_e^k, first derivative only (forces)"""

    @staticmethod
    def forward(ctx, r, coef, r0: float, inv_h: float):
        u = (r - r0) * inv_h
        i = u.floor().clamp_(0, coef.shape[0] - 1).to(torch.int64)
        t = u - i
        rows = coef.index_select(0, i)  # (E, 4, W)
        t2 = t * t
        basis = torch.stack([torch.ones_like(t), t, t2, t2 * t], 1)  # (E, 4)
        ctx.save_for_backward(i, t, coef)
        ctx.inv_h = inv_h
        return torch.bmm(basis.unsqueeze(1), rows).squeeze(1)

    @staticmethod
    def backward(ctx, grad_w):
        i, t, coef = ctx.saved_tensors
        rows = coef.index_select(0, i)
        dbasis = torch.stack(
            [torch.zeros_like(t), torch.ones_like(t), 2 * t, 3 * t * t], 1
        )
        dw = torch.bmm(dbasis.unsqueeze(1), rows).squeeze(1)  # dw/dt
        return (dw * grad_w).sum(1) * ctx.inv_h, None, None, None


_KERNELS = None


def _kernels():
    """(forward, backward) cupy RawKernels: one launch each"""
    global _KERNELS
    if _KERNELS is None:
        import cupy as cp

        fwd = cp.RawKernel(r"""
extern "C" __global__ void radial_fwd(
    const float* r, const float* coef, float* out,
    const float r0, const float inv_h, const int K1, const int W,
    const long long E)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E * W) return;
    long long e = idx / W; int w = (int)(idx - e * W);
    float u = (r[e] - r0) * inv_h;
    int i = (int)floorf(u);
    i = i < 0 ? 0 : (i > K1 - 1 ? K1 - 1 : i);
    float t = u - (float)i;
    const float* c = coef + ((long long)i * 4) * W + w;
    out[idx] = c[0] + t * (c[W] + t * (c[2 * W] + t * c[3 * W]));
}""", 'radial_fwd')
        bwd = cp.RawKernel(r"""
extern "C" __global__ void radial_bwd(
    const float* r, const float* coef, const float* grad_w, float* grad_r,
    const float r0, const float inv_h, const int K1, const int W,
    const long long E)
{
    // one warp per edge
    long long e = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (e >= E) return;
    float u = (r[e] - r0) * inv_h;
    int i = (int)floorf(u);
    i = i < 0 ? 0 : (i > K1 - 1 ? K1 - 1 : i);
    float t = u - (float)i;
    const float* c = coef + ((long long)i * 4) * W;
    const float* g = grad_w + e * W;
    float acc = 0.0f;
    for (int w = lane; w < W; w += 32)
        acc += g[w] * (c[W + w] + t * (2.0f * c[2 * W + w] + 3.0f * t * c[3 * W + w]));
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) grad_r[e] = acc * inv_h;
}""", 'radial_bwd')
        _KERNELS = (cp, fwd, bwd)
    return _KERNELS


class _HermiteKernel(torch.autograd.Function):
    """_Hermite as one CUDA kernel per pass (cupy RawKernel)"""

    @staticmethod
    def forward(ctx, r, coef, r0: float, inv_h: float):
        cp, fwd, _ = _kernels()
        E, (K1, _, W) = r.shape[0], coef.shape
        out = torch.empty((E, W), dtype=r.dtype, device=r.device)
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream(r.device).cuda_stream)
        with stream:
            n = E * W
            fwd(((n + 255) // 256,), (256,), (
                cp.asarray(r.detach()), cp.asarray(coef.detach()), cp.asarray(out),
                cp.float32(r0), cp.float32(inv_h), cp.int32(K1), cp.int32(W),
                cp.int64(E),
            ))
        ctx.save_for_backward(r, coef)
        ctx.args = (r0, inv_h)
        return out

    @staticmethod
    def backward(ctx, grad_w):
        cp, _, bwd = _kernels()
        r, coef = ctx.saved_tensors
        r0, inv_h = ctx.args
        E, (K1, _, W) = r.shape[0], coef.shape
        grad_r = torch.empty_like(r)
        grad_w = grad_w.contiguous()
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream(r.device).cuda_stream)
        with stream:
            bwd(((E * 32 + 255) // 256,), (256,), (
                cp.asarray(r.detach()), cp.asarray(coef.detach()), cp.asarray(grad_w.detach()),
                cp.asarray(grad_r), cp.float32(r0), cp.float32(inv_h),
                cp.int32(K1), cp.int32(W), cp.int64(E),
            ))
        return grad_r, None, None, None


class RadialTable(nn.Module):
    """drop-in for weight_nn: input the edge lengths (E,), output (E, W)"""

    def __init__(self, coef: torch.Tensor, r0: float, h: float, kernel=False):
        super().__init__()
        self.kernel = kernel
        self.register_buffer('coef', coef)  # (K - 1, 4, W)
        self.r0 = r0
        self.inv_h = 1.0 / h

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if r.shape[0] == 0:
            return r.new_zeros(0, self.coef.shape[2])
        fn = _HermiteKernel if self.kernel and r.is_cuda else _Hermite
        return fn.apply(r.contiguous(), self.coef, self.r0, self.inv_h)


def _table(radial, r0: float, r1: float, num_knots: int, dtype):
    """Hermite coefficients (K - 1, 4, W) of radial: (K,) fp64 -> (K, W)"""
    r = torch.linspace(r0, r1, num_knots, dtype=torch.float64)
    h = (r1 - r0) / (num_knots - 1)
    v, d = torch.autograd.functional.jvp(radial, r, torch.ones_like(r))
    d = d * h
    v0, v1, d0, d1 = v[:-1], v[1:], d[:-1], d[1:]
    coef = torch.stack(
        [v0, d0, 3 * (v1 - v0) - 2 * d0 - d1, 2 * (v0 - v1) + d0 + d1], 1
    )
    return coef.to(dtype), h


def tabulate_radial(
    model: nn.Module, num_knots: int = 2048, r_min: float = 0.25,
    kernel: bool = False,
) -> List[str]:
    """
    Replaces, in place, the radial MLP of every convolution of model by a
    RadialTable on [r_min, cutoff] (shorter edges: the first interval's cubic,
    extrapolated). kernel: one cupy CUDA kernel per pass instead of torch ops
    (fp32, CUDA). Returns the names of the convolutions changed.
    """
    emb = model._modules['edge_embedding']
    if getattr(emb, 'distance_transform', None) is not None:
        raise ValueError('radial input is not a function of r only')
    ref = next(model.parameters())
    emb64 = copy.deepcopy(emb).to('cpu', torch.float64)
    changed = []
    for name, conv in model._modules.items():
        conv = getattr(conv, 'conv', conv)  # MLIAPWrappedConvolution
        if not name.endswith('_convolution'):
            continue
        if conv.key_weight_input != KEY.EDGE_EMBEDDING:
            raise ValueError(f'{name}: radial input {conv.key_weight_input}')
        mlp64 = copy.deepcopy(conv.weight_nn).to('cpu', torch.float64)

        def radial(r, mlp64=mlp64):
            return mlp64(
                emb64.basis_function(r) * emb64.cutoff_function(r).unsqueeze(-1)
            )

        coef, h = _table(radial, r_min, float(model.cutoff), num_knots, ref.dtype)
        conv.weight_nn = RadialTable(coef.to(ref.device), r_min, h, kernel)
        conv.key_weight_input = KEY.EDGE_LENGTH
        changed.append(name)
    return changed
