"""Frozen radial networks for inference.

For radial weights that depend only on edge length, cubic Hermite tables
give values and their first derivatives. The folded MLP keeps the exact
network and absorbs e3nn normalization and the convolution denominator.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

import sevenn._keys as KEY
from sevenn._const import DEFAULT_RADIAL_TABLE_KNOTS, DEFAULT_RADIAL_TABLE_R_MIN
from sevenn.nn.edge_embedding import BesselBasis


def _copy_cpu_double(module: nn.Module) -> nn.Module:
    # Older PyTorch cannot pickle e3nn's scripted functions during deepcopy.
    script_functions = {
        id(value): value
        for child in module.modules()
        for value in vars(child).values()
        if isinstance(value, torch.jit.ScriptFunction)
    }
    return copy.deepcopy(module, script_functions).to('cpu', torch.float64)


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
    long long e; int w;
    if (E * W < 4294967296LL) {  // 32-bit division: several times faster
        unsigned int e32 = (unsigned int)idx / (unsigned int)W;
        e = e32; w = (int)((unsigned int)idx - e32 * (unsigned int)W);
    } else {
        e = idx / W; w = (int)(idx - e * W);
    }
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
        acc += g[w] * (c[W + w] + t * (2.0f * c[2 * W + w]
                                      + 3.0f * t * c[3 * W + w]));
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
        stream = cp.cuda.ExternalStream(
            torch.cuda.current_stream(r.device).cuda_stream
        )
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
        stream = cp.cuda.ExternalStream(
            torch.cuda.current_stream(r.device).cuda_stream
        )
        with stream:
            bwd(((E * 32 + 255) // 256,), (256,), (
                cp.asarray(r.detach()), cp.asarray(coef.detach()),
                cp.asarray(grad_w.detach()),
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


class FoldedRadialMLP(nn.Module):
    """e3nn FullyConnectedNet with every constant in the frozen weights"""

    def __init__(self, net, scale: float = 1.0):
        super().__init__()
        layers = list(net)
        self.acts = [layer.act.f if layer.act is not None else None
                     for layer in layers]
        carry = 1.0  # constant factor of the previous layer's output
        for n, layer in enumerate(layers):
            w = layer.weight.detach().to(torch.float64) * carry
            if layer.act is not None:
                w = w / (layer.h_in * layer.var_in) ** 0.5
                cst = 1.0 if layer.act._is_id else layer.act.cst
                carry = cst * layer.var_out ** 0.5
            else:
                w = w / (layer.h_in * layer.var_in / layer.var_out) ** 0.5
                carry = 1.0
            if n == len(layers) - 1:
                w = w * (carry * scale)
            self.register_buffer(f'w{n}', w.to(layer.weight.dtype))
        self.num_layers = len(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n in range(self.num_layers):
            x = x @ getattr(self, f'w{n}')
            if self.acts[n] is not None:
                x = self.acts[n](x)
        return x


def _convolutions(model):
    for name, conv in model._modules.items():
        if name.endswith('_convolution'):
            yield name, getattr(conv, 'conv', conv)  # MLIAPWrappedConvolution


def _radial_embedding(embedding: nn.Module, r: torch.Tensor) -> torch.Tensor:
    basis = embedding.basis_function
    if isinstance(basis, BesselBasis):
        ur = r.unsqueeze(-1)
        at_origin = ur == 0
        denominator = torch.where(at_origin, 1.0, ur)
        values = basis.prefactor * torch.where(
            at_origin, basis.coeffs, torch.sin(basis.coeffs * ur) / denominator,
        )
    else:
        values = basis(r)
    return values * embedding.cutoff_function(r).unsqueeze(-1)


def supports_radial_table(model: nn.Module) -> bool:
    emb = model._modules.get('edge_embedding')
    return (
        emb is not None
        and getattr(emb, 'distance_transform', None) is None
        and all(
            getattr(conv, 'key_weight_input', None) == KEY.EDGE_EMBEDDING
            for _, conv in _convolutions(model)
        )
    )


def _cubics(v: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Hermite coefficients (K - 1, W, 4) with slopes in interval t."""
    dif = v[1:] - v[:-1]
    a2 = 3.0 * dif - 2.0 * d[:-1] - d[1:]
    a3 = -2.0 * dif + d[:-1] + d[1:]
    return torch.stack([v[:-1], d[:-1], a2, a3], 2)


def _table(radial, r0: float, r1: float, num_knots: int):
    r = torch.linspace(r0, r1, num_knots, dtype=torch.float64)
    h = (r1 - r0) / (num_knots - 1)
    v, d = torch.autograd.functional.jvp(radial, r, torch.ones_like(r))
    return _cubics(v, d * h), h


def tabulate_radial(
    model: nn.Module,
    num_knots: int = DEFAULT_RADIAL_TABLE_KNOTS,
    r_min: float = DEFAULT_RADIAL_TABLE_R_MIN,
    kernel: bool = False,
    *,
    exclude=(),
) -> list[str]:
    """Replace each radial MLP with a table on [r_min, cutoff].

    Shorter distances extrapolate using the first cubic. The tables provide
    first derivatives for forces. Set kernel for the FP32 CuPy CUDA path.
    """
    convs = [
        (name, conv) for name, conv in _convolutions(model) if name not in exclude
    ]
    if not convs:
        return []
    emb = model._modules['edge_embedding']
    if getattr(emb, 'distance_transform', None) is not None:
        raise ValueError('radial input is not a function of r only')
    cutoff = float(model.cutoff)
    if num_knots < 2:
        raise ValueError('radial_knots must be at least 2')
    if not 0 <= r_min < cutoff:
        raise ValueError(f'radial_r_min must be in [0, {cutoff})')
    ref = next(model.parameters())
    emb64 = _copy_cpu_double(emb)
    changed = []
    for name, conv in convs:
        if conv.key_weight_input != KEY.EDGE_EMBEDDING:
            raise ValueError(f'{name}: radial input {conv.key_weight_input}')
        mlp64 = _copy_cpu_double(conv.weight_nn)
        scale = 1.0
        if not getattr(conv, 'denominator_folded', True):
            scale = 1.0 / float(conv.denominator.detach().double().item())
            conv.denominator_folded = True

        def radial(r, mlp64=mlp64, scale=scale):
            return scale * mlp64(_radial_embedding(emb64, r))

        coef, h = _table(radial, r_min, cutoff, num_knots)
        coef = coef.transpose(1, 2).to(device=ref.device, dtype=ref.dtype)
        conv.weight_nn = RadialTable(coef.contiguous(), r_min, h, kernel)
        conv.key_weight_input = KEY.EDGE_LENGTH
        changed.append(name)
    return changed
