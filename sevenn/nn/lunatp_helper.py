import itertools
import os
import subprocess
import sys
import tempfile
import warnings
import weakref
from typing import Optional, Union

import torch
from e3nn.o3 import FullyConnectedTensorProduct

from .convolution import IrrepsConvolution, IrrepsScatterGatterFusedConvolution
from .equivariant_gate import EquivariantGate
from .linear import IrrepsLinear
from .self_connection import SelfConnectionIntro, SelfConnectionLinearIntro

try:
    from LunaEquivariance import (
        LunaEqGate,
        LunaEqLinear,
        TensorProduct,
        TensorProductConv,
        make_graph,
    )

    class LunaTPFullyConnectedTensorProduct(TensorProduct):
        """
        e3nn FullyConnectedTensorProduct computed by LunaTP (uvw mode).
        TensorProduct registers the e3nn module as the submodule tp. The e3nn
        module is kept outside the module tree and its weight and output_mask
        are registered here, so the parameter and state dict names are those of
        e3nn and checkpoints and reEWC Fisher information match across backends.
        """

        def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
            tp = FullyConnectedTensorProduct(
                irreps_in1, irreps_in2, irreps_out, **kwargs
            )
            super().__init__(tp)
            del self._modules['tp']
            object.__setattr__(self, 'tp', tp)
            self._parameters['weight'] = tp.weight
            self._buffers['output_mask'] = tp.output_mask

        @property
        def weight(self):
            return self._parameters['weight']

    _LUNATP_AVAILABLE = True
except ImportError:
    _LUNATP_AVAILABLE = False


def is_lunatp_available() -> bool:
    return _LUNATP_AVAILABLE and torch.cuda.is_available()


def lunatp_needed(func):
    def wrapper(*args, **kwargs):
        if is_lunatp_available():
            return func(*args, **kwargs)
        raise ImportError('LunaTP is not available (no GPU found or import fail)')

    return wrapper


class LunaTPConvolution(torch.nn.Module):
    """
    Wrapper around LunaEquivariance.TensorProductConv to match
    IrrepsScatterGatterFusedConvolution.convolution_cls interface.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights: bool = False,
        internal_weights: bool = False,
    ):
        super().__init__()
        if not is_lunatp_available():
            raise ImportError('LunaTP is not available')
        self.tp_conv = TensorProductConv(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )

    # (weakref to edge_filter, num_nodes, edge order, graph) of the last graph,
    # shared by all instances. The convolution layers of one forward receive the
    # same KEY.EDGE_ATTR tensor as edge_filter, so the layers after the first
    # reuse its edge order and graph.
    _cache = (None, -1, None, None)

    def forward(self, x, edge_filter, weight, edge_src, edge_dst):
        ref, num_nodes, order, graph = LunaTPConvolution._cache
        if ref is None or ref() is not edge_filter or num_nodes != x.shape[0]:
            # LunaTP requires edges sorted by destination
            order = None
            if not bool(torch.all(edge_dst[1:] >= edge_dst[:-1])):
                order = torch.argsort(edge_dst, stable=True)
                edge_src, edge_dst = edge_src[order], edge_dst[order]
            graph = make_graph(edge_src, edge_dst, x.shape[0])
            LunaTPConvolution._cache = (
                weakref.ref(edge_filter), x.shape[0], order, graph
            )
        if order is not None:
            edge_filter, weight = edge_filter[order], weight[order]
        return self.tp_conv(x, edge_filter, weight, graph=graph)


@lunatp_needed
def patch_convolution(irreps_convolution: IrrepsConvolution):
    assert not irreps_convolution.layer_instantiated, (
        'Convolution layer already instantiated; cannot patch'
    )
    ret = IrrepsScatterGatterFusedConvolution.from_irreps_convolution(
        irreps_convolution
    )
    ret.convolution_cls = LunaTPConvolution
    return ret


@lunatp_needed
def patch_linear(module: Union[IrrepsLinear, SelfConnectionLinearIntro]):
    assert not module.layer_instantiated, (
        'Linear layer already instantiated; cannot patch'
    )
    module.linear_cls = LunaEqLinear
    return module


@lunatp_needed
def patch_fully_connected(module: SelfConnectionIntro):
    assert not module.layer_instantiated, (
        'Fully connected tensor product already instantiated; cannot patch'
    )
    module.fc_tensor_product_cls = LunaTPFullyConnectedTensorProduct
    return module


@lunatp_needed
def patch_gate(module: EquivariantGate):
    # EquivariantGate builds its e3nn Gate in __init__, so the instance is replaced
    module.gate = LunaEqGate(gate=module.gate)
    return module


def _run_once(module, device: torch.device, dtype: torch.dtype, order: int):
    # Forward and backward on two nodes (and two edges) build the module's kernels.
    # LunaEquivariance compiles a variant per set of tensors that require grad, so
    # every combination of inputs and parameters (frozen or trainable) runs once.
    # order 2 also differentiates first derivatives once more, as a force loss
    # does, with an output gradient that does and does not require grad.
    if isinstance(module, LunaTPConvolution):
        tp = module.tp_conv
        dims = [tp.irreps_in1.dim, tp.irreps_in2.dim, tp.weight_numel]
        src = torch.tensor([1, 0], device=device, dtype=torch.int32)
        extra = [src, src.flip(0)]
    elif isinstance(module, LunaTPFullyConnectedTensorProduct):
        dims = [module.irreps_in1.dim]
        onehot = torch.zeros(2, module.irreps_in2.dim, device=device, dtype=dtype)
        onehot[:, 0] = 1.0
        extra = [onehot]
    else:  # LunaEqLinear, LunaEqGate
        dims = [module.irreps_in.dim]
        extra = []
    params = [p for p in module.parameters() if p.numel() > 0]
    trainable = [p.requires_grad for p in params]
    flag_sets = itertools.product((True, False), repeat=len(dims) + len(params[:1]))
    try:
        for flags in flag_sets:
            for p in params:
                p.requires_grad_(flags[-1])
            inputs = [
                torch.randn(2, d, device=device, dtype=dtype, requires_grad=f)
                for d, f in zip(dims, flags)
            ]
            wrt = [t for t in inputs + params if t.requires_grad]
            out = module(*inputs, *extra)
            if not wrt or not out.requires_grad:
                continue
            torch.autograd.grad(out.sum(), wrt, allow_unused=True)
            if order < 2:
                continue
            # the first derivative requests any subset of wrt (a force requests
            # the inputs only), then its result is differentiated once more
            requests = [
                c for r in range(1, len(wrt) + 1)
                for c in itertools.combinations(wrt, r)
            ]
            for grad_out_flag, requested in itertools.product((1, 0), requests):
                grad_out = torch.randn_like(out, requires_grad=bool(grad_out_flag))
                out = module(*inputs, *extra)
                grads = torch.autograd.grad(
                    out, requested, grad_out, allow_unused=True, create_graph=True
                )
                loss = sum(
                    g.square().sum()
                    for g in grads
                    if g is not None and g.requires_grad
                )
                if isinstance(loss, torch.Tensor):
                    targets = wrt + [grad_out] * grad_out_flag
                    torch.autograd.grad(loss, targets, allow_unused=True)
    finally:
        for p, f in zip(params, trainable):
            p.requires_grad_(f)


def _precompile_worker(checkpoint_path: str, rank: str, size: str, order: str):
    from sevenn.util import load_checkpoint

    model = load_checkpoint(checkpoint_path).build_model(enable_lunatp=True)
    model.to('cuda')
    param = next(model.parameters())
    luna_types = (
        LunaTPConvolution,
        LunaTPFullyConnectedTensorProduct,
        LunaEqLinear,
        LunaEqGate,
    )
    targets = [m for m in model.modules() if isinstance(m, luna_types)]
    for module in targets[int(rank) :: int(size)]:
        _run_once(module, param.device, param.dtype, int(order))


def precompile(
    checkpoint: str, order: int = 1, max_workers: Optional[int] = None
) -> None:
    """
    Build the LunaEquivariance kernels of the model of a checkpoint (path or
    pretrained name) before its first call, in max_workers processes (default
    min(8, available CPUs)) that each run a share of the LunaTP modules once.
    Processes, not threads: LunaEquivariance kernels must not run from several
    threads of one process.
    order=2 also builds the kernels of second derivatives (force training).
    """
    if not is_lunatp_available():
        return
    from sevenn.util import load_checkpoint

    checkpoint_path = load_checkpoint(checkpoint).checkpoint_path
    workers = max_workers or min(8, len(os.sched_getaffinity(0)))
    # one compiler process per worker: a kernel's main.cpp and cuda.cu in turn
    env = {**os.environ, 'MAX_JOBS': '1'}
    code = (
        'import sys; from sevenn.nn.lunatp_helper import _precompile_worker; '
        '_precompile_worker(*sys.argv[1:])'
    )
    with tempfile.TemporaryDirectory() as tmp:
        procs = []
        for rank in range(workers):
            log = open(os.path.join(tmp, f'{rank}.log'), 'w')
            cmd = [sys.executable, '-c', code, checkpoint_path, str(rank)]
            cmd += [str(workers), str(order)]
            procs.append(
                subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            )
            log.close()
        failed = [rank for rank, proc in enumerate(procs) if proc.wait() != 0]
        if failed:
            with open(os.path.join(tmp, f'{failed[0]}.log')) as f:
                tail = f.read()[-2000:]
            warnings.warn(
                f'LunaTP precompile workers {failed} failed; their kernels '
                f'build at the first call.\n{tail}'
            )
