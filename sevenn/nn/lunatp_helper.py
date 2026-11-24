import fcntl
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import warnings
from typing import Optional, Union

import torch
from e3nn.o3 import FullyConnectedTensorProduct

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .convolution import IrrepsConvolution, IrrepsScatterGatterFusedConvolution
from .equivariant_gate import EquivariantGate
from .linear import IrrepsLinear
from .self_connection import SelfConnectionIntro, SelfConnectionLinearIntro

_WORKERS_ENV = 'SEVENNET_LUNATP_PRECOMPILE_WORKERS'

try:
    from LunaEquivariance import (
        LunaEqGate,
        LunaEqLinear,
        TensorProduct,
        TensorProductConv,
        make_graph,
    )
    from LunaEquivariance.utils import jit as _lunaeq_jit
    from torch.utils import cpp_extension as _cpp_extension

    class _DigestVersionedCppExtension:
        """
        torch.utils.cpp_extension for LunaEquivariance.utils.jit, with a kernel
        library version equal to its build digest. Torch's version counts the
        same-named libraries loaded before, so a process that loads kernels in
        another order would compile them again.
        """

        class JIT_EXTENSION_VERSIONER:
            @staticmethod
            def bump_version_if_changed(name, build_arguments, **_):
                return int(build_arguments[0][0], 16)

        def __getattr__(self, attr):
            return getattr(_cpp_extension, attr)

    if 'JIT_EXTENSION_VERSIONER' in _lunaeq_jit.load_extension.__code__.co_names:
        _lunaeq_jit.cpp_extension = _DigestVersionedCppExtension()

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

    @staticmethod
    def get_graph(key, edge_src, edge_dst, num_nodes: int):
        """
        (edge order or None, LunaTPGraph), cached on the key tensor. make_graph
        rejects edges not sorted by destination, which are then sorted here.
        """
        cached = getattr(key, '_lunatp_graph', None)
        if cached is not None and cached[0] == num_nodes:
            return cached[1], cached[2]
        order = None
        try:
            graph = make_graph(edge_src, edge_dst, num_nodes)
        except ValueError:
            if bool(torch.all(edge_dst[1:] >= edge_dst[:-1])):
                raise
            order = torch.argsort(edge_dst, stable=True)
            graph = make_graph(edge_src[order], edge_dst[order], num_nodes)
        key._lunatp_graph = (num_nodes, order, graph)
        return order, graph

    def forward(
        self,
        x,
        edge_filter,
        weight,
        edge_src,
        edge_dst,
        graph_key: Optional[torch.Tensor] = None,
    ):
        key = edge_filter if graph_key is None else graph_key
        order, graph = self.get_graph(key, edge_src, edge_dst, x.shape[0])
        if order is not None:
            edge_filter, weight = edge_filter[order], weight[order]
        return self.tp_conv(x, edge_filter, weight, graph=graph)


class LunaTPSharedGraphConvolution(IrrepsScatterGatterFusedConvolution):
    """
    Passes the input edge index tensor as the LunaTP graph cache key: the layers
    of a forward share one graph, and a caller that passes the same edge index
    tensor again reuses it.
    """

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        assert self.convolution is not None, 'Convolution is not instantiated'
        assert self.weight_nn is not None, 'Weight_nn is not instantiated'

        x = data[self.key_x]
        weight = self.weight_nn(data[self.key_weight_input])
        if self.is_parallel:
            x = torch.cat([x, data[KEY.NODE_FEATURE_GHOST]])

        edge_src = data[self.key_edge_idx][1]
        edge_dst = data[self.key_edge_idx][0]
        edge_filter = data[self.key_filter]

        if edge_src.numel() == 0:
            x = x.new_zeros(x.shape[0], self._out_dim)
            x = x + (edge_filter.sum() + weight.sum()) * 0
        else:
            x = self.convolution(
                x,
                edge_filter,
                weight,
                edge_src,
                edge_dst,
                graph_key=data[self.key_edge_idx],
            )

        x = x.div(self.denominator)

        if self.is_parallel:
            x = torch.tensor_split(x, data[KEY.NLOCAL])[0]
        data[self.key_x] = x
        return data


@lunatp_needed
def patch_convolution(irreps_convolution: IrrepsConvolution):
    assert not irreps_convolution.layer_instantiated, (
        'Convolution layer already instantiated; cannot patch'
    )
    ret = LunaTPSharedGraphConvolution.from_irreps_convolution(irreps_convolution)
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
    """
    Build the module's kernels: one forward and backward per combination of
    tensors that require grad, the no-grad combination first (later no-grad
    calls would reuse the fused forward+backward kernel and leave the
    forward-only one unbuilt). order 2 differentiates once more and runs
    linears on 65 rows (kernel selected by n <= 64).
    """
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
    else:
        dims = [module.irreps_in.dim]
        extra = []
    sizes = (2, 65) if order >= 2 and isinstance(module, LunaEqLinear) else (2,)
    params = [p for p in module.parameters() if p.numel() > 0]
    trainable = [p.requires_grad for p in params]
    flag_sets = itertools.product((False, True), repeat=len(dims) + len(params[:1]))
    try:
        for flags, n in itertools.product(flag_sets, sizes):
            for p in params:
                p.requires_grad_(flags[-1])
            inputs = [
                torch.randn(n, d, device=device, dtype=dtype, requires_grad=f)
                for d, f in zip(dims, flags)
            ]
            wrt = [t for t in inputs + params if t.requires_grad]
            out = module(*inputs, *extra)
            if not wrt or not out.requires_grad:
                continue
            torch.autograd.grad(out.sum(), wrt, allow_unused=True)
            if order < 2:
                continue
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


def _precompile_worker(config_path: str, rank: str, size: str, order: str):
    """
    A convolution compiles its forward kernel when it is built, so each worker
    first builds its share of them instead of waiting for them in the model
    build.
    """
    from sevenn.model_build import build_E3_equivariant_model

    config = torch.load(config_path, weights_only=False)
    e3nn_model = build_E3_equivariant_model({**config, KEY.USE_LUNATP: False})
    convolutions = [
        m for m in e3nn_model.modules() if isinstance(m, IrrepsConvolution)
    ]
    for conv in convolutions[int(rank) :: int(size)]:
        LunaTPConvolution(**conv.convolution_kwargs)
    model = build_E3_equivariant_model(dict(config))
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


def _precompile_marker(config: dict, order: int) -> str:
    from importlib.metadata import version

    from torch.utils import cpp_extension

    root = os.environ.get('LUNAEQ_CACHE_DIR') or os.path.join(
        os.environ.get('TORCH_EXTENSIONS_DIR')
        or cpp_extension.get_default_build_root(),
        'lunaeq',
    )
    key = [
        config,
        order,
        torch.__version__,
        version('LunaEquivariance'),
        torch.cuda.get_device_capability(),
    ]
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, default=str).encode()
    ).hexdigest()[:24]
    return os.path.join(os.path.expanduser(root), 'sevenn_precompiled', digest)


def precompile(
    model: Union[str, dict],
    order: int = 1,
    max_workers: Optional[int] = None,
    skip_done: bool = False,
) -> None:
    """
    Build the LunaEquivariance kernels of a model (checkpoint path, pretrained
    name or model config) in max_workers processes (default min(8, available
    CPUs)) that each run a share of the LunaTP modules once.
    order=2 also builds the kernels of second derivatives (force training).
    skip_done returns at once if an earlier call finished for the same model,
    order and environment.
    """
    if not is_lunatp_available():
        return
    if isinstance(model, dict):
        config = dict(model)
    else:
        from sevenn.util import load_checkpoint

        config = dict(load_checkpoint(model).config)
    config.update({
        KEY.USE_LUNATP: True,
        KEY.USE_OEQ: False,
        KEY.USE_FLASH_TP: False,
        KEY.CUEQUIVARIANCE_CONFIG: {'use': False},
    })
    marker = _precompile_marker(config, order)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    workers = max_workers or min(8, len(os.sched_getaffinity(0)))
    env = {**os.environ, 'MAX_JOBS': '1', _WORKERS_ENV: '0'}
    code = (
        'import sys; from sevenn.nn.lunatp_helper import _precompile_worker; '
        '_precompile_worker(*sys.argv[1:])'
    )
    with open(marker + '.lock', 'a') as lock, tempfile.TemporaryDirectory() as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if skip_done and os.path.exists(marker):
            return
        if skip_done:
            print(
                f'LunaTP: building kernels in {workers} processes '
                f'(once per model; {_WORKERS_ENV}=0 disables)',
                file=sys.stderr,
            )
        config_path = os.path.join(tmp, 'config.pt')
        torch.save(config, config_path)

        def run(ranks):
            procs = {}
            for rank in ranks:
                log = open(os.path.join(tmp, f'{rank}.log'), 'w')
                cmd = [sys.executable, '-c', code, config_path, str(rank)]
                cmd += [str(workers), str(order)]
                procs[rank] = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, env=env
                )
                log.close()
            return [rank for rank, proc in procs.items() if proc.wait() != 0]

        failed = run(range(workers))
        if failed:  # a shared GPU can be out of memory for a moment
            failed = run(failed)
        if failed:
            with open(os.path.join(tmp, f'{failed[0]}.log')) as f:
                tail = f.read()[-2000:]
            warnings.warn(
                f'LunaTP precompile workers {failed} failed; their kernels '
                f'build at the first call.\n{tail}'
            )
        else:
            open(marker, 'w').close()


def auto_precompile(config: dict) -> None:
    """
    precompile for a model about to be built, skipped if already done.
    SEVENNET_LUNATP_PRECOMPILE_WORKERS sets the number of processes (0: off),
    SEVENNET_LUNATP_PRECOMPILE_ORDER=2 adds the kernels for training.
    """
    workers = int(os.environ.get(_WORKERS_ENV, -1))
    if workers == 0:
        return
    order = int(os.environ.get('SEVENNET_LUNATP_PRECOMPILE_ORDER', 1))
    precompile(config, order, workers if workers > 0 else None, skip_done=True)
