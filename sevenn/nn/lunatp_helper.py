import fcntl
import hashlib
import itertools
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import threading
import warnings
from typing import Optional, Union

import torch
from e3nn.o3 import FullyConnectedTensorProduct

import sevenn._keys as KEY
from sevenn._const import (
    DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG,
    MODEL_CONFIG_CONDITION,
    AtomGraphDataType,
)

from .convolution import IrrepsConvolution, IrrepsScatterGatterFusedConvolution
from .equivariant_gate import EquivariantGate
from .linear import IrrepsLinear
from .self_connection import SelfConnectionIntro, SelfConnectionLinearIntro

_WORKERS_ENV = 'SEVENNET_LUNATP_PRECOMPILE_WORKERS'
PREPARED_GRAPH_KEY = '_lunatp_prepared_graph'
_SHARED_GRAPH_KEY = '_lunatp_graph'
_GRAPH_BUILD_LOCK = threading.Lock()

try:
    from LunaEquivariance import (
        LunaEqGate,
        LunaEqLinear,
        TensorProduct,
        TensorProductConv,
        make_graph,
    )
    from LunaEquivariance.utils import jit as _lunaeq_jit
    from LunaEquivariance.utils.extension import (
        find_cutlass_include as _find_cutlass_include,
    )
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


def _make_graph(edge_src, edge_dst, num_nodes, segment_edges=0):
    with _GRAPH_BUILD_LOCK:
        if not segment_edges:
            return make_graph(edge_src, edge_dst, num_nodes)
        import LunaEquivariance.lunatp.graph as luna_graph

        # LunaEquivariance 0.3.0 reads the segment size from the module.
        default = luna_graph._SOURCE_SEGMENT_EDGES
        luna_graph._SOURCE_SEGMENT_EDGES = segment_edges
        try:
            return make_graph(edge_src, edge_dst, num_nodes)
        finally:
            luna_graph._SOURCE_SEGMENT_EDGES = default


def prepare_graph(edge_index: torch.Tensor, num_nodes: int, segment_edges=0):
    """Return the edge permutation and LunaTP graph for this topology."""
    edge_dst, edge_src = edge_index
    try:
        return None, _make_graph(edge_src, edge_dst, num_nodes, segment_edges)
    except ValueError:
        if bool(torch.all(edge_dst[1:] >= edge_dst[:-1])):
            raise
        order = torch.argsort(edge_dst, stable=True)
        return order, _make_graph(
            edge_src[order], edge_dst[order], num_nodes, segment_edges
        )


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

    def forward(self, x, edge_filter, weight, topology):
        order, graph = topology
        if order is not None:
            edge_filter, weight = edge_filter[order], weight[order]
        return self.tp_conv(x, edge_filter, weight, graph=graph)


class LunaTPSharedGraphConvolution(IrrepsScatterGatterFusedConvolution):
    starts_graph = True
    ends_graph = False

    def _begin_graph(self, data: AtomGraphDataType) -> None:
        if self.starts_graph:
            if _SHARED_GRAPH_KEY in data:
                del data[_SHARED_GRAPH_KEY]
            # The supplied graph must match the edge order, node count and device.
            if PREPARED_GRAPH_KEY in data:
                topology = data[PREPARED_GRAPH_KEY]
                del data[PREPARED_GRAPH_KEY]
                if data[self.key_edge_idx].numel():
                    data[_SHARED_GRAPH_KEY] = topology

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        self._begin_graph(data)
        return super().forward(data)

    def _prepare_graph(self, edge_index, num_nodes):
        return prepare_graph(edge_index, num_nodes)

    def _take_graph(self, data, num_nodes):
        if self.starts_graph and _SHARED_GRAPH_KEY not in data:
            data[_SHARED_GRAPH_KEY] = self._prepare_graph(
                data[self.key_edge_idx], num_nodes
            )
        topology = data[_SHARED_GRAPH_KEY]
        if self.ends_graph:
            del data[_SHARED_GRAPH_KEY]
        return topology

    def _convolve(self, data, x, edge_filter, weight):
        topology = self._take_graph(data, x.shape[0])
        return self.convolution(x, edge_filter, weight, topology)


@lunatp_needed
def patch_convolution(
    irreps_convolution: IrrepsConvolution, starts_graph: bool = True
):
    assert not irreps_convolution.layer_instantiated, (
        'Convolution layer already instantiated; cannot patch'
    )
    ret = LunaTPSharedGraphConvolution.from_irreps_convolution(irreps_convolution)
    ret.convolution_cls = LunaTPConvolution
    ret.starts_graph = starts_graph
    ret.ends_graph = True
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
    Run no-grad first to build the forward-only kernel before the fused one.
    For linear double backward, 65 rows exercise the kernel for n > 64.
    """
    if isinstance(module, LunaTPConvolution):
        tp = module.tp_conv
        dims = [tp.irreps_in1.dim, tp.irreps_in2.dim, tp.weight_numel]
        edges = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.int32)
        extra = [prepare_graph(edges, 2)]
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

    config, device, dtype = torch.load(config_path, weights_only=False)
    torch.cuda.set_device(device)
    torch.set_default_dtype(dtype)
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

    root = os.environ.get('LUNAEQ_CACHE_DIR') or os.path.join(
        os.environ.get('TORCH_EXTENSIONS_DIR')
        or _cpp_extension.get_default_build_root(),
        'lunaeq',
    )
    defaults = DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG
    model_keys = defaults.keys() | MODEL_CONFIG_CONDITION.keys() | {
        KEY.MODEL_TYPE, KEY.DTYPE, KEY.NUM_SPECIES,
        KEY.USE_MODALITY, KEY.NUM_MODALITIES,
    }
    model = {
        key: config.get(key, defaults.get(key))
        for key in model_keys - {KEY.CONV_DENOMINATOR}
    }
    if not model[KEY.DROP_UNUSED_TP_IRREPS]:
        del model[KEY.DROP_UNUSED_TP_IRREPS]

    cuda_home = _cpp_extension.CUDA_HOME
    nvcc = os.path.join(cuda_home, 'bin', 'nvcc') if cuda_home else 'nvcc'
    compilers = (
        _cpp_extension.get_cxx_compiler(),
        os.environ.get('PYTORCH_NVCC', nvcc),
        os.environ.get('CC'),
    )
    toolchain = []
    for compiler in filter(None, compilers):
        try:
            toolchain.append(
                _lunaeq_jit.compiler_identity(compiler, os.environ.get('PATH', ''))
            )
        except (RuntimeError, OSError, subprocess.SubprocessError):
            toolchain.append((compiler, None))

    cutlass = None
    try:
        include = _find_cutlass_include()
    except RuntimeError:
        pass
    else:
        header = include / 'cutlass' / 'version.h'
        cutlass = (str(include), header.read_text() if header.is_file() else None)

    # Keep the build environment in sync with LunaEquivariance 0.3.0's JIT key.
    build_env = (
        'CC', 'NVCC_CCBIN', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS',
        'CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH',
        'PYTORCH_NVCC', 'TORCH_CUDA_ARCH_LIST',
    )
    key = [
        model,
        order,
        torch.__version__,
        torch.version.cuda,
        torch.__file__,
        torch._C._GLIBCXX_USE_CXX11_ABI,
        torch.get_default_dtype(),
        sys.version,
        sys.prefix,
        sysconfig.get_config_var('SOABI'),
        version('e3nn'),
        version('LunaEquivariance'),
        torch.cuda.get_device_capability(),
        [
            torch.cuda.get_device_capability(i)
            for i in range(torch.cuda.device_count())
        ],
        cuda_home,
        toolchain,
        cutlass,
        {name: os.environ.get(name, '') for name in build_env},
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
    order=1 uses the inference layout. order=2 uses the checkpoint layout
    and builds second derivatives for force training.
    skip_done returns at once if an earlier call finished for the same model,
    order and environment.
    """
    if not is_lunatp_available():
        return
    if isinstance(model, dict):
        config = dict(model)
    else:
        from sevenn.util import load_checkpoint

        checkpoint = load_checkpoint(model)
        if order == 1:
            checkpoint = checkpoint.compact_for_inference()
        config = checkpoint.config
    config.update({
        KEY.USE_LUNATP: True,
        KEY.USE_OEQ: False,
        KEY.USE_FLASH_TP: False,
        KEY.CUEQUIVARIANCE_CONFIG: {'use': False},
    })
    marker = _precompile_marker(config, order)
    if skip_done and os.path.isfile(marker):
        return
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    workers = max_workers or min(8, len(os.sched_getaffinity(0)))
    env = {**os.environ, 'MAX_JOBS': '1', _WORKERS_ENV: '0'}
    code = (
        'import sys; from sevenn.nn.lunatp_helper import _precompile_worker; '
        '_precompile_worker(*sys.argv[1:])'
    )
    with open(marker + '.lock', 'a') as lock, tempfile.TemporaryDirectory() as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if skip_done and os.path.isfile(marker):
            return
        if skip_done:
            print(
                f'LunaTP: building kernels in {workers} processes '
                f'(once per model; {_WORKERS_ENV}=0 disables)',
                file=sys.stderr,
            )
        config_path = os.path.join(tmp, 'config.pt')
        worker_config = (
            config, torch.cuda.current_device(), torch.get_default_dtype(),
        )
        torch.save(worker_config, config_path)

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
