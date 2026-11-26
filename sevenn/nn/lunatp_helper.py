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
from pathlib import Path
from types import SimpleNamespace
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

        @staticmethod
        def _write_ninja_file_and_build_library(**kwargs):
            build = _cpp_extension._write_ninja_file_and_build_library
            sources = kwargs['sources']
            if len(sources) == 2:
                # The CUDA source already defines the bindings.
                cpp = Path(sources[0])
                if cpp.name == 'main.cpp' and cpp.read_text().strip() in (
                    '', '#include <torch/extension.h>',
                ):
                    kwargs['sources'] = sources[1:]
            cuda = Path(sources[-1])
            if cuda.suffix != '.cu':
                return build(**kwargs)
            source = cuda.read_text()
            lean_source = source.replace(
                '#include <torch/extension.h>',
                '#include <torch/types.h>\n#include <torch/csrc/utils/pybind.h>',
            )
            if lean_source == source:
                return build(**kwargs)
            cuda.write_text(lean_source)
            try:
                return build(**kwargs)
            except RuntimeError:
                cuda.write_text(source)
                return build(**kwargs)

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
    train_params = order >= 2 and bool(params)
    flag_sets = itertools.product((False, True), repeat=len(dims) + train_params)
    try:
        for flags, n in itertools.product(flag_sets, sizes):
            for p in params:
                p.requires_grad_(train_params and flags[-1])
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


def _claim_jobs(root, rank, count):
    pending = root / f'{rank}.job'
    if pending.is_file():
        yield int(pending.read_text())
        pending.unlink()
    with (root / 'next').open('a+') as queue:
        while True:
            fcntl.flock(queue, fcntl.LOCK_EX)
            queue.seek(0)
            index = int(queue.read() or '0')
            if index >= count:
                return
            pending.write_text(str(index))
            queue.seek(0)
            queue.truncate()
            queue.write(str(index + 1))
            queue.flush()
            fcntl.flock(queue, fcntl.LOCK_UN)
            yield index
            pending.unlink()


def _precompile_worker(config_path: str, rank: str, order: str):
    from sevenn.model_build import _model_layers, patch_lunatp, patch_modality
    from sevenn.nn.inference_specialization import (
        _folded_modal_linear,
        _self_connection_for_species,
    )
    from sevenn.nn.sequential import _instantiate_modules

    config, device, dtype, base_kernels, widths = torch.load(
        config_path, weights_only=False,
    )
    torch.cuda.set_device(device)
    torch.set_default_dtype(dtype)
    order = int(order)
    layers, _ = _model_layers(config, False)
    layers = patch_modality(layers, config)
    luna_types = (
        LunaTPConvolution,
        LunaTPFullyConnectedTensorProduct,
        LunaEqLinear,
        LunaEqGate,
    )
    jobs = [
        ('base', (name, None))
        for name, layer in layers.items()
        if base_kernels and isinstance(layer, (
            IrrepsConvolution, IrrepsLinear, SelfConnectionIntro,
            SelfConnectionLinearIntro, EquivariantGate,
        ))
    ]
    if order == 1 and base_kernels:
        jobs.append(('mliap', None))
    if order == 1 and dtype == torch.float32:
        jobs = _fused_jobs(layers, widths, base_kernels) + jobs
        # Queue the backward and multi-kernel jobs before standalone forwards.
        jobs.sort(key=lambda job: (
            job[0] in ('geometry', 'first', 'radial')
            or (job[0] == 'fused' and job[1][1] == 'fwd')
        ))
    root = Path(config_path).parent
    manifest = root / f'{rank}.json'
    libraries = (
        set(json.loads(manifest.read_text())) if manifest.is_file() else set()
    )
    for index in _claim_jobs(root, rank, len(jobs)):
        kind, spec = jobs[index]
        if kind not in ('base', 'modal', 'species'):
            _precompile_fused(layers, kind, spec)
        else:
            name, width = spec
            current = patch_lunatp({name: layers[name]}, config)
            if kind == 'modal':
                target = _folded_modal_linear(current[name])
            elif kind == 'species':
                target = _self_connection_for_species(current[name], width)
            else:
                _instantiate_modules(current)
                target = current[name]
            for module in target.to(device).modules():
                if isinstance(module, luna_types):
                    _run_once(module, device, dtype, order)
        libraries.update(
            mod.__file__ for mod in _lunaeq_jit.MODULES.values()
            if getattr(mod, '__file__', None)
        )
        pending = manifest.with_suffix('.tmp')
        pending.write_text(json.dumps(sorted(libraries)))
        pending.replace(manifest)


def _species_jobs(layers, widths):
    jobs = []
    seen = set()
    for name, layer in layers.items():
        if not isinstance(layer, SelfConnectionIntro):
            continue
        key = (
            layer.irreps_in1, layer.irreps_in2, layer.irreps_out,
            repr(layer.fc_tensor_product_kwargs),
        )
        if key in seen:
            continue
        seen.add(key)
        jobs.extend(
            ('species', (name, width))
            for width in widths if width < layer.irreps_in2.dim
        )
    return jobs


def _fused_jobs(layers, widths, base_kernels):
    emb = layers['edge_embedding']
    if getattr(emb, 'distance_transform', None) is not None:
        return []
    convolutions = [
        (name, conv) for name, conv in layers.items()
        if isinstance(conv, IrrepsConvolution)
        and not conv.is_parallel
        and conv.key_weight_input == KEY.EDGE_EMBEDDING
        and conv.key_filter == KEY.EDGE_ATTR
        and len(conv.weight_nn_kwargs['hs']) == 4
    ]
    if not base_kernels:
        return _species_jobs(layers, widths) if convolutions else []
    if not convolutions:
        return [('radial', None)]

    jobs = []
    seen = set()
    for name, conv in convolutions:
        key = repr(conv.convolution_kwargs)
        if key in seen:
            continue
        seen.add(key)
        jobs.extend(
            ('fused', (name, stage, hermite, wide))
            for stage in ('bwd', 'fwd')
            for hermite in (False, True)
            for wide in (False, True)
        )
    jobs.extend(_species_jobs(layers, widths))
    jobs.extend((('geometry', None), ('first', None)))
    jobs.extend(
        ('modal', (name, None)) for name, layer in layers.items()
        if type(layer) is IrrepsLinear and layer.num_modalities > 1
    )
    jobs.append(('radial', None))
    return jobs


def _precompile_fused(layers, kind, spec):
    from .lunatp_fused import (
        _FIRST_SOURCE,
        FusedEdgeGeometry,
        _load_cuda,
        _load_fused_conv,
    )

    if kind == 'geometry':
        FusedEdgeGeometry(layers['edge_embedding'])
    elif kind == 'first':
        _load_cuda('sevenn_first_layer_fp32', _FIRST_SOURCE)
    elif kind == 'mliap':
        from ._mliap_cuda_graph import _load_prep_kernel

        _load_prep_kernel()
    elif kind == 'radial':
        from importlib.util import find_spec

        if find_spec('cupy') is not None:
            from .radial_table import _kernels

            _, fwd, bwd = _kernels()
            fwd.compile()
            bwd.compile()
    else:
        from LunaEquivariance import TensorProductConfig
        from LunaEquivariance.lunatp.utils.layout import TensorProductPaths
        from LunaEquivariance.lunatp.uvu_conv.kernels.utils import (
            build_uvu_conv_metadata,
        )

        name, stage, hermite, wide = spec
        tp = TensorProductConfig(**layers[name].convolution_kwargs)
        paths = TensorProductPaths.from_tensor_product(tp)
        meta = build_uvu_conv_metadata(paths, torch.float32)
        op = SimpleNamespace(
            **meta, compute_dtype=torch.float32,
            forward_degree_tiers=meta['max_channels'] > 128,
        )
        _load_fused_conv(op, stage, hermite, wide=wide)


def _precompile_marker(config: dict, order: int) -> str:
    from importlib.metadata import PackageNotFoundError, version

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
        4 if order == 1 else 3,
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
    if order == 1:
        cupy_versions = []
        for package in ('cupy', 'cupy-cuda12x', 'cupy-cuda13x'):
            try:
                cupy_versions.append(version(package))
            except PackageNotFoundError:
                pass
        key.extend((os.environ.get('CUPY_CACHE_DIR'), cupy_versions))
        for name in (
            'lunatp_fused.py', 'radial_table.py', 'inference_specialization.py',
            '_mliap_cuda_graph.py',
        ):
            source = Path(__file__).with_name(name).read_bytes()
            key.append(hashlib.sha256(source).hexdigest())
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, default=str).encode()
    ).hexdigest()[:24]
    return os.path.join(os.path.expanduser(root), 'sevenn_precompiled', digest)


def _read_precompiled(marker):
    try:
        manifest = json.loads(Path(marker).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or not {'libraries', 'species_widths'} <= set(
        manifest
    ):
        return None
    if all(Path(library).is_file() for library in manifest['libraries']):
        return manifest
    return None


def _lunatp_config(config):
    return {
        **config,
        KEY.USE_LUNATP: True,
        KEY.USE_OEQ: False,
        KEY.USE_FLASH_TP: False,
        KEY.CUEQUIVARIANCE_CONFIG: {'use': False},
    }


def precompile(
    model: Union[str, dict],
    order: int = 1,
    max_workers: Optional[int] = None,
    skip_done: bool = True,
    *,
    species_widths=(),
    all_species_widths: bool = False,
) -> bool:
    """
    Build kernels for a checkpoint, pretrained model name, or model config.
    order=1 covers plain, fused (32- and 64-bit offsets) and radially
    tabulated inference. order=2 uses the checkpoint layout and builds
    second derivatives.
    skip_done returns at once if an earlier call finished for the same model,
    order and environment.
    """
    if not is_lunatp_available():
        return False
    if isinstance(model, dict):
        config = dict(model)
    else:
        from sevenn.util import load_checkpoint

        checkpoint = load_checkpoint(model)
        if order == 1:
            checkpoint = checkpoint.compact_for_inference()
        config = checkpoint.config
    config = _lunatp_config(config)
    requested = set()
    if order == 1:
        full_width = config[KEY.NUM_SPECIES]
        if any(width < 1 or width > full_width for width in species_widths):
            raise ValueError(f'species widths must be in [1, {full_width}]')
        widths = range(1, full_width) if all_species_widths else species_widths
        requested.update(width for width in widths if width < full_width)
    marker = _precompile_marker(config, order)
    manifest = _read_precompiled(marker) if skip_done else None
    if manifest and requested.issubset(manifest['species_widths']):
        return True
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    cpus = len(os.sched_getaffinity(0))
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        cpus = min(cpus, int(os.environ['SLURM_CPUS_PER_TASK']))
    workers = max_workers if max_workers is not None and max_workers > 0 else (
        min(16, cpus)
    )
    env = {**os.environ, 'MAX_JOBS': '1', _WORKERS_ENV: '0'}
    code = (
        'import sys; from sevenn.nn.lunatp_helper import _precompile_worker; '
        '_precompile_worker(*sys.argv[1:])'
    )
    with open(marker + '.lock', 'a') as lock, tempfile.TemporaryDirectory() as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _read_precompiled(marker) if skip_done else None
        cached = set(manifest['species_widths']) if manifest else set()
        if manifest and requested.issubset(cached):
            return True
        missing = sorted(requested - cached)
        if manifest:
            from sevenn.model_build import _model_layers

            layers, _ = _model_layers(config, False)
            workers = min(workers, len(_fused_jobs(layers, missing, False)))
            widths = ', '.join(map(str, missing))
            if workers:
                print(
                    'LunaTP fusion: building self-connection kernels for species '
                    f'width {widths} in {workers} parallel processes. '
                    'This may take a minute.',
                    file=sys.stderr,
                )
        elif skip_done:
            print(
                f'LunaTP: building kernels in {workers} processes '
                f'(once per model; {_WORKERS_ENV}=0 disables)',
                file=sys.stderr,
            )
        config_path = os.path.join(tmp, 'config.pt')
        worker_config = (
            config, torch.cuda.current_device(), torch.get_default_dtype(),
            manifest is None, tuple(missing),
        )
        torch.save(worker_config, config_path)

        def run(ranks):
            procs = {}
            for rank in ranks:
                log = open(os.path.join(tmp, f'{rank}.log'), 'w')
                cmd = [sys.executable, '-c', code, config_path, str(rank)]
                cmd.append(str(order))
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
            return False
        libraries = {
            library
            for worker_manifest in Path(tmp).glob('*.json')
            for library in json.loads(worker_manifest.read_text())
        }
        pending = Path(marker + '.tmp')
        if manifest:
            libraries.update(manifest['libraries'])
        pending.write_text(json.dumps({
            'libraries': sorted(libraries),
            'species_widths': sorted(cached | requested),
        }))
        pending.replace(marker)
        return True


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


def precompile_species_width(config: dict, width: int) -> None:
    workers = int(os.environ.get(_WORKERS_ENV, -1))
    if workers:
        precompile(
            config, max_workers=workers if workers > 0 else None,
            species_widths=(width,),
        )
