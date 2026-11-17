import itertools
import os
import subprocess
import sys
import tempfile
import warnings
from typing import Optional, Union

import torch
import torch.nn.functional as F
from e3nn.o3 import FullyConnectedTensorProduct

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .bilinear_gate import BilinearGate
from .convolution import IrrepsConvolution, IrrepsScatterGatterFusedConvolution
from .edge_attention import EdgeAttention
from .equivariant_gate import EquivariantGate
from .linear import IrrepsLinear
from .normalization import EquivariantRMSNorm
from .self_connection import SelfConnectionIntro, SelfConnectionLinearIntro

try:
    from LunaEquivariance import (
        LunaEqGate,
        LunaEqLinear,
        TensorProduct,
        TensorProductConv,
        make_graph,
    )
    from LunaEquivariance.nn.equ_norm import EquNormOperator
    from LunaEquivariance.utils import jit as _lunaeq_jit
    from torch.utils import cpp_extension as _cpp_extension

    class _DigestVersionedCppExtension:
        """
        torch.utils.cpp_extension as LunaEquivariance.utils.jit sees it, with a
        JIT_EXTENSION_VERSIONER that numbers a kernel library by its build
        digest. Every other attribute is torch.utils.cpp_extension's.

        LunaEquivariance 0.2.0 ("Release LunaEquivariance 0.2.0", unchanged by
        "Fix cuda13.x compatibility (#3)") stores a kernel library as
        <LUNAEQ_CACHE_DIR>/<digest>/<name>_v<version>.so, with the version of
        torch's process-wide JIT_EXTENSION_VERSIONER: the number of libraries
        of the same base name that this process loaded before (no suffix for
        version 0). A process that loads the kernels in another order looks
        for another file name and compiles an identical kernel again
        (sevenn lunatp_precompile workers each load a share of the modules).
        Here the version is the digest as an integer: one name per digest in
        every process, and kernels of one base name with different digests
        keep different module names in one process (the pybind11 module name
        is the library name). Only LunaEquivariance's jit module sees this
        object; torch's versioner and its other callers are unchanged.
        LunaEquivariance 0.1.0 names a library <name>_<digest> without the
        versioner and is left as is.
        """

        class JIT_EXTENSION_VERSIONER:
            @staticmethod
            def bump_version_if_changed(name, build_arguments, **_):
                return int(build_arguments[0][0], 16)  # ((digest,),)

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

    class LunaTPTensorProduct(TensorProduct):
        """
        An existing e3nn TensorProduct computed by LunaTP (BilinearGate: 'uuu',
        no or per-node weights). As in LunaTPFullyConnectedTensorProduct, the
        e3nn module is kept outside the module tree and its parameters, buffers and
        submodules (w3j constants) are registered here under the e3nn names,
        so state dicts match the e3nn backend.
        Kernels: LunaEquivariance 0.1.0 (HPC-AI-Team, BSD-3-Clause,
        github.com/HPC-AI-Team/LunaEquivariance), fused 'uuu' tensor product
        (LunaEquivariance/lunatp/uuu) with CUDA derivatives through third order.
        """

        def __init__(self, tp):
            super().__init__(tp)
            del self._modules['tp']
            object.__setattr__(self, 'tp', tp)
            for name, value in tp.named_parameters(recurse=False):
                self._parameters[name] = value
            for name, value in tp.named_buffers(recurse=False):
                self._buffers[name] = value
            for name, child in tp.named_children():
                self._modules[name] = child

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
        (edge order or None, LunaTPGraph) of the dst-sorted edges, cached as
        an attribute of the key tensor: built once per key tensor, freed with
        it. The key is the forward's input edge index (graph_key, see
        LunaTPSharedGraphConvolution), so all layers of a forward share one
        graph, and a caller that keeps the same edge index tensor across calls
        (the calculator: CUDA graph, and eager calls of an unchanged topology)
        builds it once. Without graph_key the key is the edge filter.
        The key tensor must not be modified in place while it is a key.
        LunaTP requires edges sorted by destination. make_graph checks the order
        in its kernel and reads the result on the host anyway, so dst-sorted
        edges (neighbor lists, the calculator) are not checked a second time
        (one host sync less); unsorted edges raise there and are sorted here.
        """
        cached = getattr(key, '_lunatp_graph', None)
        if cached is not None and cached[0] == num_nodes:
            return cached[1], cached[2]
        order = None
        try:
            graph = make_graph(edge_src, edge_dst, num_nodes)
        except ValueError:
            if bool(torch.all(edge_dst[1:] >= edge_dst[:-1])):
                raise  # not an order error
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
    IrrepsScatterGatterFusedConvolution with LunaTPConvolution that passes the
    forward's input edge index tensor (data[key_edge_idx]) as the graph cache
    key, so all layers of a forward share one LunaTP graph, also layers whose
    edge filter is not KEY.EDGE_ATTR (edge attention: the attention-weighted
    spherical harmonics, a new tensor per layer). The edge index is an input
    of the forward, so a caller that passes the same tensor again (CUDA graph
    capture after a warm-up) finds the graph without a host sync.
    Same math, bitwise identical outputs.
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

        if edge_src.numel() == 0:  # as IrrepsScatterGatterFusedConvolution
            x = x.new_zeros(x.shape[0], self._out_dim)
            x = x + (edge_filter.sum() + weight.sum()) * 0
        else:
            # int64 indices: read only by make_graph (int32 tables), once per key
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


class LunaTPEdgeAttention(EdgeAttention):
    """
    EdgeAttention on the LunaTP backend: the attention envelope s5(r)^2
    depends on the edge lengths only, so it is computed by the first attention
    layer of a forward and reused by the others (stored in the data dict next
    to the KEY.EDGE_LENGTH tensor it was computed from; recomputed whenever
    that tensor is a different one, e.g. a new forward on the same data).
    Same forward values (bitwise); the gradients of the envelope of all layers
    are summed before its backward instead of after (round-off only).
    Everything else is EdgeAttention.forward unchanged.
    LunaTP models are not TorchScript-compiled (LunaTP is ML-IAP / Python
    only), the TorchScript EdgeAttention of the other backends is untouched.
    """

    _KEY_ENV2 = 'attn_env2'
    _KEY_ENV2_R = 'attn_env2_r'  # the KEY.EDGE_LENGTH tensor env2 belongs to

    def _envelope2(self, data: AtomGraphDataType) -> torch.Tensor:
        r = data[KEY.EDGE_LENGTH]
        if self._KEY_ENV2_R in data and data[self._KEY_ENV2_R] is r:
            return data[self._KEY_ENV2]
        x = r * self.inv_cutoff
        poly = 1.0 + x * (4.0 + x * (10.0 + x * (20.0 + 35.0 * x)))
        s5 = (1.0 - x).pow(4) * poly
        env2 = s5 * s5
        data[self._KEY_ENV2] = env2
        data[self._KEY_ENV2_R] = r
        return env2

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        # EdgeAttention.forward with the shared envelope (keep in sync)
        s = data[self.key_x][:, self.s0: self.s1]
        s_src = s
        if self.is_parallel:
            g = data[KEY.NODE_FEATURE_GHOST][:, self.s0: self.s1]
            s_src = torch.cat([s, g])
        q = self._rms(self.q(s), self.gq)
        k = self._rms(self.k(s_src), self.gk)
        dst = data[KEY.EDGE_IDX][0]
        src = data[KEY.EDGE_IDX][1]
        qk = q.index_select(0, dst) * k.index_select(0, src)
        logit = qk.sum(-1) * self.scale
        if self.use_rb:
            z = data[self.key_atom_type]
            logit = logit + self.rb(data[KEY.EDGE_RBF]).squeeze(-1)
            t_src = F.embedding(z, self.t_src).squeeze(-1)
            t_dst = F.embedding(z, self.t_dst).squeeze(-1)
            logit = logit + t_src.index_select(0, src) + t_dst.index_select(0, dst)
        logit = self.cap * torch.tanh(logit / self.cap)
        a = self._envelope2(data) * torch.exp(logit)
        den = torch.zeros(s.shape[0], dtype=a.dtype, device=a.device)
        den = den.index_add(0, dst, a)
        if self.use_null:
            den = den + F.softplus(self.zeta)
        else:
            den = den + self.eps
        w = a / den.index_select(0, dst)
        data[self.key_filter_out] = data[KEY.EDGE_ATTR] * w.unsqueeze(-1)
        if self.use_gate:
            data[self.key_gate] = 2.0 * torch.sigmoid(self.og(s))
        return data


@lunatp_needed
def patch_edge_attention(module: EdgeAttention) -> EdgeAttention:
    """LunaTPEdgeAttention in place (same parameters, buffers, state dict)"""
    module.__class__ = LunaTPEdgeAttention
    return module


@lunatp_needed
def patch_convolution(
    irreps_convolution: IrrepsConvolution, share_graph: bool = True
):
    """
    share_graph: every layer keys the LunaTP graph cache on the forward's
    edge index (LunaTPSharedGraphConvolution); False keeps the per-filter key
    (one graph per attention layer).
    """
    assert not irreps_convolution.layer_instantiated, (
        'Convolution layer already instantiated; cannot patch'
    )
    cls = IrrepsScatterGatterFusedConvolution
    if share_graph:
        cls = LunaTPSharedGraphConvolution
    ret = cls.from_irreps_convolution(irreps_convolution)
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


def _scalar_blocks(irreps):
    return [
        s for (_, ir), s in zip(irreps, irreps.slices()) if ir.l == 0 and ir.p == 1
    ]


class LunaTPEquivariantRMSNorm(EquivariantRMSNorm):
    """
    EquivariantRMSNorm computed by the fused LunaEquivariance equivariant norm
    (one kernel per forward, backward and double backward instead of ~100
    small PyTorch kernels over the three passes of force training). Same
    function, parameters and buffers (state dict) as EquivariantRMSNorm:
    LunaEq 'merged' grouping with 'component' weighting is the balanced RMS
    over all irrep blocks, its scalar centering and bias act on the 0e block.
    LunaEq centers each 0e block separately, EquivariantRMSNorm centers all
    0e channels together, so only irreps with at most one 0e block are patched
    (see patch_rms_norm).

    Sources: fused kernel = LunaEquivariance 0.1.0 EquNormOperator
    (github.com/HPC-AI-Team/LunaEquivariance, BSD-3-Clause,
    LunaEquivariance/nn/equ_norm/interface.py; docs operators/
    equivariant_norm.html), cite Y. Zhou et al., LunaTP: Enabling Efficient
    and Scalable Tensor Product for Equivariant MLIPs, arXiv:2610.18393
    (2026). Only called, no code copied. The normalization itself follows
    EquiformerV2 (arXiv:2306.12059) / eSEN (arXiv:2502.12147), see
    sevenn/nn/normalization.py.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        scalar_blocks = _scalar_blocks(self.irreps)
        if self.centering and len(scalar_blocks) != 1:
            raise ValueError('fused RMSNorm needs exactly one 0e block to center')
        self.scalar_start = scalar_blocks[0].start if scalar_blocks else 0
        self.equ_norm = EquNormOperator(
            self.irreps,
            grouping='merged',
            center_scalars=self.centering,
            weighting='component',
            layout='mul_ir',
            eps=self.eps,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        beta = None
        if self.bias is not None and self.centering:
            # bias holds one entry per feature, only those of the 0e block act
            beta = self.bias.narrow(0, self.scalar_start, self.num_scalars)
        data[self.key_x] = self.equ_norm(data[self.key_x], self.weight, beta)
        return data


@lunatp_needed
def patch_rms_norm(module: EquivariantRMSNorm):
    """
    Return the fused (LunaEq) version of an EquivariantRMSNorm, or the module
    itself if its irreps have more than one 0e block (centering differs).
    """
    if module.centering and len(_scalar_blocks(module.irreps)) != 1:
        return module
    fused = LunaTPEquivariantRMSNorm(
        module.irreps,
        module.key_x,
        eps=module.eps,
        centering=module.centering,
        affine=module.affine,
    )
    reference = module.balance  # floating buffer: device and dtype of the module
    fused.to(device=reference.device, dtype=reference.dtype)
    fused.load_state_dict(module.state_dict())
    return fused


@lunatp_needed
def patch_bilinear_gate(module: BilinearGate):
    """
    BilinearGate's three ops (e3nn Gate, 'uuu' TensorProduct on the full gate
    input/output buffers, Linear) replaced by LunaEqGate, LunaTP and
    LunaEqLinear; the forward (Gate -> TP -> Linear -> add) is unchanged and
    state dict names are those of e3nn.
    """
    module.gate = LunaEqGate(gate=module.gate)
    module.tp = LunaTPTensorProduct(module.tp)
    module.linear = LunaEqLinear(module.linear)
    return module


def _run_once(module, device: torch.device, dtype: torch.dtype, order: int):
    # Forward and backward on two nodes (and two edges) build the module's kernels.
    # LunaEquivariance compiles a variant per set of tensors that require grad, so
    # every combination of inputs and parameters (frozen or trainable) runs once.
    # The combination without grad runs first: after a differentiable call,
    # LunaEqLinear serves later no-grad calls from the fused forward+backward
    # library (o3/linear/kernels/utils.py, run_kernel; LunaEquivariance 0.1.0 and
    # a045a88), while a calculator whose first call of a frozen linear is no-grad
    # (the atom-type embedding) builds the forward-only library linear_fwd.
    # order 2 also differentiates first derivatives once more, as a force loss
    # does, with an output gradient that does and does not require grad.
    if isinstance(module, LunaTPEquivariantRMSNorm):
        # equ_norm kernels depend on the derivative order only (not on which
        # operands require grad): forward, backward and double backward
        x = torch.randn(2, module.irreps.dim, device=device, dtype=dtype)
        x.requires_grad_(True)
        out = module({module.key_x: x})[module.key_x]
        grad_out = torch.randn_like(out, requires_grad=True)
        (grad,) = torch.autograd.grad(out, x, grad_out, create_graph=order >= 2)
        if order >= 2:
            grad.square().sum().backward()
        module.zero_grad(set_to_none=True)
        return
    if isinstance(module, LunaTPConvolution):
        tp = module.tp_conv
        dims = [tp.irreps_in1.dim, tp.irreps_in2.dim, tp.weight_numel]
        src = torch.tensor([1, 0], device=device, dtype=torch.int32)
        extra = [src, src.flip(0)]
    elif isinstance(module, LunaTPTensorProduct):
        dims = [module.irreps_in1.dim, module.irreps_in2.dim]
        if module.weight_numel > 0 and not module.internal_weights:
            dims.append(module.weight_numel)  # per-node path weights
        extra = []
    elif isinstance(module, LunaTPFullyConnectedTensorProduct):
        dims = [module.irreps_in1.dim]
        onehot = torch.zeros(2, module.irreps_in2.dim, device=device, dtype=dtype)
        onehot[:, 0] = 1.0
        extra = [onehot]
    else:  # LunaEqLinear, LunaEqGate
        dims = [module.irreps_in.dim]
        extra = []
    # LunaEqLinear selects its second and third derivative kernels by the number
    # of rows (n <= 64 or not, run_kernel; 0.1.0: fp32 and fp64, a045a88: fp64
    # only), so order 2 also runs 65 rows (force training of a batch above 64
    # atoms). No other a045a88 library used here depends on the number of rows
    # or edges: the 64-bit index variant for large counts is in the same library
    # as the 32-bit one, and the uvw forward selected by 1024 rows is that of
    # per-row weights (the self-connection weights are shared).
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

    # LunaEquivariance a045a88 compiles the forward kernel of a convolution when
    # the convolution is built (lunatp/uvu_conv/kernels/utils.py,
    # configure_uvu_conv), so in build_model every worker waits for each of these
    # compiles in turn. Each worker first builds its share of them alone, from
    # the e3nn model (build_model changes the checkpoint's config: one each).
    convolutions = [
        m for m in load_checkpoint(checkpoint_path).build_model().modules()
        if isinstance(m, IrrepsConvolution)
    ]
    for conv in convolutions[int(rank) :: int(size)]:
        LunaTPConvolution(**conv.convolution_kwargs)
    model = load_checkpoint(checkpoint_path).build_model(enable_lunatp=True)
    model.to('cuda')
    param = next(model.parameters())
    luna_types = (
        LunaTPConvolution,
        LunaTPFullyConnectedTensorProduct,
        LunaTPTensorProduct,
        LunaEqLinear,
        LunaEqGate,
        LunaTPEquivariantRMSNorm,
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

        def run(ranks):
            procs = {}
            for rank in ranks:
                log = open(os.path.join(tmp, f'{rank}.log'), 'w')
                cmd = [sys.executable, '-c', code, checkpoint_path, str(rank)]
                cmd += [str(workers), str(order)]
                procs[rank] = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, env=env
                )
                log.close()
            return [rank for rank, proc in procs.items() if proc.wait() != 0]

        failed = run(range(workers))
        if failed:  # once more: a shared GPU can be out of memory for a moment
            failed = run(failed)
        if failed:
            with open(os.path.join(tmp, f'{failed[0]}.log')) as f:
                tail = f.read()[-2000:]
            warnings.warn(
                f'LunaTP precompile workers {failed} failed; their kernels '
                f'build at the first call.\n{tail}'
            )
