"""Fused modules on the Apple GPU: same weights, same results to float32 noise.

enable(model) switches supported modules to subclasses that run fused Metal kernels
(sevenn/mps/kernels.py). Convolutions run the kernels in eval and train mode, with
derivatives to any order, so force training works. Linear, self-connection and gate
layers run kernels in eval mode, and in train mode a few PyTorch operations built
from the same tables, which autograd differentiates to any order; a layout they
do not cover keeps the e3nn code. Nothing is registered in the state dict;
disable(model) undoes it, and enable() again starts from scratch. runtime.py calls
both when a model moves to or from the Apple GPU.

In eval mode the kernels treat the weights of linear and self-connection layers as
constants: derivatives with respect to the inputs (positions, strain) are complete,
parameter gradients are not. For those use train mode or move the model off the
Apple GPU. Changes through load_state_dict, an optimizer or in-place operations on a
parameter are noticed before the next call (parameter identity and version counter),
and module conversions (.to, .half, .float) drop the derived tables; writes through
param.data are not noticed, move the model to the CPU and back after those.

"""

import copy
import warnings
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from e3nn.nn import Gate
from e3nn.o3 import FullyConnectedTensorProduct, Linear, TensorProduct

from sevenn.nn.convolution import IrrepsConvolution
from sevenn.nn.node_embedding import OnehotEmbedding
from sevenn.nn.self_connection import SelfConnectionIntro

from . import _policy
from .kernels import (
    CsrApply,
    CsrMap,
    EdgePlan,
    FusedTP,
    FusedUvu,
    GateApply,
    GateTables,
)

ALL_PARTS = frozenset({'conv', 'linear', 'selfconn', 'gate'})
# train mode: up to this many atom types, multiply by every type's matrix and select
SELFCONN_ALL_TYPES = 8
# above this many nonzeros PyTorch's dense mm beats the sparse kernel (measured on
# an M5 Pro with a larger model: 133 -> 100 ms per step at 324 atoms)
LINEAR_MAX_NNZ = 100_000


def _matrix_triplets(fn, ni: int, batch: int = 2048):
    """Matrix of the linear map fn as (in, out, value) triplets, read off with
    one-hot inputs on the CPU."""
    II, OO, VV = [], [], []
    with torch.no_grad():
        for s in range(0, ni, batch):
            k = min(batch, ni - s)
            x = torch.zeros(k, ni, dtype=torch.float32)
            x[torch.arange(k), s + torch.arange(k)] = 1.0
            r = fn(x).numpy()
            i, o = np.nonzero(r)
            II.append(i + s)
            OO.append(o)
            VV.append(r[i, o])
    return np.concatenate(II), np.concatenate(OO), np.concatenate(VV)


def _weight_map(param, run):
    """Which entry of `param` and which constant give each nonzero output of the
    linear map `run`, read off by probing: weights all one give the constants,
    weights 1, 2, 3, ... the index, and random weights check that each nonzero output
    is one weight times a constant. Returns (flat output positions, constants, weight
    indices) on the CPU."""
    with torch.no_grad():
        param.fill_(1.0)
        a1 = run()
        param.copy_(
            torch.arange(1, param.numel() + 1, dtype=param.dtype).view(param.shape)
        )
        a2 = run()
        w = torch.randn(
            param.shape,
            dtype=param.dtype,
            generator=torch.Generator().manual_seed(0),
        )
        param.copy_(w)
        a3 = run()
    nz = (a1 != 0).reshape(-1)
    pos = torch.nonzero(nz).squeeze(1)
    coef = a1.reshape(-1)[pos]
    idx = torch.round(a2.reshape(-1)[pos] / coef) - 1
    in_range = bool((idx >= 0).all()) and bool((idx < param.numel()).all())
    if (
        not in_range
        or not torch.allclose(
            a3.reshape(-1)[pos],
            coef * w.reshape(-1)[idx.long()],
            rtol=1e-6,
            atol=1e-9,
        )
        or bool((a3.reshape(-1)[~nz] != 0).any())
    ):
        raise NotImplementedError('not one weight times a constant per entry')
    return pos, coef, idx.long()


def _by_rows(fn, ni: int, chunk: int = 128):
    """fn applied to the rows of the (ni, ni) identity in chunks, so that probing a
    large layer needs little memory."""
    eye = torch.eye(ni, dtype=torch.float64)
    return torch.cat([fn(eye[s : s + chunk]) for s in range(0, ni, chunk)])


class _TrainMatrix:
    """Train mode of a fused linear layer: its matrix rebuilt from the live weights
    at each call, with one indexed write, so that autograd differentiates the layer
    to any order with a few large operations."""

    def __init__(self, pos, coef, idx, shape, device):
        self.pos, self.idx = pos.to(device), idx.to(device)
        self.coef = coef.to(device, torch.float32)
        self.shape = shape

    def build(self, w):
        n = 1
        for d in self.shape:
            n *= d
        return (
            torch.zeros(n, dtype=w.dtype, device=w.device)
            .index_put((self.pos,), self.coef * w.reshape(-1)[self.idx])
            .view(self.shape)
        )


class _TypedTrainMatrix:
    """Train mode of a fused self-connection: every atom type's matrix has the same
    positions and constants, and reads the weight at idx0 + t * stride, so one small
    table serves all types."""

    def __init__(self, pos, coef, idx0, stride, nt, shape, device):
        self.pos, self.idx0, self.stride = (
            pos.to(device),
            idx0.to(device),
            stride.to(device),
        )
        self.coef = coef.to(device, torch.float32)
        self.nt, self.shape = nt, shape

    def build(self, w, t: int):
        n = self.shape[0] * self.shape[1]
        v = self.coef * w.reshape(-1)[self.idx0 + t * self.stride]
        return (
            torch.zeros(n, dtype=w.dtype, device=w.device)
            .index_put((self.pos,), v)
            .view(self.shape)
        )

    def build_all(self, w):
        """(nt, ni, no): all types at once, for few types."""
        n = self.shape[0] * self.shape[1]
        t = torch.arange(self.nt, device=w.device)[:, None]
        v = self.coef * w.reshape(-1)[self.idx0 + t * self.stride]
        return (
            torch.zeros(self.nt * n, dtype=w.dtype, device=w.device)
            .index_put(((self.pos + t * n).reshape(-1),), v.reshape(-1))
            .view(self.nt, *self.shape)
        )


def _param_key(*params):
    # identity as well as version: deepcopy and load_state_dict(assign=True) make new
    # tensors with fresh counters
    return tuple((id(p), p._version) for p in params)


class _DerivedTables:
    """Fused modules keep tables derived from their weights outside the state dict.
    A module conversion (.to, .half, .float, ...) can change the weights while
    keeping their identity and version, so it drops the tables and the next call
    rebuilds them."""

    _DERIVED = ('_map', '_ver', '_train')

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        _drop(self, *self._DERIVED)
        return out

    def _set_map(self, triplets, bias, ni, no) -> None:
        """Map from the train tables: same positions at every rebuild, so after the
        first only the values change."""
        old = self.__dict__.get('_map')
        if old is not None and old.order is not None:
            old.set_values(np.concatenate([t[2] for t in triplets]), bias)
        else:
            self._map = CsrMap(
                triplets, bias, ni, no, self.weight.device, keep_order=True
            )


def _fused_ok(x, *more) -> bool:
    """The kernels and the train paths read float32 only; under autocast PyTorch
    would make some products float16 further down."""
    return (
        x.device.type == 'mps'
        and x.dtype == torch.float32
        and all(t.dtype == torch.float32 and t.device == x.device for t in more)
        and not torch.is_autocast_enabled('mps')
    )


class FusedLinear(_DerivedTables, Linear):
    """e3nn Linear; in eval mode one kernel forward and one backward, with the matrix
    read off the module."""

    def _build(self):
        train = self.__dict__.get('_train')
        if train is not None:
            # trained since the last build: the train tables give the matrix
            # from the weights directly, without probing the module again
            mat, vec = train
            no = mat.shape[1]
            pos = mat.pos.cpu().numpy()
            with torch.no_grad():
                val = (mat.coef * self.weight.reshape(-1)[mat.idx]).cpu().numpy()
                bias = (
                    np.zeros(no)
                    if vec is None
                    else vec.build(self.bias).cpu().numpy()
                )
            self._set_map([(pos // no, pos % no, val)], bias, mat.shape[0], no)
            self._ver = _param_key(self.weight, self.bias)
            return
        ref = copy.deepcopy(self).cpu()
        ref.__class__ = Linear
        trip = _matrix_triplets(
            lambda x: ref(x, None, torch.zeros_like(ref.bias)), ref.irreps_in.dim
        )
        zero = torch.zeros(1, ref.irreps_in.dim, dtype=torch.float32)
        bias = ref(zero)[0].detach().numpy()
        self._map = CsrMap(
            [trip], bias, ref.irreps_in.dim, ref.irreps_out.dim, self.weight.device
        )
        self._ver = _param_key(self.weight, self.bias)

    def _train_tables(self):
        """Matrix and bias of the layer as functions of its weights, for train mode;
        None keeps e3nn there. Built at the first call in train mode, so inference
        never pays for it."""
        ni, no = self.irreps_in.dim, self.irreps_out.dim
        probe = copy.deepcopy(self).cpu().double()
        probe.__class__ = Linear
        for n in ('_map', '_ver', '_train'):
            probe.__dict__.pop(n, None)
        try:
            with torch.no_grad():
                if probe.bias.numel():
                    probe.bias.zero_()
            mat = _TrainMatrix(
                *_weight_map(probe.weight, lambda: _by_rows(probe, ni)),
                (ni, no),
                self.weight.device,
            )
            vec = None
            if probe.bias.numel():
                with torch.no_grad():
                    probe.weight.zero_()
                vec = _TrainMatrix(
                    *_weight_map(
                        probe.bias,
                        lambda: probe(torch.zeros(1, ni, dtype=torch.float64))[0],
                    ),
                    (no,),
                    self.weight.device,
                )
            return mat, vec
        except NotImplementedError as e:
            warnings.warn(f'linear layer trains with the e3nn code: {e}')
            return None

    def forward(self, features, weight=None, bias=None):
        if (
            weight is not None
            or bias is not None
            or features.dim() != 2
            or not _fused_ok(features, self.weight)
        ):
            return Linear.forward(self, features, weight, bias)
        if self.training:
            if '_train' not in self.__dict__:
                self._train = self._train_tables()
            if self._train is None:
                return Linear.forward(self, features)
            mat, vec = self._train
            out = features @ mat.build(self.weight)
            return out if vec is None else out + vec.build(self.bias)
        if self.__dict__.get('_ver') != _param_key(self.weight, self.bias):
            self._build()  # e.g. load_state_dict after enable
        return CsrApply.apply(features, self._map, None)


def _selfconn_triplets(ref: FullyConnectedTensorProduct) -> List:
    """Per atom type the matrix of a self-connection, from its weights: a scalar
    operand maps each irrep component to itself."""
    probe = copy.deepcopy(ref)
    with torch.no_grad():
        probe.weight.fill_(1.0)
    sl1, slo = ref.irreps_in1.slices(), ref.irreps_out.slices()
    nt, ni = ref.irreps_in2.dim, ref.irreps_in1.dim
    II, OO, VV = (
        [[] for _ in range(nt)],
        [[] for _ in range(nt)],
        [[] for _ in range(nt)],
    )
    for _, ins, W in ref.weight_views(yield_instruction=True):
        if ins.connection_mode != 'uvw' or W.shape[1] != nt:
            raise NotImplementedError(f'self-connection instruction {ins}')
        d = ref.irreps_in1[ins.i_in1].ir.dim
        x = torch.zeros(1, ni, dtype=torch.float32)
        x[0, sl1[ins.i_in1].start] = 1.0
        z = torch.zeros(1, nt, dtype=torch.float32)
        z[0, 0] = 1.0
        with torch.no_grad():
            c = float(
                probe(x, z)[0, slo[ins.i_out].start]
            )  # normalisation of this path
        W = W.detach().numpy()
        for t in range(nt):
            u, v = np.nonzero(W[:, t, :])
            for m in range(d):
                II[t].append(sl1[ins.i_in1].start + u * d + m)
                OO[t].append(slo[ins.i_out].start + v * d + m)
                VV[t].append(c * W[u, t, v])
    return [
        (np.concatenate(a), np.concatenate(b), np.concatenate(c))
        for a, b, c in zip(II, OO, VV)
    ]


class FusedSelfConnection(_DerivedTables, FullyConnectedTensorProduct):
    """x (N, D) with one-hot atom types z: one matrix per type, one kernel forward
    and one backward."""

    def _build(self):
        train = self.__dict__.get('_train')
        if train is not None:
            # trained since the last build: every type's matrix from the train
            # tables, without probing the module again (seconds for many types)
            ni, no = train.shape
            pos = train.pos.cpu().numpy()
            with torch.no_grad():
                t = torch.arange(train.nt, device=train.pos.device)[:, None]
                w = self.weight.reshape(-1)
                val = (train.coef * w[train.idx0 + t * train.stride]).cpu().numpy()
            trips = [(pos // no, pos % no, v) for v in val]
            self._set_map(trips, np.zeros(no), ni, no)
            self._ver = _param_key(self.weight)
            return
        ref = copy.deepcopy(self).cpu()
        ref.__class__ = FullyConnectedTensorProduct
        ni, nt, no = ref.irreps_in1.dim, ref.irreps_in2.dim, ref.irreps_out.dim
        trips = _selfconn_triplets(ref)
        # the matrices must reproduce the module
        x = torch.randn(
            3, ni, dtype=torch.float32, generator=torch.Generator().manual_seed(0)
        )  # leaves the global generator alone
        for t in {0, nt // 2, nt - 1}:
            z = torch.zeros(3, nt, dtype=torch.float32)
            z[:, t] = 1.0
            r = np.zeros((3, no))
            np.add.at(r.T, trips[t][1], (x[:, trips[t][0]].numpy() * trips[t][2]).T)
            if not np.allclose(r, ref(x, z).detach().numpy(), atol=1e-3, rtol=1e-3):
                raise NotImplementedError('self-connection is not a per-type matrix')
        self._map = CsrMap(trips, np.zeros(no), ni, no, self.weight.device)
        self._ver = _param_key(self.weight)

    def _train_tables(self):
        """Every type's matrix as a function of the weights, for train mode; None
        keeps e3nn there. Read off the weight layout (weight_views of 0, 1, 2, ...)
        and the path constants, then checked against the module for every type at
        once with random weights. Built at the first call in train mode, so inference
        never pays for it."""
        ni, nt, no = self.irreps_in1.dim, self.irreps_in2.dim, self.irreps_out.dim
        ref = copy.deepcopy(self).cpu().double()
        ref.__class__ = FullyConnectedTensorProduct
        for n in ('_map', '_ver', '_train'):
            ref.__dict__.pop(n, None)
        sl1, slo = ref.irreps_in1.slices(), ref.irreps_out.slices()
        try:
            P, CF, I0, ST = [], [], [], []
            with torch.no_grad():
                ref.weight.fill_(1.0)
                z0 = torch.zeros(1, nt, dtype=torch.float64)
                z0[0, 0] = 1.0
                ids = torch.arange(ref.weight.numel(), dtype=torch.float64)
                for _, ins, W in ref.weight_views(
                    weight=ids, yield_instruction=True
                ):
                    if (
                        ins.connection_mode != 'uvw'
                        or W.dim() != 3
                        or W.shape[1] != nt
                    ):
                        raise NotImplementedError(
                            f'self-connection instruction {ins}'
                        )
                    W = W.long()
                    d = ref.irreps_in1[ins.i_in1].ir.dim
                    x = torch.zeros(1, ni, dtype=torch.float64)
                    x[0, sl1[ins.i_in1].start] = 1.0
                    c = float(
                        ref(x, z0)[0, slo[ins.i_out].start]
                    )  # normalisation of this path
                    stride = (
                        W[:, 1, :] - W[:, 0, :]
                        if nt > 1
                        else torch.zeros_like(W[:, 0, :])
                    )
                    if not torch.equal(
                        W,
                        W[:, :1, :]
                        + torch.arange(nt)[None, :, None] * stride[:, None, :],
                    ):
                        raise NotImplementedError(
                            'weights of a type are not at a fixed stride'
                        )
                    u, v = torch.meshgrid(
                        torch.arange(W.shape[0]),
                        torch.arange(W.shape[2]),
                        indexing='ij',
                    )
                    for m in range(d):
                        P.append(
                            (sl1[ins.i_in1].start + u * d + m) * no
                            + slo[ins.i_out].start
                            + v * d
                            + m
                        )
                        I0.append(W[:, 0, :])
                        ST.append(stride)
                        CF.append(torch.full(u.shape, c, dtype=torch.float64))
            pos, idx0, stride, coef = (
                torch.cat([a.reshape(-1) for a in L]) for L in (P, I0, ST, CF)
            )
            if pos.unique().numel() != pos.numel():
                raise NotImplementedError('two weights write one entry')
            # check: random weights and inputs, every type in one call of the module
            g = torch.Generator().manual_seed(0)
            w = torch.randn(ref.weight.shape, dtype=torch.float64, generator=g)
            x = torch.randn(nt, ni, dtype=torch.float64, generator=g)
            with torch.no_grad():
                ref.weight.copy_(w)
                want = ref(x, torch.eye(nt, dtype=torch.float64))
            got = torch.zeros(nt, no, dtype=torch.float64)
            for s0 in range(0, nt, 32):
                t = torch.arange(s0, min(nt, s0 + 32))[:, None]
                val = coef * w.reshape(-1)[idx0 + t * stride] * x[t, pos // no]
                flat = (t - s0) * no + pos % no
                acc = torch.zeros(t.shape[0] * no, dtype=torch.float64)
                got[s0 : s0 + t.shape[0]] = acc.index_add_(
                    0, flat.reshape(-1), val.reshape(-1)
                ).view(-1, no)
            if not torch.allclose(
                got, want, rtol=1e-9, atol=1e-9 * float(want.abs().max())
            ):
                raise NotImplementedError(
                    'self-connection is not one weight times a constant per entry'
                )
        except NotImplementedError as e:
            warnings.warn(f'self-connection trains with the e3nn code: {e}')
            return None
        return _TypedTrainMatrix(
            pos, coef, idx0, stride, nt, (ni, no), self.weight.device
        )

    def forward(self, x, y, weight=None):
        if (
            weight is not None
            or x.dim() != 2
            or not _fused_ok(x, y, self.weight)
            or y.requires_grad
            or y.shape != (x.shape[0], self.irreps_in2.dim)
        ):  # a type index has no derivative
            return FullyConnectedTensorProduct.forward(self, x, y, weight)
        if self.training:
            if '_train' not in self.__dict__:
                self._train = self._train_tables()
            if self._train is None:
                return FullyConnectedTensorProduct.forward(self, x, y)
            if self._train.nt <= SELFCONN_ALL_TYPES:
                # few types: every type's product, then the one-hot operand picks
                # one; no read back to the CPU
                return torch.einsum(
                    'nt,tno->no',
                    y,
                    torch.matmul(x, self._train.build_all(self.weight)),
                )
            types = y.argmax(dim=1)
            out = x.new_zeros((x.shape[0], self.irreps_out.dim))
            for t in torch.unique(
                types
            ).tolist():  # many types: one small product for each type present
                rows = torch.nonzero(types == t).squeeze(1)
                out = out.index_copy(
                    0,
                    rows,
                    x.index_select(0, rows) @ self._train.build(self.weight, t),
                )
            return out
        if self.__dict__.get('_ver') != _param_key(self.weight):
            self._build()
        return CsrApply.apply(x, self._map, y.argmax(dim=1).to(torch.int32))


class FusedConvolution(IrrepsConvolution):
    def forward(self, data):
        x = data[self.key_x]
        y = data[self.key_filter]
        # autocast would make the weights float16 further down, and the kernels read
        # float32 only
        if (
            x.device.type != 'mps'
            or torch.is_autocast_enabled('mps')
            or x.dtype != torch.float32
            or y.dtype != torch.float32
            or data[self.key_weight_input].dtype != torch.float32
        ):
            return IrrepsConvolution.forward(self, data)
        plan = self._plan.get(data[self.key_edge_idx], x.shape[0])
        weight = self.weight_nn(data[self.key_weight_input])
        x = FusedUvu.apply(x, y, weight, self._kernels, plan)
        data[self.key_x] = x.div(self.denominator)
        return data


def _gate_act_code(act) -> int:
    """Code of a normalize2mom-wrapped activation in the gate kernel (silu or tanh),
    or an error."""
    f = getattr(act, 'f', None)
    if f is torch.nn.functional.silu:
        return 0
    if f is torch.tanh:
        return 1
    raise NotImplementedError(f'gate activation {f}')


def _gate_tables(gate: Gate, device) -> GateTables:
    """Slot tables of an e3nn Gate, read off the module: which input slots feed each
    output slot."""
    n_in = gate.irreps_in.dim

    def acts(activation):
        out = []
        for mul, (l, _), a in activation.paths:
            if l != 0 or a is None:
                raise NotImplementedError(
                    'gate without activation or on non-scalars'
                )
            out += [(_gate_act_code(a), 1.0 if a._is_id else a.cst)] * mul
        return out

    act_s, act_g = acts(gate.act_scalars), acts(gate.act_gates)
    with torch.no_grad():
        labels = torch.arange(n_in, dtype=torch.float32, device=device)[None]
        scal, gates, gated = (
            t[0].round().long().cpu().numpy() for t in gate.sc(labels)
        )
        src, gsl, code, cst = [], [], [], []
        for s, (c, k) in zip(scal.tolist(), act_s):
            src.append(s)
            gsl.append(-1)
            code.append(c)
            cst.append(k)
        if len(gated):
            ng, nk = len(gated), len(gates)
            f32 = dict(dtype=torch.float32, device=device)  # not the default dtype
            # elementwise product: which gated slot and gate feed each output slot,
            # and its numeric factor
            to_gated = (
                gate.mul(
                    torch.eye(ng, **f32), torch.ones(ng, nk, **f32)
                )
                .cpu()
                .numpy()
            )
            to_gate = (
                gate.mul(
                    torch.ones(nk, ng, **f32), torch.eye(nk, **f32)
                )
                .cpu()
                .numpy()
            )
            for o in range(to_gated.shape[1]):
                ig = np.nonzero(to_gated[:, o])[0]
                ik = np.nonzero(to_gate[:, o])[0]
                if len(ig) != 1 or len(ik) != 1:
                    raise NotImplementedError('gate mapping')
                c, k = act_g[ik[0]]
                src.append(int(gates[ik[0]]))
                gsl.append(int(gated[ig[0]]))
                code.append(c)
                cst.append(k * float(to_gated[ig[0], o]))
    return GateTables(src, gsl, code, cst, n_in, gate.irreps_out.dim, device)


def _gate_train_tables(tables: GateTables):
    """The gate's slot tables as index tensors and masks, for its train path."""
    src, gated, code, cst = tables.tables
    return (
        src.long(),
        gated.clamp(min=0).long(),
        (code == 1)[None],
        (gated >= 0)[None],
        cst[None],
    )


class FusedGate(_DerivedTables, Gate):
    _DERIVED = ('_tables', '_train')

    def forward(self, features):
        if features.dim() != 2 or not _fused_ok(features):
            return Gate.forward(self, features)
        if '_tables' not in self.__dict__:  # dropped by a module conversion
            self._tables = _gate_tables(self, features.device)
        if self.training:
            # out[o] = cst[o] * act(x[src[o]]) * (x[gated[o]] if gated[o] >= 0 else
            # 1): a few elementwise operations that
            # autograd differentiates to any order, from the same tables as the
            # kernel
            if '_train' not in self.__dict__:
                self._train = _gate_train_tables(self._tables)
            src, gated, is_tanh, has_gate, cst = self._train
            v = features.index_select(1, src)
            v = cst * torch.where(
                is_tanh, torch.tanh(v), torch.nn.functional.silu(v)
            )
            return torch.where(has_gate, v * features.index_select(1, gated), v)
        return GateApply.apply(features, self._tables)


def _fuse(m, cls):
    """Switch m to cls, or leave it as it was: a module without its tables could not
    run at all."""
    original = m.__class__
    m.__class__ = cls
    try:
        m._build()
    except (NotImplementedError, ValueError) as e:
        m.__class__ = original
        raise NotImplementedError(str(e)) from e


def enable(model, parts='all') -> Dict[str, int]:
    """Switch supported modules of `model`, which must be on the Apple GPU, to fused
    kernels. Convolutions use them in eval and train mode; linear, self-connection
    and gate use them in eval mode and a few PyTorch operations in train mode.
    Returns how many modules run which way. Unsupported modules keep the e3nn code on
    the GPU, never the CPU."""
    want: Set[str] = set(ALL_PARTS if parts == 'all' else parts.split('+'))
    device = next(model.parameters()).device
    if device.type != 'mps':
        raise ValueError(
            f'fused kernels need a model on the Apple GPU, this one is on {device}'
        )
    other = {p.dtype for p in model.parameters() if p.is_floating_point()} - {
        torch.float32
    }
    if other:
        raise ValueError(
            f'fused kernels need float32 weights, this model has '
            f'{sorted(map(str, other))}'
        )
    # a second call starts from scratch, so the report, the shared edge plan and the
    # set of parts are those of this call
    disable(model)
    plan = EdgePlan()
    model._mps_plan = plan
    report = dict(
        conv_fused=0,
        conv_e3nn=0,
        linear_fused=0,
        linear_e3nn=0,
        selfconn_fused=0,
        selfconn_e3nn=0,
        gate_fused=0,
        gate_e3nn=0,
    )
    onehots = {
        m.key_additional_output
        for m in model.modules()
        if isinstance(m, OnehotEmbedding)
    }
    for m in list(model.modules()):
        if 'conv' in want and type(m) is IrrepsConvolution:
            try:
                if m.is_parallel or type(m.convolution) is not TensorProduct:
                    raise NotImplementedError('parallel or non-e3nn convolution')
                m._kernels, m._plan = FusedTP(m.convolution), plan
                m.__class__ = FusedConvolution
                report['conv_fused'] += 1
            except (NotImplementedError, ValueError, RuntimeError) as e:
                # RuntimeError: the generated shader did not compile
                warnings.warn(f'convolution keeps the e3nn code on the GPU: {e}')
                report['conv_e3nn'] += 1
        elif (
            'linear' in want
            and type(m) is Linear
            and m.irreps_out.dim > 0
            and m.irreps_in.dim > 0
            and m.weight.dim() == 1
        ):
            try:
                _fuse(m, FusedLinear)
            except NotImplementedError as e:
                warnings.warn(f'linear layer keeps the e3nn code on the GPU: {e}')
                report['linear_e3nn'] += 1
                continue
            if m._map.fwd[1].shape[0] > LINEAR_MAX_NNZ:
                # large dense blocks: PyTorch's mm is faster than the sparse kernel
                m.__class__ = Linear
                _drop(m, '_map', '_ver', '_train')
                report['linear_e3nn'] += 1
            else:
                report['linear_fused'] += 1
        elif 'selfconn' in want and type(m) is SelfConnectionIntro:
            # the kernel selects a matrix per atom type, which needs a one-hot
            # operand
            try:
                if (
                    m.key_operand not in onehots
                    or type(m.fc_tensor_product) is not FullyConnectedTensorProduct
                ):
                    raise NotImplementedError('operand is not one-hot')
                _fuse(m.fc_tensor_product, FusedSelfConnection)
                report['selfconn_fused'] += 1
            except NotImplementedError as e:
                # _fuse restores the class it changed; anything else is untouched
                warnings.warn(f'self-connection keeps the e3nn code on the GPU: {e}')
                report['selfconn_e3nn'] += 1
        elif 'gate' in want and type(m) is Gate:
            try:
                m._tables = _gate_tables(m, device)
                m.__class__ = FusedGate
                report['gate_fused'] += 1
            except (NotImplementedError, ValueError) as e:
                warnings.warn(f'gate keeps the e3nn code on the GPU: {e}')
                report['gate_e3nn'] += 1
    return report


def mps_torch_supported() -> bool:
    return _policy.torch_supported()


def check_torch(purpose: str) -> None:
    """Raise a clear error, before any work, where the Apple GPU cannot run
    SevenNet with this PyTorch."""
    if not mps_torch_supported():
        raise RuntimeError(
            f'{purpose} on the Apple GPU needs PyTorch 2.6 or later and an '
            f'available MPS device: torch {torch.__version__} cannot run the '
            f'message scatter (scatter_reduce) on MPS. The fused kernels need '
            f'2.7 or later.'
        )


def _drop(m, *names) -> None:
    for n in names:
        m.__dict__.pop(n, None)


def disable(model) -> None:
    for m in model.modules():
        if isinstance(m, FusedConvolution):
            m.__class__ = IrrepsConvolution
            del m._kernels, m._plan
        elif isinstance(m, FusedLinear):
            m.__class__ = Linear
            _drop(m, '_map', '_ver', '_train')
        elif isinstance(m, FusedSelfConnection):
            m.__class__ = FullyConnectedTensorProduct
            _drop(m, '_map', '_ver', '_train')
        elif isinstance(m, FusedGate):
            m.__class__ = Gate
            _drop(m, '_tables', '_train')
    model.__dict__.pop('_mps_plan', None)
