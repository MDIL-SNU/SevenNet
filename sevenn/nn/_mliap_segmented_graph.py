"""Capture the Omni model between its ordinary LAMMPS ghost exchanges."""
import warnings
from types import SimpleNamespace

import torch

import sevenn._keys as KEY
from sevenn.mliap import _LOCAL_ATOM_KEYS, MLIAPWrappedConvolution
from sevenn.nn.lunatp_helper import (
    PREPARED_GRAPH_KEY,
    LunaTPSharedGraphConvolution,
    prepare_graph,
)


class TraceInputs(dict):
    """Find the input tensors read before a stage overwrites them."""

    def __init__(self, data):
        super().__init__(data)
        self.initial = data
        self.reads = {}

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if (
            isinstance(value, torch.Tensor)
            and value.is_cuda
            and self.initial.get(key) is value
        ):
            self.reads[key] = value
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default

    def pop(self, key, *default):
        if key not in self:
            return super().pop(key, *default)
        value = self[key]
        del self[key]
        return value


class ReplayStage:
    def __init__(self, modules, data):
        def apply(state):
            for module in modules:
                state = module(state)
            return state

        sample = TraceInputs(data)
        result = apply(sample)
        inputs = tuple(sample.reads)
        outputs = tuple(
            key for key, value in result.items()
            if isinstance(value, torch.Tensor) and data.get(key) is not value
        )
        self.deleted = tuple(key for key in data if key not in result)
        self.fixed_outputs = {
            key: value for key, value in result.items()
            if not isinstance(value, torch.Tensor) and data.get(key) is not value
        }
        self.inputs, self.outputs = inputs, outputs

        # Hold immutable metadata across captures. Every live CUDA input to
        # the stage is supplied through a replay input slot.
        template = {
            key: None if isinstance(value, torch.Tensor) and value.is_cuda else value
            for key, value in data.items()
        }
        del result, sample

        def run(*tensors):
            current = dict(template)
            current.update(zip(inputs, tensors))
            current = apply(current)
            return tuple(current[key] for key in outputs)

        # Stage inputs must be leaves. The previous segment's autograd history
        # belongs to the caller's stream and must stay outside this capture.
        samples = tuple(
            data[key].detach().clone().requires_grad_(data[key].requires_grad)
            for key in inputs
        )
        self.replay = torch.cuda.make_graphed_callables(
            run, samples, num_warmup_iters=2, allow_unused_input=True,
        )

    def __call__(self, data):
        values = self.replay(*(data[key] for key in self.inputs))
        for key in self.deleted:
            del data[key]
        data.update(self.fixed_outputs)
        data.update(zip(self.outputs, values))


def exchange(wrapper, data):
    nlocal = data[KEY.LAMMPS_DATA].nlocal
    data[KEY.NODE_FEATURE] = data[KEY.NODE_FEATURE][:nlocal]
    wrapper.ghost_exchange(data)


def local_only(nlocal):
    def narrow(data):
        for key in _LOCAL_ATOM_KEYS:
            values = data[key]
            if values.size(0) != nlocal:
                data[key] = values[:nlocal]
        return data

    return narrow


class SegmentedReplay:
    """Run captured model segments between unchanged ML-IAP ghost exchanges."""

    def __init__(self):
        self.stages = []
        self.topology = None
        self.edges = None
        self.shape = None
        self.captures = 0
        self.replays = 0
        self.disabled = False

    def __call__(self, model, data):
        if self.disabled or not data[KEY.EDGE_IDX].numel():
            return model(data)

        lmp_data = data[KEY.LAMMPS_DATA]
        nlocal, ntotal = lmp_data.nlocal, lmp_data.ntotal
        edges = data[KEY.EDGE_IDX]
        shape = (nlocal, ntotal, edges.shape)
        if self.shape != shape or not torch.equal(self.edges, edges):
            # Record the new ordered edge set, and let it stabilize before
            # capturing. Release the old input and activation storage first.
            self.stages.clear()
            self.topology = None
            self.shape, self.edges = shape, edges.detach().clone()
            return model(data)

        if self.stages:
            if self.topology is not None:
                data[PREPARED_GRAPH_KEY] = self.topology
            for stage in self.stages:
                if isinstance(stage, ReplayStage):
                    stage(data)
                else:
                    exchange(stage, data)
            self.replays += 1
            return data

        if self.captures >= 4 and self.replays < 50 * self.captures:
            self.disabled = True
            return model(data)

        self.captures += 1
        try:
            return self._capture(model, dict(data), nlocal, ntotal)
        except (RuntimeError, ValueError) as error:
            if torch.cuda.is_current_stream_capturing():
                raise
            warnings.warn(
                f'ML-IAP CUDA Graph capture failed with ghost folding off: {error}',
                RuntimeWarning,
            )
        self.disabled = True
        self.stages.clear()
        self.topology = None
        return model(data)

    def _capture(self, model, data, nlocal, ntotal):
        stages, modules = [], [model._preprocess]
        needs_topology = False
        for module in model:
            if not isinstance(module, MLIAPWrappedConvolution):
                modules.append(module)
                continue
            needs_topology |= isinstance(module.conv, LunaTPSharedGraphConvolution)
            stages.extend([tuple(modules), module])
            modules = [module.conv, local_only(nlocal)]
        stages.append(tuple(modules))
        if len(stages) < 3:
            raise ValueError('Expected at least one ML-IAP ghost exchange')
        if needs_topology:
            self.topology = prepare_graph(data[KEY.EDGE_IDX], ntotal)
            data[PREPARED_GRAPH_KEY] = self.topology

        for stage in stages:
            if isinstance(stage, tuple):
                # The wrapped first linear layer reads only the local count.
                sample = dict(data)
                sample[KEY.LAMMPS_DATA] = SimpleNamespace(nlocal=nlocal)
                replay = ReplayStage(stage, sample)
                replay(data)
                self.stages.append(replay)
            else:
                exchange(stage, data)
                self.stages.append(stage)
        return data
