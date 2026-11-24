import ctypes
import dataclasses
import math
import os
import pathlib
import warnings
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.train.dataload import unlabeled_atoms_to_graph

_OUTPUT_KEYS = (
    KEY.PRED_TOTAL_ENERGY,
    KEY.ATOMIC_ENERGY,
    KEY.PRED_FORCE,
    KEY.PRED_STRESS,
    KEY.PRED_ATOMIC_VIRIAL,  # only with compute_atomic_virial
)


def _sort_edges(data):
    """
    Puts the edges in one order: by destination, then source (stable). The
    neighbor list orders the same edges of a periodic cell differently after
    small moves; in this order an unchanged topology gives an equal edge index
    (CUDA graph key, eager topology reuse), and the edges are sorted by
    destination as LunaTP needs. Same function as the neighbor-list order, fp32
    summation order only.
    """
    ei = data[KEY.EDGE_IDX]
    num_atoms = len(data[KEY.ATOMIC_NUMBERS])
    order = torch.argsort(ei[0] * num_atoms + ei[1], stable=True)
    # index_select: CPU advanced indexing costs ~0.5 ms at 3k edges
    data[KEY.EDGE_IDX] = ei.index_select(1, order)
    data[KEY.EDGE_VEC] = data[KEY.EDGE_VEC].index_select(0, order)


def _to_device_packed(data, device):
    """
    Moves the tensors of data to device with one pinned buffer and one
    non-blocking host-to-device copy per dtype (the tensors become views of
    it): two copies instead of one per tensor, same values.
    """
    groups: Dict[torch.dtype, list] = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            groups.setdefault(v.dtype, []).append(k)
    for keys in groups.values():
        flat = torch.cat([data[k].reshape(-1) for k in keys])
        dev = flat.pin_memory().to(device, non_blocking=True)
        offset = 0
        for k in keys:
            n = data[k].numel()
            data[k] = dev[offset:offset + n].view(data[k].shape)
            offset += n


def _segments_per_source(edge_index, num_atoms: int, segment: int):
    """
    (LunaTP source segments of each source: ceil(degree / segment), >= 1,
    source degrees). numpy: torch CPU ops on these small arrays cost ~0.1 ms
    each, numpy ~2 us
    """
    deg = np.bincount(edge_index[1].numpy(), minlength=num_atoms)
    return np.maximum((deg + segment - 1) // segment, 1), deg


def _num_segments(edge_index, num_atoms: int, segment: int) -> int:
    """number of LunaTP source segments of an edge index"""
    return int(_segments_per_source(edge_index, num_atoms, segment)[0].sum())


def _pad_edges(tensors, num_edges: int, length: float, segment: Optional[int]):
    """
    tensors (host) with edges added up to num_edges: self-loops i -> i with
    the edge vector (length, 0, 0), length = the cutoff. There the cutoff
    function and its derivative vanish, so every edge term of the model is 0:
    a padded edge adds 0 to messages and coordination, and dE/dr_ij of it is
    exactly 0 (checked once per calculator, SevenNetCalculator._pads_neutral).
    With segment (LunaTP), a padded edge goes to a source with room below its
    next multiple of segment edges: the number of source segments stays.
    The edges are put in one order again (_sort_edges).
    """
    edge_index, edge_vec = tensors[KEY.EDGE_IDX], tensors[KEY.EDGE_VEC]
    num_atoms = tensors[KEY.ATOMIC_NUMBERS].shape[0]
    k = num_edges - edge_index.shape[1]
    if segment is None:
        nodes = torch.zeros(k, dtype=edge_index.dtype)
    else:
        nseg, deg = _segments_per_source(edge_index, num_atoms, segment)
        room = segment * nseg - deg
        before = np.cumsum(room) - room
        take = np.clip(np.minimum(room, k - before), 0, None)
        nodes = torch.from_numpy(np.repeat(np.arange(num_atoms), take))
    pad_vec = edge_vec.new_zeros(k, 3)
    pad_vec[:, 0] = length
    out = dict(tensors)
    out[KEY.EDGE_IDX] = torch.cat([edge_index, torch.stack([nodes, nodes])], 1)
    out[KEY.EDGE_VEC] = torch.cat([edge_vec, pad_vec])
    _sort_edges(out)
    return out


def _with_topology(data, topology):
    """data with a prepared LunaTP topology for the next model call, if any"""
    if topology is not None:
        from sevenn.nn.lunatp_helper import PREPARED_GRAPH_KEY

        data[PREPARED_GRAPH_KEY] = topology
    return data


def _tensor_pairs(a, b, out: list) -> bool:
    """
    Walks two objects of one structure (tensors in tuples, lists, dicts and
    dataclasses) and appends (tensor of a, tensor of b) to out for every tensor
    leaf. False if a tensor shape or dtype, the structure, or a non-tensor leaf
    (e.g. a node count) differs.
    """
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        out.append((a, b))
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(
            _tensor_pairs(x, y, out) for x, y in zip(a, b)
        )
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(
            _tensor_pairs(a[k], b[k], out) for k in a
        )
    if dataclasses.is_dataclass(a):
        return all(
            _tensor_pairs(getattr(a, f.name), getattr(b, f.name), out)
            for f in dataclasses.fields(a)
        )
    return a == b


def _pack_outputs(output):
    """The calculator outputs as one flat tensor (one device-to-host copy)
    and their (key, shape, dtype)"""
    keys = [k for k in _OUTPUT_KEYS if k in output]
    tensors = [output[k].detach() for k in keys]
    flat = torch.cat([t.reshape(-1) for t in tensors])
    return flat, [(k, t.shape, t.dtype) for k, t in zip(keys, tensors)]


def _unpack_outputs(flat: np.ndarray, meta, num_atoms: int, num_edges: int):
    out = {}
    offset = 0
    for k, shape, dtype in meta:
        n = shape.numel()
        np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
        out[k] = flat[offset:offset + n].astype(np_dtype).reshape(shape)
        offset += n
    energy = out[KEY.PRED_TOTAL_ENERGY].reshape(-1)[0].item()
    results: Dict[str, Any] = {
        'free_energy': energy,
        'energy': energy,
        'energies': out[KEY.ATOMIC_ENERGY].flatten(),
        'forces': out[KEY.PRED_FORCE][:num_atoms, :],
        'stress': -out[KEY.PRED_STRESS][[0, 1, 2, 4, 5, 3]],  # as voigt notation
        'num_edges': num_edges,
    }
    if KEY.PRED_ATOMIC_VIRIAL in out:
        results['stresses'] = out[KEY.PRED_ATOMIC_VIRIAL][:num_atoms, :]
    return results


class SevenNetCalculator(Calculator):
    """Supporting properties:
    'free_energy', 'energy', 'forces', 'stress', 'energies'
    free_energy equals energy. 'energies' stores atomic energy.

    Multi-GPU acceleration is not supported with ASE calculator.
    You should use LAMMPS for the acceleration.
    """

    def __init__(
        self,
        model: Union[str, pathlib.PurePath, AtomGraphSequential] = '7net-0',
        file_type: str = 'checkpoint',
        device: Union[torch.device, str] = 'auto',
        modal: Optional[str] = None,
        enable_cueq: bool = False,
        enable_flash: bool = False,
        enable_oeq: bool = False,
        enable_lunatp: bool = False,
        compute_atomic_virial: bool = False,
        sevennet_config: Optional[Dict] = None,  # Not used in logic, just meta info
        fused: bool = False,
        cuda_graph: bool = False,
        **kwargs,
    ) -> None:
        """Initialize SevenNetCalculator.

        Parameters
        ----------
        model: str | Path | AtomGraphSequential, default='7net-0'
            Name of pretrained models (7net-omni, 7net-nano-5.5, 7net-mf-ompa,
            7net-omat, 7net-l3i5, 7net-0) or path to the checkpoint, deployed model
            or the model itself
        file_type: str, default='checkpoint'
            one of 'checkpoint' | 'model_instance'
        device: str | torch.device, default='auto'
            if not given, use CUDA if available
        modal: str | None, default=None
            modal (fidelity) if given model is multi-modal model. for 7net-mf-ompa,
            it should be one of 'mpa' (MPtrj + sAlex) or 'omat24' (OMat24)
            for 7net-omni, use one of 'mpa', 'omat24', 'matpes_pbe', 'matpes_r2scan',
            'mp_r2scan', 'oc20', 'oc22', 'odac23', 'omol25_low', 'omol25_high',
            'spice', 'qcml', 'pet_mad'
            case insensitive
        enable_cueq: bool, default=False
            if True, use cuEquivariant to accelerate inference.
        enable_flash: bool, default=False
            if True, use FlashTP to accelerate inference.
        enable_oeq: bool, default=False
            if True, use OpenEquivariance to accelerate inference.
        enable_lunatp: bool, default=False
            if True, use LunaTP to accelerate inference.
        sevennet_config: dict | None, default=None
            Not used, but can be used to carry meta information of this calculator
        compute_atomic_virial: bool, default=False
            If True, request per-atom virial output (`stresses`) at runtime.
        fused: bool, default=False
            Use frozen-model kernels for supported LunaTP checkpoints on CUDA.
        cuda_graph: bool, default=False
            Capture and replay repeated CUDA calls with compatible atom and
            edge shapes. Run eagerly when capture is unavailable.
        """
        super().__init__(**kwargs)
        self.sevennet_config = None
        self.compute_atomic_virial = compute_atomic_virial

        if isinstance(model, pathlib.PurePath):
            model = str(model)

        allowed_file_types = ['checkpoint', 'model_instance']
        file_type = file_type.lower()
        if file_type not in allowed_file_types:
            if file_type == 'torchscript':
                raise ValueError(
                    'torchscript file_type is no longer supported. '
                    'Use checkpoint or model_instance instead.'
                )
            raise ValueError(f'file_type not in {allowed_file_types}')

        enable_cueq = os.getenv('SEVENNET_ENABLE_CUEQ') == '1' or enable_cueq
        enable_flash = os.getenv('SEVENNET_ENABLE_FLASH') == '1' or enable_flash
        enable_oeq = os.getenv('SEVENNET_ENABLE_OEQ') == '1' or enable_oeq
        enable_lunatp = os.getenv('SEVENNET_ENABLE_LUNATP') == '1' or enable_lunatp

        if not (enable_cueq or enable_flash or enable_oeq or enable_lunatp):
            util.warn_no_tp_accelerator('SevenNetCalculator')

        if enable_cueq and file_type == 'model_instance':
            warnings.warn(
                'file_type should be checkpoint to enable cueq. cueq set to False'
            )
            enable_cueq = False

        if enable_oeq and file_type == 'model_instance':
            warnings.warn(
                'file_type should be checkpoint to enable oeq. oeq set to False'
            )
            enable_oeq = False

        if enable_lunatp and file_type == 'model_instance':
            warnings.warn(
                'file_type should be checkpoint to enable lunatp. '
                'lunatp set to False'
            )
            enable_lunatp = False

        if isinstance(device, str):  # TODO: do we really need this?
            if device == 'auto':
                self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        if file_type == 'checkpoint' and isinstance(model, str):
            cp = util.load_checkpoint(model).compact_for_inference()

            model_loaded = cp.build_model(
                enable_cueq=enable_cueq,
                enable_flash=enable_flash,
                enable_oeq=enable_oeq,
                enable_lunatp=enable_lunatp,
            )
            model_loaded.set_is_batch_data(False)
            # inference only: forces and stress are gradients w.r.t. edge vectors,
            # which stay differentiable; freezing the weights skips dE/dW in the
            # force backward (same results, faster on every backend). A
            # 'model_instance' given by the caller is left as it is.
            model_loaded.requires_grad_(False)

            self.type_map = cp.config[KEY.TYPE_MAP]
            self.cutoff = cp.config[KEY.CUTOFF]
            self.sevennet_config = cp.config

        elif isinstance(model, AtomGraphSequential):
            if model.type_map is None:
                raise ValueError(
                    'Model must have the type_map to be used with calculator'
                )
            if model.cutoff == 0.0:
                raise ValueError('Model cutoff seems not initialized')
            model.eval_type_map = torch.tensor(True)  # ?
            model.set_is_batch_data(False)
            model_loaded = model
            self.type_map = model.type_map
            self.cutoff = model.cutoff
        else:
            raise ValueError('Unexpected input combinations')

        if self.sevennet_config is None and sevennet_config is not None:
            self.sevennet_config = sevennet_config

        self.model = model_loaded
        if self.compute_atomic_virial:
            force_output = self.model._modules.get('force_output')
            if force_output is None or not hasattr(
                force_output, 'use_atomic_virial'
            ):
                raise ValueError(
                    'compute_atomic_virial=True but model does not have '
                    'a force_output module with use_atomic_virial support.'
                )
            force_output.use_atomic_virial = True

        self.modal = None
        if isinstance(self.model, AtomGraphSequential):
            modal_map = self.model.modal_map
            if modal_map:
                modal_ava = list(modal_map.keys())
                if not modal:
                    raise ValueError(f'modal argument missing (avail: {modal_ava})')
                elif modal not in modal_ava:
                    raise ValueError(f'unknown modal {modal} (not in {modal_ava})')
                self.modal = modal
            elif not self.model.modal_map and modal:
                warnings.warn(f'modal={modal} is ignored as model has no modal_map')

        self.model.to(self.device)
        self.model.eval()

        self.fused = []
        self._fused_species = set()
        if fused or os.getenv('SEVENNET_ENABLE_FUSED') == '1':
            if not (
                enable_lunatp and file_type == 'checkpoint'
                and isinstance(model, str) and self.device.type == 'cuda'
            ):
                warnings.warn(
                    'fused needs enable_lunatp, a checkpoint and CUDA: ignored'
                )
            else:
                from sevenn.nn.inference_specialization import fold_modal
                from sevenn.nn.lunatp_fused import (
                    collapse_first_layer,
                    fuse_radial_conv,
                )

                self.fused = fuse_radial_conv(self.model, knots=2048)
                if self.fused:
                    self.model.requires_grad_(False)
                    if self.modal is not None:
                        self.fused += fold_modal(self.model, self.modal)
                    reference = cp.build_model()
                    reference.delete_module_by_key('force_output')
                    self.fused += collapse_first_layer(
                        self.model, reference, self.modal
                    )
                else:
                    warnings.warn('fused: no convolution of this model is covered')


        self.cuda_graph = cuda_graph and self.device.type == 'cuda'
        self._graphs: OrderedDict = OrderedDict()  # topology -> entry or None
        self._graphs_seen: OrderedDict = OrderedDict()  # topologies seen once
        self._capture_stream = None  # one stream: cuBLAS keeps a workspace each
        self._n_captures = 0  # captures tried (failed ones included)
        self._n_replays = 0  # replays of a kept graph (capture calls excluded)
        self._n_refreshes = 0  # replays with another edge index copied in
        self._pad = None  # padded edges: None unchecked, True, False (off)
        self._segment = -1  # LunaTP source segment size, None (e3nn); -1 unknown
        # (number of atoms, host edge index, device edge index) of the last eager call
        self._last_edge_index = (-1, None, None, None)
        self._has_lunatp: Optional[bool] = None
        self.implemented_properties = [
            'free_energy',
            'energy',
            'forces',
            'stress',
            'stresses',
            'energies',
        ]

    def set_atoms(self, atoms: Atoms) -> None:
        # called by ase, when atoms.calc = calc
        zs = tuple(set(atoms.get_atomic_numbers()))
        for z in zs:
            if z not in self.type_map:
                sp = list(self.type_map.keys())
                raise ValueError(
                    f'Model do not know atomic number: {z}, (knows: {sp})'
                )

    def output_to_results(self, output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        flat, meta = _pack_outputs(output)
        return _unpack_outputs(
            flat.cpu().numpy(),
            meta,
            output[KEY.ATOMIC_NUMBERS].shape[0],  # = num_atoms, no device read
            output[KEY.EDGE_IDX].shape[1],
        )

    def _fused_for(self, atoms: Atoms) -> None:
        """the fused kernels' species tables cover the species of atoms"""
        numbers = set(atoms.get_atomic_numbers().tolist())
        if numbers <= self._fused_species:
            return
        from sevenn.nn.lunatp_fused import specialize_fused_species

        species = self._fused_species | numbers
        specialize_fused_species(
            self.model, [self.type_map[z] for z in sorted(species)]
        )
        self._fused_species = species
        # Specializing new species replaces weights held by earlier captures.
        self._graphs.clear()
        self._graphs_seen.clear()

    _MAX_GRAPHS = 16
    _MAX_SEEN = 64
    # thrash guard: after _FREE_CAPTURES captures, capture only while replays
    # >= _REPLAYS_PER_CAPTURE * captures. A capture call costs ~2-3 eager calls
    # and a replay saves ~0.8 eager call (A5000, LunaTP, dimer/Si64).
    _FREE_CAPTURES = 8
    _REPLAYS_PER_CAPTURE = 2

    def _graph_results(self, data: AtomGraphData) -> Optional[Dict[str, Any]]:
        """Replay a compatible capture or return None for eager evaluation."""
        if self._segment == -1:
            self._segment = self._lunatp_segment()
        if self._pad is None and float(np.float32(self.cutoff)) != self.cutoff:
            self._pad = False
        pad = self._pad is not False
        tensors = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
        floats = [k for k, v in tensors.items() if v.is_floating_point()]
        edge_index = tensors[KEY.EDGE_IDX]
        num_atoms = tensors[KEY.ATOMIC_NUMBERS].shape[0]
        num_edges = edge_index.shape[1]
        segments = (
            _num_segments(edge_index, num_atoms, self._segment)
            if self._segment else None
        )
        edge_keys = (KEY.EDGE_IDX, KEY.EDGE_VEC)
        # Capture fixes non-tensor inputs, integer data other than the edge
        # index, and the shapes of the tensors and LunaTP segments.
        key = tuple(
            (k, None if pad and k in edge_keys else v.shape,
             None if k in floats or k == KEY.EDGE_IDX else v.numpy().tobytes())
            for k, v in tensors.items()
        ) + tuple((k, repr(v)) for k, v in data.items() if k not in tensors)
        key += (('segments', segments),)

        entry = self._graphs.get(key, False)  # False: no kept graph
        if entry is not False:
            self._graphs.move_to_end(key)
            if entry is None:  # capture failed
                return None
            if num_edges > entry['num_edges']:  # more edges than the graph
                del self._graphs[key]
                self._mark_seen(key, num_edges)
                entry = False
        if entry is False:
            if key not in self._graphs_seen or not self._capture_pays():
                self._mark_seen(key, num_edges)  # first call, or guard: eager
                return None
            e_pad = num_edges
            if pad:
                e_max = max(self._graphs_seen[key], num_edges)
                e_pad = e_max + max(16, e_max // 8)
                if self._segment:
                    e_pad = min(e_pad, self._segment * segments)
            del self._graphs_seen[key]  # second call: capture
            padded = self._padded(tensors, e_pad)
            if self._pad is None and e_pad > num_edges:
                self._pad = self._pads_neutral(data, tensors, padded)
                if not self._pad:
                    return None  # the next calls key on the number of edges
            self._n_captures += 1
            entry = self._capture(data, padded, floats)
            self._graphs[key] = entry
            if len(self._graphs) > self._MAX_GRAPHS:
                self._graphs.popitem(last=False)
            if entry is None:
                return None
            entry['num_edges'] = e_pad
        else:
            padded = self._padded(tensors, entry['num_edges'])
            new_edge_index = padded[KEY.EDGE_IDX]
            if not torch.equal(entry['edge_index'], new_edge_index):
                if not self._refresh(entry, new_edge_index):
                    # backend topology of other shapes (LunaTP segments)
                    del self._graphs[key]
                    self._mark_seen(key, num_edges)
                    return None
                entry['edge_index'] = new_edge_index
                self._n_refreshes += 1
            self._n_replays += 1
        tensors = padded

        graph, dev_in, host_in = entry['graph'], entry['dev_in'], entry['host_in']
        dev_out, host_out = entry['dev_out'], entry['host_out']
        torch.cat([tensors[k].reshape(-1) for k in floats], out=host_in)
        dev_in.copy_(host_in, non_blocking=True)
        graph.replay()
        host_out.copy_(dev_out, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        return _unpack_outputs(
            host_out.numpy(),  # astype copies: results outlive host_out
            entry['meta'],
            num_atoms,
            num_edges,
        )

    def _lunatp_segment(self) -> Optional[int]:
        """source segment size (edges) of the LunaTP graph, None without LunaTP"""
        from sevenn.nn.lunatp_helper import LunaTPConvolution

        if not any(isinstance(m, LunaTPConvolution) for m in self.model.modules()):
            return None
        from LunaEquivariance.lunatp.uvu_conv.kernels.utils import (
            _SOURCE_SEGMENT_EDGES,
        )

        return int(_SOURCE_SEGMENT_EDGES)

    def _padded(self, tensors, num_edges: int):
        if num_edges == tensors[KEY.EDGE_IDX].shape[1]:
            return tensors
        return _pad_edges(tensors, num_edges, self.cutoff, self._segment)

    def _pads_neutral(self, data, tensors, padded) -> bool:
        """Check once that cutoff-length self-loops preserve energy and forces."""
        energies, pad_grads, finite = [], 0.0, True
        for t in (tensors, padded):
            d = AtomGraphData(**{k: v.to(self.device) for k, v in t.items()})
            for k, v in data.items():
                if k not in t:
                    d[k] = v
            vec = d[KEY.EDGE_VEC].requires_grad_(True)
            grads = []
            vec.register_hook(grads.append)  # dE/dr_ij (force output)
            energies.append(float(self.model(d)[KEY.PRED_TOTAL_ENERGY].sum().detach()))
            if t is padded:
                ei, ev = t[KEY.EDGE_IDX], t[KEY.EDGE_VEC]
                pad = (ei[0] == ei[1]) & (ev.norm(dim=1) >= self.cutoff)
                g = grads[0] if grads else torch.zeros_like(vec)
                pad_grads = g[pad.to(self.device)].abs().sum().item()
                finite = bool(torch.isfinite(g).all())
        e0, e1 = energies
        finite = finite and math.isfinite(e0) and math.isfinite(e1)
        if (
            not finite
            or pad_grads != 0.0
            or abs(e1 - e0) > 1e-5 + 1e-6 * abs(e0)
        ):
            warnings.warn(
                'padded edges change the model outputs (sum |dE/dr_ij| '
                f'{pad_grads:.2e}, energy {e1 - e0:+.2e}); the CUDA graphs of '
                'this calculator key on the number of edges'
            )
            return False
        return True

    def _mark_seen(self, key, num_edges: int) -> None:
        self._graphs_seen[key] = max(num_edges, self._graphs_seen.get(key, 0))
        self._graphs_seen.move_to_end(key)
        if len(self._graphs_seen) > self._MAX_SEEN:
            self._graphs_seen.popitem(last=False)

    def _refresh(self, entry, edge_index) -> bool:
        """Update the capture's topology in place if its buffer shapes match."""
        static_edge_index = entry['static'][KEY.EDGE_IDX]
        new = edge_index.pin_memory().to(self.device, non_blocking=True)
        topology = self._lunatp_topology(new, entry['num_atoms'])
        pairs: list = []
        if not _tensor_pairs(entry['topology'], topology, pairs):
            return False
        static_edge_index.copy_(new)
        copied = set()
        for dst, src in pairs:  # graph._tables holds the same tensors again
            if id(dst) not in copied:
                copied.add(id(dst))
                dst.copy_(src)
        return True

    def _capture_pays(self) -> bool:
        return (
            self._n_captures < self._FREE_CAPTURES
            or self._n_replays >= self._REPLAYS_PER_CAPTURE * self._n_captures
        )

    def _capture(self, data, tensors, floats):
        """
        Capture the model call on static device inputs: the float inputs are
        views of one device buffer (filled by one copy per replay), the integer
        inputs are device copies kept with the graph, and the LunaTP topology
        of the static edge index is prepared here and passed to every model
        call of the capture. Returns the cache entry, or None with a warning if
        the capture fails.
        """
        flat_in = torch.cat([tensors[k].reshape(-1) for k in floats])
        host_in = flat_in.pin_memory()
        dev_in = flat_in.to(self.device)
        static = {}
        offset = 0
        for k, v in tensors.items():
            if k in floats:
                static[k] = dev_in[offset:offset + v.numel()].view(v.shape)
                offset += v.numel()
            else:
                static[k] = v.to(self.device)
        extra = {k: v for k, v in data.items() if k not in tensors}
        num_atoms = static[KEY.ATOMIC_NUMBERS].shape[0]
        topology = self._lunatp_topology(static[KEY.EDGE_IDX], num_atoms)

        def run():
            d = AtomGraphData(**static)
            for k, v in extra.items():
                d[k] = v
            return _pack_outputs(self.model(_with_topology(d, topology)))

        # one memory pool for the kept graphs; a pool whose graphs are all freed
        # is gone, so its handle is not kept: a new pool if no graph is kept
        pool = next((e['graph'].pool() for e in self._graphs.values() if e), None)
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(self.device)
        current = torch.cuda.current_stream(self.device)
        side = self._capture_stream
        side.wait_stream(current)
        try:
            with torch.cuda.stream(side):
                run()  # warm-up: lazy inits and caches of the static inputs
                graph.capture_begin(pool=pool)
                try:
                    dev_out, meta = run()
                finally:
                    graph.capture_end()
        except Exception as e:
            current.wait_stream(side)
            warnings.warn(
                f'CUDA graph capture failed ({type(e).__name__}: {e}); '
                'this topology runs eagerly'
            )
            return None
        current.wait_stream(side)
        host_out = torch.empty_like(dev_out, device='cpu').pin_memory()
        return {
            'graph': graph, 'dev_in': dev_in, 'host_in': host_in,
            'dev_out': dev_out, 'host_out': host_out, 'meta': meta,
            'static': static, 'edge_index': tensors[KEY.EDGE_IDX],
            'topology': topology, 'num_atoms': num_atoms,
        }

    def _lunatp_topology(self, edge_index, num_atoms: int):
        """(edge order, LunaTP graph) of a device edge index, None without LunaTP"""
        from sevenn.nn.lunatp_helper import LunaTPConvolution, prepare_graph

        if self._has_lunatp is None:
            self._has_lunatp = any(
                isinstance(m, LunaTPConvolution) for m in self.model.modules()
            )
        if not self._has_lunatp or edge_index.numel() == 0:
            return None
        return prepare_graph(edge_index, num_atoms)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # call parent class to set necessary atom attributes
        Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            raise ValueError('No atoms to evaluate')
        if self.fused:
            self._fused_for(atoms)
        data = AtomGraphData.from_numpy_dict(
            unlabeled_atoms_to_graph(atoms, self.cutoff, with_shift=False)
        )
        if self.modal:
            data[KEY.DATA_MODALITY] = self.modal
        _sort_edges(data)

        if self.cuda_graph:
            results = self._graph_results(data)
            if results is not None:
                self.results = results
                return

        # same topology as the previous eager call: its device edge index and
        # LunaTP topology
        ei = data[KEY.EDGE_IDX]
        num_atoms = data[KEY.ATOMIC_NUMBERS].shape[0]
        last = self._last_edge_index
        reuse = last[0] == num_atoms and torch.equal(last[1], ei)
        if reuse:
            del data[KEY.EDGE_IDX]
        if self.device.type == 'cuda':
            _to_device_packed(data, self.device)
        else:
            data.to(self.device)  # type: ignore
        if reuse:
            data[KEY.EDGE_IDX] = last[2]
        else:
            topology = self._lunatp_topology(data[KEY.EDGE_IDX], num_atoms)
            last = (num_atoms, ei, data[KEY.EDGE_IDX], topology)
            self._last_edge_index = last

        # a plain dict: the ~110 writes and ~210 reads of a forward skip the
        # torch_geometric Data storage layer (same tensors)
        output = self.model(_with_topology(data.to_dict(), last[3]))
        self.results = self.output_to_results(output)


class SevenNetD3Calculator(SumCalculator):
    def __init__(
        self,
        model: Union[str, pathlib.PurePath, AtomGraphSequential] = '7net-0',
        file_type: str = 'checkpoint',
        device: Union[torch.device, str] = 'auto',
        modal: Optional[str] = None,
        enable_cueq: bool = False,
        enable_flash: bool = False,
        enable_oeq: bool = False,
        enable_lunatp: bool = False,
        sevennet_config: Optional[Any] = None,
        damping_type: str = 'damp_bj',
        functional_name: str = 'pbe',
        vdw_cutoff: float = 9000,  # au^2, 0.52917726 angstrom = 1 au
        cn_cutoff: float = 1600,  # au^2, 0.52917726 angstrom = 1 au
        **kwargs,  # pass extra kwargs to both calculators
    ) -> None:
        """Initialize SevenNetD3Calculator. CUDA required.

        Parameters
        ----------
        model: str | Path | AtomGraphSequential
            Name of pretrained models (7net-omni, 7net-nano-5.5, 7net-mf-ompa,
            7net-omat, 7net-l3i5, 7net-0) or path to the checkpoint, deployed model
            or the model itself
        file_type: str, default='checkpoint'
            one of 'checkpoint' | 'model_instance'
        device: str | torch.device, default='auto'
            if not given, use CUDA if available
        modal: str | None, default=None
            modal (fidelity) if given model is multi-modal model.
            for 7net-mf-ompa, it should be one of 'mpa' or 'omat24'.
            for 7net-omni, use one of 'mpa', 'omat24', 'matpes_pbe',
            'matpes_r2scan', 'mp_r2scan', 'oc20', 'oc22', 'odac23',
            'omol25_low', 'omol25_high', 'spice', 'qcml', 'pet_mad'
        enable_cueq: bool, default=False
            if True, use cuEquivariance backend.
        enable_flash: bool, default=False
            if True, use flashTP backend.
        enable_oeq: bool, default=False
            if True, use OpenEquivariance backend.
        enable_lunatp: bool, default=False
            if True, use LunaTP backend.
        sevennet_config: dict | None, default=None
            hold meta information
        damping_type: str, default='damp_bj'
            Damping type of D3, one of 'damp_bj' | 'damp_zero'
        functional_name: str, default='pbe'
            Target functional name of D3 parameters.
        vdw_cutoff: float, default=9000
            vdw cutoff of D3 calculator in au^2
        cn_cutoff: float, default=1600
            cn cutoff of D3 calculator in au^2
        """
        if kwargs.get('compute_atomic_virial', False):
            warnings.warn(
                'D3Calculator does not support per-atom stress. '
                'Atomic stress from SevenNetD3Calculator will not '
                'include D3 contributions.'
            )

        d3_calc = D3Calculator(
            damping_type=damping_type,
            functional_name=functional_name,
            vdw_cutoff=vdw_cutoff,
            cn_cutoff=cn_cutoff,
            **kwargs,
        )

        sevennet_calc = SevenNetCalculator(
            model=model,
            file_type=file_type,
            device=device,
            modal=modal,
            enable_cueq=enable_cueq,
            enable_flash=enable_flash,
            enable_oeq=enable_oeq,
            enable_lunatp=enable_lunatp,
            sevennet_config=sevennet_config,
            **kwargs,
        )

        super().__init__([sevennet_calc, d3_calc])

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        # To match the precision of SevenNetCalculator output
        super().calculate(atoms, properties, system_changes)
        for key in ('forces', 'stress', 'stresses'):
            if key in self.results:
                self.results[key] = np.asarray(self.results[key], dtype=np.float32)


def _load(name: str) -> ctypes.CDLL:
    from torch.utils.cpp_extension import LIB_EXT, _get_build_directory, load

    # Load the library from the candidate locations

    package_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        return ctypes.CDLL(os.path.join(package_dir, f'{name}{LIB_EXT}'))
    except OSError:
        pass

    cache_dir = _get_build_directory(name, verbose=False)
    try:
        return ctypes.CDLL(os.path.join(cache_dir, f'{name}{LIB_EXT}'))
    except OSError:
        pass

    # Compile the library if it is not found

    if os.access(package_dir, os.W_OK):
        compile_dir = package_dir
    else:
        print('Warning: package directory is not writable. Using cache directory.')
        compile_dir = cache_dir

    # Matching CUDA architectures
    # If TORCH_CUDA_ARCH_LIST is set, we respect the user setting but warn if
    # it does not include the detected architecture.
    # If not set, we set it to a reasonable default including the detected arch.
    # We do not worry about CPU-only setup (not implemented yet).

    import torch
    major, minor = torch.cuda.get_device_capability()
    detected_arch = f'{major}.{minor}'

    if os.environ.get('TORCH_CUDA_ARCH_LIST'):
        base_archs = os.environ['TORCH_CUDA_ARCH_LIST']
        if detected_arch not in base_archs:
            print(
                f'Warning: TORCH_CUDA_ARCH_LIST={base_archs} does not include '
                f'detected GPU architecture {detected_arch}. '
                f'It may lead to compile error or NoKernelImageForDevice error.'
            )
    else:
        base_archs = '6.1;7.0;7.5;8.0;8.6;8.9;9.0'
        if detected_arch not in base_archs:
            base_archs += f';{detected_arch}'
        os.environ['TORCH_CUDA_ARCH_LIST'] = base_archs
        print(
            f'Info: TORCH_CUDA_ARCH_LIST is not set. '
            f'Detected GPU architecture {detected_arch}, '
            f'setting TORCH_CUDA_ARCH_LIST={base_archs} for compilation.'
        )

    load(
        name=name,
        sources=[os.path.join(package_dir, 'pair_e3gnn', 'pair_d3_for_ase.cu')],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-fmad=false'],
        build_directory=compile_dir,
        verbose=True,
        is_python_module=False,
    )

    return ctypes.CDLL(os.path.join(compile_dir, f'{name}{LIB_EXT}'))


class PairD3(ctypes.Structure):
    pass  # Opaque structure; only used as a pointer


class D3Calculator(Calculator):
    """ASE calculator for accelerated D3 van der Waals (vdW) correction.

    Example:
        from ase.calculators.mixing import SumCalculator
        calc_1 = SevenNetCalculator()
        calc_2 = D3Calculator()
        return SumCalculator([calc_1, calc_2])

    This calculator interfaces with the `libpaird3.so` library,
    which is compiled by nvcc during the package installation.
    If you encounter any errors, please verify
    the installation process and the compilation options in `setup.py`.
    Note: Multi-GPU parallel MD is not supported in this mode.
    Note: Cffi could be used, but it was avoided to reduce dependencies.
    """

    # Here, free_energy = energy
    implemented_properties = ['free_energy', 'energy', 'forces', 'stress']

    def __init__(
        self,
        damping_type: str = 'damp_bj',  # damp_bj, damp_zero
        functional_name: str = 'pbe',  # check the source code
        vdw_cutoff: float = 9000,  # au^2, 0.52917726 angstrom = 1 au
        cn_cutoff: float = 1600,  # au^2, 0.52917726 angstrom = 1 au
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if not torch.cuda.is_available():
            raise NotImplementedError('CPU + D3 is not implemented yet')

        self.rthr = vdw_cutoff
        self.cnthr = cn_cutoff
        self.damp_name = damping_type.lower()
        self.func_name = functional_name.lower()

        if self.damp_name not in ['damp_bj', 'damp_zero']:
            raise ValueError('Error: Invalid damping type.')

        self._lib = _load('pair_d3')

        self._lib.pair_init.restype = ctypes.POINTER(PairD3)
        self.pair = self._lib.pair_init()

        self._lib.pair_set_atom.argtypes = [
            ctypes.POINTER(PairD3),  # PairD3* pair
            ctypes.c_int,  # int natoms
            ctypes.c_int,  # int ntypes
            ctypes.POINTER(ctypes.c_int),  # int* types
            ctypes.POINTER(ctypes.c_double),  # double* x
        ]
        self._lib.pair_set_atom.restype = None

        self._lib.pair_set_domain.argtypes = [
            ctypes.POINTER(PairD3),  # PairD3* pair
            ctypes.c_int,  # int xperiodic
            ctypes.c_int,  # int yperiodic
            ctypes.c_int,  # int zperiodic
            ctypes.POINTER(ctypes.c_double),  # double* boxlo
            ctypes.POINTER(ctypes.c_double),  # double* boxhi
            ctypes.c_double,  # double xy
            ctypes.c_double,  # double xz
            ctypes.c_double,  # double yz
        ]
        self._lib.pair_set_domain.restype = None

        self._lib.pair_run_settings.argtypes = [
            ctypes.POINTER(PairD3),  # PairD3* pair
            ctypes.c_double,  # double rthr
            ctypes.c_double,  # double cnthr
            ctypes.c_char_p,  # const char* damp_name
            ctypes.c_char_p,  # const char* func_name
        ]
        self._lib.pair_run_settings.restype = None

        self._lib.pair_run_coeff.argtypes = [
            ctypes.POINTER(PairD3),  # PairD3* pair
            ctypes.POINTER(ctypes.c_int),  # int* atomic_numbers
        ]
        self._lib.pair_run_coeff.restype = None

        self._lib.pair_run_compute.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_run_compute.restype = None

        self._lib.pair_get_energy.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_get_energy.restype = ctypes.c_double

        self._lib.pair_get_force.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_get_force.restype = ctypes.POINTER(ctypes.c_double)

        self._lib.pair_get_stress.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_get_stress.restype = ctypes.POINTER(ctypes.c_double * 6)

        self._lib.pair_fin.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_fin.restype = None

    def _idx_to_numbers(self, Z_of_atoms):
        unique_numbers = list(dict.fromkeys(Z_of_atoms))
        return unique_numbers

    def _idx_to_types(self, Z_of_atoms):
        unique_numbers = list(dict.fromkeys(Z_of_atoms))
        mapping = {num: idx + 1 for idx, num in enumerate(unique_numbers)}
        atom_types = [mapping[num] for num in Z_of_atoms]
        return atom_types

    def _convert_domain_ase2lammps(self, cell):
        qtrans, ltrans = np.linalg.qr(cell.T, mode='complete')
        lammps_cell = ltrans.T
        signs = np.sign(np.diag(lammps_cell))
        lammps_cell = lammps_cell * signs
        qtrans = qtrans * signs
        lammps_cell = lammps_cell[(0, 1, 2, 1, 2, 2), (0, 1, 2, 0, 0, 1)]
        rotator = qtrans.T
        return lammps_cell, rotator

    def _stress2tensor(self, stress):
        tensor = np.array(
            [
                [stress[0], stress[3], stress[4]],
                [stress[3], stress[1], stress[5]],
                [stress[4], stress[5], stress[2]],
            ]
        )
        return tensor

    def _tensor2stress(self, tensor):
        stress = -np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[1, 2],
                tensor[0, 2],
                tensor[0, 1],
            ]
        )
        return stress

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            raise ValueError('No atoms to evaluate')

        has_pbc = atoms.get_pbc().any()

        if atoms.get_cell().sum() == 0:
            print(
                'Warning: D3Calculator requires a cell.\n'
                'Warning: An orthogonal cell large enough is generated.'
            )
            positions = atoms.get_positions()
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            max_cutoff = np.sqrt(max(self.rthr, self.cnthr)) * 0.52917726

            cell_lengths = max_pos - min_pos + max_cutoff + 1.0  # extra margin
            cell = np.eye(3) * cell_lengths

            atoms.set_cell(cell)
            atoms.set_pbc([True, True, True])  # for minus positions

        cell, rotator = self._convert_domain_ase2lammps(atoms.get_cell())

        Z_of_atoms = atoms.get_atomic_numbers()
        natoms = len(atoms)
        ntypes = len(set(Z_of_atoms))
        types = (ctypes.c_int * natoms)(*self._idx_to_types(Z_of_atoms))

        positions = atoms.get_positions() @ rotator.T
        x_flat = (ctypes.c_double * (natoms * 3))(*positions.flatten())

        atomic_numbers = (ctypes.c_int * ntypes)(*self._idx_to_numbers(Z_of_atoms))

        boxlo = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
        boxhi = (ctypes.c_double * 3)(cell[0], cell[1], cell[2])
        xy = cell[3]
        xz = cell[4]
        yz = cell[5]
        xperiodic, yperiodic, zperiodic = atoms.get_pbc()

        lib = self._lib
        assert lib is not None
        lib.pair_set_atom(self.pair, natoms, ntypes, types, x_flat)

        xperiodic = xperiodic.astype(int)
        yperiodic = yperiodic.astype(int)
        zperiodic = zperiodic.astype(int)
        lib.pair_set_domain(
            self.pair, xperiodic, yperiodic, zperiodic, boxlo, boxhi, xy, xz, yz
        )

        lib.pair_run_settings(
            self.pair,
            self.rthr,
            self.cnthr,
            self.damp_name.encode('utf-8'),
            self.func_name.encode('utf-8'),
        )

        lib.pair_run_coeff(self.pair, atomic_numbers)
        lib.pair_run_compute(self.pair)

        result_E = lib.pair_get_energy(self.pair)

        result_F_ptr = lib.pair_get_force(self.pair)
        result_F_size = natoms * 3
        result_F = np.ctypeslib.as_array(
            result_F_ptr, shape=(result_F_size,)
        ).reshape((natoms, 3))
        result_F = np.array(result_F)
        result_F = result_F @ rotator

        result_S = lib.pair_get_stress(self.pair)
        result_S = np.array(result_S.contents)
        if has_pbc:
            result_S = (
                self._tensor2stress(
                    rotator.T @ self._stress2tensor(result_S) @ rotator
                )
                / atoms.get_volume()
            )
        else:
            result_S = np.zeros(6)

        self.results = {
            'free_energy': result_E,
            'energy': result_E,
            'forces': result_F,
            'stress': result_S,
        }

    def __del__(self):
        if self._lib is not None:
            self._lib.pair_fin(self.pair)
            self._lib = None
            self.pair = None
