"""
CUDA graph replay of the SevenNet model call in LAMMPS ML-IAP (single rank).
Kept free of the lammps import (as sevenn/nn/_ghost_exchange.py), so it can be
tested with an emulated lmp_data.

The ML-IAP pair list changes at nearly every MD step (pairs are filtered by
the cutoff), so one capture has to serve every topology of a run. The captured
call has fixed shapes, and everything that changes is copied into static
device tensors:

- ghosts: with one rank every ghost is a periodic image of a local atom, so
  the source index of an edge is folded to its owner (found by one
  forward_exchange of the local indices) and the plain serial model is
  called: no ghost exchange inside the call.
- edges: padded to a fixed number with self-loops of length r_c on a few fake
  atoms appended after the local atoms. Fake atoms share no edge with a real
  atom and their energies are not summed, so the padding cannot change an
  output (no assumption on the cutoff function). Real edges keep the LAMMPS
  pair order (sorted by i, as LunaTP needs), so the pair forces are returned
  without a permutation.
- LunaTP: the graph tables are rebuilt for a changed pair list and copied in
  place; the source segment tables are padded with empty segments to a fixed
  number.

More edges / segments / atoms than the capture holds: captured again with new
room. Several ranks (a ghost owned by another rank): not supported, the
caller runs the eager ghost-exchange model.
"""
import atexit
import dataclasses
import os
import time
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

import sevenn._keys as KEY


def _walk_tensors(obj, out: List[torch.Tensor], seen: set) -> None:
    """unique tensors of dataclasses / dicts / tuples / lists, in a fixed order"""
    if isinstance(obj, torch.Tensor):
        if id(obj) not in seen:
            seen.add(id(obj))
            out.append(obj)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            _walk_tensors(obj[k], out, seen)
    elif isinstance(obj, (tuple, list)):
        for v in obj:
            _walk_tensors(v, out, seen)
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            _walk_tensors(getattr(obj, f.name), out, seen)


def _graph_tensors(graph) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    _walk_tensors(graph, out, set())
    return out


_PREP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
namespace {
// one step's bookkeeping before a replay: the edge index of the pairs (ghosts
// folded to their owners) padded with self-loops on the fake atoms, and
// flags[0] a ghost of another rank, [1] the edge index differs from ref,
// [2] a species that is not allowed, [3] number of pairs between atoms
// closer than near in index
__global__ void sevenn_mliap_prep_kernel(
    const double* __restrict__ recv, const int32_t* __restrict__ pair_i,
    const int32_t* __restrict__ pair_j, const int32_t* __restrict__ elems,
    const bool* __restrict__ allowed, const int64_t* __restrict__ ref,
    int64_t* __restrict__ cand, int32_t* __restrict__ flags, double token,
    int64_t nlocal, int64_t ntotal, int64_t npairs, int64_t edges,
    int64_t num_fake, int64_t near, bool prune) {
    const int64_t p = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < edges) {
        int64_t a, b;
        if (p < npairs) {
            a = pair_i[p];
            b = int64_t(recv[2 * int64_t(pair_j[p])]);
            const int64_t d = a > b ? a - b : b - a;
            if (d < near) atomicAdd(flags + 3, 1);
        } else {
            a = b = nlocal + ((p - npairs) * num_fake) / (edges - npairs);
        }
        cand[p] = a;
        cand[edges + p] = b;
        if (a != ref[p] || b != ref[edges + p]) flags[1] = 1;
    }
    if (p < ntotal && recv[2 * p + 1] != token) flags[0] = 1;
    if (prune && p < nlocal && !allowed[elems[p]]) flags[2] = 1;
}
}  // namespace
void sevenn_mliap_prep(torch::Tensor recv, torch::Tensor pair_i,
                       torch::Tensor pair_j, torch::Tensor elems,
                       torch::Tensor allowed, torch::Tensor ref,
                       torch::Tensor cand, torch::Tensor flags, double token,
                       int64_t nlocal, int64_t npairs, int64_t num_fake,
                       int64_t near, bool prune) {
    const int64_t ntotal = recv.size(0), edges = cand.size(1);
    const int64_t n = std::max(edges, ntotal);
    if (n == 0) return;
    sevenn_mliap_prep_kernel<<<(n + 255) / 256, 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        recv.data_ptr<double>(), pair_i.data_ptr<int32_t>(),
        pair_j.data_ptr<int32_t>(), elems.data_ptr<int32_t>(),
        allowed.data_ptr<bool>(), ref.data_ptr<int64_t>(),
        cand.data_ptr<int64_t>(), flags.data_ptr<int32_t>(), token, nlocal,
        ntotal, npairs, edges, num_fake, near, prune);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sevenn_mliap_prep", &sevenn_mliap_prep);
}
"""


def _make_graph(edge_index: torch.Tensor, num_nodes: int, segment_edges: int):
    """make_graph with source segments of at most segment_edges (0: LunaTP's)"""
    import LunaEquivariance.lunatp.graph as luna_graph

    default = luna_graph._SOURCE_SEGMENT_EDGES
    if segment_edges:
        luna_graph._SOURCE_SEGMENT_EDGES = segment_edges
    try:
        return luna_graph.make_graph(edge_index[1], edge_index[0], num_nodes)
    finally:
        luna_graph._SOURCE_SEGMENT_EDGES = default


def _lunatp_graph(
    edge_index: torch.Tensor, num_nodes: int, num_segments: int,
    segment_edges: int = 0,
):
    """
    LunaTPGraph of a dst-sorted edge index whose source segment tables are
    padded to num_segments with empty segments of the last node, or None if
    the graph has more segments.
    """
    graph = _make_graph(edge_index, num_nodes, segment_edges)
    seg = graph.source_segments
    k = num_segments - seg.nodes.numel()
    if k < 0:
        return None
    if k > 0:
        num_edges = edge_index.shape[1]
        rowptr = seg.rowptr.clone()
        rowptr[-1] = num_segments
        seg = dataclasses.replace(
            seg,
            nodes=torch.cat([seg.nodes, seg.nodes.new_full((k,), num_nodes - 1)]),
            begin=torch.cat([seg.begin, seg.begin.new_full((k,), num_edges)]),
            end=torch.cat([seg.end, seg.end.new_full((k,), num_edges)]),
            rowptr=rowptr,
            order=torch.cat([
                seg.order,
                torch.arange(
                    num_segments - k, num_segments,
                    device=seg.order.device, dtype=seg.order.dtype,
                ),
            ]),
        )
        graph = dataclasses.replace(graph, source_segments=seg)
        graph.store_topology()
    return graph


class MLIAPCudaGraph:
    """
    model: plain serial model (not patch_model_for_mliap) as built from a
    checkpoint (the modal is read from the data: no prepare_modal_deploy,
    whose host-to-device copy cannot be captured), force_output deleted,
    eval, on device, weights frozen. modal: of a multi-modal model.
    compute(lmp_data) -> True if energies and pair forces were written to
    lmp_data; False: the caller must run the eager call.
    """

    # room above the sizes at capture
    # edges: E / EDGE_ROOM (at least MIN_EDGE_ROOM); halved (down to
    # MIN_EDGE_ROOM_DIV) whenever a run outgrows a capture
    EDGE_ROOM = 32
    MIN_EDGE_ROOM_DIV = 2
    MIN_EDGE_ROOM = 128
    SEGMENT_ROOM = 8  # LunaTP source segments: N / SEGMENT_ROOM
    PAD_EDGES_PER_FAKE_ATOM = 32
    # thrash guard: after FREE_CAPTURES captures, a capture needs
    # REPLAYS_PER_CAPTURE replays per capture so far, else eager for good
    FREE_CAPTURES = 4
    REPLAYS_PER_CAPTURE = 50
    # fused kernels: index launch order above this index_coherence (random
    # atom order: 0.44), for cells large enough for the order to matter
    INDEX_ORDER_COHERENCE = 0.6
    INDEX_ORDER_MIN_ATOMS = 256

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cutoff: float,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        modal: Optional[str] = None,
    ):
        self.model = model
        self.modal = modal
        self.device = device
        self.cutoff = float(cutoff)
        self.dtype = dtype
        self.verbose = verbose or os.getenv('SEVENN_MLIAP_GRAPH_VERBOSE') == '1'
        self.enabled = device.type == 'cuda'
        # no capture: the eager call of the folded serial model
        self.eager_fold = os.getenv('SEVENN_MLIAP_FOLD_EAGER') == '1'
        self.n_captures = 0
        self.n_replays = 0
        self.n_refreshes = 0
        self.n_eager = 0
        self._entry: Optional[Dict[str, Any]] = None
        self._edge_room = self.EDGE_ROOM
        # SEVENN_MLIAP_GRAPH_TIMING=1: wall time per phase of a call (a
        # synchronize after each phase, so the call itself gets slower),
        # printed at exit
        self._times: Optional[Dict[str, float]] = None
        if os.getenv('SEVENN_MLIAP_GRAPH_TIMING') == '1':
            self._times = {}
            atexit.register(self._print_times)
        self._stream = None
        # a value the owner of a ghost must return: another rank has another one
        self._token = float(int.from_bytes(os.urandom(6), 'little'))
        from sevenn.nn.lunatp_helper import LunaTPConvolution

        self._lunatp = any(
            isinstance(m, LunaTPConvolution) for m in model.modules()
        )
        from sevenn.nn.lunatp_fused import LunaTPFusedRadialConvolution

        # fused radial convolutions: weight rows per distinct edge length. A
        # full neighbor list holds every length twice (i-j, j-i); the captured
        # call has rows for half of its pairs plus slack, or for all of them
        # once that was not enough
        self._fused = any(
            isinstance(m, LunaTPFusedRadialConvolution) for m in model.modules()
        )
        self._unique_all = False
        # species self connection pruned to the species seen so far
        # (lunatp_fused.prune_self_connection); a new species prunes again
        self._prune = bool(getattr(model, 'prune_species', False))
        self._owner_buf = None
        # the step bookkeeping as one kernel (built by LunaEquivariance's JIT)
        self._prep = None
        if self._lunatp and device.type == 'cuda' and (
            os.getenv('SEVENN_MLIAP_GRAPH_PREP', '1') != '0'
        ):
            try:
                from sevenn.nn.lunatp_fused import _load_cuda

                self._prep = _load_cuda('sevenn_mliap_prep', _PREP_SOURCE)
            except Exception as e:  # no compiler: the torch operations
                warnings.warn(f'ML-IAP step kernel not built: {e}')
        self._allowed = torch.zeros(256, dtype=torch.bool, device=device)

    def _tick(self, phase: str, since: float) -> float:
        if self._times is None:
            return since
        torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self._times[phase] = self._times.get(phase, 0.0) + now - since
        return now

    def _print_times(self) -> None:
        n = max(1, self.n_replays)
        total = sum(v for k, v in self._times.items() if k != 'capture')
        print(f'[SevenNet] CUDA graph phases, ms per replay ({n} replays, '
              f'{self.n_refreshes} refreshes, {self.n_captures} captures):')
        for k, v in self._times.items():
            print(f'  {k:10s} {v / n * 1e3:8.3f}')
        print(f'  {"sum":10s} {total / n * 1e3:8.3f}  (without capture)', flush=True)

    # ---- per call ----------------------------------------------------------
    def _owners(self, lmp_data, nlocal: int, ntotal: int):
        """(owner local index of every atom, flag: a ghost of another rank)"""
        buf = self._owner_buf
        if buf is None or buf[0] != (nlocal, ntotal):
            send = torch.full(
                (ntotal, 2), -1.0, dtype=torch.float64, device=self.device
            )
            send[:nlocal, 0] = torch.arange(
                nlocal, dtype=torch.float64, device=self.device
            )
            send[:nlocal, 1] = self._token
            buf = self._owner_buf = ((nlocal, ntotal), send, torch.empty_like(send))
        _, send, recv = buf  # send is constant for (nlocal, ntotal)
        lmp_data.forward_exchange(send, recv, 2)
        return recv

    def _prepare(self, lmp_data, entry, recv, nlocal: int, npairs: int):
        """
        One kernel for the step's bookkeeping when the captured call may fit:
        fills entry['cand'], returns [foreign, changed, unseen, reorder] (one
        host sync), or None if the kernel does not apply.
        """
        if self._prep is None:
            return None
        dev = self.device
        pair_i = torch.as_tensor(lmp_data.pair_i, device=dev)
        pair_j = torch.as_tensor(lmp_data.pair_j, device=dev)
        elems = torch.as_tensor(lmp_data.elems, device=dev)
        if not (
            pair_i.dtype == pair_j.dtype == elems.dtype == torch.int32
            and pair_i.is_contiguous() and pair_j.is_contiguous()
            and elems.is_contiguous()
        ):
            return None
        flags = entry.get('flags')
        if flags is None:
            flags = entry['flags'] = torch.zeros(4, dtype=torch.int32, device=dev)
        else:
            flags.zero_()
        self._prep.sevenn_mliap_prep(
            recv, pair_i, pair_j, elems, self._allowed, entry['edge_index'],
            entry['cand'], flags, self._token, nlocal, npairs,
            entry['num_fake'], max(1, nlocal // 4), self._prune,
        )
        foreign, changed, unseen, near = flags.tolist()  # the one host sync
        reorder = False
        if self._fused and nlocal >= self.INDEX_ORDER_MIN_ATOMS:
            coherent = near / npairs > self.INDEX_ORDER_COHERENCE
            reorder = coherent != entry['index_order']
        return [bool(foreign), bool(changed), bool(unseen), reorder]

    def compute(self, lmp_data) -> bool:
        if not self.enabled:
            return False
        nlocal, ntotal, npairs = lmp_data.nlocal, lmp_data.ntotal, lmp_data.npairs
        if npairs <= 1 or nlocal == 0:
            return False
        dev = self.device
        t = time.perf_counter()
        recv = self._owners(lmp_data, nlocal, ntotal)
        t = self._tick('owners', t)
        entry = self._entry
        fits = (
            entry is not None
            and entry['nlocal'] == nlocal
            and npairs <= entry['num_edges']
        )
        flags = None
        if fits and not self.eager_fold:
            flags = self._prepare(lmp_data, entry, recv, nlocal, npairs)
        elems = torch.as_tensor(lmp_data.elems, device=dev)[:nlocal]
        if flags is not None:
            cand = entry['cand']  # filled: the pairs, ghosts as their owners
            pair_i, src = cand[0, :npairs], cand[1, :npairs]
            foreign = unseen = None
        else:
            owner = recv[:, 0].to(torch.int64)
            foreign = (recv[:, 1] != self._token).any()
            pair_i = torch.as_tensor(lmp_data.pair_i, device=dev)
            pair_j = torch.as_tensor(lmp_data.pair_j, device=dev)
            src = owner.index_select(0, pair_j.to(torch.int64))
            unseen = foreign
            if self._prune:
                unseen = (~self._allowed[elems.to(torch.int64)]).any()
        if self.eager_fold:
            if bool(foreign):
                return self._disable('a ghost atom is owned by another rank')
            if self._prune and bool(unseen):
                self._new_species(elems)
            return self._eager(lmp_data, nlocal, pair_i, src, elems)

        if fits:
            cand = entry['cand']
            if flags is None:
                self._fill_edge_index(cand, pair_i, src, npairs, entry)
                reorder = foreign
                if self._fused and nlocal >= self.INDEX_ORDER_MIN_ATOMS:
                    # the atom order became (in)coherent, e.g. LAMMPS sorted
                    d = (cand[0, :npairs] - cand[1, :npairs]).abs()
                    coherent = (d < nlocal // 4).float().mean() > (
                        self.INDEX_ORDER_COHERENCE
                    )
                    reorder = coherent != entry['index_order']
                flags = torch.stack(
                    [foreign, (cand != entry['edge_index']).any(), unseen, reorder]
                ).tolist()  # the one host sync before the replay
            t = self._tick('edges', t)
            if flags[0]:
                return self._disable('a ghost atom is owned by another rank')
            if self._prune and flags[2]:
                self._new_species(elems)
                fits = False  # other weights: capture again
            elif self._fused and flags[3]:
                fits = False  # other launch order: capture again
            elif flags[1] and not self._refresh(entry, cand):
                fits = False
            t = self._tick('refresh', t)
        elif bool(foreign):
            return self._disable('a ghost atom is owned by another rank')
        elif self._prune and bool(unseen):
            self._new_species(elems)
        if not fits:
            if entry is not None and entry['nlocal'] == nlocal:
                self._edge_room = max(self.MIN_EDGE_ROOM_DIV, self._edge_room // 2)
            if not self._capture_pays():
                return self._disable(
                    f'{self.n_captures} captures for {self.n_replays} replays'
                )
            entry = self._capture(nlocal, npairs, pair_i, src, elems)
            self._entry = entry
            if entry is None:
                return self._disable('capture failed')
            if self.verbose:
                print(
                    f'[SevenNet] CUDA graph captured: {nlocal} atoms '
                    f'(+{entry["num_nodes"] - nlocal} fake), {npairs} pairs '
                    f'(room {entry["num_edges"]}, weight rows '
                    f'{entry["unique_cap"]}, index coherence '
                    f'{entry.get("coherence", 0.0):.2f}), capture '
                    f'{self.n_captures}',
                    flush=True,
                )
            t = self._tick('capture', t)
        else:
            self.n_replays += 1

        for attempt in range(2):
            with torch.no_grad():
                entry['z'][:nlocal] = elems
                vec = entry['edge_vec']
                vec[:npairs] = torch.as_tensor(lmp_data.rij, device=dev)
                vec[npairs:] = entry['pad_vec']
            t = self._tick('copy_in', t)
            entry['graph'].replay()
            t = self._tick('replay', t)
            if not self._fused or int(entry['n_unique']) <= entry['unique_cap']:
                break
            if self.verbose:
                print(
                    f'[SevenNet] {int(entry["n_unique"])} distinct edge lengths '
                    f'of {npairs} pairs, weight rows {entry["unique_cap"]}',
                    flush=True,
                )
            # more distinct edge lengths than weight rows: rows for all edges
            if attempt == 1 or self._unique_all:
                return self._disable('edge lengths do not fit the weight rows')
            self._unique_all = True
            entry = self._capture(nlocal, npairs, pair_i, src, elems)
            self._entry = entry
            if entry is None:
                return self._disable('capture failed')
            t = self._tick('capture', t)

        torch.as_tensor(lmp_data.eatoms).copy_(entry['e_atoms'])
        lmp_data.energy = entry['e_total']
        lmp_data.update_pair_forces_gpu(entry['pair_forces'])
        self._tick('outputs', t)
        return True

    def _eager(self, lmp_data, nlocal, pair_i, src, elems) -> bool:
        dev = self.device
        edge_vec = torch.as_tensor(lmp_data.rij, dtype=self.dtype, device=dev)
        edge_vec.requires_grad_(True)
        data = {
            KEY.EDGE_IDX: torch.stack([pair_i.to(torch.int64), src]),
            KEY.EDGE_VEC: edge_vec,
            KEY.ATOMIC_NUMBERS: elems.to(torch.int64),
            KEY.NUM_ATOMS: torch.tensor(nlocal, dtype=torch.int64, device=dev),
        }
        if self.modal is not None:
            data[KEY.DATA_MODALITY] = self.modal
        if self._fused:
            from sevenn.nn.lunatp_fused import set_unique_cap

            set_unique_cap(self.model, None)
        output = self.model(data)
        e_atoms = output[KEY.ATOMIC_ENERGY].view(-1)
        e_total = e_atoms.sum()
        (grad,) = torch.autograd.grad(e_total, [edge_vec])
        torch.as_tensor(lmp_data.eatoms).copy_(e_atoms.detach())
        lmp_data.energy = e_total.detach()
        lmp_data.update_pair_forces_gpu(grad.to(torch.float64))
        self.n_eager += 1
        return True

    # ---- helpers -----------------------------------------------------------
    def _new_species(self, elems) -> None:
        from sevenn.nn.lunatp_fused import prune_self_connection

        self._allowed[elems.to(torch.int64)] = True
        numbers = torch.nonzero(self._allowed).view(-1).tolist()
        names = prune_self_connection(
            self.model, [self.model.type_map[z] for z in numbers]
        )
        if self.verbose:
            print(
                f'[SevenNet] self connection of {len(names)} layers pruned to '
                f'atomic numbers {numbers}', flush=True,
            )

    def _disable(self, why: str) -> bool:
        self.enabled = False
        self._entry = None
        warnings.warn(f'SevenNet ML-IAP CUDA graph is off ({why}); eager calls')
        return False

    def _capture_pays(self) -> bool:
        return (
            self.n_captures < self.FREE_CAPTURES
            or self.n_replays >= self.REPLAYS_PER_CAPTURE * self.n_captures
        )

    @staticmethod
    def _fill_edge_index(out, pair_i, src, npairs: int, entry) -> None:
        """out (2, E_pad): the real edges, then self-loops on the fake atoms"""
        out[0, :npairs] = pair_i
        out[1, :npairs] = src
        k = out.shape[1] - npairs
        if k > 0:
            # pad p of k -> fake atom p * M // k (ascending: dst stays sorted)
            fake = entry['nlocal'] + torch.div(
                entry['arange'][:k] * entry['num_fake'], k, rounding_mode='floor'
            )
            out[0, npairs:] = fake
            out[1, npairs:] = fake

    def _refresh(self, entry, cand) -> bool:
        """
        The edge index cand becomes the edge index of the captured call. False
        if its LunaTP graph does not fit the captured tables.
        """
        if self._lunatp:
            graph = _lunatp_graph(
                cand, entry['num_nodes'], entry['num_segments'],
                entry['segment_edges'],
            )
            if graph is None:
                return False
            new = _graph_tensors(graph)
            old = entry['graph_tensors']
            if len(new) != len(old) or any(
                a.shape != b.shape or a.dtype != b.dtype for a, b in zip(old, new)
            ):
                return False
            for a, b in zip(old, new):
                a.copy_(b)
        entry['edge_index'].copy_(cand)
        self.n_refreshes += 1
        return True

    def _capture(self, nlocal, npairs, pair_i, src, elems):
        self.n_captures += 1
        self._entry = None  # frees the previous graph and its pool
        dev = self.device
        room = max(self.MIN_EDGE_ROOM, npairs // self._edge_room)
        num_edges = npairs + room
        num_fake = max(1, -(-2 * room // self.PAD_EDGES_PER_FAKE_ATOM))
        num_nodes = nlocal + num_fake
        entry: Dict[str, Any] = {
            'nlocal': nlocal, 'num_edges': num_edges, 'num_nodes': num_nodes,
            'num_fake': num_fake,
            'arange': torch.arange(num_edges, dtype=torch.int64, device=dev),
        }
        edge_index = torch.empty((2, num_edges), dtype=torch.int64, device=dev)
        self._fill_edge_index(edge_index, pair_i, src, npairs, entry)
        z = torch.empty(num_nodes, dtype=torch.int64, device=dev)
        z[:nlocal] = elems
        z[nlocal:] = elems[0]
        pad_vec = torch.tensor(
            [self.cutoff, 0.0, 0.0], dtype=self.dtype, device=dev
        )
        edge_vec = pad_vec.repeat(num_edges, 1).requires_grad_(True)
        num_atoms = torch.tensor(num_nodes, dtype=torch.int64, device=dev)

        if self._lunatp:
            # small graphs of the fused kernels: shorter source segments
            # (see lunatp_fused.SMALL_GRAPH_EDGES), 4 times the count
            seg_edges = 0
            if self._fused:
                from sevenn.nn.lunatp_fused import SMALL_GRAPH_EDGES

                if num_edges < SMALL_GRAPH_EDGES:
                    seg_edges = 8
            n_seg = _make_graph(
                edge_index, num_nodes, seg_edges
            ).source_segments.nodes.numel()
            scale = 4 if seg_edges else 1
            num_segments = n_seg + scale * (
                max(16, nlocal // self.SEGMENT_ROOM)
                + room // self.PAD_EDGES_PER_FAKE_ATOM + num_fake
            )
            entry['segment_edges'] = seg_edges
            graph = _lunatp_graph(edge_index, num_nodes, num_segments, seg_edges)
            # the key LunaTPSharedGraphConvolution looks the graph up by
            edge_index._lunatp_graph = (num_nodes, None, graph)
            entry['num_segments'] = num_segments
            entry['graph_tensors'] = _graph_tensors(graph)

        unique_cap = 0
        if self._fused:
            from sevenn.nn.lunatp_fused import KEY_N_UNIQUE, set_unique_cap

            unique_cap = num_edges
            if not self._unique_all:
                # half of the real pairs (+ slack), one row for the pads
                unique_cap = min(num_edges, npairs // 2 + npairs // 64 + 64)
            set_unique_cap(self.model, unique_cap)
            from sevenn.nn.lunatp_fused import index_coherence, set_index_order

            # launch order of the kernel blocks: node index order pays if
            # the atoms are in a spatially coherent order (see lunatp_fused)
            coherence = 1.0
            if nlocal >= self.INDEX_ORDER_MIN_ATOMS:
                coherence = index_coherence(edge_index[:, :npairs], nlocal)
            entry['coherence'] = coherence
            entry['index_order'] = coherence > self.INDEX_ORDER_COHERENCE
            set_index_order(self.model, entry['index_order'])
        entry['unique_cap'] = unique_cap

        def run():
            data = {
                KEY.EDGE_IDX: edge_index,
                KEY.EDGE_VEC: edge_vec,
                KEY.ATOMIC_NUMBERS: z,
                KEY.NUM_ATOMS: num_atoms,
            }
            if self.modal is not None:
                data[KEY.DATA_MODALITY] = self.modal
            output = self.model(data)
            e_atoms = output[KEY.ATOMIC_ENERGY].view(-1)[:nlocal]
            e_total = e_atoms.sum()
            (grad,) = torch.autograd.grad(e_total, [edge_vec])
            return (
                e_atoms.detach().to(torch.float64),
                e_total.detach().to(torch.float64),
                grad.to(torch.float64),
                output[KEY_N_UNIQUE] if self._fused else None,
            )

        graph = torch.cuda.CUDAGraph()
        if self._stream is None:
            self._stream = torch.cuda.Stream(dev)
        current = torch.cuda.current_stream(dev)
        side = self._stream
        side.wait_stream(current)
        try:
            with torch.cuda.stream(side):
                run()  # warm-up: lazy inits, caches on the static inputs
                graph.capture_begin()
                try:
                    e_atoms, e_total, pair_forces, n_unique = run()
                finally:
                    graph.capture_end()
        except Exception as e:
            current.wait_stream(side)
            warnings.warn(f'CUDA graph capture failed ({type(e).__name__}: {e})')
            return None
        current.wait_stream(side)
        entry.update(
            graph=graph, edge_index=edge_index, edge_vec=edge_vec, z=z,
            pad_vec=pad_vec, cand=edge_index.clone(),
            e_atoms=e_atoms, e_total=e_total, pair_forces=pair_forces,
            n_unique=n_unique,
        )
        return entry
