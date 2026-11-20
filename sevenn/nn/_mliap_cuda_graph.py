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
import dataclasses
import os
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


def _lunatp_graph(edge_index: torch.Tensor, num_nodes: int, num_segments: int):
    """
    LunaTPGraph of a dst-sorted edge index whose source segment tables are
    padded to num_segments with empty segments of the last node, or None if
    the graph has more segments.
    """
    from LunaEquivariance import make_graph

    graph = make_graph(edge_index[1], edge_index[0], num_nodes)
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
        self._stream = None
        # a value the owner of a ghost must return: another rank has another one
        self._token = float(int.from_bytes(os.urandom(6), 'little'))
        from sevenn.nn.lunatp_helper import LunaTPConvolution

        self._lunatp = any(
            isinstance(m, LunaTPConvolution) for m in model.modules()
        )

    # ---- per call ----------------------------------------------------------
    def _owners(self, lmp_data, nlocal: int, ntotal: int):
        """(owner local index of every atom, flag: a ghost of another rank)"""
        send = torch.full((ntotal, 2), -1.0, dtype=torch.float64, device=self.device)
        send[:nlocal, 0] = torch.arange(
            nlocal, dtype=torch.float64, device=self.device
        )
        send[:nlocal, 1] = self._token
        recv = torch.empty_like(send)
        lmp_data.forward_exchange(send, recv, 2)
        return recv[:, 0].to(torch.int64), (recv[:, 1] != self._token).any()

    def compute(self, lmp_data) -> bool:
        if not self.enabled:
            return False
        nlocal, ntotal, npairs = lmp_data.nlocal, lmp_data.ntotal, lmp_data.npairs
        if npairs <= 1 or nlocal == 0:
            return False
        dev = self.device
        owner, foreign = self._owners(lmp_data, nlocal, ntotal)
        pair_i = torch.as_tensor(lmp_data.pair_i, device=dev)
        pair_j = torch.as_tensor(lmp_data.pair_j, device=dev)
        src = owner.index_select(0, pair_j.to(torch.int64))
        elems = torch.as_tensor(lmp_data.elems, device=dev)[:nlocal]
        if self.eager_fold:
            if bool(foreign):
                return self._disable('a ghost atom is owned by another rank')
            return self._eager(lmp_data, nlocal, pair_i, src, elems)

        entry = self._entry
        fits = (
            entry is not None
            and entry['nlocal'] == nlocal
            and npairs <= entry['num_edges']
        )
        if fits:
            cand = entry['cand']
            self._fill_edge_index(cand, pair_i, src, npairs, entry)
            flags = torch.stack(
                [foreign, (cand != entry['edge_index']).any()]
            ).tolist()  # the one host sync before the replay
            if flags[0]:
                return self._disable('a ghost atom is owned by another rank')
            if flags[1] and not self._refresh(entry, cand):
                fits = False
        elif bool(foreign):
            return self._disable('a ghost atom is owned by another rank')
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
                    f'(room {entry["num_edges"]}), capture {self.n_captures}',
                    flush=True,
                )
        else:
            self.n_replays += 1

        with torch.no_grad():
            entry['z'][:nlocal] = elems
            vec = entry['edge_vec']
            vec[:npairs] = torch.as_tensor(lmp_data.rij, device=dev)
            vec[npairs:] = entry['pad_vec']
        entry['graph'].replay()

        torch.as_tensor(lmp_data.eatoms).copy_(entry['e_atoms'])
        lmp_data.energy = entry['e_total']
        lmp_data.update_pair_forces_gpu(entry['pair_forces'])
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
            graph = _lunatp_graph(cand, entry['num_nodes'], entry['num_segments'])
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
            exact = _lunatp_graph(edge_index, num_nodes, 0)
            n_seg = 0
            if exact is None:  # always: a graph has at least one segment
                from LunaEquivariance import make_graph

                n_seg = make_graph(
                    edge_index[1], edge_index[0], num_nodes
                ).source_segments.nodes.numel()
            num_segments = (
                n_seg + max(16, nlocal // self.SEGMENT_ROOM)
                + room // self.PAD_EDGES_PER_FAKE_ATOM + num_fake
            )
            graph = _lunatp_graph(edge_index, num_nodes, num_segments)
            # the key LunaTPSharedGraphConvolution looks the graph up by
            edge_index._lunatp_graph = (num_nodes, None, graph)
            entry['num_segments'] = num_segments
            entry['graph_tensors'] = _graph_tensors(graph)

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
                    e_atoms, e_total, pair_forces = run()
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
        )
        return entry
