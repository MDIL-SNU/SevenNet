"""Edge padding on the Apple GPU. PyTorch's MPS backend compiles a graph for every
new tensor shape and keeps it for the life of the process; in MD almost every step,
and in training almost every shuffled batch, has a new number of edges. Without
padding a training process grows by one to two gigabytes a minute.

Pad edges go from atom 0 to atom 0 with vector (cutoff, 0, 0), after the real edges.
The cutoff function vanishes there, so their radial weights are exactly zero
(padding_is_safe checks this on the actual model), and the fused kernels do not
visit them at all (the edge plan is told how many edges are real). Energy, forces,
stress, every other output and every gradient are those of the real edges; unpad()
drops the pad edges from the outputs.
"""

import warnings
from typing import Optional

import torch

import sevenn._keys as KEY


def _is_edge_key(key: str) -> bool:
    """Per-edge tensors of a graph or an output, by name: never a per-atom tensor,
    even when a small batch has as many atoms as edges."""
    return key in (
        KEY.EDGE_VEC,
        KEY.EDGE_LENGTH,
        KEY.EDGE_ATTR,
        KEY.EDGE_EMBEDDING,
        KEY.CELL_SHIFT,
    ) or key.startswith('edge_')


def pad(data, cutoff: float, bucket: int, plan) -> Optional[int]:
    """Pad the edges of `data` (a graph or a batch, in place) to a multiple of
    `bucket` and tell the edge plan. Returns the number of real edges, or None if
    nothing was padded."""
    idx = data[KEY.EDGE_IDX]
    n_real = int(idx.shape[1])
    n_pad = (-n_real) % bucket if bucket > 1 else 0
    if n_pad == 0 or n_real == 0:
        return None
    for key in list(data.keys()):
        t = data[key]
        if (
            key == KEY.EDGE_IDX
            or not _is_edge_key(key)
            or not isinstance(t, torch.Tensor)
            or t.dim() == 0
            or t.shape[0] != n_real
        ):
            continue
        filler = t.new_zeros((n_pad,) + tuple(t.shape[1:]))
        if key == KEY.EDGE_VEC:
            filler[:, 0] = cutoff
        data[key] = torch.cat([t, filler], 0)
    data[KEY.EDGE_IDX] = torch.cat([idx, idx.new_zeros((2, n_pad))], 1)
    plan.expect_padded = (n_real, n_real + n_pad)
    return n_real


def unpad(out, n_real: int):
    """Drop the pad edges from a model output (in place); n_real is the edge count
    before padding."""
    n_all = int(out[KEY.EDGE_IDX].shape[1])
    if n_all == n_real:
        return out
    for key in list(out.keys()):
        t = out[key]
        if key == KEY.EDGE_IDX:
            out[key] = t[:, :n_real]
        elif (
            _is_edge_key(key)
            and isinstance(t, torch.Tensor)
            and t.dim() > 0
            and t.shape[0] == n_all
        ):
            out[key] = t[:n_real]
    return out


def padding_is_safe(model, cutoff: float, all_fused: bool = False) -> bool:
    """True if a pad edge (r = cutoff) gives exactly zero radial weights in every
    convolution, with finite derivatives.

    These derivatives are float32 noise, not 0 (seen up to 5e-5 in 7net-l3i5). With
    all convolutions fused the noise goes nowhere, because the kernels do not visit
    pad edges. A convolution left on e3nn would multiply it by dE/dw (forces cancel,
    as source = destination, but the stress does not), so without all_fused the
    derivatives must also stay below 1e-5.

    """
    # derivatives are part of the check: also when the model moves inside no_grad
    with torch.enable_grad():
        return _padding_is_safe(model, cutoff, all_fused)


def _padding_is_safe(model, cutoff: float, all_fused: bool) -> bool:
    device = next(model.parameters()).device
    vec = torch.tensor(
        [[cutoff, 0.0, 0.0]], dtype=torch.float32, device=device, requires_grad=True
    )
    if 'edge_embedding' not in model._modules:
        warnings.warn('edge padding disabled: the model has no edge_embedding')
        return False
    data = model._modules['edge_embedding']({KEY.EDGE_VEC: vec})
    emb = data[KEY.EDGE_EMBEDDING]
    tensors = [emb] + [
        m.weight_nn(emb)
        for m in model.modules()
        if getattr(m, 'weight_nn', None) is not None
    ]
    for t in tensors:
        if float(t.detach().abs().max()) != 0.0:
            return False
        for i in range(
            0, t.shape[1], max(t.shape[1] // 8, 1)
        ):  # a sample of components; the sum could cancel
            d = torch.autograd.grad(t[:, i].sum(), vec, retain_graph=True)[0]
            if not bool(torch.isfinite(d).all()) or (
                not all_fused and float(d.abs().max()) > 1e-5
            ):
                return False
    return True
