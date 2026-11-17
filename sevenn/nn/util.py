import torch


def broadcast(
    src: torch.Tensor,
    other: torch.Tensor,
    dim: int
) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


@torch.jit.unused
def frozen_product(w: torch.Tensor, scale: float) -> torch.Tensor:
    """
    w * scale, computed once while w is frozen (requires_grad False, e.g. the
    calculator's model) and kept on w, keyed by its storage address, version
    counter (bumped by every in-place update, e.g. load_state_dict) and scale:
    the same tensor values as w * scale, one kernel less per call. Recomputed
    every call when w is trainable (training) or in inference mode (a tensor
    made there cannot enter a later autograd graph). Python only: TorchScript
    callers compute w * scale (torch.jit.is_scripting()).
    """
    if w.requires_grad or torch.is_inference_mode_enabled():
        return w * scale
    return _frozen(w, ('mul', scale), lambda: w * scale)


@torch.jit.unused
def frozen_tanh(w: torch.Tensor) -> torch.Tensor:
    """torch.tanh(w), kept on a frozen w as frozen_product does"""
    if w.requires_grad or torch.is_inference_mode_enabled():
        return torch.tanh(w)
    return _frozen(w, ('tanh',), lambda: torch.tanh(w))


def _frozen(w, op, compute):
    key = (w.data_ptr(), w._version)
    cache = w.__dict__.setdefault('_sevenn_frozen', {})
    cached = cache.get(op)
    if cached is None or cached[0] != key:
        cached = cache[op] = (key, compute())
    return cached[1]
