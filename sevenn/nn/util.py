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
    key = (w.data_ptr(), w._version, scale)
    cached = getattr(w, '_sevenn_frozen_product', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    out = w * scale
    w._sevenn_frozen_product = (key, out)
    return out
