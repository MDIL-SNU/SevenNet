"""When the Apple GPU code runs: the opt-out and the PyTorch requirements. Imports
nothing from torch.mps, so it is safe to import on any machine."""

import os
from typing import Optional

import torch

# SEVENN_DISABLE_MPS=1 (or true, yes) keeps the reference e3nn code on the Apple GPU,
# for reproducibility or debugging.
OPT_OUT = 'SEVENN_DISABLE_MPS'


def disabled() -> bool:
    return os.environ.get(OPT_OUT, '').strip().lower() in ('1', 'true', 'yes')


_TORCH_OK: Optional[bool] = None


def torch_supported() -> bool:
    """Whether this PyTorch can run SevenNet on the Apple GPU at all. The message
    scatter of the e3nn code (scatter_reduce) has an MPS kernel from PyTorch 2.6."""
    global _TORCH_OK
    if _TORCH_OK is None:
        try:
            x = torch.zeros(2, dtype=torch.float32, device='mps')
            idx = torch.zeros(1, dtype=torch.long, device='mps')
            one = torch.ones(1, dtype=torch.float32, device='mps')
            x.scatter_reduce_(0, idx, one, reduce='sum')
            _TORCH_OK = True
        except (NotImplementedError, RuntimeError):
            _TORCH_OK = False
    return _TORCH_OK
