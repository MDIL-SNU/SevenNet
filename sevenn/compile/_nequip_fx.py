"""Util code copied from NequIP 0.9.0

Source: https://github.com/mir-group/nequip
Paper: https://arxiv.org/abs/2504.16068
"""

import contextlib

import torch
from torch.fx.experimental.proxy_tensor import make_fx


@contextlib.contextmanager
def fx_duck_shape(enabled: bool):
    """
    For our use of `make_fx` to unfold the autograd graph, we must set the following
    `use_duck_shape` parameter to `False` (it's `True` by default).
    It forces dynamic batch dims (num_frames, num_atoms, num_edges) to shape
    specialize if the batch dim is the same as that of a static dim.
    E.g. in training, shape specialization would occur if a weight tensor has a
    dimension with shape (16,) and we use a batch size of 16
    (so the dynamic batch dim `num_frames` is 16) because of the duck shaping.
    """
    # save previous state
    init_duck_shape = torch.fx.experimental._config.use_duck_shape
    # set mode variables
    torch.fx.experimental._config.use_duck_shape = enabled
    try:
        yield
    finally:
        # restore state
        torch.fx.experimental._config.use_duck_shape = init_duck_shape


def nequip_make_fx(model, inputs):
    with fx_duck_shape(False):
        return make_fx(
            model,
            tracing_mode='symbolic',
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[i.clone() for i in inputs])
