from typing import Callable

import torch.cuda

from sevenn.nn.convolution import IrrepsConvolution

try:
    from flashTP_e3nn import uvu_TP

    _FLASH_AVAILABLE = True

except ImportError:
    _FLASH_AVAILABLE = False


def is_flash_available() -> bool:
    return _FLASH_AVAILABLE and torch.cuda.is_available()


def flash_needed(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if is_flash_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(
                'FlashTP is not available (no GPU found or import fail)'
            )

    return wrapper


@flash_needed
def patch_convolution(
    irreps_convolution: IrrepsConvolution, _flash_lammps: bool = False
):
    from sevenn.nn.convolution import IrrepsScatterGatterFusedConvolution

    assert not irreps_convolution.layer_instantiated

    ret = IrrepsScatterGatterFusedConvolution.from_irreps_convolution(
        irreps_convolution
    )
    ret.convolution_cls = uvu_TP  # type: ignore
    ret.convolution_kwargs['use_lammps'] = _flash_lammps
    del ret.convolution_kwargs['shared_weights']
    del ret.convolution_kwargs['internal_weights']

    return ret
