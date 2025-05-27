from typing import Callable

from sevenn.nn.convolution import IrrepsConvolution

try:
    from flashTP_e3nn import uvu_TP

    _FLASH_AVAILABLE = True

except ImportError:
    _FLASH_AVAILABLE = False


def is_flash_available() -> bool:
    return _FLASH_AVAILABLE


def flash_needed(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if is_flash_available():
            return func(*args, **kwargs)
        else:
            raise ImportError('cue is not available')

    return wrapper


@flash_needed
def patch_convolution(irreps_convolution: IrrepsConvolution):

    assert not irreps_convolution.layer_instantiated

    irreps_convolution.convolution_cls = uvu_TP  # type: ignore

    del irreps_convolution.convolution_kwargs['shared_weights']
    del irreps_convolution.convolution_kwargs['internal_weights']
    return irreps_convolution
