import pathlib
from typing import Optional, Union

import torch.cuda

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified

    _MLIAP_AVAILABLE = True

except ImportError:
    _MLIAP_AVAILABLE = False

_DEPLOY_MLIAP = False  # passed to sevenn.nn.convolution


def is_mliap_available() -> bool:
    print('_MLIAP_AVAILABLE:', _MLIAP_AVAILABLE)
    return _MLIAP_AVAILABLE and torch.cuda.is_available()


def deploy_mliap(
    checkpoint: Union[pathlib.Path, str],
    fname='deployed_serial',
    modal: Optional[str] = None,
    use_flash: bool = False,
    use_cueq: bool = False,
) -> None:
    from sevenn.lmp_mliap_wrapper import SevenNetMLIAPWrapper

    if fname.endswith('.pt') is False:
        fname += '.pt'

    mliap_module = SevenNetMLIAPWrapper(
        model_path=checkpoint,
        modal=modal,
        use_cueq=use_cueq,
        use_flash=use_flash,
    )
    torch.save(mliap_module, fname)
