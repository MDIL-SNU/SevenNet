import pathlib
from typing import Optional, Union

import torch.cuda

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified

    _MLIAP_AVAILABLE = True

except ImportError:
    _MLIAP_AVAILABLE = False


def is_mliap_available() -> bool:
    return _MLIAP_AVAILABLE and torch.cuda.is_available()


def deploy_mliap(
    checkpoint: Union[pathlib.Path, str],
    fname='deployed_serial',
    modal: Optional[str] = None,
    use_flash: bool = False,
    use_cueq: bool = False,
) -> None:
    from .lmp_mliap_wrapper import SevenNetMLIAPWrapper

    if fname.endswith('.pt') is False:
        fname = fname.with_suffix(fname.suffix + '.pt')

    fname.parent.mkdir(parents=True, exist_ok=True)
    mliap_module = SevenNetMLIAPWrapper(
        model_path=checkpoint,
        modal=modal,
        use_cueq=use_cueq,
        use_flash=use_flash,
    )
    torch.save(mliap_module, fname)
