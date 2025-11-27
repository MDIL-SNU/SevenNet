import os
from importlib.metadata import version
from warnings import warn

from packaging.version import Version

__version__ = version('sevenn')

from e3nn import __version__ as e3nn_ver

if Version(e3nn_ver) < Version('0.5.0'):
    raise ValueError(
        'The e3nn version MUST be 0.5.0 or later due to changes in CG coefficient '
        'convention.'
    )


if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
    warn(
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is enabled. '
        'This may alter numerical behavior of sevennet.'
    )
