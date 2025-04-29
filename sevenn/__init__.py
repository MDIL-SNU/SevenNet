from importlib.metadata import version

from packaging.version import Version

__version__ = version('sevenn')

from e3nn import __version__ as e3nn_ver

if Version(e3nn_ver) < Version('0.5.0'):
    raise ValueError(
        'The e3nn version MUST be 0.5.0 or later due to changes in CG coefficient '
        'convention.'
    )
