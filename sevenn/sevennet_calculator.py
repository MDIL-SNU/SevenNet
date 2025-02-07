import warnings

from .calculator import *  # noqa: F403

warnings.warn('Please use sevenn.calculator instead of sevenn.sevennet_calculator',
              DeprecationWarning, stacklevel=2)
