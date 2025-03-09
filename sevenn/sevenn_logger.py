import warnings

from .logger import *  # noqa: F403

warnings.warn('Please use sevenn.logger instead of sevenn.sevenn_logger',
              DeprecationWarning, stacklevel=2)
