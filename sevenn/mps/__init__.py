"""Apple GPU (MPS) backend of SevenNet: fused Metal kernels for the e3nn layers, edge
padding, and the LAMMPS pair style. Nothing here is imported unless a model moves to
the Apple GPU: AtomGraphSequential._apply calls runtime.apply() for that move, and
moving the model away undoes it. SEVENN_DISABLE_MPS=1 keeps the reference e3nn code.
"""

from ._policy import disabled

__all__ = ['disabled', 'prepare', 'report', 'restore']


def __getattr__(name):
    # the kernels load on first use, not on `import sevenn.mps._policy`
    if name in ('prepare', 'report', 'restore'):
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
