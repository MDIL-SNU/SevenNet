"""`sevenn patch_lammps` on a Mac: copy the Apple GPU pair style into a LAMMPS source
tree and link it to this Python environment (src/patch_lammps.sh)."""

import os.path as osp
import subprocess
import sys

import torch

from .. import _policy

SRC = osp.join(osp.dirname(osp.abspath(__file__)), 'src')


def detected(cuda_only_options: bool) -> bool:
    """A Mac with an Apple GPU that this PyTorch can run on, no CUDA, no CUDA-only
    option (--d3, --enable_flash, --enable_oeq) and no SEVENN_DISABLE_MPS."""
    if cuda_only_options or sys.platform != 'darwin' or torch.cuda.is_available():
        return False
    if not torch.backends.mps.is_available() or _policy.disabled():
        return False
    return _policy.torch_supported()


def patch(lammps_dir: str) -> int:
    root = osp.abspath(osp.join(SRC, '..', '..', '..', '..'))
    script = osp.join(SRC, 'patch_lammps.sh')
    cmd = ['bash', script, lammps_dir, sys.executable, root]
    return subprocess.run(cmd).returncode
