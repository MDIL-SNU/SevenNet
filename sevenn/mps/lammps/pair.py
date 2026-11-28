"""Python side of pair style e3gnn on the Apple GPU (src/): owns the
calculator and computes for LAMMPS.

pair_coeff * * checkpoint.pth [keywords] element1 element2 ...
Keywords as in the NVIDIA pair style: modal NAME selects the fidelity of a
multi-fidelity model (see KEYWORDS for the others).

"""

from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers

from sevenn.calculator import SevenNetCalculator

_VIRIAL = [
    0,
    1,
    2,
    5,
    4,
    3,
]  # ASE voigt xx yy zz yz xz xy to LAMMPS xx yy zz xy xz yz


def _cell(box) -> np.ndarray:
    lo, hi, xy, xz, yz, _ = box
    return np.array(
        [
            [hi[0] - lo[0], 0.0, 0.0],
            [xy, hi[1] - lo[1], 0.0],
            [xz, yz, hi[2] - lo[2]],
        ]
    )


def _modal(args: List[str], i: int, out: dict) -> int:
    if i + 1 == len(args):
        raise ValueError('e3gnn: modal requires a name')
    out['modal'] = args[i + 1]
    return i + 2


# pair_coeff keywords before the element list: name -> reader(args, i, out) returning
# the index after its values
KEYWORDS: Dict[str, Callable[[List[str], int, dict], int]] = {'modal': _modal}


def _plain(calc, opt: dict):
    """The calculator LAMMPS calls, and a note for the log."""
    return calc, ''


WRAP = _plain


def _parse(args: List[str]) -> dict:
    out: dict = dict(modal=None)
    i = 0
    while i < len(args) and args[i] in KEYWORDS:
        i = KEYWORDS[args[i]](args, i, out)
    out['symbols'] = args[i:]
    bad = [s for s in out['symbols'] if s not in atomic_numbers]
    if bad:
        raise ValueError(
            f'e3gnn: {bad[0]} is neither a keyword ({", ".join(KEYWORDS)}) nor an '
            f'element symbol'
        )
    return out


class PairDriver:
    def __init__(self, args: List[str], ntypes: int, box: Sequence) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                'e3gnn: this pair style needs the Apple GPU, and MPS is not '
                'available'
            )
        path, opt = args[0], _parse(args[1:])
        if path.endswith('.pt'):
            raise ValueError(
                f'e3gnn: {path} looks like a deployed model. TorchScript cannot '
                f'hold the Metal kernels: give the .pth checkpoint or a pretrained '
                f'name'
            )
        if len(opt['symbols']) < ntypes:
            raise ValueError(
                f'e3gnn: not enough elements in pair_coeff: {ntypes} atom types, '
                f'got {opt["symbols"]}'
            )
        self.base = SevenNetCalculator(path, device='mps', modal=opt['modal'])
        known = set(self.base.type_map)
        unknown = [s for s in opt['symbols'] if atomic_numbers.get(s) not in known]
        if unknown:
            raise ValueError(f'e3gnn: elements unknown to this model: {unknown}')
        self.calc, note = WRAP(self.base, opt)
        self.numbers = np.array([0] + [atomic_numbers[s] for s in opt['symbols']])
        self.cutoff = float(self.base.cutoff)
        from sevenn.mps import report

        run = report(self.base.model) or 'reference code'
        self.banner = (
            f'e3gnn on the Apple GPU: {path}, cutoff {self.cutoff} A{note}, '
            f'elements {" ".join(opt["symbols"])}\n{run}'
        )

    def compute(self, nlocal: int, x, f, types, eatom, virial, box) -> float:
        if (
            nlocal == 0
        ):  # all atoms lost during a run, with thermo_modify lost ignore
            return 0.0
        lo, _, _, _, _, periodic = box
        forces = np.frombuffer(f, dtype=np.float64).reshape(nlocal, 3)
        out = np.frombuffer(virial, dtype=np.float64)
        atoms = Atoms(
            numbers=self.numbers[np.frombuffer(types, dtype=np.int32)],
            positions=np.frombuffer(x, dtype=np.float64).reshape(nlocal, 3)
            - np.array(lo),
            cell=_cell(box),
            pbc=[bool(p) for p in periodic],
        )
        atoms.calc = self.calc
        energy = atoms.get_potential_energy()
        forces += atoms.get_forces()
        out[:] = -(atoms.get_stress(voigt=True) * atoms.get_volume())[_VIRIAL]
        if eatom is not None:
            np.frombuffer(eatom, dtype=np.float64)[:] += atoms.calc.results[
                'energies'
            ]
        return float(energy)
