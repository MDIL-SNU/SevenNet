"""Serial LAMMPS with a SevenNet calculator through `fix external`, driven from
Python. For plain `lmp -in` on a Mac see the pair style in src/.

LAMMPS runs the dynamics and the calculator gives energy, forces and virial at each
step, so anything an ASE calculator supports works, among them the Metal kernels.
In the input, replace the pair style by:

    pair_style   zero 6.0
    pair_coeff   * *
    fix          sevenn all external pf/callback 1 1
    fix_modify   sevenn energy yes virial yes

The fix needs the callback before the first `run`, so the input goes through
run_input() line by line. Serial only.

If the calculator fails, LAMMPS cannot see the Python error: the callback then asks
LAMMPS to stop the run at the next step, writes NaN forces, and run_input() raises
the original error once the command returns. After attach() alone, call the
function it returns after each command.

"""

import sys
import traceback
from typing import Callable, List, Sequence

import numpy as np
from ase import Atoms


def attach(
    lmp, calc, symbols: Sequence[str], fix_id: str = 'sevenn'
) -> Callable[[], None]:
    """Make fix `fix_id` of `lmp` use `calc`. symbols[i] is the element of LAMMPS
    atom type i + 1. Returns a function that raises the calculator's error, if
    there was one; call it after each command that runs dynamics."""
    if lmp.extract_setting('world_size') != 1:
        raise ValueError(
            'the fix external bridge is serial: run LAMMPS on one MPI rank'
        )
    units = lmp.extract_global('units')
    if units != 'metal':
        raise ValueError(f'the fix external bridge needs units metal, got {units}')
    types = np.array(['X'] + list(symbols))
    failed: List = []

    def callback(_caller, step, nlocal, _tag, x, f):
        if failed:  # the run is stopping: nothing valid to give
            lmp.force_timeout()
            return
        try:
            energy, forces, virial = evaluate(nlocal, x)
            forces = np.asarray(forces, dtype=np.float64)
            if forces.shape != (nlocal, 3) or len(virial) != 6:
                raise ValueError(
                    f'the calculator gave forces of shape {forces.shape} and '
                    f'{len(virial)} virial components for {nlocal} atoms'
                )
            # published only once all results exist and have the right shape
            f[:nlocal] = forces
            lmp.fix_external_set_energy_global(fix_id, energy)
            lmp.fix_external_set_virial_global(fix_id, virial)
        except Exception as e:  # an exception would not reach LAMMPS
            failed.append((step, e))
            traceback.print_exc(file=sys.stderr)
            f[:nlocal] = np.full((nlocal, 3), np.nan)
            lmp.force_timeout()

    def evaluate(nlocal, x):
        if nlocal == 0:  # all atoms lost, with thermo_modify lost ignore
            return 0.0, np.zeros((0, 3)), [0.0] * 6
        boxlo, boxhi, xy, yz, xz, periodicity, _ = lmp.extract_box()
        lo, hi = np.array(boxlo), np.array(boxhi)
        cell = np.array(
            [
                [hi[0] - lo[0], 0.0, 0.0],
                [xy, hi[1] - lo[1], 0.0],
                [xz, yz, hi[2] - lo[2]],
            ]
        )
        ctype = lmp.extract_atom('type')
        sym = types[
            np.fromiter((ctype[i] for i in range(nlocal)), dtype=int, count=nlocal)
        ]
        atoms = Atoms(
            symbols=sym.tolist(),
            positions=np.array(x[:nlocal]) - lo,
            cell=cell,
            pbc=[bool(p) for p in periodicity],
        )
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        s = (
            atoms.get_stress(voigt=True) * atoms.get_volume()
        )  # xx yy zz yz xz xy, eV
        return energy, forces, [-s[0], -s[1], -s[2], -s[5], -s[4], -s[3]]

    def check() -> None:
        if failed:
            step, e = failed[0]
            raise RuntimeError(
                f'the calculator failed at step {step}: {e}. LAMMPS stopped the '
                f'run there; that step has NaN forces and is not valid'
            ) from e

    lmp.set_fix_external_callback(fix_id, callback)
    return check


def run_input(
    lmp, path: str, calc, symbols: Sequence[str], fix_id: str = 'sevenn'
) -> List[str]:
    """Run a LAMMPS input file, attaching `calc` right after the fix is defined.
    Returns the commands run."""
    done: List[str] = []
    attached = False
    check: Callable[[], None] = lambda: None  # noqa: E731
    pending = ''
    with open(path) as handle:
        for raw in handle:
            line = pending + raw.rstrip('\n')
            if line.rstrip().endswith('&'):
                pending = line.rstrip()[:-1]
                continue
            pending = ''
            words = line.split('#')[
                0
            ].split()  # the full line goes to LAMMPS, which knows about quotes
            if not words:
                continue
            lmp.command(line)
            check()
            done.append(line)
            if words[0] == 'fix' and len(words) > 1 and words[1] == fix_id:
                check = attach(lmp, calc, symbols, fix_id)
                attached = True
    if not attached:
        raise ValueError(f'no "fix {fix_id} all external pf/callback 1 1" in {path}')
    return done
