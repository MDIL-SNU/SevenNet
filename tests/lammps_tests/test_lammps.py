import copy
import logging
import pathlib
import subprocess

import ase.calculators.lammps
import ase.io.lammpsdata
import numpy as np
import pytest
from ase.build import bulk, surface
from ase.calculators.singlepoint import SinglePointCalculator

from sevenn.scripts.deploy import deploy, deploy_parallel
from sevenn.sevennet_calculator import SevenNetCalculator
from sevenn.util import model_from_checkpoint

logger = logging.getLogger('test_lammps')

cutoff = 4.0

lmp_script_path = str(
    (pathlib.Path(__file__).parent / 'scripts' / 'skel.lmp').resolve()
)

data_root = (pathlib.Path(__file__).parent.parent / 'data').resolve()
hfo2_path = str(data_root / 'systems' / 'hfo2.extxyz')
cp_0_path = str(data_root / 'checkpoints' / 'cp_0.pth')  # knows Hf, O


@pytest.fixture(scope='module')
def serial_potential_path(tmp_path_factory):
    model, config = model_from_checkpoint(cp_0_path)
    tmp = tmp_path_factory.mktemp('serial_potential')
    pot_path = str(tmp / 'deployed_serial.pt')
    deploy(model.state_dict(), config, pot_path)
    return pot_path


@pytest.fixture(scope='module')
def parallel_potential_path(tmp_path_factory):
    model, config = model_from_checkpoint(cp_0_path)
    tmp = tmp_path_factory.mktemp('paralllel_potential')
    pot_path = str(tmp / 'deployed_parallel')
    deploy_parallel(model.state_dict(), config, pot_path)
    return ' '.join(['3', pot_path])


@pytest.fixture(scope='module')
def ref_calculator():
    return SevenNetCalculator(cp_0_path)


def hfo2_bulk(replicate=(2, 2, 2), a=3.0):
    atoms = bulk('HfO', 'rocksalt', a, orthorhombic=True)
    atoms = atoms * replicate
    atoms.rattle(stdev=0.10)
    return atoms


def hf_surface(replicate=(3, 3, 1), layers=4, vacuum=0.5):
    atoms = surface('Al', (1, 0, 0), layers=layers, vacuum=vacuum)
    atoms.set_atomic_numbers([72] * len(atoms))  # Hf
    atoms = atoms * replicate
    atoms.rattle(stdev=0.10)
    return atoms


def get_system(system_name, **kwargs):
    if system_name == 'bulk':
        return hfo2_bulk(**kwargs)
    elif system_name == 'surface':
        return hf_surface(**kwargs)
    else:
        raise ValueError()


def assert_atoms(atoms1, atoms2, rtol=1e-5, atol=1e-6):
    def acl(a, b, rtol=rtol, atol=atol):
        return np.allclose(a, b, rtol=rtol, atol=atol)

    assert len(atoms1) == len(atoms2)
    assert acl(atoms1.get_cell(), atoms2.get_cell())
    assert acl(atoms1.get_potential_energy(), atoms2.get_potential_energy())
    assert acl(atoms1.get_forces(), atoms2.get_forces(), rtol * 10, atol * 10)
    assert acl(
        atoms1.get_stress(voigt=False),
        atoms2.get_stress(voigt=False),
        rtol * 10,
        atol * 10,
    )
    # assert acl(atoms1.get_potential_energies(), atoms2.get_potential_energies())


def _lammps_results_to_atoms(lammps_log, force_dump):
    with open(lammps_log, 'r') as f:
        lines = f.readlines()
    lmp_log = None
    for i, line in enumerate(lines):
        if not line.startswith('Per MPI rank memory allocation'):
            continue
        lmp_log = {
            k: eval(v) for k, v in zip(lines[i + 1].split(), lines[i + 2].split())
        }
        break

    assert lmp_log is not None and 'PotEng' in lmp_log

    latoms_list = ase.io.read(force_dump, format='lammps-dump-text', index=':')
    assert isinstance(latoms_list, list)
    latoms = latoms_list[0]
    assert latoms.calc is not None
    latoms.calc.results['energy'] = lmp_log['PotEng']
    latoms.calc.results['free_energy'] = lmp_log['PotEng']
    latoms.info = {
        'data_from': 'lammps',
        'lmp_log': lmp_log,
        'lmp_dump': force_dump,
    }
    # atomic energy read
    latoms.calc.results['energies'] = latoms.arrays['c_pa'][:, 0]
    stress = np.array(
        [
            [lmp_log['Pxx'], lmp_log['Pxy'], lmp_log['Pxz']],
            [lmp_log['Pxy'], lmp_log['Pyy'], lmp_log['Pyz']],
            [lmp_log['Pxz'], lmp_log['Pyz'], lmp_log['Pzz']],
        ]
    )
    stress = -1 * stress / 1602.1766208 / 1000  # convert bars to eV/A^3
    latoms.calc.results['stress'] = stress

    return latoms


def _run_lammps(atoms, pair_style, potential, wd, command, test_name):
    wd = wd.resolve()
    pbc = atoms.get_pbc()
    pbc_str = ' '.join(['p' if x else 'f' for x in pbc])
    chem = list(set(atoms.get_chemical_symbols()))
    # Way to ase handle lammps structure

    prism = ase.calculators.lammps.coordinatetransform.Prism(
        atoms.get_cell(), pbc=pbc
    )
    lmp_stct = wd / 'lammps_structure'
    ase.io.lammpsdata.write_lammps_data(
        lmp_stct, atoms, prismobj=prism, specorder=chem
    )

    with open(lmp_script_path, 'r') as f:
        cont = f.read()

    lammps_log = str(wd / 'log.lammps')
    force_dump = str(wd / 'force.dump')

    var_dct = {}
    var_dct['__ELEMENT__'] = ' '.join(chem)
    var_dct['__LMP_STCT__'] = str(lmp_stct.resolve())
    var_dct['__PAIR_STYLE__'] = pair_style
    var_dct['__POTENTIALS__'] = potential
    var_dct['__BOUNDARY__'] = pbc_str
    var_dct['__FORCE_DUMP_PATH__'] = force_dump
    for key, val in var_dct.items():
        cont = cont.replace(key, val)

    input_script_path = str(wd / 'in.lmp')
    with open(input_script_path, 'w') as f:
        f.write(cont)

    command = f'{command} -in {input_script_path} -log {lammps_log}'
    subprocess_routine(command.split(), test_name)

    lmp_atoms = _lammps_results_to_atoms(lammps_log, force_dump)
    assert lmp_atoms.calc is not None

    rot_mat = prism.rot_mat
    results = copy.deepcopy(lmp_atoms.calc.results)
    r_force = np.dot(results['forces'], rot_mat.T)
    results['forces'] = r_force
    if 'stress' in results:
        # see ase.calculators.lammpsrun.py
        stress_tensor = results['stress']
        stress_atoms = np.dot(np.dot(rot_mat, stress_tensor), rot_mat.T)
        results['stress'] = stress_atoms
    r_cell = lmp_atoms.get_cell() @ rot_mat.T
    lmp_atoms.set_cell(r_cell, scale_atoms=True)
    lmp_atoms = SinglePointCalculator(lmp_atoms, **results).get_atoms()

    return lmp_atoms


def serial_lammps_run(atoms, potential, wd, test_name, lammps_cmd):
    command = lammps_cmd
    return _run_lammps(atoms, 'e3gnn', potential, wd, command, test_name)


def parallel_lammps_run(
    atoms, potential, wd, test_name, ncores, lammps_cmd, mpirun_cmd
):
    command = f'{mpirun_cmd} -np {ncores} {lammps_cmd}'
    return _run_lammps(atoms, 'e3gnn/parallel', potential, wd, command, test_name)


def subprocess_routine(cmd, name):
    res = subprocess.run(cmd, capture_output=True, timeout=30)
    if res.returncode != 0:
        logger.error(f'Subprocess {name} failed return code: {res.returncode}')
        logger.error(res.stderr.decode('utf-8'))
        raise RuntimeError(f'{name} failed')

    logger.info(f'stdout of {name}:')
    logger.info(res.stdout.decode('utf-8'))


@pytest.mark.parametrize(
    'system',
    ['bulk', 'surface'],
)
def test_serial(system, serial_potential_path, ref_calculator, lammps_cmd, tmp_path):
    atoms = get_system(system)
    atoms_lammps = serial_lammps_run(
        atoms=atoms,
        potential=serial_potential_path,
        wd=tmp_path,
        test_name='serial lmp test',
        lammps_cmd=lammps_cmd,
    )
    atoms.calc = ref_calculator
    assert_atoms(atoms, atoms_lammps)


@pytest.mark.parametrize(
    'system,ncores',
    [
        ('bulk', 1),
        ('bulk', 2),
        ('bulk', 4),
        ('surface', 1),
        ('surface', 2),
        ('surface', 3),
        ('surface', 4),
    ],
)
def test_parallel(
    system,
    ncores,
    parallel_potential_path,
    ref_calculator,
    lammps_cmd,
    mpirun_cmd,
    tmp_path,
):
    if system == 'bulk':
        rep = (6, 6, 3)
    elif system == 'surface':
        rep = (4, 4, 1)
    else:
        assert False
    atoms = get_system(system, replicate=rep)
    atoms_lammps = parallel_lammps_run(
        atoms=atoms,
        potential=parallel_potential_path,
        wd=tmp_path,
        test_name='parallel lmp test',
        lammps_cmd=lammps_cmd,
        mpirun_cmd=mpirun_cmd,
        ncores=ncores,
    )
    atoms.calc = ref_calculator
    assert_atoms(atoms, atoms_lammps)
