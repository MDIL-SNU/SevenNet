import copy
import logging
import pathlib
import subprocess
from unittest import mock

import ase.calculators.lammps
import ase.io.lammpsdata
import numpy as np
import pytest
import torch
from ase.build import bulk, surface
from ase.calculators.singlepoint import SinglePointCalculator

import sevenn
from sevenn.calculator import SevenNetCalculator
from sevenn.main.sevenn_get_model import main as mliap_cli
from sevenn.nn.cue_helper import is_cue_available
from sevenn.nn.flash_helper import is_flash_available
from sevenn.util import pretrained_name_to_path

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
    _MLIAP_AVAILABLE = True
except ImportError:
    _MLIAP_AVAILABLE = False

if not _MLIAP_AVAILABLE:
    pytest.skip('ML-IAP not available', allow_module_level=True)

logger = logging.getLogger('test_mliap')

mliap_script_path = str(
    (pathlib.Path(__file__).parent / 'scripts' / 'mliap_skel.lmp').resolve()
)

data_root = (pathlib.Path(__file__).parent.parent / 'data').resolve()
cp_7net0_path = pretrained_name_to_path('7net-0')
cp_mf_path = pretrained_name_to_path('7net-mf-0')


@pytest.fixture(scope='module')
def mliap_potential_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('mliap_pt')
    pt_path = str((tmp / 'base.pt').resolve())

    argv = [
        'sevenn_get_model',
        cp_7net0_path,
        '-o', pt_path,
        '--use_mliap'
    ]
    with mock.patch('sys.argv', argv):
        mliap_cli()
    return pt_path


@pytest.fixture(scope='module')
def mliap_modal_potential_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('mliap_modal_pt')
    pt_path = str((tmp / 'modal.pt').resolve())

    argv = [
        'sevenn_get_model',
        cp_mf_path,
        '-o', pt_path,
        '-m', 'PBE',
        '--use_mliap'
    ]
    with mock.patch('sys.argv', argv):
        mliap_cli()
    return pt_path


@pytest.fixture(scope='module')
def mliap_cueq_potential_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('mliap_cueq_pt')
    pt_path = str((tmp / 'cueq.pt').resolve())

    argv = [
        'sevenn_get_model',
        cp_7net0_path,
        '-o', pt_path,
        '--enable_cueq',
        '--use_mliap'
    ]
    with mock.patch('sys.argv', argv):
        mliap_cli()
    return pt_path


@pytest.fixture(scope='module')
def mliap_flash_potential_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('mliap_flash_pt')
    pt_path = str((tmp / 'flash.pt').resolve())

    argv = [
        'sevenn_get_model',
        cp_7net0_path,
        '-o', pt_path,
        '--enable_flashTP',
        '--use_mliap'
    ]
    with mock.patch('sys.argv', argv):
        mliap_cli()
    return pt_path


@pytest.fixture(scope='module')
def ref_7net0_calculator():
    return SevenNetCalculator(cp_7net0_path)


@pytest.fixture(scope='module')
def ref_modal_calculator():
    return SevenNetCalculator(cp_mf_path, modal='PBE')


def hfo2_bulk(replicate=(2, 2, 2), a=4.0):
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


def _run_mliap(atoms, pair_style, wd, command, test_name):
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

    with open(mliap_script_path, 'r') as f:
        cont = f.read()

    lammps_log = str(wd / 'log.lammps')
    force_dump = str(wd / 'force.dump')

    var_dct = {}
    var_dct['__ELEMENT__'] = ' '.join(chem)
    var_dct['__LMP_STCT__'] = str(lmp_stct.resolve())
    var_dct['__PAIR_STYLE__'] = pair_style
    var_dct['__BOUNDARY__'] = pbc_str
    var_dct['__FORCE_DUMP_PATH__'] = force_dump
    for key, val in var_dct.items():
        cont = cont.replace(key, val)

    input_script_path = str(wd / 'in.lmp')
    with open(input_script_path, 'w') as f:
        f.write(cont)

    command = (
        f'{command} -in {input_script_path}'
        f'-k on g 1 -sf kk -pk kokkos neigh half -log {lammps_log}'
    )
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


def mliap_lammps_run(
    atoms, pt_path, wd, test_name, lammps_cmd
):
    command = lammps_cmd
    pair_style = f'{pt_path}'
    return _run_mliap(atoms, pair_style, wd, command, test_name)


def subprocess_routine(cmd, name):
    res = subprocess.run(cmd, capture_output=True, timeout=30)
    if res.returncode != 0:
        logger.error(f'Subprocess {name} failed return code: {res.returncode}')
        logger.error('==== STDERR ====')
        logger.error(res.stderr.decode('utf-8'))
        logger.error('==== STDOUT ====')
        logger.error(res.stdout.decode('utf-8'))
        raise RuntimeError(f'{name} failed')

    logger.info(f'stdout of {name}:')
    logger.info(res.stdout.decode('utf-8'))


@pytest.mark.parametrize('system', ['bulk', 'surface'])
def test_mliap(
    system,
    mliap_potential_path,
    ref_7net0_calculator,
    lammps_cmd,
    tmp_path
):
    atoms = get_system(system)
    atoms_lmp = mliap_lammps_run(
        atoms=atoms,
        pt_path=mliap_potential_path,
        wd=tmp_path,
        test_name='mliap serial',
        lammps_cmd=lammps_cmd,
    )
    atoms.calc = ref_7net0_calculator
    assert_atoms(atoms, atoms_lmp, atol=1e-5)


@pytest.mark.parametrize('system', ['bulk', 'surface'])
def test_mliap_modal(
    system,
    mliap_modal_potential_path,
    ref_modal_calculator,
    lammps_cmd,
    tmp_path
):
    atoms = get_system(system)
    atoms_lmp = mliap_lammps_run(
        atoms=atoms,
        pt_path=mliap_modal_potential_path,
        wd=tmp_path,
        test_name='mliap modal',
        lammps_cmd=lammps_cmd,
    )
    atoms.calc = ref_modal_calculator
    assert_atoms(atoms, atoms_lmp, atol=1e-5)


@pytest.mark.skipif(not is_cue_available(), reason='cueq not available')
@pytest.mark.parametrize('system', ['bulk', 'surface'])
def test_mliap_cueq(
    system,
    mliap_cueq_potential_path,
    ref_7net0_calculator,
    lammps_cmd,
    tmp_path
):
    atoms = get_system(system)
    atoms_lmp = mliap_lammps_run(
        atoms=atoms,
        pt_path=mliap_cueq_potential_path,
        wd=tmp_path,
        test_name='mliap cueq',
        lammps_cmd=lammps_cmd,
    )
    atoms.calc = ref_7net0_calculator
    assert_atoms(atoms, atoms_lmp, atol=1e-5)


@pytest.mark.skipif(
    not is_flash_available() or not torch.cuda.is_available(),
    reason='flashTP or gpu is not available',
)
@pytest.mark.parametrize('system', ['bulk', 'surface'])
def test_mliap_flash(
    system,
    mliap_flash_potential_path,
    ref_7net0_calculator,
    lammps_cmd,
    tmp_path
):
    atoms = get_system(system)
    atoms_lmp = mliap_lammps_run(
        atoms=atoms,
        pt_path=mliap_flash_potential_path,
        wd=tmp_path,
        test_name='mliap flash',
        lammps_cmd=lammps_cmd,
    )
    atoms.calc = ref_7net0_calculator
    assert_atoms(atoms, atoms_lmp, atol=1e-5)
