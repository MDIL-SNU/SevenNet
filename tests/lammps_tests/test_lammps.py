import copy
import logging
import pathlib
import subprocess

import ase.calculators.lammps
import ase.io.lammpsdata
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, surface
from ase.calculators.singlepoint import SinglePointCalculator

import sevenn
from sevenn.calculator import SevenNetCalculator
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.cue_helper import is_cue_available
from sevenn.nn.flash_helper import is_flash_available
from sevenn.scripts.deploy import deploy, deploy_parallel
from sevenn.util import chemical_species_preprocess, pretrained_name_to_path

logger = logging.getLogger('test_lammps')

cutoff = 4.0

lmp_script_path = str(
    (pathlib.Path(__file__).parent / 'scripts' / 'skel.lmp').resolve()
)

data_root = (pathlib.Path(__file__).parent.parent / 'data').resolve()
cp_0_path = str(data_root / 'checkpoints' / 'cp_0.pth')  # knows Hf, O
cp_7net0_path = pretrained_name_to_path('7net-0')
cp_mf_path = pretrained_name_to_path('7net-mf-0')


@pytest.fixture(scope='module')
def serial_potential_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('serial_potential')
    pot_path = str(tmp / 'deployed_serial.pt')
    deploy(cp_0_path, pot_path)
    return pot_path


@pytest.fixture(scope='module')
def serial_potential_path_7net0(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('serial_potential_7net0')
    pot_path = str(tmp / 'deployed_serial.pt')
    deploy(cp_7net0_path, pot_path)
    return pot_path


@pytest.fixture(scope='module')
def serial_potential_path_flash(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('serial_potential_flash')
    pot_path = str(tmp / 'deployed_serial.pt')
    deploy(cp_7net0_path, pot_path, use_flash=True)
    return pot_path


@pytest.fixture(scope='module')
def parallel_potential_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('paralllel_potential')
    pot_path = str(tmp / 'deployed_parallel')
    deploy_parallel(cp_0_path, pot_path)
    return ' '.join(['3', pot_path])


@pytest.fixture(scope='module')
def serial_modal_potential_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('serial_modal_potential')
    pot_path = str(tmp / 'deployed_serial.pt')
    deploy(cp_mf_path, pot_path, 'PBE')
    return pot_path


@pytest.fixture(scope='module')
def parallel_modal_potential_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('paralllel_modal_potential')
    pot_path = str(tmp / 'deployed_parallel')
    deploy_parallel(cp_mf_path, pot_path, 'PBE')
    return ' '.join(['5', pot_path])


@pytest.fixture(scope='module')
def ref_calculator():
    return SevenNetCalculator(cp_0_path)


@pytest.fixture(scope='module')
def ref_7net0_calculator():
    return SevenNetCalculator(cp_7net0_path)


@pytest.fixture(scope='module')
def ref_modal_calculator():
    return SevenNetCalculator(cp_mf_path, modal='PBE')


def get_model_config():
    config = {
        'cutoff': cutoff,
        'channel': 8,
        'lmax': 2,
        'is_parity': True,
        'num_convolution_layer': 3,
        'self_connection_type': 'linear',  # not NequIp
        'interaction_type': 'nequip',
        'radial_basis': {
            'radial_basis_name': 'bessel',
        },
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'weight_nn_hidden_neurons': [64, 64],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': 30.0,
        'train_denominator': False,
        'shift': -10.0,
        'scale': 10.0,
        'train_shift_scale': False,
        'irreps_manual': False,
        'lmax_edge': -1,
        'lmax_node': -1,
        'readout_as_fcn': False,
        'use_bias_in_linear': False,
        '_normalize_sph': True,
    }
    config.update(chemical_species_preprocess(['Hf', 'O']))
    return config


def get_model(config_overwrite=None, use_cueq=False, cueq_config=None):
    cf = get_model_config()
    if config_overwrite is not None:
        cf.update(config_overwrite)

    cueq_config = cueq_config or {'cuequivariance_config': {'use': use_cueq}}
    cf.update(cueq_config)

    model = build_E3_equivariant_model(cf, parallel=False)
    assert not isinstance(model, list)
    return model


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


@pytest.mark.skipif(not is_flash_available(), reason='flash not available')
@pytest.mark.parametrize(
    'system',
    ['bulk', 'surface'],
)
def test_serial_flash(
    system, serial_potential_path_flash, ref_7net0_calculator, lammps_cmd, tmp_path
):
    atoms = get_system(system)
    atoms_lammps = serial_lammps_run(
        atoms=atoms,
        potential=serial_potential_path_flash,
        wd=tmp_path,
        test_name='serial lmp test',
        lammps_cmd=lammps_cmd,
    )
    atoms.calc = ref_7net0_calculator
    assert_atoms(atoms, atoms_lammps, atol=1e-5)


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


@pytest.mark.parametrize(
    'system',
    ['bulk', 'surface'],
)
def test_modal_serial(
    system, serial_modal_potential_path, ref_modal_calculator, lammps_cmd, tmp_path
):
    atoms = get_system(system)
    atoms_lammps = serial_lammps_run(
        atoms=atoms,
        potential=serial_modal_potential_path,
        wd=tmp_path,
        test_name='serial lmp test',
        lammps_cmd=lammps_cmd,
    )
    atoms.calc = ref_modal_calculator
    assert_atoms(atoms, atoms_lammps)


@pytest.mark.parametrize(
    'system,ncores',
    [
        ('bulk', 2),
        ('surface', 2),
    ],
)
def test_modal_parallel(
    system,
    ncores,
    parallel_modal_potential_path,
    ref_modal_calculator,
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
        potential=parallel_modal_potential_path,
        wd=tmp_path,
        test_name='parallel lmp test',
        lammps_cmd=lammps_cmd,
        mpirun_cmd=mpirun_cmd,
        ncores=ncores,
    )
    atoms.calc = ref_modal_calculator
    assert_atoms(atoms, atoms_lammps)


@pytest.mark.filterwarnings('ignore:.*is not found from.*')
@pytest.mark.skipif(not is_cue_available(), reason='cueq not available')
def test_cueq_serial(lammps_cmd, tmp_path):
    """
    TODO: Use already saved cueq enabled checkpoint after cueq becomes stable
    """
    cueq = True
    model = get_model(use_cueq=cueq)
    ref_calc = SevenNetCalculator(model, file_type='model_instance')
    atoms = get_system('bulk')

    cfg = get_model_config()
    cfg.update(
        {'cuequivariance_config': {'use': cueq}, 'version': sevenn.__version__}
    )

    cp_path = str(tmp_path / 'cp.pth')
    torch.save(
        {'model_state_dict': model.state_dict(), 'config': cfg},
        cp_path,
    )

    pot_path = str(tmp_path / 'deployed_from_cueq_serial.pt')
    deploy(cp_path, pot_path)

    atoms_lammps = serial_lammps_run(
        atoms=atoms,
        potential=pot_path,
        wd=tmp_path,
        test_name='cueq checkpoint serial lmp run test',
        lammps_cmd=lammps_cmd,
    )
    atoms.calc = ref_calc
    assert_atoms(atoms, atoms_lammps)


@pytest.mark.filterwarnings('ignore:.*is not found from.*')
@pytest.mark.skipif(not is_cue_available(), reason='cueq not available')
def test_cueq_parallel(lammps_cmd, mpirun_cmd, tmp_path):
    """
    TODO: Use already saved cueq enabled checkpoint after cueq becomes stable
    """
    cueq = True
    model = get_model(use_cueq=cueq)
    ref_calc = SevenNetCalculator(model, file_type='model_instance')
    atoms = get_system('surface', replicate=(4, 4, 1))

    cfg = get_model_config()
    cfg.update(
        {'cuequivariance_config': {'use': cueq}, 'version': sevenn.__version__}
    )

    cp_path = str(tmp_path / 'cp.pth')
    torch.save(
        {'model_state_dict': model.state_dict(), 'config': cfg},
        cp_path,
    )

    pot_path = str(tmp_path / 'deployed_from_cueq_parallel')
    deploy_parallel(cp_path, pot_path)

    atoms_lammps = parallel_lammps_run(
        atoms=atoms,
        potential=' '.join([str(cfg['num_convolution_layer']), pot_path]),
        wd=tmp_path,
        test_name='cueq checkpoint parallel lmp run test',
        lammps_cmd=lammps_cmd,
        mpirun_cmd=mpirun_cmd,
        ncores=2,
    )
    atoms.calc = ref_calc
    assert_atoms(atoms, atoms_lammps)


def _get_disconnected_system(name):
    """Build Atoms object for a disconnected test structure."""
    if name.startswith('single_o'):
        positions = [[10, 10, 10]]
        symbols = 'O'
    elif name.startswith('two_o'):
        positions = [[5, 10, 10], [15, 10, 10]]
        symbols = 'OO'
    elif name.startswith('three_o'):
        positions = [[10, 10, 10], [12, 10, 10], [10, 10, 20]]
        symbols = 'OOO'
    else:
        raise ValueError(f'Unknown disconnected structure: {name}')

    return Atoms(symbols, positions=positions, cell=[20, 20, 20], pbc=True)


_disconnected_systems = [
    'single_o',
    'two_o_disconnected',
    'three_o_partial',
]


@pytest.mark.parametrize('system', _disconnected_systems)
def test_disconnected_serial(
    system,
    serial_potential_path_7net0,
    ref_7net0_calculator,
    lammps_cmd,
    tmp_path,
):
    atoms = _get_disconnected_system(system)
    atoms_lammps = serial_lammps_run(
        atoms=atoms,
        potential=serial_potential_path_7net0,
        wd=tmp_path,
        test_name=f'disconnected serial {system}',
        lammps_cmd=lammps_cmd,
    )
    atoms.calc = ref_7net0_calculator
    assert_atoms(atoms, atoms_lammps)


@pytest.mark.skipif(not is_flash_available(), reason='flash not available')
@pytest.mark.parametrize('system', _disconnected_systems)
def test_disconnected_serial_flash(
    system,
    serial_potential_path_flash,
    ref_7net0_calculator,
    lammps_cmd,
    tmp_path,
):
    atoms = _get_disconnected_system(system)
    atoms_lammps = serial_lammps_run(
        atoms=atoms,
        potential=serial_potential_path_flash,
        wd=tmp_path,
        test_name=f'disconnected serial flash {system}',
        lammps_cmd=lammps_cmd,
    )
    atoms.calc = ref_7net0_calculator
    assert_atoms(atoms, atoms_lammps, atol=1e-5)
