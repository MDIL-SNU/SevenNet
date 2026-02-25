import copy

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

from sevenn.calculator import D3Calculator, SevenNetCalculator
from sevenn.nn.cue_helper import is_cue_available
from sevenn.nn.flash_helper import is_flash_available
from sevenn.scripts.deploy import deploy
from sevenn.util import model_from_checkpoint, pretrained_name_to_path


@pytest.fixture
def atoms_pbc():
    atoms1 = bulk('NaCl', 'rocksalt', a=5.63)
    atoms1.set_cell([[1.0, 2.815, 2.815], [2.815, 0.0, 2.815], [2.815, 2.815, 0.0]])
    atoms1.set_positions([[0.0, 0.0, 0.0], [2.815, 0.0, 0.0]])
    return atoms1


@pytest.fixture
def atoms_mol():
    atoms2 = molecule('H2O')
    atoms2.set_positions([[0.0, 0.2, 0.12], [0.0, 0.76, -0.48], [0.0, -0.76, -0.48]])
    return atoms2


@pytest.fixture(scope='module')
def sevennet_0_cal():
    return SevenNetCalculator('7net-0_11July2024')


@pytest.fixture(scope='module')
def sevennet_0_cueq_cal():
    cpp = pretrained_name_to_path('7net-0_11July2024')
    model, _ = model_from_checkpoint(cpp, enable_cueq=True)
    return SevenNetCalculator(model)


@pytest.fixture(scope='module')
def sevennet_0_flash_cal():
    cpp = pretrained_name_to_path('7net-0_11July2024')
    model, _ = model_from_checkpoint(cpp, enable_flash=True)
    return SevenNetCalculator(model)


@pytest.fixture(scope='module')
def d3_cal():
    try:
        return D3Calculator()
    except NotImplementedError as e:
        pytest.skip(f'{e}')


def test_sevennet_0_cal_pbc(atoms_pbc, sevennet_0_cal):
    atoms1_ref = {
        'energy': -3.779199,
        'energies': [-1.8493923, -1.9298072],
        'force': [
            [12.666697, 0.04726403, 0.04775861],
            [-12.666697, -0.04726403, -0.04775861],
        ],
        'stress': [
            [
                -0.6439122,
                -0.03643947,
                -0.03643981,
                0.00599139,
                0.04544507,
                0.04543639,
            ]
        ],
    }

    atoms_pbc.calc = sevennet_0_cal
    assert np.allclose(atoms_pbc.get_potential_energy(), atoms1_ref['energy'])
    assert np.allclose(
        atoms_pbc.get_potential_energy(force_consistent=True), atoms1_ref['energy']
    )
    assert np.allclose(atoms_pbc.get_forces(), atoms1_ref['force'])
    assert np.allclose(atoms_pbc.get_stress(), atoms1_ref['stress'])
    assert np.allclose(atoms_pbc.get_potential_energies(), atoms1_ref['energies'])


def test_sevennet_0_cal_mol(atoms_mol, sevennet_0_cal):
    atoms2_ref = {
        'energy': -12.782808303833008,
        'energies': [-6.2493525, -3.141562, -3.3918958],
        'force': [
            [0.0, -1.3619621e01, 7.5937047e00],
            [0.0, 9.3918495e00, -1.0172190e01],
            [0.0, 4.2277718e00, 2.5784855e00],
        ],
    }
    atoms_mol.calc = sevennet_0_cal
    assert np.allclose(atoms_mol.get_potential_energy(), atoms2_ref['energy'])
    assert np.allclose(
        atoms_mol.get_potential_energy(force_consistent=True), atoms2_ref['energy']
    )
    assert np.allclose(atoms_mol.get_forces(), atoms2_ref['force'])
    assert np.allclose(atoms_mol.get_potential_energies(), atoms2_ref['energies'])


def test_sevennet_0_cal_deployed_consistency(tmp_path, atoms_pbc):
    fname = str(tmp_path / '7net_0.pt')
    deploy(pretrained_name_to_path('7net-0_11July2024'), fname)

    calc_script = SevenNetCalculator(fname, file_type='torchscript')
    calc_cp = SevenNetCalculator(pretrained_name_to_path('7net-0_11July2024'))

    atoms_pbc.calc = calc_cp
    atoms_pbc.get_potential_energy()
    res_cp = copy.copy(atoms_pbc.calc.results)

    atoms_pbc.calc = calc_script
    atoms_pbc.get_potential_energy()
    res_script = copy.copy(atoms_pbc.calc.results)

    for k in res_cp:
        assert np.allclose(res_cp[k], res_script[k])


def test_sevennet_0_cal_as_instance_consistency(atoms_pbc):
    model, _ = model_from_checkpoint(
        pretrained_name_to_path('7net-0_11July2024')
    )

    calc_cp = SevenNetCalculator(pretrained_name_to_path('7net-0_11July2024'))
    calc_instance = SevenNetCalculator(model, file_type='model_instance')

    atoms_pbc.calc = calc_cp
    atoms_pbc.get_potential_energy()
    res_cp = copy.copy(atoms_pbc.calc.results)

    atoms_pbc.calc = calc_instance
    atoms_pbc.get_potential_energy()
    res_script = copy.copy(atoms_pbc.calc.results)

    for k in res_cp:
        assert np.allclose(res_cp[k], res_script[k])


@pytest.mark.skipif(not is_cue_available(), reason='cueq not available')
def test_sevennet_0_cal_cueq(atoms_pbc, sevennet_0_cueq_cal):
    atoms1_ref = {
        'energy': -3.779199,
        'energies': [-1.8493923, -1.9298072],
        'force': [
            [12.666697, 0.04726403, 0.04775861],
            [-12.666697, -0.04726403, -0.04775861],
        ],
        'stress': [
            [
                -0.6439122,
                -0.03643947,
                -0.03643981,
                0.00599139,
                0.04544507,
                0.04543639,
            ]
        ],
    }

    atoms_pbc.calc = sevennet_0_cueq_cal

    assert np.allclose(atoms_pbc.get_potential_energy(), atoms1_ref['energy'])
    assert np.allclose(
        atoms_pbc.get_potential_energy(force_consistent=True), atoms1_ref['energy']
    )
    assert np.allclose(atoms_pbc.get_forces(), atoms1_ref['force'])
    assert np.allclose(atoms_pbc.get_stress(), atoms1_ref['stress'])
    assert np.allclose(atoms_pbc.get_potential_energies(), atoms1_ref['energies'])


@pytest.mark.skipif(not is_flash_available(), reason='flash not available')
def test_sevennet_0_cal_flash(atoms_pbc, sevennet_0_flash_cal):
    atoms1_ref = {
        'energy': -3.779199,
        'energies': [-1.8493923, -1.9298072],
        'force': [
            [12.666697, 0.04726403, 0.04775861],
            [-12.666697, -0.04726403, -0.04775861],
        ],
        'stress': [
            [
                -0.6439122,
                -0.03643947,
                -0.03643981,
                0.00599139,
                0.04544507,
                0.04543639,
            ]
        ],
    }

    atoms_pbc.calc = sevennet_0_flash_cal

    assert np.allclose(atoms_pbc.get_potential_energy(), atoms1_ref['energy'])
    assert np.allclose(
        atoms_pbc.get_potential_energy(force_consistent=True), atoms1_ref['energy']
    )
    assert np.allclose(atoms_pbc.get_forces(), atoms1_ref['force'])
    assert np.allclose(atoms_pbc.get_stress(), atoms1_ref['stress'])
    assert np.allclose(atoms_pbc.get_potential_energies(), atoms1_ref['energies'])


def test_d3_cal_pbc(atoms_pbc, d3_cal):
    atoms1_ref = {
        'energy': -0.531393751583389,
        'force': [
            [-0.00570205, 0.00107457, 0.00107459],
            [0.00570205, -0.00107457, -0.00107459],
        ],
        'stress': [
            [
                1.52403705e-02,
                1.50417333e-02,
                1.50417321e-02,
                -3.22684163e-05,
                -5.05532863e-05,
                -5.05586994e-05,
            ]
        ],
    }

    atoms_pbc.calc = d3_cal

    assert np.allclose(atoms_pbc.get_potential_energy(), atoms1_ref['energy'])
    assert np.allclose(
        atoms_pbc.get_potential_energy(force_consistent=True), atoms1_ref['energy']
    )
    assert np.allclose(atoms_pbc.get_forces(), atoms1_ref['force'])
    assert np.allclose(atoms_pbc.get_stress(), atoms1_ref['stress'])


def test_d3_cal_mol(atoms_mol, d3_cal):
    atoms2_ref = {
        'energy': -0.009889134535170716,
        'force': [
            [0.0, 2.04263840e-03, 1.27477674e-03],
            [0.0, -9.90038901e-05, 1.18046682e-06],
            [0.0, -1.94363451e-03, -1.27595721e-03],
        ],
    }

    atoms_mol.calc = d3_cal

    assert np.allclose(atoms_mol.get_potential_energy(), atoms2_ref['energy'])
    assert np.allclose(
        atoms_mol.get_potential_energy(force_consistent=True), atoms2_ref['energy']
    )
    assert np.allclose(atoms_mol.get_forces(), atoms2_ref['force'])


REF_VALUES = {
    'single_o_pbc': {
        'energy': -1.9413528442382812,
        'energies': [-1.9413528442382812],
        'forces': [[0.0, 0.0, 0.0]],
        'stress': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'pbc': True,
    },
    'two_o_disconnected_pbc': {
        'energy': -3.882704734802246,
        'energies': [-1.941352367401123, -1.941352367401123],
        'forces': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        'stress': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'pbc': True,
    },
    'three_o_partial_pbc': {
        'energy': -6.802117824554443,
        'energies': [-2.43038272857666, -2.43038272857666, -1.941352367401123],
        'forces': [
            [3.8830623626708984, 0.0, 0.0],
            [-3.8830623626708984, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        'stress': [0.0009707655990496278, 0.0, 0.0, 0.0, 0.0, 0.0],
        'pbc': True,
    },
}


def _get_disconnected_system(name):
    """Build Atoms object for a disconnected test structure."""
    if name.startswith('single_o_'):
        positions = [[10, 10, 10]]
        symbols = 'O'
    elif name.startswith('two_o_'):
        positions = [[5, 10, 10], [15, 10, 10]]
        symbols = 'OO'
    elif name.startswith('three_o_'):
        positions = [[10, 10, 10], [12, 10, 10], [10, 10, 20]]
        symbols = 'OOO'
    else:
        raise ValueError(f'Unknown structure: {name}')

    is_pbc = name.endswith('_pbc')
    if is_pbc:
        return Atoms(symbols, positions=positions, cell=[20, 20, 20], pbc=True)
    else:
        return Atoms(symbols, positions=positions)


_disconnected_systems = [
    'single_o_pbc',
    'single_o_mol',
    'two_o_disconnected_pbc',
    'two_o_disconnected_mol',
    'three_o_partial_pbc',
    'three_o_partial_mol',
]


@pytest.mark.parametrize('system', _disconnected_systems)
def test_disconnected_e3nn(system, sevennet_0_cal):
    atoms = _get_disconnected_system(system)
    _test_pbc = system.endswith('_pbc')
    ref_key = system if _test_pbc else system.replace('_mol', '_pbc')
    ref = REF_VALUES[ref_key]
    atoms.calc = sevennet_0_cal

    assert np.allclose(atoms.get_potential_energy(), ref['energy'])
    assert np.allclose(atoms.get_forces(), ref['forces'])
    assert np.allclose(atoms.get_potential_energies(), ref['energies'])
    if _test_pbc:
        assert np.allclose(atoms.get_stress(), ref['stress'])


@pytest.mark.skipif(not is_flash_available(), reason='flash not available')
@pytest.mark.parametrize('system', _disconnected_systems)
def test_disconnected_flash(system, sevennet_0_flash_cal):
    atoms = _get_disconnected_system(system)
    _test_pbc = system.endswith('_pbc')
    ref_key = system if _test_pbc else system.replace('_mol', '_pbc')
    ref = REF_VALUES[ref_key]
    atoms.calc = sevennet_0_flash_cal

    assert np.allclose(atoms.get_potential_energy(), ref['energy'])
    assert np.allclose(atoms.get_forces(), ref['forces'])
    assert np.allclose(atoms.get_potential_energies(), ref['energies'])
    if _test_pbc:
        assert np.allclose(atoms.get_stress(), ref['stress'])


@pytest.mark.skipif(not is_cue_available(), reason='cueq not available')
@pytest.mark.parametrize('system', _disconnected_systems)
def test_disconnected_cueq(system, sevennet_0_cueq_cal):
    atoms = _get_disconnected_system(system)
    _test_pbc = system.endswith('_pbc')
    ref_key = system if _test_pbc else system.replace('_mol', '_pbc')
    ref = REF_VALUES[ref_key]
    atoms.calc = sevennet_0_cueq_cal

    assert np.allclose(atoms.get_potential_energy(), ref['energy'])
    assert np.allclose(atoms.get_forces(), ref['forces'])
    assert np.allclose(atoms.get_potential_energies(), ref['energies'])
    if _test_pbc:
        assert np.allclose(atoms.get_stress(), ref['stress'])
