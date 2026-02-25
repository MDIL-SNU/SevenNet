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
    atoms_pbc.rattle(stdev=0.01, seed=42)
    atoms1_ref = {
        'energy': -3.647711753845215,
        'energies': [-1.7780534029006958, -1.8696582317352295],
        'force': [
            [13.095220565795898, 0.05549357831478119, 0.10542003065347672],
            [-13.095221519470215, -0.055493563413619995, -0.1054200679063797],
        ],
        'stress': [
            [
                -0.6614749431610107,
                -0.03719595819711685,
                -0.03681188449263573,
                0.005672863684594631,
                0.04221367835998535,
                0.04504658654332161,
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
    atoms_mol.rattle(stdev=0.01, seed=42)
    atoms2_ref = {
        'energy': -12.870156288146973,
        'energies': [-6.2914958000183105, -3.1829171180725098, -3.3957436084747314],
        'force': [
            [-0.11430990695953369, -12.89616584777832, 6.915047645568848],
            [0.16116246581077576, 8.810967445373535, -9.560930252075195],
            [-0.04685257002711296, 4.085198402404785, 2.6458816528320312],
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
    atoms_pbc.rattle(stdev=0.01, seed=42)
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
        assert np.allclose(res_cp[k], res_script[k], rtol=1e-4, atol=1e-4)


def test_sevennet_0_cal_as_instance_consistency(atoms_pbc):
    atoms_pbc.rattle(stdev=0.01, seed=42)
    model, _ = model_from_checkpoint(pretrained_name_to_path('7net-0_11July2024'))

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
    atoms_pbc.rattle(stdev=0.01, seed=42)
    atoms1_ref = {
        'energy': -3.647711753845215,
        'energies': [-1.7780534029006958, -1.8696582317352295],
        'force': [
            [13.095220565795898, 0.05549357831478119, 0.10542003065347672],
            [-13.095221519470215, -0.055493563413619995, -0.1054200679063797],
        ],
        'stress': [
            [
                -0.6614749431610107,
                -0.03719595819711685,
                -0.03681188449263573,
                0.005672863684594631,
                0.04221367835998535,
                0.04504658654332161,
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
    atoms_pbc.rattle(stdev=0.01, seed=42)
    atoms1_ref = {
        'energy': -3.647711753845215,
        'energies': [-1.7780534029006958, -1.8696582317352295],
        'force': [
            [13.095220565795898, 0.05549357831478119, 0.10542003065347672],
            [-13.095221519470215, -0.055493563413619995, -0.1054200679063797],
        ],
        'stress': [
            [
                -0.6614749431610107,
                -0.03719595819711685,
                -0.03681188449263573,
                0.005672863684594631,
                0.04221367835998535,
                0.04504658654332161,
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
