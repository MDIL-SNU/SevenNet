import copy

import numpy as np
import pytest
from ase.build import bulk, molecule

from sevenn.scripts.deploy import deploy
from sevenn.sevennet_calculator import SevenNetCalculator
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


def test_sevennet_0_cal_deployed(tmp_path, atoms_pbc):
    model, config = model_from_checkpoint(
        pretrained_name_to_path('7net-0_11July2024')
    )

    fname = str(tmp_path / '7net_0.pt')
    deploy(model.state_dict(), config, fname)

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


def test_sevennet_0_cal_as_instnace(atoms_pbc):
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
