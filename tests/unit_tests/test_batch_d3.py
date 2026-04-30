"""Test BatchD3 against serial D3Calculator."""
# TODO: check stress-things with non-pbc input
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

from sevenn.calculator import D3Calculator

try:
    from sevenn.torchsim_d3 import BatchD3
    BATCH_D3_AVAILABLE = True
except Exception:
    BATCH_D3_AVAILABLE = False

pytestmark = pytest.mark.skipif(not BATCH_D3_AVAILABLE, reason='BatchD3 unavailable')

# Reference values from test_calculator.py
REF_NACL_PBC = {
    'energy': -0.531393751583389,
    'forces': np.array([
        [-0.00570205, 0.00107457, 0.00107459],
        [0.00570205, -0.00107457, -0.00107459],
    ]),
    'stress': np.array([
        1.52403705e-02, 1.50417333e-02, 1.50417321e-02,
        -3.22684163e-05, -5.05532863e-05, -5.05586994e-05,
    ]),
}

REF_H2O_MOL = {
    'energy': -0.009889134535170716,
    'forces': np.array([
        [0.0, 2.04263840e-03, 1.27477674e-03],
        [0.0, -9.90038901e-05, 1.18046682e-06],
        [0.0, -1.94363451e-03, -1.27595721e-03],
    ]),
}


def make_nacl():
    atoms = bulk('NaCl', 'rocksalt', a=5.63)
    atoms.set_cell([[1.0, 2.815, 2.815], [2.815, 0.0, 2.815], [2.815, 2.815, 0.0]])
    atoms.set_positions([[0.0, 0.0, 0.0], [2.815, 0.0, 0.0]])
    return atoms


def make_h2o():
    atoms = molecule('H2O')
    atoms.set_positions([[0.0, 0.2, 0.12], [0.0, 0.76, -0.48], [0.0, -0.76, -0.48]])
    atoms.set_pbc(False)
    return atoms


def atoms_to_batch(atoms_list, vdw_cutoff=9000, cn_cutoff=1600):
    B = len(atoms_list)
    natoms_each = np.array([len(a) for a in atoms_list], dtype=np.int32)
    atomic_numbers = np.concatenate([a.get_atomic_numbers() for a in atoms_list])
    positions = np.concatenate([a.get_positions() for a in atoms_list])
    pbc = np.array([a.get_pbc().astype(int) for a in atoms_list], dtype=np.int32)

    cells = []
    for a in atoms_list:
        if a.get_cell().sum() == 0:
            pos = a.get_positions()
            max_cutoff = np.sqrt(max(vdw_cutoff, cn_cutoff)) * 0.52917726
            lengths = pos.max(axis=0) - pos.min(axis=0) + max_cutoff + 1.0
            cells.append(np.diag(lengths))
        else:
            cells.append(a.get_cell().array)
    cells = np.array(cells)

    return B, natoms_each, atomic_numbers, positions, cells, pbc


def virial_to_voigt_stress(virial_3x3, volume):
    """[3,3] extensive virial -> Voigt intensive stress (D3Calculator convention)."""
    v = virial_3x3
    return -np.array([
        v[0, 0], v[1, 1], v[2, 2], v[1, 2], v[0, 2], v[0, 1]
    ]) / volume


def serial_d3(atoms_list):
    """Run D3Calculator on each atoms, return list of (E, F, S)."""
    calc = D3Calculator()
    results = []
    for atoms in atoms_list:
        a = atoms.copy()
        a.calc = calc
        e = a.get_potential_energy()
        f = a.get_forces().copy()
        s = a.get_stress().copy()
        results.append((e, f, s))
    return results


@pytest.fixture(scope='module')
def batch_d3():
    try:
        return BatchD3()
    except NotImplementedError as e:
        pytest.skip(str(e))


def test_batch_pbc_replicated(batch_d3):
    nacl = make_nacl()
    atoms_list = [nacl] * 4
    B, natoms_each, Z, pos, cells, pbc = atoms_to_batch(atoms_list)
    energy, forces, stress = batch_d3.compute(B, natoms_each, Z, pos, cells, pbc)

    for i in range(1, 4):
        assert energy[i] == energy[0]
        np.testing.assert_array_equal(forces[i * 2:(i + 1) * 2], forces[:2])
        np.testing.assert_array_equal(stress[i], stress[0])

    np.testing.assert_allclose(energy[0], REF_NACL_PBC['energy'], rtol=1e-5)
    np.testing.assert_allclose(forces[:2], REF_NACL_PBC['forces'], rtol=1e-4)
    vol = nacl.get_volume()
    stress_voigt = virial_to_voigt_stress(stress[0], vol)
    np.testing.assert_allclose(
        stress_voigt, REF_NACL_PBC['stress'], rtol=1e-4, atol=1e-8
    )


def test_batch_mol_replicated(batch_d3):
    h2o = make_h2o()
    atoms_list = [h2o] * 4
    B, natoms_each, Z, pos, cells, pbc = atoms_to_batch(atoms_list)
    energy, forces, stress = batch_d3.compute(B, natoms_each, Z, pos, cells, pbc)

    for i in range(1, 4):
        assert energy[i] == energy[0]
        np.testing.assert_array_equal(forces[i * 3:(i + 1) * 3], forces[:3])
        np.testing.assert_array_equal(stress[i], stress[0])

    np.testing.assert_allclose(energy[0], REF_H2O_MOL['energy'], rtol=1e-5)
    np.testing.assert_allclose(forces[:3], REF_H2O_MOL['forces'], rtol=1e-4)


def test_batch_mixed(batch_d3):
    nacl = make_nacl()
    h2o = make_h2o()
    atoms_list = [nacl, h2o, nacl, h2o]
    B, natoms_each, Z, pos, cells, pbc = atoms_to_batch(atoms_list)
    energy, forces, stress = batch_d3.compute(B, natoms_each, Z, pos, cells, pbc)

    serial = serial_d3(atoms_list)
    offset = 0
    for i, (ref_e, ref_f, _ref_s) in enumerate(serial):
        n = natoms_each[i]
        np.testing.assert_allclose(energy[i], ref_e, rtol=1e-5)
        np.testing.assert_allclose(forces[offset:offset + n], ref_f, rtol=1e-4)
        offset += n

    vol = nacl.get_volume()
    for batch_idx, serial_idx in [(0, 0), (2, 2)]:
        stress_voigt = virial_to_voigt_stress(stress[batch_idx], vol)
        np.testing.assert_allclose(
            stress_voigt, serial[serial_idx][2], rtol=1e-4, atol=1e-8
        )
