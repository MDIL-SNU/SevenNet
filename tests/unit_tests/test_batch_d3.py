"""Test BatchD3 against serial D3Calculator."""
# TODO: check stress-things with non-pbc input
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

try:
    from sevenn.torchsim import BatchD3
    BATCH_D3_AVAILABLE = True
except Exception:
    BATCH_D3_AVAILABLE = False

pytestmark = pytest.mark.skipif(not BATCH_D3_AVAILABLE, reason='BatchD3 unavailable')

atol = 1e-7
rtol = 0

# Reference values (from BatchD3, PBE, damp-bj)
REF_NACL_PBC = {
    'energy': -0.53141738353661261,
    'forces': np.array([
        [-0.00570207347292894, 0.00107461457140288, 0.00107461681627128],
        [0.00570207347292894, -0.00107461457140288, -0.00107461681627128],
    ]),
    'stress': np.array([
        1.52406481701966405e-02,
        1.50420038287462279e-02,
        1.50420038287462279e-02,
        -3.22732735330434251e-05,
        -5.05585799284106584e-05,
        -5.05586197267584965e-05,
    ]),
}

REF_H2O_MOL = {
    'energy': -0.0098891317633263368,
    'forces': np.array([
        [0.0, 2.0425725520247292e-03, 1.2747388259690125e-03],
        [0.0, -9.9002383155458572e-05, 1.1804777827983966e-06],
        [0.0, -1.9435701688692707e-03, -1.2759193037518110e-03],
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


def atoms_to_batch(atoms_list):
    B = len(atoms_list)
    natoms_each = np.array([len(a) for a in atoms_list], dtype=np.int32)
    atomic_numbers = np.concatenate([a.get_atomic_numbers() for a in atoms_list])
    positions = np.concatenate([a.get_positions() for a in atoms_list])
    pbc = np.array([a.get_pbc().astype(int) for a in atoms_list], dtype=np.int32)
    cells = np.array([a.get_cell().array for a in atoms_list])
    return B, natoms_each, atomic_numbers, positions, cells, pbc


def virial_to_voigt_stress(virial_3x3, volume):
    """[3,3] extensive virial -> Voigt intensive stress (D3Calculator convention)."""
    v = virial_3x3
    return -np.array([
        v[0, 0], v[1, 1], v[2, 2], v[1, 2], v[0, 2], v[0, 1]
    ]) / volume


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

    np.testing.assert_allclose(
        energy[0], REF_NACL_PBC['energy'], rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        forces[:2], REF_NACL_PBC['forces'], rtol=rtol, atol=atol
    )
    vol = nacl.get_volume()
    stress_voigt = virial_to_voigt_stress(stress[0], vol)
    np.testing.assert_allclose(
        stress_voigt, REF_NACL_PBC['stress'], rtol=rtol, atol=atol
    )


def test_batch_mol_replicated(batch_d3):
    h2o = make_h2o()
    atoms_list = [h2o] * 4
    B, natoms_each, Z, pos, cells, pbc = atoms_to_batch(atoms_list)
    energy, forces, stress = batch_d3.compute(B, natoms_each, Z, pos, cells, pbc)

    for i in range(1, 4):
        assert energy[i] == energy[0]
        np.testing.assert_array_equal(forces[i * 3:(i + 1) * 3], forces[:3])

    np.testing.assert_allclose(
        energy[0], REF_H2O_MOL['energy'], rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        forces[:3], REF_H2O_MOL['forces'], rtol=rtol, atol=atol
    )


def test_batch_mixed(batch_d3):
    nacl = make_nacl()
    h2o = make_h2o()
    atoms_list = [nacl, h2o, nacl, h2o]
    B, natoms_each, Z, pos, cells, pbc = atoms_to_batch(atoms_list)
    energy, forces, stress = batch_d3.compute(B, natoms_each, Z, pos, cells, pbc)

    refs = [REF_NACL_PBC, REF_H2O_MOL, REF_NACL_PBC, REF_H2O_MOL]
    offset = 0
    for i, ref in enumerate(refs):
        n = natoms_each[i]
        np.testing.assert_allclose(
            energy[i], ref['energy'], rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            forces[offset:offset + n], ref['forces'], rtol=rtol, atol=atol
        )
        offset += n

    vol = nacl.get_volume()
    for batch_idx in (0, 2):  # nacl entries
        stress_voigt = virial_to_voigt_stress(stress[batch_idx], vol)
        np.testing.assert_allclose(
            stress_voigt, REF_NACL_PBC['stress'], rtol=rtol, atol=atol
        )
