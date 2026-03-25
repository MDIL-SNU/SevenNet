import numpy as np
from ase.build import bulk

import sevenn._keys as KEY
from sevenn.calculator import SevenNetCalculator
from sevenn.scripts.deploy import deploy_ts
from sevenn.util import pretrained_name_to_path


def _get_atoms_pbc():
    atoms = bulk('NaCl', 'rocksalt', a=5.63)
    atoms.set_cell([[1.0, 2.815, 2.815], [2.815, 0.0, 2.815], [2.815, 2.815, 0.0]])
    atoms.set_positions([[0.0, 0.0, 0.0], [2.815, 0.0, 0.0]])
    return atoms


def test_atomic_virial_is_exposed_in_python_torchscript_path(tmp_path):
    model_path = str(tmp_path / '7net_0_atomic_virial.pt')
    deploy_ts(pretrained_name_to_path('7net-0_11July2024'), model_path, atomic_virial=True)

    calc = SevenNetCalculator(model_path, file_type='torchscript')
    atoms = _get_atoms_pbc()
    atoms.calc = calc
    _ = atoms.get_potential_energy()

    assert 'stresses' in calc.results
    atomic_virial = np.asarray(calc.results['stresses'])

    assert atomic_virial.shape == (len(atoms), 6)
    assert np.isfinite(atomic_virial).all()
    assert np.any(np.abs(atomic_virial) > 0.0)
    # Full 6-component per-atom reference check.
    # Sort rows for deterministic comparison when atom-wise ordering changes.
    atomic_virial_ref = np.array(
        [
            [10.06461800, -0.07430478, -0.07463801, -0.38235345, -0.04390856, -0.47967120],
            [13.55999100, 1.41123860, 1.41158500, -1.28466740, -0.17591128, -1.18766780],
        ]
    )
    atomic_virial_sorted = atomic_virial[np.argsort(atomic_virial[:, 0])]
    atomic_virial_ref_sorted = atomic_virial_ref[np.argsort(atomic_virial_ref[:, 0])]
    assert np.allclose(atomic_virial_sorted, atomic_virial_ref_sorted, atol=1e-4)

    # Internal model stress ordering is [xx, yy, zz, xy, yz, zx].
    # Calculator exposes ASE stress as -stress_internal with
    # component order [xx, yy, zz, yz, zx, xy].
    virial_sum = atomic_virial.sum(axis=0)
    virial_sum_ref = np.array([
        23.62459886,
        1.33693361,
        1.33694608,
        -1.66702306,
        -0.21981908,
        -1.66734152,
    ])
    assert np.allclose(virial_sum, virial_sum_ref, atol=1e-4)

    stress_internal_from_virial = virial_sum / atoms.get_volume()
    stress_ase_from_virial = -stress_internal_from_virial[[0, 1, 2, 4, 5, 3]]

    assert np.allclose(calc.results['stress'], stress_ase_from_virial, atol=1e-5)
