"""
SevenNetCalculator copies its inputs with one host-to-device copy per dtype and
its outputs with one device-to-host copy: the results equal, bitwise and in dtype,
those of a model call on per-tensor copies read back tensor by tensor (the former
calculator path), for a periodic cell (stress) and a molecule, on CUDA and CPU.
"""
import numpy as np
import pytest
import torch
from ase.build import bulk, molecule

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator, _sort_edges
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import pretrained_name_to_path

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def _reference(calc, atoms):
    data = AtomGraphData.from_numpy_dict(
        unlabeled_atoms_to_graph(atoms, calc.cutoff)
    )
    _sort_edges(data)  # the calculator's edge order (fp32 summation order)
    data.to(calc.device)
    out = calc.model(data)
    n = out['num_atoms'].item()
    return {
        'energy': out[KEY.PRED_TOTAL_ENERGY].detach().cpu().item(),
        'energies': out[KEY.ATOMIC_ENERGY].detach().cpu().numpy().flatten(),
        'forces': out[KEY.PRED_FORCE].detach().cpu().numpy()[:n, :],
        'stress': np.array(
            (-out[KEY.PRED_STRESS]).detach().cpu().numpy()[[0, 1, 2, 4, 5, 3]]
        ),
    }


@pytest.mark.parametrize('device', DEVICES)
def test_packed_io_equals_per_tensor_io(device):
    torch.use_deterministic_algorithms(True, warn_only=True)
    calc = SevenNetCalculator(
        pretrained_name_to_path('7net-0_11July2024'), device=device
    )
    for atoms in (bulk('Si', 'diamond', a=5.43, cubic=True), molecule('H2O')):
        atoms.calc = calc
        atoms.get_potential_energy()
        ref = _reference(calc, atoms)
        assert calc.results['energy'] == ref['energy']
        for k in ('energies', 'forces', 'stress'):
            got = np.asarray(calc.results[k])
            assert got.dtype == ref[k].dtype, k
            assert np.array_equal(got, ref[k]), k
