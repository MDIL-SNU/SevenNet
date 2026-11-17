"""
A model call with its inputs on the GPU makes no host synchronization
(torch.cuda.set_sync_debug_mode('error') raises on one): single-modal 7net-0 and
multi-modal 7net-mf-0 (ModalWiseRescale), periodic cell with stress.
"""
import pytest
import torch
from ase.build import bulk

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import pretrained_name_to_path

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


@pytest.mark.parametrize('name, modal', [('7net-0_11July2024', None), ('7net-mf-0', 'R2SCAN')])
def test_model_call_no_host_sync(name, modal):
    calc = SevenNetCalculator(pretrained_name_to_path(name), modal=modal)
    atoms = bulk('Si', 'diamond', a=5.43, cubic=True)
    data = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms, calc.cutoff))
    if modal:
        data[KEY.DATA_MODALITY] = modal
    data.to(calc.device)
    calc.model(data.clone())  # warm-up (lazy initialization may synchronize)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        out = calc.model(data)
    finally:
        torch.cuda.set_sync_debug_mode('default')
    assert out[KEY.PRED_FORCE].shape == (len(atoms), 3)
