"""
tabulate_radial (sevenn/nn/radial_table.py): a model whose radial MLPs are
replaced by cubic Hermite tables in r gives the energy and forces of the
model (fp32 round-off level), with torch ops (CPU, CUDA) and with the CUDA
kernels; a model whose radial input is not the edge embedding is refused.
"""
import copy

import pytest
import torch
from ase.build import bulk

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.radial_table import (
    fold_radial,
    optimize_radial,
    tabulate_radial,
)
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import model_from_checkpoint, pretrained_name_to_path

CASES = [('cpu', False)]
if torch.cuda.is_available():
    CASES += [('cuda', False), ('cuda', True)]


def _data(cutoff, device):
    atoms = bulk('NaCl', 'rocksalt', a=5.63) * (3, 3, 3)
    atoms.rattle(0.1, seed=1)
    data = AtomGraphData.from_numpy_dict(
        unlabeled_atoms_to_graph(atoms, cutoff, with_shift=False)
    )
    return data.to(device).to_dict()


@pytest.mark.parametrize('device,kernel', CASES)
def test_radial_table(device, kernel):
    if kernel:
        pytest.importorskip('cupy')
    model, config = model_from_checkpoint(pretrained_name_to_path('7net-0'))
    model.set_is_batch_data(False)
    model.to(device).eval()
    table = copy.deepcopy(model)
    names = tabulate_radial(table, kernel=kernel)
    assert len(names) == config[KEY.NUM_CONVOLUTION]
    ref = model(_data(config[KEY.CUTOFF], device))
    out = table(_data(config[KEY.CUTOFF], device))
    e_ref = ref[KEY.PRED_TOTAL_ENERGY].item()
    assert abs(out[KEY.PRED_TOTAL_ENERGY].item() - e_ref) < 2e-6 * abs(e_ref)
    f_ref = ref[KEY.PRED_FORCE]
    assert (out[KEY.PRED_FORCE] - f_ref).abs().max().item() < 2e-5
    assert f_ref.abs().max().item() > 0.1


@pytest.mark.parametrize('width', [0, 500, 10**6])
def test_optimize_radial(width):
    """folded MLP / table per layer by width: same outputs"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config = model_from_checkpoint(pretrained_name_to_path('7net-0'))
    model.set_is_batch_data(False)
    model.to(device).eval()
    opt = copy.deepcopy(model)
    names = optimize_radial(opt, width, kernel=False)
    assert len(names) == config[KEY.NUM_CONVOLUTION]
    kinds = {n.split(':')[1] for n in names}
    assert kinds == ({'fold'} if width == 0 else
                     {'table'} if width == 10**6 else {'fold', 'table'})
    assert fold_radial(opt) == []  # denominators are folded once
    ref = model(_data(config[KEY.CUTOFF], device))
    out = opt(_data(config[KEY.CUTOFF], device))
    f_ref = ref[KEY.PRED_FORCE]
    assert (out[KEY.PRED_FORCE] - f_ref).abs().max().item() < 2e-5
    e_ref = ref[KEY.PRED_TOTAL_ENERGY].item()
    assert abs(out[KEY.PRED_TOTAL_ENERGY].item() - e_ref) < 2e-6 * abs(e_ref)


def test_radial_table_refused():
    model, _ = model_from_checkpoint(pretrained_name_to_path('7net-0'))
    conv = next(m for n, m in model._modules.items() if n.endswith('_convolution'))
    conv.key_weight_input = KEY.EDGE_ATTR
    with pytest.raises(ValueError, match='radial input'):
        tabulate_radial(model)
