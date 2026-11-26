import pytest
import torch
from ase.build import bulk

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.nn.radial_table import RadialTable, tabulate_radial
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import model_from_checkpoint, pretrained_name_to_path

CASES = [('cpu', False)]
if torch.cuda.is_available():
    CASES += [('cuda', False), ('cuda', True)]


@pytest.fixture(autouse=True)
def native_cpu_kernels():
    # Legacy oneDNN kernels can exceed the FP32 comparison tolerance.
    with torch.backends.mkldnn.flags(enabled=False):
        yield


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
    ref = model(_data(config[KEY.CUTOFF], device))
    names = tabulate_radial(model, kernel=kernel)
    assert len(names) == config[KEY.NUM_CONVOLUTION]
    table = model._modules[names[0]].weight_nn
    origin = table.coef.new_zeros(1).requires_grad_()
    at_origin = table(origin)
    assert torch.isfinite(at_origin).all()
    (slope,) = torch.autograd.grad(at_origin.sum(), origin)
    torch.testing.assert_close(slope, torch.zeros_like(slope))
    out = model(_data(config[KEY.CUTOFF], device))
    e_ref = ref[KEY.PRED_TOTAL_ENERGY].item()
    assert abs(out[KEY.PRED_TOTAL_ENERGY].item() - e_ref) < 2e-6 * abs(e_ref)
    f_ref = ref[KEY.PRED_FORCE]
    assert (out[KEY.PRED_FORCE] - f_ref).abs().max().item() < 2e-5
    assert f_ref.abs().max().item() > 0.1


def test_radial_table_refused():
    model, _ = model_from_checkpoint(pretrained_name_to_path('7net-0'))
    conv = next(m for n, m in model._modules.items() if n.endswith('_convolution'))
    conv.key_weight_input = KEY.EDGE_ATTR
    with pytest.raises(ValueError, match='radial input'):
        tabulate_radial(model)


@pytest.mark.parametrize('device', sorted({device for device, _ in CASES}))
def test_ase_defaults_to_radial_table_without_tp_acceleration(device):
    default = SevenNetCalculator('7net-0', device=device)
    exact = SevenNetCalculator('7net-0', device=device, radial_table=False)
    tables = [m for m in default.model.modules() if isinstance(m, RadialTable)]
    assert tables
    assert {(m.coef.shape[0], m.r0) for m in tables} == {(2047, 0.0)}
    assert not any(isinstance(m, RadialTable) for m in exact.model.modules())
    atoms = bulk('NaCl', 'rocksalt', a=5.63) * (2, 2, 2)
    atoms.rattle(0.1, seed=1)
    default.calculate(atoms)
    exact.calculate(atoms)
    torch.testing.assert_close(
        torch.as_tensor(default.results['forces']),
        torch.as_tensor(exact.results['forces']),
        rtol=1e-5, atol=2e-5,
    )
