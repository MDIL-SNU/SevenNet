# TODO: add gradient test from total loss after double precision.
#       so far, it is empirically checked by seeing learning curves
import copy
import os
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ase.build import bulk
from torch_geometric.loader.dataloader import Collater

import sevenn
import sevenn._keys as KEY
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import SevenNetCalculator
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.lunatp_helper import is_lunatp_available
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.util import chemical_species_preprocess, model_from_checkpoint

cutoff = 4.0

_atoms = bulk('NaCl', 'rocksalt', a=4.00) * (2, 2, 2)
_avg_num_neigh = 30.0
_atoms.rattle()

_graph = AtomGraphData.from_numpy_dict(dl.unlabeled_atoms_to_graph(_atoms, cutoff))


def get_graphs(batched):
    # batch size 2
    cloned = [_graph.clone().to('cuda'), _graph.clone().to('cuda')]
    if not batched:
        return cloned
    else:
        return Collater(cloned)(cloned)


def get_model_config():
    config = {
        'cutoff': cutoff,
        'channel': 32,
        'lmax': 2,
        'is_parity': True,
        'num_convolution_layer': 3,
        'self_connection_type': 'nequip',  # not NequIp
        'interaction_type': 'nequip',
        'radial_basis': {
            'radial_basis_name': 'bessel',
        },
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'weight_nn_hidden_neurons': [64, 64],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': _avg_num_neigh,
        'train_denominator': False,
        'shift': -10.0,
        'scale': 10.0,
        'train_shift_scale': False,
        'irreps_manual': False,
        'lmax_edge': -1,
        'lmax_node': -1,
        'readout_as_fcn': False,
        'use_bias_in_linear': False,
        '_normalize_sph': True,
    }
    chems = set()
    chems.update(_atoms.get_chemical_symbols())
    config.update(**chemical_species_preprocess(list(chems)))
    return config


def get_model(config_overwrite=None, use_lunatp=False):
    cf = get_model_config()
    if config_overwrite is not None:
        cf.update(config_overwrite)

    cf['use_lunatp'] = use_lunatp

    model = build_E3_equivariant_model(cf, parallel=False)
    assert isinstance(model, AtomGraphSequential)
    model.to('cuda')
    return model


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize(
    'cf',
    [
        ({}),
        ({'lmax': 3}),
        ({'num_interaction_layer': 2}),
        ({'num_interaction_layer': 4}),
    ],
)
def test_model_output(cf):
    torch.manual_seed(777)
    model_e3nn = get_model(cf)
    torch.manual_seed(777)
    model_lunatp = get_model(cf, use_lunatp=True)

    model_e3nn.set_is_batch_data(True)
    model_lunatp.set_is_batch_data(True)

    e3nn_out = model_e3nn._preprocess(get_graphs(batched=True))
    lunatp_out = model_lunatp._preprocess(get_graphs(batched=True))

    for k, e3nn_f in model_e3nn._modules.items():
        lunatp_f = model_lunatp._modules[k]
        e3nn_out = e3nn_f(e3nn_out)  # type: ignore
        lunatp_out = lunatp_f(lunatp_out)  # type: ignore
        assert torch.allclose(e3nn_out.x, lunatp_out.x, atol=1e-6), (
            f'{k} \n\n {e3nn_f} \n\n {lunatp_f}'
        )

    assert torch.allclose(
        e3nn_out.inferred_total_energy, lunatp_out.inferred_total_energy
    )
    assert torch.allclose(e3nn_out.atomic_energy, lunatp_out.atomic_energy)
    assert torch.allclose(
        e3nn_out.inferred_force, lunatp_out.inferred_force, atol=1e-5
    )
    assert torch.allclose(
        e3nn_out.inferred_stress, lunatp_out.inferred_stress, atol=1e-5
    )


@pytest.mark.filterwarnings('ignore:.*is not found from.*')
@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize(
    'start_from_lunatp',
    [
        (True),
        (False),
    ],
)
def test_checkpoint_convert(tmp_path, start_from_lunatp):
    torch.manual_seed(123)
    model_from = get_model(use_lunatp=start_from_lunatp)

    cfg = get_model_config()
    cfg.update(
        {
            'use_lunatp': start_from_lunatp,
            'version': sevenn.__version__,
        }
    )
    torch.save(
        {'model_state_dict': model_from.state_dict(), 'config': cfg},
        tmp_path / 'cp_from.pth',
    )

    model_to, _ = model_from_checkpoint(
        str(tmp_path / 'cp_from.pth'), enable_lunatp=(not start_from_lunatp)
    )
    model_to.to('cuda')

    model_from.set_is_batch_data(True)
    model_to.set_is_batch_data(True)

    from_out = model_from(get_graphs(batched=True))
    to_out = model_to(get_graphs(batched=True))

    assert torch.allclose(
        from_out.inferred_total_energy, to_out.inferred_total_energy
    )
    assert torch.allclose(from_out.atomic_energy, to_out.atomic_energy)
    assert torch.allclose(from_out.inferred_force, to_out.inferred_force, atol=1e-5)
    assert torch.allclose(
        from_out.inferred_stress, to_out.inferred_stress, atol=1e-5
    )


@pytest.mark.filterwarnings('ignore:.*is not found from.*')
@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.parametrize(
    'start_from_lunatp',
    [
        (True),
        (False),
    ],
)
def test_checkpoint_convert_no_batch(tmp_path, start_from_lunatp):
    torch.manual_seed(123)
    model_from = get_model(use_lunatp=start_from_lunatp)

    cfg = get_model_config()
    cfg.update(
        {
            'use_lunatp': start_from_lunatp,
            'version': sevenn.__version__,
        }
    )
    torch.save(
        {'model_state_dict': model_from.state_dict(), 'config': cfg},
        tmp_path / 'cp_from.pth',
    )

    model_to, _ = model_from_checkpoint(
        str(tmp_path / 'cp_from.pth'), enable_lunatp=(not start_from_lunatp)
    )
    model_to.to('cuda')

    model_from.set_is_batch_data(False)
    model_to.set_is_batch_data(False)

    from_out = model_from(get_graphs(batched=False)[0])
    to_out = model_to(get_graphs(batched=False)[0])

    assert torch.allclose(
        from_out.inferred_total_energy, to_out.inferred_total_energy
    )
    assert torch.allclose(from_out.atomic_energy, to_out.atomic_energy)
    assert torch.allclose(from_out.inferred_force, to_out.inferred_force, atol=1e-5)
    assert torch.allclose(
        from_out.inferred_stress, to_out.inferred_stress, atol=1e-5
    )


def assert_atoms(atoms1, atoms2, rtol=1e-5, atol=1e-6):
    def acl(a, b, rtol=rtol, atol=atol):
        return np.allclose(a, b, rtol=rtol, atol=atol)

    assert len(atoms1) == len(atoms2)
    assert acl(atoms1.get_cell(), atoms2.get_cell())
    assert acl(atoms1.get_potential_energy(), atoms2.get_potential_energy())
    assert acl(atoms1.get_forces(), atoms2.get_forces(), rtol * 10, atol * 10)
    assert acl(
        atoms1.get_stress(voigt=False),
        atoms2.get_stress(voigt=False),
        rtol * 10,
        atol * 10,
    )
    # assert acl(atoms1.get_potential_energies(), atoms2.get_potential_energies())


@pytest.mark.filterwarnings('ignore:.*is not found from.*')
@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_calculator(tmp_path):
    lunatp = True
    model = get_model(use_lunatp=lunatp)
    ref_calc = SevenNetCalculator(model, file_type='model_instance')
    atoms = copy.deepcopy(_atoms)
    atoms.calc = ref_calc

    cfg = get_model_config()
    cfg.update(
        {'use_lunatp': lunatp, 'version': sevenn.__version__}
    )

    cp_path = str(tmp_path / 'cp.pth')
    torch.save(
        {'model_state_dict': model.state_dict(), 'config': cfg},
        cp_path,
    )

    calc2 = SevenNetCalculator(cp_path, enable_lunatp=False)
    atoms2 = copy.deepcopy(_atoms)
    atoms2.calc = calc2

    assert_atoms(atoms, atoms2)


_CALCULATOR_CALLS = """
import sys
from ase import Atoms
from ase.build import bulk
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator(sys.argv[1], enable_lunatp=True)
dimer = Atoms('NaCl', positions=[[0, 0, 0], [0, 0, 2.4]])
for atoms in (dimer, bulk('NaCl', 'rocksalt', a=4.0)):
    atoms.calc = calc
    atoms.get_potential_energy()
"""


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
@pytest.mark.skipif(
    os.environ.get('SEVENNET_TEST_PRECOMPILE') != '1',
    reason='compiles all kernels of a model; set SEVENNET_TEST_PRECOMPILE=1',
)
def test_precompile_covers_calculator(tmp_path, monkeypatch):
    from sevenn.nn.lunatp_helper import precompile

    monkeypatch.setenv('LUNAEQ_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('SEVENNET_LUNATP_PRECOMPILE_WORKERS', '0')
    cfg = get_model_config()
    cfg.update({
        'channel': 4, 'lmax': 1, 'num_convolution_layer': 2,
        'use_lunatp': False, 'version': sevenn.__version__,
    })
    model = build_E3_equivariant_model(cfg, parallel=False)
    cp_path = str(tmp_path / 'cp.pth')
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, cp_path)

    precompile(cp_path)
    built = set(tmp_path.rglob('*.so'))
    assert built
    subprocess.run(
        [sys.executable, '-c', _CALCULATOR_CALLS, cp_path], check=True
    )
    added = set(tmp_path.rglob('*.so')) - built
    assert not added, f'kernels built after precompile: {sorted(added)}'


_LOAD_EQU_NORM = """
import sys
import torch
from e3nn.o3 import Irreps
import sevenn.nn.lunatp_helper
from LunaEquivariance.nn.equ_norm import EquNormOperator

for irreps in sys.argv[1:]:
    EquNormOperator(irreps)(torch.randn(2, Irreps(irreps).dim, device='cuda'))
"""


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_kernel_cache_independent_of_load_order(tmp_path):
    env = {**os.environ, 'LUNAEQ_CACHE_DIR': str(tmp_path)}
    configs = ['2x0e+1x1o', '3x0e+1x1o']

    def libraries_after(irreps):
        cmd = [sys.executable, '-c', _LOAD_EQU_NORM, *irreps]
        subprocess.run(cmd, check=True, env=env)
        return sorted(str(p) for p in tmp_path.rglob('*.so'))

    built = libraries_after(configs)
    assert len(built) == 2
    assert libraries_after(configs[::-1]) == built


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_unsorted_edges():
    # LunaTP needs edges sorted by destination, LunaTPConvolution sorts them
    torch.manual_seed(777)
    model_e3nn = get_model()
    torch.manual_seed(777)
    model_lunatp = get_model(use_lunatp=True)
    model_e3nn.set_is_batch_data(False)
    model_lunatp.set_is_batch_data(False)

    graph = _graph.clone().to('cuda')
    perm = torch.randperm(graph.edge_index.shape[1], device='cuda')
    graph.edge_index = graph.edge_index[:, perm]
    graph.edge_vec = graph.edge_vec[perm]
    dst = graph.edge_index[0]
    assert not bool(torch.all(dst[1:] >= dst[:-1]))

    e3nn_out = model_e3nn(graph.clone())
    lunatp_out = model_lunatp(graph.clone())
    assert torch.allclose(
        e3nn_out.inferred_total_energy, lunatp_out.inferred_total_energy
    )
    assert torch.allclose(
        e3nn_out.inferred_force, lunatp_out.inferred_force, atol=1e-5
    )
    assert torch.allclose(
        e3nn_out.inferred_stress, lunatp_out.inferred_stress, atol=1e-5
    )


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_graph_is_shared_only_within_one_forward(monkeypatch):
    from sevenn.nn import lunatp_helper as helper

    monkeypatch.setenv('SEVENNET_LUNATP_PRECOMPILE_WORKERS', '0')
    model = get_model({'channel': 4, 'lmax': 1, 'num_convolution_layer': 2}, True)
    model.set_is_batch_data(False)
    data = _graph.clone().to('cuda')
    edge_index = data[KEY.EDGE_IDX]
    num_nodes = data[KEY.ATOMIC_NUMBERS].numel()
    prepared = helper.prepare_graph(edge_index, num_nodes)
    prepare = Mock(wraps=helper.prepare_graph)
    monkeypatch.setattr(helper, 'prepare_graph', prepare)

    original = model(data)[KEY.PRED_FORCE]
    assert prepare.call_count == 1
    data[helper.PREPARED_GRAPH_KEY] = prepared
    torch.testing.assert_close(model(data)[KEY.PRED_FORCE], original)
    assert prepare.call_count == 1
    assert helper.PREPARED_GRAPH_KEY not in data

    changed_edges = edge_index.clone()
    changed_edges[1] = (changed_edges[1] + 1) % num_nodes
    edge_index.copy_(changed_edges)
    changed = model(data)[KEY.PRED_FORCE]
    assert prepare.call_count == 2
    fresh = _graph.clone().to('cuda')
    fresh[KEY.EDGE_IDX] = changed_edges
    torch.testing.assert_close(changed, model(fresh)[KEY.PRED_FORCE])
    assert not torch.allclose(original, changed)

    data[KEY.EDGE_IDX] = edge_index[:, :0]
    data[KEY.EDGE_VEC] = data[KEY.EDGE_VEC][:0].detach()
    data[helper.PREPARED_GRAPH_KEY] = prepared
    kernel = Mock(side_effect=AssertionError('convolution on empty edges'))
    for layer in model:
        if isinstance(layer, helper.LunaTPSharedGraphConvolution):
            monkeypatch.setattr(layer.convolution, 'forward', kernel)
    prepare.reset_mock()
    model(data)
    prepare.assert_not_called()
    kernel.assert_not_called()
    assert helper.PREPARED_GRAPH_KEY not in data


def test_no_fallback(monkeypatch):
    import sevenn.nn.lunatp_helper as lunatp_helper
    from sevenn.checkpoint import _solve_accelerator

    monkeypatch.setattr(lunatp_helper, 'is_lunatp_available', lambda: False)
    with pytest.raises(ImportError):
        _solve_accelerator(False, False, False, True)
    with pytest.raises(ValueError):
        _solve_accelerator(False, False, True, True)


@pytest.mark.skipif(not is_lunatp_available(), reason='lunatp not available')
def test_auto_precompile_runs_once(tmp_path, monkeypatch):
    import sevenn.nn.lunatp_helper as helper

    popen = Mock()
    popen.return_value.wait.return_value = 0
    monkeypatch.setattr(helper.subprocess, 'Popen', popen)
    monkeypatch.setattr(
        helper._lunaeq_jit, 'compiler_identity', Mock(return_value=())
    )
    monkeypatch.setenv('LUNAEQ_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('SEVENNET_LUNATP_PRECOMPILE_WORKERS', '3')
    cfg = get_model_config()

    helper.auto_precompile({**cfg, KEY.RANDOM_SEED: 42, KEY.CONV_DENOMINATOR: 4.0})
    assert popen.call_count == 3
    helper.auto_precompile(cfg)
    assert popen.call_count == 3
    helper.auto_precompile({**cfg, 'lmax': 3})
    assert popen.call_count == 6
    monkeypatch.setenv('LD_LIBRARY_PATH', '/opt/another-env')
    helper.auto_precompile(cfg)
    assert popen.call_count == 9
    monkeypatch.setenv('SEVENNET_LUNATP_PRECOMPILE_WORKERS', '0')
    helper.auto_precompile({**cfg, 'lmax': 1})
    assert popen.call_count == 9
