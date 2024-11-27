# TODO: add gradient test from total loss after double precision.
#       so far, it is empirically checked by seeing learning curves
import copy

import numpy as np
import pytest
import torch
from ase.build import bulk
from torch_geometric.loader.dataloader import Collater

import sevenn
import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.sevennet_calculator import SevenNetCalculator
from sevenn.util import (
    chemical_species_preprocess,
    model_from_checkpoint_with_backend,
)

cutoff = 4.0

_atoms = bulk('NaCl', 'rocksalt', a=4.00) * (2, 2, 2)
_avg_num_neigh = 30.0
_atoms.rattle()

_graph = AtomGraphData.from_numpy_dict(dl.unlabeled_atoms_to_graph(_atoms, cutoff))
print(_graph)


def get_graphs(batched):
    # batch size 2
    cloned = [_graph.clone(), _graph.clone()]
    if not batched:
        return cloned
    else:
        return Collater(cloned)(cloned)


def get_model_config():
    config = {
        'cutoff': cutoff,
        'channel': 8,
        'lmax': 2,
        'is_parity': True,
        'num_convolution_layer': 3,
        'self_connection_type': 'linear',  # not NequIp
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


def get_model(config_overwrite=None, use_cueq=False, cueq_config=None):
    cf = get_model_config()
    if config_overwrite is not None:
        cf.update(config_overwrite)

    cueq_config = cueq_config or {'cuequivariance_config': {'use': use_cueq}}
    cf.update(cueq_config)

    model = build_E3_equivariant_model(cf, parallel=False)
    assert isinstance(model, AtomGraphSequential)
    return model


@pytest.mark.parametrize(
    'cf',
    [
        ({}),
        ({'is_parity': False}),
        ({'channel': 7}),
        ({'lmax': 3}),
        ({'num_interaction_layer': 2}),
        ({'num_interaction_layer': 4}),
        # ({'self_connection_type': 'nequip'}),
    ],
)
def test_model_output(cf):
    torch.manual_seed(777)
    model_e3nn = get_model(cf)
    torch.manual_seed(777)
    model_cueq = get_model(cf, use_cueq=True)

    model_e3nn.set_is_batch_data(True)
    model_cueq.set_is_batch_data(True)

    e3nn_out = model_e3nn._preprocess(get_graphs(batched=True))
    cueq_out = model_cueq._preprocess(get_graphs(batched=True))

    for k, e3nn_f in model_e3nn._modules.items():
        cueq_f = model_cueq._modules[k]
        e3nn_out = e3nn_f(e3nn_out)  # type: ignore
        cueq_out = cueq_f(cueq_out)  # type: ignore
        assert torch.allclose(
            e3nn_out.x, cueq_out.x, atol=1e-6
        ), f'{k} \n\n {e3nn_f} \n\n {cueq_f}'

    assert torch.allclose(
        e3nn_out.inferred_total_energy, cueq_out.inferred_total_energy
    )
    assert torch.allclose(e3nn_out.atomic_energy, cueq_out.atomic_energy)
    assert torch.allclose(
        e3nn_out.inferred_force, cueq_out.inferred_force, atol=1e-5
    )
    assert torch.allclose(
        e3nn_out.inferred_stress, cueq_out.inferred_stress, atol=1e-5
    )


@pytest.mark.parametrize(
    'start_from_cueq',
    [
        (True),
        (False),
    ],
)
def test_checkpoint_convert(tmp_path, start_from_cueq):
    torch.manual_seed(123)
    model_from = get_model(use_cueq=start_from_cueq)

    cfg = get_model_config()
    cfg.update(
        {
            'cuequivariance_config': {'use': start_from_cueq},
            'version': sevenn.__version__,
        }
    )
    torch.save(
        {'model_state_dict': model_from.state_dict(), 'config': cfg},
        tmp_path / 'cp_from.pth',
    )

    backend = 'e3nn' if start_from_cueq else 'cueq'
    model_to, _ = model_from_checkpoint_with_backend(
        str(tmp_path / 'cp_from.pth'), backend
    )

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


@pytest.mark.parametrize(
    'start_from_cueq',
    [
        (True),
        (False),
    ],
)
def test_checkpoint_convert_no_batch(tmp_path, start_from_cueq):
    torch.manual_seed(123)
    model_from = get_model(use_cueq=start_from_cueq)

    cfg = get_model_config()
    cfg.update(
        {
            'cuequivariance_config': {'use': start_from_cueq},
            'version': sevenn.__version__,
        }
    )
    torch.save(
        {'model_state_dict': model_from.state_dict(), 'config': cfg},
        tmp_path / 'cp_from.pth',
    )

    backend = 'e3nn' if start_from_cueq else 'cueq'
    model_to, _ = model_from_checkpoint_with_backend(
        str(tmp_path / 'cp_from.pth'), backend
    )

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


def test_calculator(tmp_path):
    cueq = True
    model = get_model(use_cueq=cueq)
    ref_calc = SevenNetCalculator(model, file_type='model_instance')
    atoms = copy.deepcopy(_atoms)
    atoms.calc = ref_calc

    cfg = get_model_config()
    cfg.update(
        {'cuequivariance_config': {'use': cueq}, 'version': sevenn.__version__}
    )

    cp_path = str(tmp_path / 'cp.pth')
    torch.save(
        {'model_state_dict': model.state_dict(), 'config': cfg},
        cp_path,
    )

    calc2 = SevenNetCalculator(cp_path, enable_cueq=False)
    atoms2 = copy.deepcopy(_atoms)
    atoms2.calc = calc2

    assert_atoms(atoms, atoms2)
