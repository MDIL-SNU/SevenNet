import pytest
import torch
from ase.build import bulk, molecule
from ase.data import chemical_symbols
from torch_geometric.loader.dataloader import Collater

import sevenn.train.dataload as dl
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.util import chemical_species_preprocess

cutoff = 4.0


_samples = {
    'bulk': bulk('NaCl', 'rocksalt', a=5.63),
    'mol': molecule('H2O'),
    'isolated': molecule('H'),
}
n_samples = len(_samples)
n_atoms_total = sum([len(at) for at in _samples.values()])

_graph_list = [
    AtomGraphData.from_numpy_dict(dl.unlabeled_atoms_to_graph(at, cutoff))
    for at in list(_samples.values())
]


def test_chemical_species_preprocess():
    chems = ['He', 'H', 'Be', 'H']
    cf = chemical_species_preprocess(chems, universal=False)
    assert cf['chemical_species'] == ['Be', 'H', 'He']
    assert cf['_number_of_species'] == 3
    assert cf['_type_map'] == {4: 0, 1: 1, 2: 2}

    cf = chemical_species_preprocess(chems, universal=True)
    assert cf['chemical_species'] == chemical_symbols
    assert cf['_number_of_species'] == len(chemical_symbols)
    assert len(cf['_type_map']) == len(chemical_symbols)
    for z, node_idx in cf['_type_map'].items():
        assert z == node_idx


def get_graphs(batched):
    cloned = [g.clone() for g in _graph_list]
    if not batched:
        return cloned
    else:
        return Collater(cloned)(cloned)


def get_model_config():
    config = {
        'cutoff': cutoff,
        'channel': 4,
        'radial_basis': {
            'radial_basis_name': 'bessel',
        },
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'interaction_type': 'nequip',
        'lmax': 2,
        'is_parity': True,
        'num_convolution_layer': 3,
        'weight_nn_hidden_neurons': [64, 64],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': 1.0,
        'train_denominator': False,
        'self_connection_type': 'nequip',
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
    for at in list(_samples.values()):
        chems.update(at.get_chemical_symbols())
    config.update(**chemical_species_preprocess(list(chems)))
    return config


def get_model(config_overwrite={}):
    cf = get_model_config()
    cf.update(**config_overwrite)
    model = build_E3_equivariant_model(cf, parallel=False)
    assert isinstance(model, AtomGraphSequential)
    return model


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('cf', [{}])
def test_shape(cf, batched):
    model = get_model(cf)
    model.set_is_batch_data(batched)

    graph = get_graphs(batched)
    if not batched:
        output_shapes = {
            'inferred_total_energy': (),
            'inferred_stress': (6,),
        }
        for g in graph:
            natoms = g['num_atoms']
            output_shapes.update(
                {
                    'atomic_energy': (natoms, 1),  # intended
                    'inferred_force': (natoms, 3),
                }
            )
            output = model(g)
            for k, shape in output_shapes.items():
                assert output[k].shape == shape, f'{k}: {output[k].shape} != {shape}'
    else:
        output_shapes = {
            'inferred_total_energy': (n_samples,),
            'atomic_energy': (n_atoms_total, 1),  # intended
            'inferred_force': (n_atoms_total, 3),
            'inferred_stress': (n_samples, 6),
        }
        output = model(graph)
        for k, shape in output_shapes.items():
            assert output[k].shape == shape, f'{k}: {output[k].shape} != {shape}'


def test_batch():
    model = get_model()
    model.set_is_batch_data(False)

    graph_list = get_graphs(batched=False)
    output_list = [model(g) for g in graph_list]

    model.set_is_batch_data(True)
    graph_batch = get_graphs(batched=True)
    output_batched = model(graph_batch)

    e_concat = torch.concat(
        [g['inferred_total_energy'].unsqueeze(-1) for g in output_list]
    )
    ae_concat = torch.concat([g['atomic_energy'].squeeze(1) for g in output_list])
    f_concat = torch.concat([g['inferred_force'] for g in output_list])
    s_concat = torch.stack([g['inferred_stress'] for g in output_list])

    assert torch.allclose(e_concat, output_batched['inferred_total_energy'])
    assert torch.allclose(ae_concat, output_batched['atomic_energy'].squeeze(1))
    assert torch.allclose(
        torch.round(f_concat, decimals=5),
        torch.round(output_batched['inferred_force'], decimals=5),
        atol=1e-5
    )

    assert torch.allclose(  # TODO, hard-coded, assumes the first structure is bulk
        torch.round(s_concat[0], decimals=5),
        torch.round(output_batched['inferred_stress'][0], decimals=5),
    )


_n_param_tests = [
    ({}, 20642),
    ({'train_denominator': True}, 20642 + 3),
    ({'train_shift_scale': True}, 20642 + 2),
    ({'shift': [1.0] * 4}, 20642),
    ({'scale': [1.0] * 4,
      'train_shift_scale': True}, 20642 + 8),
    ({'num_convolution_layer': 4}, 33458),
    ({'lmax': 3}, 26866),
    ({'channel': 2}, 16883),
    ({'is_parity': False}, 20386),
    ({'self_connection_type': 'linear'}, 20114),
]


@pytest.mark.parametrize('cf,ref', _n_param_tests)
def test_num_params(cf, ref):
    model = get_model(cf)
    param = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert param == ref, f'ref: {ref} != given: {param}'


# TODO: test_irreps, test_gard, test_equivariance
