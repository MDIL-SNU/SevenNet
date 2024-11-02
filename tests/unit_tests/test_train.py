import pathlib

import ase.io
import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

import sevenn.train.graph_dataset as graph_ds
from sevenn._const import NUM_UNIV_ELEMENT
from sevenn.scripts.processing_continue import processing_continue_v2
from sevenn.scripts.processing_epoch import processing_epoch_v2
from sevenn.sevenn_logger import Logger
from sevenn.train.dataload import graph_build
from sevenn.train.graph_dataset import from_config as dataset_from_config
from sevenn.train.trainer import Trainer
from sevenn.util import (
    chemical_species_preprocess,
    get_error_recorder,
    pretrained_name_to_path,
)

cutoff = 4.0

data_root = (pathlib.Path(__file__).parent.parent / 'data').resolve()

hfo2_path = str(data_root / 'systems' / 'hfo2.extxyz')
cp_0_path = str(data_root / 'checkpoints' / 'cp_0.pth')
sevennet_0_path = pretrained_name_to_path('7net-0_11July2024')

known_elements = ['Hf', 'O']
_elemwise_ref_energy_dct = {72: -17.379337, 8: -34.7499924}

Logger()  # init


@pytest.fixture()
def HfO2_atoms():
    atoms = ase.io.read(hfo2_path)
    return atoms


@pytest.fixture(scope='module')
def HfO2_loader():
    atoms = ase.io.read(hfo2_path, index=':')
    assert isinstance(atoms, list)
    graphs = graph_build(atoms, cutoff, y_from_calc=True)
    return DataLoader(graphs, batch_size=2)


@pytest.fixture()
def graph_dataset_path(tmp_path):
    ds = graph_ds.SevenNetGraphDataset(
        cutoff=cutoff, root=tmp_path, files=[hfo2_path], processed_name='tmp.pt'
    )
    return ds.processed_paths[0]


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
        'conv_denominator': 'avg_num_neigh',
        'train_denominator': False,
        'self_connection_type': 'nequip',
        'train_shift_scale': False,
        'irreps_manual': False,
        'lmax_edge': -1,
        'lmax_node': -1,
        'readout_as_fcn': False,
        'use_bias_in_linear': False,
        '_normalize_sph': True,
    }
    config.update(**chemical_species_preprocess(known_elements))
    return config


def get_train_config():
    config = {
        'random_seed': 1,
        'epoch': 2,
        'loss': 'mse',
        'loss_param': {},
        'optimizer': 'adam',
        'optim_param': {},
        'scheduler': 'exponentiallr',
        'scheduler_param': {'gamma': 0.99},
        'force_loss_weight': 1.0,
        'stress_loss_weight': 0.1,
        'per_epoch': 1,
        'continue': {
            'checkpoint': False,
            'reset_optimizer': False,
            'reset_scheduler': False,
            'reset_epoch': False,
        },
        'is_train_stress': True,
        'train_shuffle': True,
        'best_metric': 'TotalLoss',
        'error_record': [
            ('Energy', 'RMSE'),
            ('Force', 'RMSE'),
            ('Stress', 'RMSE'),
            ('TotalLoss', 'None'),
        ],
        'device': 'cpu',
        'is_ddp': False,
    }
    return config


def get_data_config():
    config = {
        'batch_size': 2,
        'shift': 'per_atom_energy_mean',
        'scale': 'force_rms',
        'preprocess_num_cores': 1,
        'data_format_args': {},
        'load_trainset_path': hfo2_path,
    }
    return config


def get_config(overwrite={}):
    cf = {}
    cf.update(get_model_config())
    cf.update(get_train_config())
    cf.update(get_data_config())
    cf.update(overwrite)
    return cf


def test_processing_continue_v2_7net0(tmp_path):
    cp = torch.load(sevennet_0_path, weights_only=False, map_location='cpu')

    cfg = get_config(
        {
            'continue': {
                'checkpoint': sevennet_0_path,
                'reset_optimizer': False,
                'reset_scheduler': True,
                'reset_epoch': False,
            }
        }
    )
    shift_ref = cp['model_state_dict']['rescale_atomic_energy.shift'].cpu().numpy()
    scale_ref = np.array([1.73] * 89)
    conv_denominator_ref = np.array([35.989574] * 5)

    with Logger().switch_file(str(tmp_path / 'log.sevenn')):
        state_dicts, epoch = processing_continue_v2(cfg)
    assert epoch == 601
    assert np.allclose(np.array(cfg['shift']), shift_ref)
    assert np.allclose(np.array(cfg['shift'])[0], -5.062768)
    assert np.allclose(np.array(cfg['scale']), scale_ref)
    assert np.allclose(np.array(cfg['conv_denominator']), conv_denominator_ref)
    assert cfg['_number_of_species'] == 89
    assert cfg['_type_map'][89] == 0  # Ac
    assert cfg['_type_map'][40] == 88  # Zr
    assert state_dicts[2] is None  # scheduler reset


@pytest.mark.parametrize(
    'cfg_overwrite,ds_names',
    [
        ({}, ['trainset']),
        ({'load_myset_path': hfo2_path}, ['trainset', 'myset']),
    ],
)
def test_dataset_from_config(cfg_overwrite, ds_names, tmp_path):
    cfg = get_config(cfg_overwrite)
    with Logger().switch_file(str(tmp_path / 'log.sevenn')):
        datasets = dataset_from_config(cfg, tmp_path)

    assert set(ds_names) == set(datasets.keys())
    for ds_name in ds_names:
        assert (tmp_path / 'sevenn_data' / f'{ds_name}.pt').is_file()
        assert (tmp_path / 'sevenn_data' / f'{ds_name}.yaml').is_file()


def test_dataset_from_config_as_it_is_load(graph_dataset_path, tmp_path):
    cfg = get_config({'load_trainset_path': graph_dataset_path})
    print(graph_dataset_path)
    new_wd = tmp_path / 'tmp_wd'
    with Logger().switch_file(str(tmp_path / 'log.sevenn')):
        _ = dataset_from_config(cfg, str(new_wd))
    print((tmp_path / 'tmp_wd' / 'sevenn_data'))
    assert not (tmp_path / 'tmp_wd' / 'sevenn_data').is_dir()


@pytest.mark.parametrize(
    'cfg_overwrite,shift,scale,conv',
    [
        (
            {},
            -28.978,
            0.113304,
            25.333333,
        ),
        (
            {
                'shift': -1.2345678,
            },
            -1.234567,
            0.113304,
            25.333333,
        ),
        (
            {
                'conv_denominator': 'sqrt_avg_num_neigh',
            },
            -28.978,
            0.113304,
            25.333333**0.5,
        ),
        (
            {
                'shift': 'force_rms',
            },
            0.113304,
            0.113304,
            25.333333,
        ),
        (
            {
                'shift': 'elemwise_reference_energies',
            },
            [
                0.0
                if z not in _elemwise_ref_energy_dct
                else _elemwise_ref_energy_dct[z]
                for z in range(NUM_UNIV_ELEMENT)
            ],
            0.113304,
            25.333333,
        ),
    ],
)
def test_dataset_from_config_statistics_init(
    cfg_overwrite, shift, scale, conv, tmp_path
):
    cfg = get_config(cfg_overwrite)
    with Logger().switch_file(str(tmp_path / 'log.sevenn')):
        _ = dataset_from_config(cfg, tmp_path)

    assert np.allclose(cfg['shift'], shift)
    assert np.allclose(cfg['scale'], scale)
    assert np.allclose(cfg['conv_denominator'], conv)


def test_dataset_from_config_chem_auto(tmp_path):
    cfg = get_config(
        {
            'chemical_species': 'auto',
            '_number_of_species': 'auto',
            '_type_map': 'auto',
        }
    )
    with Logger().switch_file(str(tmp_path / 'log.sevenn')):
        _ = dataset_from_config(cfg, tmp_path)
    assert cfg['chemical_species'] == ['Hf', 'O']
    assert cfg['_number_of_species'] == 2
    assert cfg['_type_map'] == {72: 0, 8: 1}


def test_run_one_epoch(HfO2_loader):
    trainer_args, _, _ = Trainer.args_from_checkpoint(cp_0_path)
    trainer = Trainer(**trainer_args)
    erc = get_error_recorder()

    ref1 = {
        'Energy_RMSE': '28.977758',
        'Force_RMSE': '0.214107',
        'Stress_RMSE': '190.014237',
    }

    ref2 = {
        'Energy_RMSE': '28.977878',
        'Force_RMSE': '0.213105',
        'Stress_RMSE': '188.772557',
    }

    trainer.run_one_epoch(HfO2_loader, is_train=False, error_recorder=erc)
    ret1 = erc.get_dct()
    erc.epoch_forward()

    for k in ref1:
        assert np.allclose(float(ret1[k]), float(ref1[k]))

    trainer.run_one_epoch(HfO2_loader, is_train=True, error_recorder=erc)
    erc.epoch_forward()

    trainer.run_one_epoch(HfO2_loader, is_train=False, error_recorder=erc)
    ret2 = erc.get_dct()
    erc.epoch_forward()

    for k in ref2:
        assert np.allclose(float(ret2[k]), float(ref2[k]))


def test_processing_epoch_v2(HfO2_loader, tmp_path):
    trainer_args, _, _ = Trainer.args_from_checkpoint(cp_0_path)
    trainer = Trainer(**trainer_args)
    erc = get_error_recorder()
    start_epoch = 10
    total_epoch = 12
    per_epoch = 1
    best_metric = 'Energy_RMSE'
    best_metric_loader_key = 'myset'
    loaders = {'trainset': HfO2_loader, 'myset': HfO2_loader}

    with Logger().switch_file(str(tmp_path / 'log.sevenn')):
        processing_epoch_v2(
            config={},
            trainer=trainer,
            loaders=loaders,
            start_epoch=start_epoch,
            error_recorder=erc,
            total_epoch=total_epoch,
            per_epoch=per_epoch,
            best_metric_loader_key=best_metric_loader_key,
            best_metric=best_metric,
            working_dir=tmp_path,
        )
    assert (tmp_path / 'checkpoint_10.pth').is_file()
    assert (tmp_path / 'checkpoint_11.pth').is_file()
    assert (tmp_path / 'checkpoint_12.pth').is_file()
    assert (tmp_path / 'checkpoint_best.pth').is_file()
    assert (tmp_path / 'lc.csv').is_file()
    with open(tmp_path / 'lc.csv', 'r') as f:
        lines = f.readlines()
    heads = [ll.strip() for ll in lines[0].split(',')]
    assert 'epoch' in heads
    assert 'lr' in heads
    assert 'trainset_Energy_RMSE' in heads
    assert 'myset_Stress_MAE' in heads
    lasts = [ll.strip() for ll in lines[-1].split(',')]
    assert lasts[0] == '12'
    assert lasts[1] == '0.000980'  # lr
    assert lasts[-2] == '0.087873'  # myset Force MAE


def _write_empty_checkpoint():
    from sevenn.model_build import build_E3_equivariant_model

    # Function I used to make empty checkpoint, to write the test
    cfg = get_config({'shift': 0.0, 'scale': 1.0, 'conv_denominator': 5.0})
    model = build_E3_equivariant_model(cfg)
    trainer = Trainer.from_config(model, cfg)  # type: ignore
    trainer.write_checkpoint('./cp_0.pth', config=cfg, epoch=0)


if __name__ == '__main__':
    _write_empty_checkpoint()
