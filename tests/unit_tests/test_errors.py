# test_errors: error recorder.py, loss.py
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn
from torch import tensor

import sevenn.error_recorder as erc
import sevenn.train.loss as loss
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.optim import loss_dict

_default_config = {
    'loss': 'mse',
    'loss_param': {},
    'error_record': [
        ('Energy', 'RMSE'),
        ('Force', 'RMSE'),
        ('Stress', 'RMSE'),
        ('Energy', 'MAE'),
        ('Force', 'MAE'),
        ('Stress', 'MAE'),
        ('TotalLoss', 'None'),
    ],
    'is_train_stress': True,
    'force_loss_weight': 1.0,
    'stress_loss_weight': 0.001,
}

_erc_test_params = [
    ('TotalEnergy', 4, 3),
    ('Energy', 4, 3),
    ('Force', 4, 3),
    ('Stress', 4, 3),
    ('Stress_GPa', 4, 3),
    ('Energy', 4, 1),
    ('Energy', 1, 1),
    ('Force', 1, 3),
    ('Stress', 1, 3),
]


def acl(a, b):
    return torch.allclose(a, b, atol=1e-6)


def config(**overwrite):  # to make it read-only
    cf = deepcopy(_default_config)
    for k, v in overwrite.items():
        cf[k] = v
    return cf


def test_per_atom_energy_loss():
    loss_f = loss.PerAtomEnergyLoss(criterion=torch.nn.MSELoss())
    ref = torch.rand(2)
    pred = torch.rand(2)
    natoms = torch.randint(1, 10, (2,))
    tmp = AtomGraphData(
        total_energy=ref,
        inferred_total_energy=pred,
        num_atoms=natoms,
    ).to_dict()
    ret = loss_f.get_loss(tmp)
    assert loss_f.criterion is not None
    assert torch.allclose(loss_f.criterion((ref / natoms), (pred / natoms)), ret)


def test_force_loss():
    loss_f = loss.ForceLoss(criterion=torch.nn.MSELoss())
    ref = torch.rand((4, 3))
    pred = torch.rand((4, 3))
    batch = tensor([0, 0, 0, 1])
    tmp = AtomGraphData(
        force_of_atoms=ref,
        inferred_force=pred,
        batch=batch,
    ).to_dict()
    ret = loss_f.get_loss(tmp)
    assert loss_f.criterion is not None
    assert torch.allclose(loss_f.criterion(ref.reshape(-1), pred.reshape(-1)), ret)


def test_stress_loss():
    loss_f = loss.StressLoss(criterion=torch.nn.MSELoss())
    ref = torch.rand((2, 6))
    pred = torch.rand((2, 6))
    tmp = AtomGraphData(
        stress=ref,
        inferred_stress=pred,
    ).to_dict()
    ret = loss_f.get_loss(tmp)
    KB = 1602.1766208
    assert loss_f.criterion is not None
    assert torch.allclose(
        loss_f.criterion(ref.reshape(-1) * KB, pred.reshape(-1) * KB), ret
    )


@pytest.mark.parametrize('conf', [config(), config(is_train_stress=False)])
def test_loss_from_config(conf):
    loss_functions = loss.get_loss_functions_from_config(conf)

    if conf['is_train_stress']:
        assert len(loss_functions) == 3
    else:
        assert len(loss_functions) == 2

    for loss_def, w in loss_functions:
        assert isinstance(loss_def, loss.LossDefinition)
        if isinstance(loss_def, loss.PerAtomEnergyLoss):
            assert w == 1.0
        elif isinstance(loss_def, loss.ForceLoss):
            assert w == conf['force_loss_weight']
        elif isinstance(loss_def, loss.StressLoss):
            assert w == conf['stress_loss_weight']
        else:
            raise ValueError(f'Unexpected loss function: {loss_def}')


@pytest.mark.parametrize('err_type,ndata,natoms', _erc_test_params)
def test_rms_error(err_type, ndata, natoms):
    err_dct = erc.get_err_type(err_type)
    err = erc.RMSError(**err_dct)
    ref = torch.rand((ndata, err.vdim)).squeeze(1)
    pred = torch.rand((ndata, err.vdim)).squeeze(1)
    natoms = torch.tensor([natoms] * ndata)
    _data = {
        err_dct['ref_key']: ref,
        err_dct['pred_key']: pred,
        'num_atoms': natoms,
    }

    tmp = AtomGraphData(**_data)
    err.update(tmp)

    _ref = ref * err.coeff
    _pred = pred * err.coeff
    if 'per_atom' in err_dct and err_dct['per_atom']:
        # natoms = natoms.unsqueeze(-1)
        _ref = _ref / natoms
        _pred = _pred / natoms
    val = torch.sqrt(((_ref - _pred) ** 2).sum() / ndata)  # not ndata*natoms
    assert np.allclose(err.get(), val.item())
    err.update(tmp)
    assert np.allclose(err.get(), val.item())


@pytest.mark.parametrize('err_type,ndata,natoms', _erc_test_params)
def test_mae_error(err_type, ndata, natoms):
    err_dct = erc.get_err_type(err_type)
    vdim = err_dct['vdim']
    err = erc.MAError(**err_dct)
    ref = torch.rand((ndata, vdim)).squeeze(1)
    pred = torch.rand((ndata, vdim)).squeeze(1)
    natoms = torch.tensor([natoms] * ndata)
    _data = {
        err_dct['ref_key']: ref,
        err_dct['pred_key']: pred,
        'num_atoms': natoms,
    }

    tmp = AtomGraphData(**_data)
    err.update(tmp)

    _ref = ref * err.coeff
    _pred = pred * err.coeff
    if 'per_atom' in err_dct and err_dct['per_atom']:
        _ref /= natoms
        _pred /= natoms

    val = abs(_ref - _pred).sum() / (ndata * vdim)
    assert np.allclose(err.get(), val.item())
    err.update(tmp)
    assert np.allclose(err.get(), val.item())


# TODO: test_component_rms_error


@pytest.mark.parametrize('err_type,ndata,natoms', _erc_test_params)
def test_custom_error(err_type, ndata, natoms):
    def func(a, b):
        return a * b

    err_dct = erc.get_err_type(err_type)
    vdim = err_dct['vdim']
    err = erc.CustomError(func, **err_dct)
    ref = torch.rand((ndata, vdim)).squeeze(1)
    pred = torch.rand((ndata, vdim)).squeeze(1)
    natoms = torch.tensor([natoms] * ndata)
    _data = {
        err_dct['ref_key']: ref,
        err_dct['pred_key']: pred,
        'num_atoms': natoms,
    }

    _ref = ref * err.coeff
    _pred = pred * err.coeff
    if 'per_atom' in err_dct and err_dct['per_atom']:
        _ref /= natoms
        _pred /= natoms

    tmp = AtomGraphData(**_data)
    err.update(tmp)
    val = func(_ref, _pred).mean()
    assert np.allclose(err.get(), val.item())
    err.update(tmp)
    assert np.allclose(err.get(), val.item())


@pytest.mark.parametrize('conf', [config(), config(is_train_stress=False)])
def test_total_loss_metric_from_config(conf):
    def func(a, b):
        return a * b

    err = erc.ErrorRecorder.init_total_loss_metric(conf, func)
    ndata = 3
    natoms = 4

    e1, e2 = torch.rand(ndata), torch.rand(ndata)
    f1, f2 = torch.rand(ndata * natoms, 3), torch.rand(ndata * natoms, 3)
    s1, s2 = torch.rand((ndata, 6)), torch.rand((ndata, 6))
    _data = {
        'total_energy': e1,
        'inferred_total_energy': e2,
        'force_of_atoms': f1,
        'inferred_force': f2,
        'stress': s1,
        'inferred_stress': s2,
        'num_atoms': torch.tensor([natoms] * ndata),
    }

    tmp = AtomGraphData(**_data)
    err.update(tmp)

    val = (func(e1 / natoms, e2 / natoms)).mean() + conf['force_loss_weight'] * func(
        f1, f2
    ).mean()
    if conf['is_train_stress']:
        KB = 1602.1766208
        val += conf['stress_loss_weight'] * func(s1 * KB, s2 * KB).mean()

    assert np.allclose(err.get(), val.item())
    err.update(tmp)
    assert np.allclose(err.get(), val.item())


@pytest.mark.parametrize(
    'conf', [config(), config(is_train_stress=False), config(loss='huber')]
)
def test_error_recorder_from_config(conf):
    recorder = erc.ErrorRecorder.from_config(conf)

    total_loss_flag = False
    for metric in recorder.metrics:
        if conf['is_train_stress'] is False:
            assert 'stress' not in metric.name
        if metric.name == 'TotalLoss':
            total_loss_flag = True
            for loss_metric, _ in metric.metrics:  # type: ignore
                print(loss_metric.func)
                assert isinstance(loss_metric.func, loss_dict[conf['loss']])
    assert total_loss_flag
