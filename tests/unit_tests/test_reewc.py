import pathlib

import ase.io
import pytest
import torch
from torch_geometric.loader import DataLoader

from sevenn.logger import Logger
from sevenn.train.dataload import graph_build
from sevenn.train.reewc.loss import EWCLoss
from sevenn.train.reewc.trainer import ReewcTrainer
from sevenn.train.trainer import Trainer

Logger()  # init singleton used by train_v2


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(3))
        self.b = torch.nn.Parameter(torch.ones(2))


def _fisher_opt(model):
    fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    opt = {n: p.detach().clone() for n, p in model.named_parameters()}
    return fisher, opt


def test_ewc_loss_zero_at_optimum():
    model = _TinyModel()
    fisher, opt = _fisher_opt(model)
    ewc = EWCLoss(fisher, opt)
    assert float(ewc.get_loss({}, model)) == pytest.approx(0.0)


def test_ewc_loss_positive_after_perturbation():
    model = _TinyModel()
    fisher, opt = _fisher_opt(model)
    ewc = EWCLoss(fisher, opt)
    with torch.no_grad():
        model.a += 2.0
    # sum_i fisher_i * (p_i - opt_i)^2 = 3 * (1 * 2^2) = 12
    assert float(ewc.get_loss({}, model)) == pytest.approx(12.0)


def test_ewc_loss_requires_model():
    model = _TinyModel()
    fisher, opt = _fisher_opt(model)
    ewc = EWCLoss(fisher, opt)
    with pytest.raises(ValueError):
        ewc.get_loss({}, None)


def test_ewc_loss_fisher_must_be_dict():
    with pytest.raises(ValueError):
        EWCLoss([1, 2, 3], {})


def test_ewc_loss_no_matching_param_raises():
    model = _TinyModel()
    fisher = {'bogus.param': torch.ones(3)}
    opt = {'bogus.param': torch.zeros(3)}
    ewc = EWCLoss(fisher, opt)
    with pytest.raises(ValueError):
        ewc.get_loss({}, model)


def test_ewc_loss_shape_mismatch_raises():
    model = _TinyModel()
    fisher = {'a': torch.ones(5)}  # model.a has shape (3,)
    opt = {'a': torch.zeros(5)}
    ewc = EWCLoss(fisher, opt)
    with pytest.raises(ValueError):
        ewc.get_loss({}, model)


def test_ewc_loss_partial_fisher_warns_and_skips():
    # a trainable param without a Fisher entry is left unconstrained, not fatal
    model = _TinyModel()
    fisher = {'a': torch.ones(3)}  # no entry for 'b'
    opt = {'a': torch.zeros(3)}
    ewc = EWCLoss(fisher, opt)
    with torch.no_grad():
        model.a += 1.0  # 'b' stays at its optimum
    with pytest.warns(UserWarning):
        loss = float(ewc.get_loss({}, model))
    assert loss == pytest.approx(3.0)  # 3 * (1 * 1^2); 'b' contributes nothing


def test_ewc_loss_opt_shape_mismatch_raises():
    model = _TinyModel()
    fisher = {'a': torch.ones(3), 'b': torch.ones(2)}
    opt = {'a': torch.zeros(3), 'b': torch.zeros(5)}  # wrong opt shape for 'b'
    ewc = EWCLoss(fisher, opt)
    with pytest.raises(ValueError):
        ewc.get_loss({}, model)


def test_ewc_loss_empty_fisher_raises():
    model = _TinyModel()
    ewc = EWCLoss({}, {})
    with pytest.raises(ValueError):  # clean error, not StopIteration
        ewc.get_loss({}, model)


def test_continue_ewc_keys_survive_parse():
    # continue.fisher_information / opt_params / ewc_lambda must survive config
    # normalization so EWC is not silently disabled on a real YAML run.
    from sevenn.parse_input import init_train_config

    cfg = {
        'optimizer': 'adam',
        'scheduler': 'exponentiallr',
        'continue': {
            'checkpoint': '7net-0',
            'fisher_information': '/x/fisher.pt',
            'opt_params': '/x/opt.pt',
            'ewc_lambda': 1000,
        },
    }
    train_meta = init_train_config(cfg)
    cont = train_meta['continue']
    assert cont['fisher_information'] == '/x/fisher.pt'
    assert cont['opt_params'] == '/x/opt.pt'
    assert cont['ewc_lambda'] == 1000


# --- Scheduler: native CosineAnnealingWarmupRestarts ---


def _make_opt():
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.0)


def test_warmup_scheduler_registered():
    from sevenn.train.optim import scheduler_dict

    assert 'cosineannealingwarmuplr' in scheduler_dict


def test_warmup_scheduler_known_lrs():
    from sevenn.train.optim import CosineAnnealingWarmupRestarts

    opt = _make_opt()
    sch = CosineAnnealingWarmupRestarts(
        opt, first_cycle_steps=10, max_lr=1.0, min_lr=0.0, warmup_steps=4
    )
    assert opt.param_groups[0]['lr'] == pytest.approx(0.0)  # init = min_lr
    sch.step()  # warmup step 1 of 4 -> max_lr * 1/4
    assert opt.param_groups[0]['lr'] == pytest.approx(0.25)
    for _ in range(3):  # warmup end (step 4) -> max_lr
        sch.step()
    assert opt.param_groups[0]['lr'] == pytest.approx(1.0)
    for _ in range(6):  # full cycle (step 10) -> restart to min_lr
        sch.step()
    assert opt.param_groups[0]['lr'] == pytest.approx(0.0)


# --- Config injection + validation (get_loss_functions_from_config) ---


def _base_loss_cfg():
    return {
        'loss': 'huber',
        'loss_param': {'delta': 0.01},
        'use_weight': False,
        'force_loss_weight': 1.0,
        'stress_loss_weight': 0.01,
        'is_train_stress': True,
        'device': 'cpu',
        'continue': {'checkpoint': False},
    }


def _write_ewc_pkls(tmp_path):
    fisher = {'p': torch.ones(2)}
    opt = {'p': torch.zeros(2)}
    fp, op = tmp_path / 'fisher.pkl', tmp_path / 'opt.pkl'
    torch.save(fisher, fp)
    torch.save(opt, op)
    return str(fp), str(op)


def test_loss_functions_inject_ewc(tmp_path):
    from sevenn.train.loss import get_loss_functions_from_config
    from sevenn.train.reewc.loss import EWCLoss

    fp, op = _write_ewc_pkls(tmp_path)
    cfg = _base_loss_cfg()
    cfg['continue'] = {
        'checkpoint': False,
        'fisher_information': fp,
        'opt_params': op,
        'ewc_lambda': 100.0,
    }
    lfs = get_loss_functions_from_config(cfg)
    ewc = [(ld, w) for ld, w in lfs if isinstance(ld, EWCLoss)]
    assert len(ewc) == 1
    assert ewc[0][1] == pytest.approx(50.0)  # ewc_lambda / 2


def test_no_ewc_keys_no_ewc_loss():
    from sevenn.train.loss import get_loss_functions_from_config
    from sevenn.train.reewc.loss import EWCLoss

    lfs = get_loss_functions_from_config(_base_loss_cfg())
    assert not any(isinstance(ld, EWCLoss) for ld, _ in lfs)


def test_loss_ewc_requires_both_fisher_and_opt(tmp_path):
    from sevenn.train.loss import get_loss_functions_from_config

    fp, _ = _write_ewc_pkls(tmp_path)
    cfg = _base_loss_cfg()
    cfg['continue'] = {
        'checkpoint': False,
        'fisher_information': fp,
        'ewc_lambda': 100.0,
    }
    with pytest.raises(ValueError):
        get_loss_functions_from_config(cfg)


def test_loss_ewc_lambda_only_raises():
    from sevenn.train.loss import get_loss_functions_from_config

    cfg = _base_loss_cfg()
    cfg['continue'] = {'checkpoint': False, 'ewc_lambda': 100.0}
    with pytest.raises(ValueError):
        get_loss_functions_from_config(cfg)


def test_loss_ewc_nonpositive_lambda_raises(tmp_path):
    from sevenn.train.loss import get_loss_functions_from_config

    fp, op = _write_ewc_pkls(tmp_path)
    cfg = _base_loss_cfg()
    cfg['continue'] = {
        'checkpoint': False,
        'fisher_information': fp,
        'opt_params': op,
        'ewc_lambda': 0,
    }
    with pytest.raises(ValueError):
        get_loss_functions_from_config(cfg)


# --- Config keys parsed + default values ---


def test_rehearsal_data_keys_parsed():
    from sevenn.parse_input import init_data_config

    data_meta = init_data_config(
        {'batch_size': 8, 'rehearsal': True, 'mem_batch_size': 4, 'mem_ratio': 0.5}
    )
    assert data_meta['rehearsal'] is True
    assert data_meta['mem_batch_size'] == 4
    assert data_meta['mem_ratio'] == 0.5


def test_rehearsal_defaults_off():
    from sevenn.parse_input import init_data_config

    data_meta = init_data_config({'batch_size': 8})
    assert data_meta['rehearsal'] is False  # default is off


# --- Replay (rehearsal) in Trainer.run_one_epoch ---

_data_root = (pathlib.Path(__file__).parent.parent / 'data').resolve()
_hfo2_path = str(_data_root / 'systems' / 'hfo2.extxyz')
_cp_0_path = str(_data_root / 'checkpoints' / 'cp_0.pth')


@pytest.fixture(scope='module')
def hfo2_loader():
    atoms = ase.io.read(_hfo2_path, index=':')
    graphs = graph_build(atoms, 4.0, y_from_calc=True)
    return DataLoader(graphs, batch_size=2)


class _CountingLoader:
    def __init__(self, base):
        self.base = base
        self.pulls = 0

    def __iter__(self):
        for batch in self.base:
            self.pulls += 1
            yield batch


def test_replay_consumed_only_in_train(hfo2_loader):
    targs, _, _ = Trainer.args_from_checkpoint(_cp_0_path)
    mem = _CountingLoader(hfo2_loader)
    trainer = ReewcTrainer(**targs, device='cpu', memory_loader=mem)
    n_batches = sum(1 for _ in hfo2_loader)
    trainer.run_one_epoch(hfo2_loader, is_train=True)
    assert mem.pulls == n_batches  # one memory batch per training batch
    mem.pulls = 0
    trainer.run_one_epoch(hfo2_loader, is_train=False)
    assert mem.pulls == 0  # replay is skipped during evaluation


def test_replay_changes_params(hfo2_loader):
    targs, _, _ = Trainer.args_from_checkpoint(_cp_0_path)
    trainer = ReewcTrainer(**targs, device='cpu', memory_loader=hfo2_loader)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.run_one_epoch(hfo2_loader, is_train=True)
    after = list(trainer.model.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_replay_records_memory_metrics(hfo2_loader):
    # the replayed memory set is recorded so it can be logged as 'memoryset'
    from sevenn.util import get_error_recorder

    targs, _, _ = Trainer.args_from_checkpoint(_cp_0_path)
    trainer = ReewcTrainer(**targs, device='cpu', memory_loader=hfo2_loader)
    mem_rec = get_error_recorder([('Energy', 'RMSE'), ('Force', 'RMSE')])
    trainer.run_one_epoch(
        hfo2_loader, is_train=True, memory_error_recorder=mem_rec
    )
    errs = mem_rec.epoch_forward()
    assert len(errs) > 0
    assert all(v == v for v in errs.values())  # finite (recorder was updated)


# --- train_v2 reEWC guards (fail-loud, no silent fallback) ---


def test_train_v2_reewc_ddp_guard(tmp_path):
    from sevenn.scripts.train import train_v2

    cfg = {'rehearsal': True, 'is_ddp': True, 'continue': {'checkpoint': False}}
    with pytest.raises(NotImplementedError):
        train_v2(cfg, str(tmp_path))


def test_train_v2_memory_path_requires_rehearsal(tmp_path):
    from sevenn.scripts.train import train_v2

    cfg = {
        'rehearsal': False,
        'load_memory_path': ['/nonexistent.sevenn_data'],
        'is_ddp': False,
        'continue': {'checkpoint': False},
    }
    with pytest.raises(ValueError):
        train_v2(cfg, str(tmp_path))


def test_train_v2_single_modal_guard(tmp_path):
    from sevenn.scripts.train import train_v2

    cfg = {
        'rehearsal': True,
        'is_ddp': False,
        'use_modality': True,
        'load_memory_path': ['/nonexistent.sevenn_data'],
        'continue': {'checkpoint': False},
    }
    with pytest.raises(ValueError):
        train_v2(cfg, str(tmp_path))


def test_train_v2_rehearsal_atoms_guard(tmp_path):
    from sevenn.scripts.train import train_v2

    cfg = {
        'rehearsal': True,
        'is_ddp': False,
        'dataset_type': 'atoms',
        'load_memory_path': ['/nonexistent.sevenn_data'],
        'continue': {'checkpoint': False},
    }
    with pytest.raises(NotImplementedError):
        train_v2(cfg, str(tmp_path))
