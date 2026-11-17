"""
train_shift_modals (model section of the yaml): only the shift rows of the
listed modalities train, every other row stays bit-identical, nothing else is
frozen. CPU, tiny model, a few optimizer steps. train_v2 end to end at the end.
Not covered on CPU: cuda_graph (the capture only holds the Muon group, the
shift is in the auxiliary Adam group) and DDP (the hook runs before the reducer
reads .grad).
"""
import copy
import pathlib
from unittest import mock

import ase.io
import numpy as np
import pytest
import torch
import yaml
from torch_geometric.loader import DataLoader

import sevenn
import sevenn._const as _const
import sevenn._keys as KEY
from sevenn.logger import Logger
from sevenn.model_build import build_E3_equivariant_model
from sevenn.parse_input import init_model_config, init_train_config
from sevenn.scripts.processing_continue import processing_continue_v2
from sevenn.train.dataload import graph_build
from sevenn.train.loss import get_modal_regularization
from sevenn.train.muon import MuonWithAuxAdam
from sevenn.train.reewc.trainer import ReewcTrainer
from sevenn.train.trainer import Trainer
from sevenn.util import load_checkpoint
from tests.unit_tests.test_arch_options import ARCHS, with_arch

hfo2_path = str(
    (pathlib.Path(__file__).parent.parent / 'data' / 'systems' / 'hfo2.extxyz')
    .resolve()
)
cutoff = 4.0
MODALS = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
TRAINED = ['b', 'd']
FIXED_ROWS = [0, 2]
TRAINED_ROWS = [1, 3]

_MUON = {'muon': {'lr': 0.01}, 'adam': {'lr': 0.005}}
_MODEL = {
    'chemical_species': ['Hf', 'O'],
    'cutoff': cutoff,
    'channel': 4,
    'lmax': 1,
    'num_convolution_layer': 2,
    'weight_nn_hidden_neurons': [8, 8],
    'self_connection_type': 'linear',
    'use_modal_node_embedding': False,
    'use_modal_self_inter_intro': True,
    'use_modal_self_inter_outro': True,
    'use_modal_output_block': True,
}

Logger()  # init


def _cfg(model=None, **train):
    """global config as train_v2 sees it after the datasets are loaded"""
    train = {
        'device': 'cpu',
        'optimizer': 'adam',
        'optim_param': {'lr': 0.01},
        'scheduler': 'exponentiallr',
        'scheduler_param': {'gamma': 0.99},
        'force_loss_weight': 1.0,
        'stress_loss_weight': 0.01,
        'use_modality': True,
        **train,
    }
    model = {KEY.TRAIN_SHIFT_MODALS: TRAINED, **(model or {})}
    cfg = init_model_config({**_MODEL, **model})
    cfg.update(init_train_config(train))
    cfg.update({
        'version': sevenn.__version__,
        KEY.USE_MODALITY: True,
        KEY.MODAL_MAP: dict(MODALS),
        KEY.NUM_MODALITIES: len(MODALS),
        KEY.USE_MODAL_WISE_SHIFT: True,
        KEY.USE_MODAL_WISE_SCALE: False,
        KEY.SHIFT: [[-1.0 - m, -2.0 - m] for m in range(len(MODALS))],
        KEY.SCALE: 1.5,
        KEY.CONV_DENOMINATOR: [5.0, 5.0],
    })
    return cfg


def _model(cfg):
    torch.manual_seed(0)
    return build_E3_equivariant_model(copy.deepcopy(cfg))


@pytest.fixture(scope='module')
def loader():
    atoms = ase.io.read(hfo2_path, index=':')
    atoms = (atoms * 4)[:8]
    graphs = graph_build(atoms, cutoff, y_from_calc=True)
    for i, g in enumerate(graphs):
        g[KEY.DATA_MODALITY] = list(MODALS)[i % len(MODALS)]
    return DataLoader(graphs, batch_size=4)


def _shift(trainer):
    return trainer.model._modules['rescale_atomic_energy'].shift


def _bytes(t):
    return t.detach().cpu().numpy().tobytes()


def _check_rows(before, shift):
    assert _bytes(shift[FIXED_ROWS]) == _bytes(before[FIXED_ROWS])
    for row in TRAINED_ROWS:
        assert not torch.equal(shift[row], before[row])


def _others(trainer):
    return {
        n: p.detach().clone() for n, p in trainer.model.named_parameters()
        if 'rescale_atomic_energy' not in n
    }


@pytest.mark.parametrize(
    'optimizer,optim_param',
    [
        ('adam', {'lr': 0.01}),
        ('adam', {'lr': 0.01, 'amsgrad': True}),
        ('adamw', {'lr': 0.01, 'weight_decay': 0.0}),
        ('radam', {'lr': 0.01}),
        # lr 0.01 with momentum 0.9 diverges (NaN) on the ungated model
        ('sgd', {'lr': 0.002, 'momentum': 0.5}),
        ('adagrad', {'lr': 0.01}),
        ('muon', _MUON),
        ('muon', {**_MUON, 'batched': False}),
    ],
)
@pytest.mark.parametrize('grad_clip', [None, 1e-3])
@pytest.mark.parametrize('arch', ARCHS)
def test_masked_rows_bit_identical(
    loader, optimizer, optim_param, grad_clip, arch
):
    """
    grad clip active (tiny threshold), L2_modal and group alignment on; both
    model build paths (gate: the modal one-hots of the linears are gated)
    """
    cfg = _cfg(
        model=with_arch({}, arch),
        optimizer=optimizer,
        optim_param=copy.deepcopy(optim_param),
        grad_clip=grad_clip,
        regularization_param={
            'modal': {'regularization_weight': 1e-2},
            'modal_group_alignment': {
                'alignment_weight': 1e-2, 'groups': {'g': ['a', 'b', 'c']},
            },
        },
    )
    model = _model(cfg)
    frozen_before = {n for n, p in model.named_parameters() if not p.requires_grad}
    trainer = Trainer.from_config(model, cfg)
    assert len(trainer.loss_functions) == 5  # e, f, s + the two regularizers

    shift = _shift(trainer)
    assert shift.requires_grad
    # nothing else frozen, only the shift became trainable
    assert (
        {n for n, p in model.named_parameters() if not p.requires_grad}
        == frozen_before - {'rescale_atomic_energy.shift'}
    )
    before, others = shift.detach().clone(), _others(trainer)
    for _ in range(3):
        trainer.run_one_epoch(loader, is_train=True)
    assert shift.dtype == torch.float64
    # a diverged run (NaN) would also break the frozen rows: NaN * 0 = NaN
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    _check_rows(before, shift)
    assert torch.all(shift.grad[FIXED_ROWS] == 0)
    assert torch.any(shift.grad[TRAINED_ROWS] != 0)
    moved = [
        n for n, p in trainer.model.named_parameters()
        if n in others and not torch.equal(p, others[n])
    ]
    assert len(moved) > 3  # the network still trains
    rescale = trainer.model._modules['rescale_atomic_energy']
    assert rescale.scale.requires_grad is False


def test_train_one_batch(loader):
    cfg = _cfg(optimizer='muon', optim_param=copy.deepcopy(_MUON), grad_clip=1.0)
    trainer = Trainer.from_config(_model(cfg), cfg)
    shift = _shift(trainer)
    before = shift.detach().clone()
    for batch in loader:
        trainer.train_one_batch(batch)
    _check_rows(before, shift)


def test_muon_routes_shift_to_auxiliary_adam():
    cfg = _cfg(optimizer='muon', optim_param=copy.deepcopy(_MUON))
    trainer = Trainer.from_config(_model(cfg), cfg)
    assert isinstance(trainer.optimizer, MuonWithAuxAdam)
    shift = _shift(trainer)
    groups = trainer.optimizer.param_groups
    aux = [g for g in groups if not g['use_muon']]
    assert len(aux) == 1 and aux[0]['weight_decay'] == 0.0
    assert any(p is shift for p in aux[0]['params'])
    assert not any(p is shift for g in groups if g['use_muon'] for p in g['params'])


def test_batched_and_loop_give_the_same_shift(loader):
    """same step in both muon paths: both read the masked .grad"""
    shifts = []
    for batched in (True, False):
        optim_param = {**copy.deepcopy(_MUON), 'batched': batched}
        cfg = _cfg(optimizer='muon', optim_param=optim_param)
        trainer = Trainer.from_config(_model(cfg), cfg)
        trainer.run_one_epoch(loader, is_train=True)
        shifts.append(_shift(trainer).detach().clone())
    assert _bytes(shifts[0]) == _bytes(shifts[1])


def test_regularizers_do_not_read_the_shift(loader):
    cfg = _cfg(
        regularization_param={
            'modal': {'regularization_weight': 1e-2},
            'modal_group_alignment': {
                'alignment_weight': 1e-2, 'groups': {'g': ['a', 'b']},
            },
        },
    )
    trainer = Trainer.from_config(_model(cfg), cfg)
    shift = _shift(trainer)
    regs = get_modal_regularization(cfg, trainer.model)
    assert len(regs) == 2
    assert not any('rescale' in k for r, _ in regs for k in r.module_keys)
    batch = next(iter(loader))
    output = trainer.model(batch)
    for reg, _ in regs:
        loss = reg.get_loss(output, trainer.model)
        assert loss.requires_grad
        assert torch.autograd.grad(loss, shift, allow_unused=True)[0] is None


def test_reewc_trainer_from_config_passes_the_key():
    cfg = _cfg()
    trainer = ReewcTrainer.from_config(_model(cfg), cfg, memory_loader=[])
    assert _shift(trainer).requires_grad


def test_train_only_modals_is_applied_first(loader):
    """the freeze keeps a frozen shift frozen, the shift rows are added after"""
    cfg = _cfg()
    cfg[KEY.TRAIN_ONLY_MODALS] = ['a']
    trainer = Trainer.from_config(_model(cfg), cfg)
    shift = _shift(trainer)
    before = shift.detach().clone()
    assert shift.requires_grad
    trainer.run_one_epoch(loader, is_train=True)
    _check_rows(before, shift)


def test_yaml_validation():
    assert _cfg()[KEY.TRAIN_SHIFT_MODALS] == TRAINED
    assert init_model_config(dict(_MODEL))[KEY.TRAIN_SHIFT_MODALS] is None
    cond = _const.MODEL_CONFIG_CONDITION[KEY.TRAIN_SHIFT_MODALS]
    assert cond(None) and cond(['a'])
    assert not any(cond(x) for x in ('a', [], [1], True))
    with pytest.raises(ValueError, match='is not valid'):
        init_model_config({**_MODEL, KEY.TRAIN_SHIFT_MODALS: 'a'})
    for key in (KEY.TRAIN_SHIFT, KEY.TRAIN_SHIFT_SCALE):
        with pytest.raises(ValueError, match='replaces'):
            init_model_config({**_MODEL, KEY.TRAIN_SHIFT_MODALS: ['a'], key: True})
    meta = init_model_config(
        {**_MODEL, KEY.TRAIN_SHIFT_MODALS: ['a'], KEY.TRAIN_SCALE: True}
    )
    assert meta[KEY.TRAIN_SCALE] is True


def test_trainer_validation():
    cfg = _cfg({KEY.TRAIN_SHIFT_MODALS: ['a', 'nope']})
    with pytest.raises(ValueError, match="'nope'"):
        Trainer.from_config(_model(cfg), cfg)

    cfg = _cfg()
    cfg[KEY.USE_MODAL_WISE_SHIFT] = False
    cfg[KEY.SHIFT] = [-1.0, -2.0]
    with pytest.raises(ValueError, match='use_modal_wise_shift'):
        Trainer.from_config(_model(cfg), cfg)

    cfg = _cfg()  # past the yaml check, e.g. a config written by hand
    cfg[KEY.TRAIN_SHIFT] = True
    with pytest.raises(ValueError, match='replaces'):
        Trainer.from_config(_model(cfg), cfg)

    cfg = _cfg()
    for k in (KEY.USE_MODALITY, KEY.MODAL_MAP, KEY.NUM_MODALITIES):
        cfg.pop(k)
    cfg[KEY.SHIFT], cfg[KEY.SCALE] = -1.0, 1.0
    cfg[KEY.USE_MODAL_SELF_INTER_INTRO] = False
    cfg[KEY.USE_MODAL_SELF_INTER_OUTRO] = False
    cfg[KEY.USE_MODAL_OUTPUT_BLOCK] = False
    with pytest.raises(ValueError, match='modal_map'):
        Trainer.from_config(_model(cfg), cfg)


@pytest.mark.parametrize(
    'optimizer,optim_param',
    [
        ('adamw', {'lr': 0.01}),  # torch default 0.01
        ('adam', {'lr': 0.01, 'weight_decay': 1e-4}),
        ('muon', {'muon': {'lr': 0.01}, 'adam': {'lr': 0.01, 'weight_decay': 1e-4}}),
    ],
)
def test_weight_decay_on_the_shift_is_refused(optimizer, optim_param):
    cfg = _cfg(optimizer=optimizer, optim_param=copy.deepcopy(optim_param))
    with pytest.raises(ValueError, match='weight_decay'):
        Trainer.from_config(_model(cfg), cfg)


def test_muon_weight_decay_of_hidden_weights_is_allowed(loader):
    optim_param = {'muon': {'lr': 0.01, 'weight_decay': 1e-2}, 'adam': {'lr': 0.01}}
    cfg = _cfg(optimizer='muon', optim_param=optim_param)
    trainer = Trainer.from_config(_model(cfg), cfg)
    shift = _shift(trainer)
    before = shift.detach().clone()
    trainer.run_one_epoch(loader, is_train=True)
    _check_rows(before, shift)


def _write_cp(tmp_path, loader, model_keys=None, **train):
    """a trained checkpoint with optimizer state; the key popped if not given"""
    cfg = _cfg({KEY.TRAIN_SHIFT_MODALS: None, **(model_keys or {})}, **train)
    trainer = Trainer.from_config(_model(cfg), cfg)
    trainer.run_one_epoch(loader, is_train=True)
    if KEY.TRAIN_SHIFT_MODALS not in (model_keys or {}):
        cfg.pop(KEY.TRAIN_SHIFT_MODALS)
    path = str(tmp_path / 'cp.pth')
    trainer.write_checkpoint(path, config=cfg, epoch=1)
    return path


def _continue(path, model=None, reset_optimizer=True, **train):
    cfg = _cfg(model, **train)
    cfg[KEY.CONTINUE] = {
        KEY.CHECKPOINT: path,
        KEY.RESET_OPTIMIZER: reset_optimizer,
        KEY.RESET_SCHEDULER: reset_optimizer,
        KEY.RESET_EPOCH: True,
    }
    state_dicts, _, _ = processing_continue_v2(cfg)
    model = build_E3_equivariant_model(cfg)
    trainer = Trainer.from_config(model, cfg)
    trainer.load_state_dicts(*state_dicts, strict=False)
    return trainer, cfg


@pytest.mark.parametrize('optimizer', ['adam', 'muon'])
def test_continue_takes_the_key_from_the_yaml(tmp_path, loader, optimizer):
    optim = {'adam': {'lr': 0.01}, 'muon': _MUON}[optimizer]
    train = {'optimizer': optimizer, 'optim_param': copy.deepcopy(optim)}
    path = _write_cp(tmp_path, loader, **train)
    cp = load_checkpoint(path)
    assert not cp.config[KEY.TRAIN_SHIFT_MODALS]  # default filled in: None
    trainer, cfg = _continue(path, **train)
    assert cfg[KEY.TRAIN_SHIFT_MODALS] == TRAINED
    shift = _shift(trainer)
    assert shift.requires_grad
    before = shift.detach().clone()
    assert torch.equal(
        before, cp.model_state_dict['rescale_atomic_energy.shift'].to(before.dtype)
    )  # the checkpoint's state went into the masked parameter
    trainer.run_one_epoch(loader, is_train=True)
    _check_rows(before, shift)


def test_continue_without_key_and_without_reset_is_refused(tmp_path, loader):
    path = _write_cp(tmp_path, loader)
    with pytest.raises(ValueError, match='reset_optimizer'):
        _continue(path, reset_optimizer=False)


def test_continue_after_full_shift_training_is_refused(tmp_path, loader):
    """same tensors in the optimizer, but old Adam moments would move the rows"""
    path = _write_cp(tmp_path, loader, {KEY.TRAIN_SHIFT: True})
    with pytest.raises(ValueError, match='reset_optimizer'):
        _continue(path, reset_optimizer=False)
    trainer, _ = _continue(path, reset_optimizer=True)
    shift = _shift(trainer)
    before = shift.detach().clone()
    trainer.run_one_epoch(loader, is_train=True)
    _check_rows(before, shift)


def test_continue_with_changed_rows_is_refused(tmp_path, loader):
    path = _write_cp(tmp_path, loader, {KEY.TRAIN_SHIFT_MODALS: ['a', 'b']})
    with pytest.raises(ValueError, match='reset_optimizer'):
        _continue(path, reset_optimizer=False)


def test_continue_same_rows_keeps_the_optimizer(tmp_path, loader):
    path = _write_cp(tmp_path, loader, {KEY.TRAIN_SHIFT_MODALS: TRAINED})
    trainer, _ = _continue(path, reset_optimizer=False)
    assert len(trainer.optimizer.state) > 0
    shift = _shift(trainer)
    before = shift.detach().clone()
    trainer.run_one_epoch(loader, is_train=True)
    _check_rows(before, shift)


def test_saved_key_is_harmless_for_inference(tmp_path, loader):
    from sevenn.calculator import SevenNetCalculator
    from sevenn.scripts.deploy import deploy

    path = _write_cp(tmp_path, loader, {KEY.TRAIN_SHIFT_MODALS: TRAINED})
    cp = load_checkpoint(path)
    assert cp.config[KEY.TRAIN_SHIFT_MODALS] == TRAINED
    model = cp.build_model()
    assert not model._modules['rescale_atomic_energy'].shift.requires_grad

    atoms = ase.io.read(hfo2_path)
    atoms.calc = SevenNetCalculator(path, modal='b', device='cpu')
    e_0 = atoms.get_potential_energy()
    cfg = cp.config
    cfg[KEY.TRAIN_SHIFT_MODALS] = None
    other = str(tmp_path / 'no_key.pth')
    torch.save({**cp.get_checkpoint_dict(), 'config': cfg}, other)
    atoms.calc = SevenNetCalculator(other, modal='b', device='cpu')
    assert atoms.get_potential_energy() == e_0

    fname = str(tmp_path / 'deployed_serial.pt')
    deploy(path, fname, modal='b')
    assert pathlib.Path(fname).is_file()


def _run_sevenn(tmp_path, yaml_dict):
    from sevenn.main.sevenn import main as sevenn_main

    path = str(tmp_path / 'input.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(yaml_dict, f)
    Logger().switch_file(str(tmp_path / 'log.sevenn'))
    cli_args = ['train', '-w', str(tmp_path), path]  # default: train_v2
    with mock.patch('sys.argv', ['sevenn'] + cli_args):
        sevenn_main()


def _yaml_dict(**train):
    modal_files = [
        {'data_modality': m, 'file_list': [{'file': hfo2_path}]} for m in MODALS
    ]
    return {
        'model': {**_MODEL, KEY.TRAIN_SHIFT_MODALS: TRAINED},
        'train': {
            'device': 'cpu',
            'epoch': 2,
            'per_epoch': 1,
            'use_modality': True,
            'optimizer': 'muon',
            'optim_param': copy.deepcopy(_MUON),
            'scheduler': 'exponentiallr',
            'scheduler_param': {'gamma': 0.99},
            'grad_clip': 1.0,
            'regularization_param': {'modal': {'regularization_weight': 1e-3}},
            **train,
        },
        'data': {
            'batch_size': 2,
            'load_trainset_path': modal_files,
            'use_modal_wise_shift': True,
            'use_modal_wise_scale': False,
            'shift': 'elemwise_reference_energies',
            'scale': 'force_rms',
        },
    }


def test_train_v2_end_to_end(tmp_path):
    """sevenn train: new model (rows masked), then continue"""
    first = tmp_path / 'first'
    first.mkdir()
    # a first run without the key gives the checkpoint to continue from
    dct = _yaml_dict()
    dct['model'].pop(KEY.TRAIN_SHIFT_MODALS)
    _run_sevenn(first, dct)
    cp_path = str(first / 'checkpoint_2.pth')
    cp = load_checkpoint(cp_path)
    shift0 = cp.model_state_dict['rescale_atomic_energy.shift']
    assert np.allclose(shift0[0].numpy(), shift0[1].numpy())

    second = tmp_path / 'second'
    second.mkdir()
    train = {KEY.CONTINUE: {KEY.CHECKPOINT: cp_path, KEY.RESET_OPTIMIZER: True,
                            KEY.RESET_SCHEDULER: True, KEY.RESET_EPOCH: True}}
    _run_sevenn(second, _yaml_dict(**train))
    cp2 = load_checkpoint(str(second / 'checkpoint_2.pth'))
    assert cp2.config[KEY.TRAIN_SHIFT_MODALS] == TRAINED
    shift2 = cp2.model_state_dict['rescale_atomic_energy.shift']
    _check_rows(shift0, shift2)
