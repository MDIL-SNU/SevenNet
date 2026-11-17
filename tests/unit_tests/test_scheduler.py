"""Warmup-stable-decay ('wsd') scheduler: lr curve, OneCycle parity, resume."""

import io
import math
from types import SimpleNamespace

import pytest
import torch

import sevenn._keys as KEY
from sevenn.scripts.processing_by_batch import update_config_for_batch_training
from sevenn.train.optim import WarmupStableDecayLR, scheduler_dict

_MAX = [0.01, 0.005]
_KW = dict(
    pct_warmup=0.01, pct_stable_end=0.35, div_factor=25.0, final_div_factor=4.0
)


def _optim():
    torch.manual_seed(0)
    w1 = torch.nn.Parameter(torch.randn(3, 3))
    w2 = torch.nn.Parameter(torch.randn(3))
    return torch.optim.Adam([{'params': [w1]}, {'params': [w2]}], lr=1.0)


def _curve(sched, n):
    lrs = []
    for _ in range(n):
        lrs.append([g['lr'] for g in sched.optimizer.param_groups])
        sched.optimizer.step()
        sched.step()
    return lrs


def _expected(step, total, lr_max):
    lr0 = lr_max / 25.0
    lr_min = lr0 / 4.0
    w_end = 0.01 * total - 1
    s_end = 0.35 * total - 1
    if step <= w_end:
        return lr0 + (lr_max - lr0) * step / w_end
    if step <= s_end:
        return lr_max
    pct = (step - s_end) / (total - 1 - s_end)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * pct))


def test_wsd_registered():
    assert scheduler_dict['wsd'] is WarmupStableDecayLR


def test_wsd_curve():
    total = 1000
    sched = WarmupStableDecayLR(_optim(), max_lr=_MAX, total_steps=total, **_KW)
    lrs = _curve(sched, total + 5)
    for step in range(total):
        for g, lr_max in enumerate(_MAX):
            assert lrs[step][g] == pytest.approx(
                _expected(step, total, lr_max), rel=1e-12
            )
    for g, lr_max in enumerate(_MAX):
        assert lrs[0][g] == pytest.approx(lr_max / 25.0)  # warmup start
        assert lrs[9][g] == pytest.approx(lr_max)  # warmup end (1 % - 1)
        assert all(lrs[s][g] == pytest.approx(lr_max) for s in range(9, 350))
        assert lrs[350][g] < lr_max  # decay has started
        assert lrs[total - 1][g] == pytest.approx(lr_max / 100.0)
        # held at the final lr after total_steps (no error, no rebound)
        assert lrs[total + 4][g] == pytest.approx(lr_max / 100.0)
        dec = [lrs[s][g] for s in range(349, total)]
        assert all(a >= b for a, b in zip(dec, dec[1:]))
        inc = [lrs[s][g] for s in range(0, 10)]
        assert all(a < b for a, b in zip(inc, inc[1:]))


def test_wsd_matches_onecycle_endpoints():
    """Same initial/peak/final lr per group as our OneCycleLR setup."""
    total = 500
    oc = torch.optim.lr_scheduler.OneCycleLR(
        _optim(),
        max_lr=_MAX,
        total_steps=total,
        pct_start=0.01,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=25.0,
        final_div_factor=4.0,
    )
    wsd = WarmupStableDecayLR(_optim(), max_lr=_MAX, total_steps=total, **_KW)
    a = _curve(oc, total)
    b = _curve(wsd, total)
    for step in (0, 4, total - 1):  # 4 = warmup end (0.01 * 500 - 1)
        for g in range(2):
            assert b[step][g] == pytest.approx(a[step][g], rel=1e-12)
    for g in range(2):  # WSD is never below OneCycle (stable plateau)
        assert all(b[s][g] >= a[s][g] - 1e-15 for s in range(4, total))


def test_wsd_epochs_and_scalar_max_lr():
    sched = WarmupStableDecayLR(
        _optim(), max_lr=0.01, epochs=2, steps_per_epoch=500, **_KW
    )
    assert sched.total_steps == 1000
    assert [g['lr'] for g in sched.optimizer.param_groups] == pytest.approx(
        [0.0004, 0.0004]
    )
    with pytest.raises(ValueError):
        WarmupStableDecayLR(_optim(), max_lr=[0.01], total_steps=10)
    with pytest.raises(ValueError):
        WarmupStableDecayLR(_optim(), max_lr=0.01)


def test_wsd_resume():
    total, cut = 400, 137
    ref = _curve(
        WarmupStableDecayLR(_optim(), max_lr=_MAX, total_steps=total, **_KW),
        total,
    )
    opt = _optim()
    sched = WarmupStableDecayLR(opt, max_lr=_MAX, total_steps=total, **_KW)
    first = _curve(sched, cut)
    buf = io.BytesIO()
    torch.save({'optim': opt.state_dict(), 'sched': sched.state_dict()}, buf)
    buf.seek(0)
    state = torch.load(buf, weights_only=False)

    # fresh objects built from the config (as Trainer does), then loaded
    opt2 = _optim()
    sched2 = WarmupStableDecayLR(opt2, max_lr=_MAX, total_steps=total, **_KW)
    opt2.load_state_dict(state['optim'])
    sched2.load_state_dict(state['sched'])
    rest = _curve(sched2, total - cut)
    flat = [lr for lrs in first + rest for lr in lrs]
    assert flat == pytest.approx([lr for lrs in ref for lr in lrs], rel=1e-15)
    assert sched2.last_epoch == total


@pytest.mark.parametrize('sched', ['onecyclelr', 'wsd'])
def test_batch_mode_total_steps(sched):
    loader = SimpleNamespace(sampler=SimpleNamespace(total_size=1000))
    config = {
        KEY.WORLD_SIZE: 2,
        KEY.BATCH_SIZE: 64,
        KEY.SCHEDULER: sched,
        KEY.SCHEDULER_PARAM: {'max_lr': _MAX, 'epochs': 1.5},
    }
    update_config_for_batch_training(config, {'trainset': loader})
    assert config[KEY.SCHEDULER_BATCH_MODE] is True
    assert config[KEY.SCHEDULER_PARAM]['total_steps'] == math.ceil(
        1.5 * math.ceil(1000 / 128)
    )
    if sched == 'wsd':  # the trainer passes scheduler_param through as kwargs
        s = WarmupStableDecayLR(_optim(), **config[KEY.SCHEDULER_PARAM])
        assert s.total_steps == 12
