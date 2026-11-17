"""MuonWithAuxAdam(batched=True) must give exactly the per-block loop's step."""

import copy

import pytest
import torch

from sevenn.train.muon import MuonWithAuxAdam, build_muon_param_groups
from tests.unit_tests.test_arch_options import with_arch
from tests.unit_tests.test_model import get_model

_modal = {
    'use_modality': True,
    '_number_of_modalities': 2,
    '_modal_map': {'x1': 0, 'x2': 1},
    'use_modal_node_embedding': False,
    'use_modal_self_inter_intro': True,
    'use_modal_self_inter_outro': True,
    'use_modal_output_block': True,
    'use_modal_wise_shift': False,
    'use_modal_wise_scale': False,
}
_partial = {
    'lmax': 2,
    'irreps_manual': ['4x0e', '4x0e+4x1o+4x2e', '4x0e+4x1o+4x2e', '4x0e'],
    'self_connection_type': 'linear',
    'drop_unused_tp_irreps': True,
}
_metaformer = {
    **_partial,
    'interaction_type': 'metaformer',
    'self_connection_type': 'identity',
    'radial_pair_embedding': True,
}
_devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
_hyper = {
    'muon': {'lr': 0.01, 'weight_decay': 0.01},
    'adam': {'lr': 0.005, 'weight_decay': 0.02},
}


def _pair(cf, device, cuda_graph=False):
    torch.manual_seed(0)
    m1 = get_model(cf).to(device)
    m2 = get_model(cf).to(device)
    m2.load_state_dict(m1.state_dict())
    o1 = MuonWithAuxAdam.from_model(m1, batched=False, **_hyper)
    o2 = MuonWithAuxAdam.from_model(
        m2, batched=True, cuda_graph=cuda_graph, **_hyper
    )
    return m1, m2, o1, o2


_variants = [('cpu', False)] + (
    [('cuda', False), ('cuda', True)] if torch.cuda.is_available() else []
)


def _set_grads(m1, m2, skip=()):
    for (name, p1), p2 in zip(m1.named_parameters(), m2.parameters()):
        if name in skip:
            p1.grad, p2.grad = None, None
            continue
        g = torch.randn_like(p1)
        p1.grad, p2.grad = g.clone(), g.clone()


def _assert_equal(m1, m2, o1, o2):
    for (name, p1), p2 in zip(m1.named_parameters(), m2.parameters()):
        assert torch.equal(p1, p2), name
        s1, s2 = o1.state[p1], o2.state[p2]
        assert s1.keys() == s2.keys(), name
        for k in s1:
            if torch.is_tensor(s1[k]):
                assert torch.equal(s1[k], s2[k]), (name, k)
            else:
                assert s1[k] == s2[k], (name, k)


@pytest.mark.parametrize('device,cuda_graph', _variants)
@pytest.mark.parametrize(
    'cf',
    [
        {**_partial, **_modal},
        {**_metaformer, **_modal},
        _metaformer,
        {**_metaformer, 'ffn_type': 'swiglu', 'layer_scale_init': 0.1},
        {**_metaformer, 'ffn_film': True},
        # isolation-gate build path (readout without bias, gated linears)
        with_arch({**_partial, **_modal}, 'isolation_gate'),
        with_arch({**_metaformer, **_modal}, 'isolation_gate'),
    ],
)
def test_batched_step_is_bitwise_equal(cf, device, cuda_graph):
    m1, m2, o1, o2 = _pair(cf, device, cuda_graph)
    groups = build_muon_param_groups(m1)
    names = {id(p): n for n, p in m1.named_parameters()}
    muon_names = [names[id(p)] for p in groups[0]['params']]
    adam_names = [names[id(p)] for p in groups[1]['params']]
    if cf.get('use_modality'):  # the mixed (modal Adam + Muon) path is covered
        assert any(
            any(b[2] for b in blocks) for blocks in groups[0]['e3nn_blocks'].values()
        )
    # some parameters without gradient at one step: skipped, and the Adam step
    # counters then differ between parameters
    skip = {muon_names[0], muon_names[-1], adam_names[0]}
    # cuda_graph: eager (0), eager (1, other parameters), eager (2), capture
    # (3), replays (4-6)
    for step in range(7):
        _set_grads(m1, m2, skip=skip if step == 1 else ())
        o1.step()
        o2.step()
        _assert_equal(m1, m2, o1, o2)
    if cuda_graph:
        assert o2._muon_graph['graph'] is not None


@pytest.mark.parametrize('device,cuda_graph', _variants)
def test_state_dict_interchangeable(device, cuda_graph):
    cf = {**_metaformer, **_modal}
    m1, m2, o1, o2 = _pair(cf, device, cuda_graph)
    for _ in range(3):  # cuda_graph: captured at the 2nd step, replayed
        _set_grads(m1, m2)
        o1.step()
        o2.step()
    # resume each from the other's state dict (e.g. old checkpoints); deep
    # copies: load_state_dict keeps the given tensors (no aliasing of states)
    sd1, sd2 = copy.deepcopy(o1.state_dict()), copy.deepcopy(o2.state_dict())
    o1.load_state_dict(sd2)
    o2.load_state_dict(sd1)
    assert o2.batched and not o1.batched
    for _ in range(3):  # new momentum buffers: eager, recapture, replay
        _set_grads(m1, m2)
        o1.step()
        o2.step()
    _assert_equal(m1, m2, o1, o2)


def test_batched_newton_schulz_probe():
    from sevenn.train.muon import (
        _batched_ns_is_exact,
        zeropower_via_newtonschulz5,
    )

    device = torch.device(_devices[-1])
    for shape in [(4, 4), (8, 3), (3, 8)]:
        exact = _batched_ns_is_exact(shape, 3, device, 5)
        x = torch.randn(3, *shape, device=device)
        same = all(
            torch.equal(zeropower_via_newtonschulz5(m, 5), r)
            for m, r in zip(x, zeropower_via_newtonschulz5(x, 5))
        )
        assert exact == same
