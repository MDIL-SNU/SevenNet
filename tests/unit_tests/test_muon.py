import copy

import pytest
import torch

from sevenn.model_build import build_E3_equivariant_model
from sevenn.train.muon import (
    MuonWithAuxAdam,
    _muon_shape_factor,
    build_muon_param_groups,
    zeropower_via_newtonschulz5,
)
from sevenn.util import chemical_species_preprocess

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
_linear = {'self_connection_type': 'linear'}
_hyper = {
    'muon': {'lr': 0.01, 'weight_decay': 0.01},
    'adam': {'lr': 0.005, 'weight_decay': 0.02},
}
_variants = [('cpu', False)] + (
    [('cuda', False), ('cuda', True)] if torch.cuda.is_available() else []
)


def get_model(config_overwrite):
    config = {
        'cutoff': 4.0,
        'channel': 4,
        'radial_basis': {'radial_basis_name': 'bessel'},
        'cutoff_function': {'cutoff_function_name': 'poly_cut'},
        'interaction_type': 'nequip',
        'lmax': 2,
        'is_parity': True,
        'num_convolution_layer': 3,
        'weight_nn_hidden_neurons': [64, 64],
        'act_radial': 'silu',
        'act_scalar': {'e': 'silu', 'o': 'tanh'},
        'act_gate': {'e': 'silu', 'o': 'tanh'},
        'conv_denominator': 30.0,
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
        **chemical_species_preprocess(['Hf', 'O']),
        **config_overwrite,
    }
    return build_E3_equivariant_model(config, parallel=False)


def _adam(state, grad, betas, eps):
    if not state:
        state.update(m=torch.zeros_like(grad), v=torch.zeros_like(grad), t=0)
    state['t'] += 1
    state['m'].lerp_(grad, 1 - betas[0])
    state['v'].lerp_(grad.square(), 1 - betas[1])
    m_hat = state['m'] / (1 - betas[0] ** state['t'])
    return m_hat / ((state['v'] / (1 - betas[1] ** state['t'])).sqrt() + eps)


class _Reference:
    """Per-block Muon and per-parameter Adam, one matrix at a time."""

    def __init__(self, model):
        self.muon, self.adam = build_muon_param_groups(model, **_hyper)
        self.state = {}

    def _muon_update(self, index, p):
        g = self.muon
        s = self.state.setdefault(p, {'buf': torch.zeros_like(p)})
        s['buf'].lerp_(p.grad, 1 - g['momentum'])
        u = p.grad.lerp(s['buf'], g['momentum'])
        lr = torch.full_like(p, g['lr'])
        wd = torch.full_like(p, g['weight_decay'])
        if index not in g['e3nn_blocks']:  # radial matrix
            fans = p.shape
            if g['matrix_layouts'][index] == 'fan_in_fan_out':
                fans = p.shape[::-1]
            factor = _muon_shape_factor(*fans)
            return factor * zeropower_via_newtonschulz5(u, g['ns_steps']), lr, wd
        out = torch.zeros_like(p)
        for (start, stop, _), shape, is_modal in g['e3nn_blocks'][index]:
            if is_modal:
                state = s.setdefault((start, stop), {})
                out[start:stop] = _adam(
                    state, p.grad[start:stop], self.adam['betas'], self.adam['eps']
                )
                lr[start:stop] = self.adam['lr']
                wd[start:stop] = self.adam['weight_decay']
                continue
            block = zeropower_via_newtonschulz5(
                u[start:stop].reshape(shape), g['ns_steps']
            )
            factor = _muon_shape_factor(shape[1], shape[0])
            out[start:stop] = (factor * block).flatten()
        return out, lr, wd

    @torch.no_grad()
    def step(self):
        for index, p in enumerate(self.muon['params']):
            if p.grad is not None:
                update, lr, wd = self._muon_update(index, p)
                p.copy_(p * (1 - lr * wd) - lr * update)
        g = self.adam
        for p in g['params']:
            if p.grad is not None:
                state = self.state.setdefault(p, {})
                update = _adam(state, p.grad, g['betas'], g['eps'])
                p.copy_(p * (1 - g['lr'] * g['weight_decay']) - g['lr'] * update)


def _set_grads(models, skip=()):
    for named in zip(*(m.named_parameters() for m in models)):
        grad = None if named[0][0] in skip else torch.randn_like(named[0][1])
        for _, p in named:
            p.grad = None if grad is None else grad.clone()


def _models(cf, device, n):
    torch.manual_seed(0)
    models = [get_model(cf).to(device) for _ in range(n)]
    for m in models[1:]:
        m.load_state_dict(models[0].state_dict())
    return models


@pytest.mark.parametrize('device,cuda_graph', _variants)
@pytest.mark.parametrize('cf', [{}, _linear, _modal, {**_linear, **_modal}])
def test_step_matches_per_block_reference(cf, device, cuda_graph):
    m_ref, m_opt = _models(cf, device, 2)
    ref = _Reference(m_ref)
    opt = MuonWithAuxAdam.from_model(m_opt, cuda_graph=cuda_graph, **_hyper)
    names = {id(p): n for n, p in m_ref.named_parameters()}
    if cf.get('use_modality'):
        blocks = ref.muon['e3nn_blocks'].values()
        assert any(is_modal for block in blocks for _, _, is_modal in block)
    skip = {
        names[id(ref.muon['params'][0])],
        names[id(ref.muon['params'][-1])],
        names[id(ref.adam['params'][0])],
    }
    # bf16 Newton-Schulz rounds differently for stacked matrices (< 2e-4 here);
    # a wrong block, scale or learning rate is off by more than 1e-3
    atol = 5e-4
    for step in range(6):
        _set_grads([m_ref, m_opt], skip=skip if step == 1 else ())
        ref.step()
        opt.step()
        for (name, p1), p2 in zip(m_ref.named_parameters(), m_opt.parameters()):
            assert torch.allclose(p1, p2, rtol=0, atol=atol), (step, name)
    if cuda_graph:
        assert opt._muon_graph['graph'] is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda not available')
def test_cuda_graph_is_bitwise_equal_and_survives_load_state_dict():
    m1, m2 = _models({**_linear, **_modal}, 'cuda', 2)
    o1 = MuonWithAuxAdam.from_model(m1, **_hyper)
    o2 = MuonWithAuxAdam.from_model(m2, cuda_graph=True, **_hyper)
    for step in range(8):
        if step == 4:  # resume each from the other's state
            sd1, sd2 = copy.deepcopy(o1.state_dict()), copy.deepcopy(o2.state_dict())
            o1.load_state_dict(sd2)
            o2.load_state_dict(sd1)
        _set_grads([m1, m2])
        o1.step()
        o2.step()
        for (name, p1), p2 in zip(m1.named_parameters(), m2.parameters()):
            assert torch.equal(p1, p2), (step, name)
    assert o2._muon_graph['graph'] is not None


def test_every_trainable_parameter_is_routed_once():
    model = get_model({**_linear, **_modal})
    muon, adam = build_muon_param_groups(model)
    routed = [id(p) for p in muon['params'] + adam['params']]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(routed) == sorted(trainable)
    names = {id(p): n for n, p in model.named_parameters()}
    for p in muon['params']:
        assert '_self_interaction_' in names[id(p)] or '.weight_nn.' in names[id(p)]
