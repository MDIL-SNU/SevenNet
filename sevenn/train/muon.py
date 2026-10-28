"""Muon optimizer support for SevenNet.

The implementation follows the released NequIP/Keller Jordan Muon update while
using a SevenNet-specific, conservative parameter routing policy.  Only hidden
e3nn Linear blocks and radial weight-network matrices use Muon.  FCTP weights,
embeddings, readouts, biases, and energy shift/scale parameters use auxiliary
Adam with independent (normally zero) weight decay.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Iterable, List, Literal, Tuple

import torch
import torch.distributed as dist
from e3nn.nn import FullyConnectedNet

E3nnBlock = Tuple[Tuple[int, int, None], Tuple[int, int], bool]
MatrixLayout = Literal['fan_out_fan_in', 'fan_in_fan_out']
FAN_OUT_FAN_IN: MatrixLayout = 'fan_out_fan_in'
FAN_IN_FAN_OUT: MatrixLayout = 'fan_in_fan_out'


def _muon_shape_factor(fan_out: int, fan_in: int) -> float:
    """Return Keller's original Muon adjustment in semantic fan order."""
    if fan_out <= 0 or fan_in <= 0:
        raise ValueError(
            f'Muon fan sizes must be positive, got out={fan_out}, in={fan_in}'
        )
    return max(1.0, fan_out / fan_in) ** 0.5


def zeropower_via_newtonschulz5(
    matrix: torch.Tensor, steps: int = 5
) -> torch.Tensor:
    """Approximate the polar factor of a matrix using the Muon quintic iteration."""
    if matrix.ndim < 2:
        raise ValueError('Muon requires a matrix-valued update')
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1.0e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.mT
    return x


def _muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float,
    ns_steps: int,
    e3nn_blocks: List[E3nnBlock] | None,
    matrix_layout: MatrixLayout = FAN_OUT_FAN_IN,
) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1.0 - momentum)
    update = grad.lerp(momentum_buffer, momentum)
    if e3nn_blocks is None:
        if update.ndim == 4:
            update = update.view(len(update), -1)
        if matrix_layout == FAN_OUT_FAN_IN:
            fan_out, fan_in = update.size(-2), update.size(-1)
        elif matrix_layout == FAN_IN_FAN_OUT:
            fan_in, fan_out = update.size(-2), update.size(-1)
        else:
            raise ValueError(f'Unknown Muon matrix layout: {matrix_layout}')
        orthogonal = zeropower_via_newtonschulz5(update, ns_steps)
        orthogonal *= _muon_shape_factor(fan_out, fan_in)
        return orthogonal

    pieces = []
    for slice_data, shape, use_modal_adam in e3nn_blocks:
        if use_modal_adam:
            raise ValueError('Modal Adam blocks require the mixed e3nn update path')
        block_slice = slice(*slice_data)
        block = update[block_slice].reshape(shape)
        orthogonal = zeropower_via_newtonschulz5(block, ns_steps)
        # e3nn Linear instruction views are stored (fan_in, fan_out).
        orthogonal *= _muon_shape_factor(shape[1], shape[0])
        pieces.append(orthogonal.flatten())
    return torch.cat(pieces)


def _adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: Tuple[float, float],
    eps: float,
) -> torch.Tensor:
    exp_avg.lerp_(grad, 1.0 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1.0 - betas[1])
    bias_corrected_avg = exp_avg / (1.0 - betas[0] ** step)
    bias_corrected_sq = exp_avg_sq / (1.0 - betas[1] ** step)
    return bias_corrected_avg / (bias_corrected_sq.sqrt() + eps)


def _mixed_e3nn_step(
    parameter: torch.nn.Parameter,
    state: Dict[str, Any],
    blocks: List[E3nnBlock],
    muon_group: Dict[str, Any],
    adam_group: Dict[str, Any],
) -> None:
    """Apply Muon and Adam to disjoint instruction blocks of one flat weight."""
    grad = parameter.grad
    if grad is None:
        return

    momentum_buffer = state['momentum_buffer']
    momentum_buffer.lerp_(grad, 1.0 - muon_group['momentum'])
    muon_base_update = grad.lerp(momentum_buffer, muon_group['momentum'])

    modal_blocks = [block for block in blocks if block[2]]
    if not modal_blocks:
        raise ValueError('Mixed e3nn update requested without a modal Adam block')
    modal_grad = torch.cat(
        [grad[slice(*slice_data)].flatten() for slice_data, _, _ in modal_blocks]
    )
    if 'modal_exp_avg' not in state:
        state['modal_exp_avg'] = torch.zeros_like(modal_grad)
        state['modal_exp_avg_sq'] = torch.zeros_like(modal_grad)
        state['modal_step'] = 0
    state['modal_step'] += 1
    modal_update = _adam_update(
        modal_grad,
        state['modal_exp_avg'],
        state['modal_exp_avg_sq'],
        state['modal_step'],
        adam_group['betas'],
        adam_group['eps'],
    )

    pieces = []
    modal_offset = 0
    for slice_data, shape, use_modal_adam in blocks:
        block_slice = slice(*slice_data)
        parameter_block = parameter[block_slice].flatten()
        if use_modal_adam:
            block_size = parameter_block.numel()
            update = modal_update.narrow(0, modal_offset, block_size)
            modal_offset += block_size
            lr = adam_group['lr']
            weight_decay = adam_group['weight_decay']
        else:
            update = zeropower_via_newtonschulz5(
                muon_base_update[block_slice].reshape(shape),
                muon_group['ns_steps'],
            )
            # e3nn Linear instruction views are stored (fan_in, fan_out).
            update *= _muon_shape_factor(shape[1], shape[0])
            update = update.flatten()
            lr = muon_group['lr']
            weight_decay = muon_group['weight_decay']
        pieces.append(parameter_block * (1.0 - lr * weight_decay) - lr * update)

    if modal_offset != modal_update.numel():
        raise ValueError(
            'Modal Adam update coverage mismatch: '
            f'used={modal_offset}, available={modal_update.numel()}'
        )
    parameter.copy_(torch.cat(pieces).reshape_as(parameter))


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Replicated-DDP Muon with an auxiliary Adam parameter group."""

    @classmethod
    def from_model(cls, model: torch.nn.Module, **kwargs) -> 'MuonWithAuxAdam':
        return cls(build_muon_param_groups(model, **kwargs))

    def __init__(self, params: Iterable[Dict[str, Any]]) -> None:
        groups = list(params)
        if not groups:
            raise ValueError('Muon requires non-empty parameter groups')
        super().__init__(groups, {})
        # Optimizer.load_state_dict replaces custom param-group metadata.  Keep
        # model-derived routing separately so legacy checkpoints remain safe.
        self._muon_runtime_routes: Dict[
            int, Tuple[List[E3nnBlock] | None, MatrixLayout]
        ] = {}
        for group in self.param_groups:
            if not group.get('use_muon', False):
                continue
            block_map = group.get('e3nn_blocks', {})
            layout_map = group.get('matrix_layouts', {})
            for index, parameter in enumerate(group['params']):
                self._muon_runtime_routes[id(parameter)] = (
                    block_map.get(index),
                    layout_map.get(index, FAN_OUT_FAN_IN),
                )
        verify_steps = os.environ.get('SEVENN_MUON_DDP_VERIFY_STEPS', '')
        self._ddp_verify_steps = {
            int(step) for step in verify_steps.split(',') if step.strip()
        }
        self._muon_step = 0

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            if not group.get('use_muon', False):
                continue
            block_map = {}
            layout_map = {}
            for index, parameter in enumerate(group['params']):
                runtime_route = self._muon_runtime_routes.get(id(parameter))
                if runtime_route is None:
                    raise ValueError('Missing immutable Muon runtime routing')
                blocks, matrix_layout = runtime_route
                if blocks is not None:
                    block_map[index] = blocks
                else:
                    layout_map[index] = matrix_layout
            group['e3nn_blocks'] = block_map
            group['matrix_layouts'] = layout_map

    def _verify_replicated_ddp_parameters(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('DDP verification requested without a process group')
        digest = hashlib.sha256()
        for group in self.param_groups:
            for parameter in group['params']:
                digest.update(
                    parameter.detach().contiguous().cpu().numpy().tobytes()
                )
        local_hash = digest.hexdigest()
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_hash)
        if len(set(gathered)) != 1:
            raise RuntimeError(
                'Muon DDP parameter mismatch at optimizer step '
                f'{self._muon_step}: {gathered}'
            )
        if dist.get_rank() == 0:
            print(
                f'MUON_DDP_HASH_OK step={self._muon_step} sha256={local_hash}',
                flush=True,
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        adam_groups = [group for group in self.param_groups if not group['use_muon']]
        if len(adam_groups) != 1:
            raise ValueError(
                'MuonWithAuxAdam requires exactly one auxiliary Adam parameter group'
            )
        adam_group = adam_groups[0]

        for group in self.param_groups:
            if group['use_muon']:
                for parameter in group['params']:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(parameter)
                    runtime_route = self._muon_runtime_routes.get(id(parameter))
                    if runtime_route is None:
                        raise ValueError('Missing immutable Muon runtime routing')
                    blocks, matrix_layout = runtime_route
                    if blocks is not None and any(block[2] for block in blocks):
                        _mixed_e3nn_step(
                            parameter,
                            state,
                            blocks,
                            group,
                            adam_group,
                        )
                        continue
                    update = _muon_update(
                        parameter.grad,
                        state['momentum_buffer'],
                        group['momentum'],
                        group['ns_steps'],
                        blocks,
                        matrix_layout,
                    )
                    parameter.mul_(1.0 - group['lr'] * group['weight_decay'])
                    parameter.add_(update.reshape_as(parameter), alpha=-group['lr'])
            else:
                for parameter in group['params']:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if not state:
                        state['exp_avg'] = torch.zeros_like(parameter)
                        state['exp_avg_sq'] = torch.zeros_like(parameter)
                        state['step'] = 0
                    state['step'] += 1
                    update = _adam_update(
                        parameter.grad,
                        state['exp_avg'],
                        state['exp_avg_sq'],
                        state['step'],
                        group['betas'],
                        group['eps'],
                    )
                    parameter.mul_(1.0 - group['lr'] * group['weight_decay'])
                    parameter.add_(update, alpha=-group['lr'])
        self._muon_step += 1
        if self._muon_step in self._ddp_verify_steps:
            self._verify_replicated_ddp_parameters()
        return loss


def _e3nn_block_metadata(
    linear: torch.nn.Module,
    modal_input_index: int | None = None,
) -> List[E3nnBlock]:
    blocks = []
    offset = 0
    for _, instruction, view in linear.weight_views(yield_instruction=True):
        if view.ndim != 2:
            raise ValueError(
                f'Expected a 2D e3nn Linear block, got {tuple(view.shape)}'
            )
        size = view.numel()
        use_modal_adam = (
            modal_input_index is not None and instruction.i_in == modal_input_index
        )
        blocks.append(
            ((offset, offset + size, None), tuple(view.shape), use_modal_adam)
        )
        offset += size
    if offset != linear.weight.numel():
        raise ValueError(
            f'e3nn block coverage mismatch: blocks={offset}, '
            f'weight={linear.weight.numel()}'
        )
    if modal_input_index is not None and not any(block[2] for block in blocks):
        raise ValueError(
            f'No e3nn instruction found for modal input index {modal_input_index}'
        )
    return blocks


def build_muon_param_groups(
    model: torch.nn.Module,
    muon: Dict[str, Any] | None = None,
    adam: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Build exhaustive, mutually exclusive reference-parity parameter groups."""
    muon_cfg = {
        'lr': 0.02,
        'momentum': 0.95,
        'weight_decay': 0.0,
        'ns_steps': 5,
        **(muon or {}),
    }
    adam_cfg = {
        'lr': 0.02,
        'betas': (0.9, 0.999),
        'eps': 1.0e-8,
        'weight_decay': 0.0,
        **(adam or {}),
    }
    modules = dict(model.named_modules())
    muon_params = []
    adam_params = []
    block_map: Dict[int, Any] = {}
    layout_map: Dict[int, MatrixLayout] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_hidden_e3nn = '_self_interaction_' in name and name.endswith(
            '.linear.weight'
        )
        is_radial_matrix = '.weight_nn.' in name and parameter.ndim == 2
        if is_hidden_e3nn:
            module_name = name.rpartition('.')[0]
            linear = modules.get(module_name)
            if linear is None or not hasattr(linear, 'weight_views'):
                raise ValueError(f'Missing e3nn Linear metadata for {name}')
            wrapper_name = module_name.rpartition('.')[0]
            wrapper = modules.get(wrapper_name)
            num_modalities = getattr(wrapper, 'num_modalities', 0)
            modal_input_index = None
            if num_modalities > 1:
                modal_input_index = len(linear.irreps_in) - 1
                modal_mul, modal_irrep = linear.irreps_in[modal_input_index]
                if (
                    modal_mul != num_modalities
                    or modal_irrep.l != 0
                    or modal_irrep.p != 1
                ):
                    raise ValueError(
                        'Expected the final e3nn input irrep to be the appended '
                        f'{num_modalities}x0e modality block for {name}, got '
                        f'{modal_mul}x{modal_irrep}'
                    )
            block_map[len(muon_params)] = _e3nn_block_metadata(
                linear,
                modal_input_index=modal_input_index,
            )
            muon_params.append(parameter)
        elif is_radial_matrix:
            owner_name = name.rpartition('.')[0]
            owner = modules.get(owner_name)
            weight_nn_name = name.split('.weight_nn.', maxsplit=1)[0] + '.weight_nn'
            weight_nn = modules.get(weight_nn_name)
            if isinstance(weight_nn, FullyConnectedNet):
                if owner is None or not all(
                    hasattr(owner, attr) for attr in ('h_in', 'h_out')
                ):
                    raise ValueError(
                        f'Missing e3nn FullyConnectedNet layer metadata for {name}'
                    )
                expected_shape = (int(owner.h_in), int(owner.h_out))
                matrix_layout = FAN_IN_FAN_OUT
            elif isinstance(owner, torch.nn.Linear):
                expected_shape = (owner.out_features, owner.in_features)
                matrix_layout = FAN_OUT_FAN_IN
            else:
                raise ValueError(
                    'Unknown radial weight storage layout for '
                    f'{name}: owner={type(owner).__name__}, '
                    f'weight_nn={type(weight_nn).__name__}'
                )
            if tuple(parameter.shape) != expected_shape:
                raise ValueError(
                    f'Radial weight shape mismatch for {name}: '
                    f'expected={expected_shape}, actual={tuple(parameter.shape)}'
                )
            layout_map[len(muon_params)] = matrix_layout
            muon_params.append(parameter)
        else:
            adam_params.append(parameter)

    all_trainable = [p for p in model.parameters() if p.requires_grad]
    routed = muon_params + adam_params
    if len({id(p) for p in routed}) != len(routed):
        raise ValueError(
            'A trainable parameter was assigned to multiple optimizer groups'
        )
    if {id(p) for p in routed} != {id(p) for p in all_trainable}:
        raise ValueError(
            'Muon parameter routing did not cover every trainable parameter'
        )
    if not muon_params or not adam_params:
        raise ValueError('Both Muon and auxiliary Adam groups must be non-empty')

    return [
        {
            'params': muon_params,
            'use_muon': True,
            'e3nn_blocks': block_map,
            'matrix_layouts': layout_map,
            **muon_cfg,
        },
        {'params': adam_params, 'use_muon': False, **adam_cfg},
    ]
