"""
Ghost Exchange modules for SevenNet (ported from NequIP) and the LAMMPS ML-IAP
model wrapping (sevenn/mliap.py). Kept free of the lammps import, so the
wrapping can be tested with an emulated lmp_data
(tests/unit_tests/test_parallel_consistency.py).
"""
import re
from collections import OrderedDict
from typing import Any, Sequence

import torch
import torch.nn as nn

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


class LAMMPSMLIAPGhostExchangeOp(torch.autograd.Function):
    """Custom autograd function for LAMMPS ML-IAP ghost exchange."""

    @staticmethod
    def forward(ctx, node_features, lmp_data):
        original_shape = node_features.shape
        node_features_flat = node_features.view(node_features.size(0), -1)
        out_flat = torch.empty_like(node_features_flat)

        # Forward exchange: fill ghost features from neighbor processes
        lmp_data.forward_exchange(node_features_flat, out_flat, out_flat.size(-1))

        # Save for backward
        ctx.original_shape = original_shape
        ctx.lmp_data = lmp_data

        return out_flat.view(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Reverse exchange: route ghost gradients back to their original processes.
        """
        grad_output_flat = grad_output.view(grad_output.size(0), -1)
        gout_flat = torch.empty_like(grad_output_flat)

        # Reverse exchange: send ghost gradients to original atoms
        ctx.lmp_data.reverse_exchange(
            grad_output_flat, gout_flat, gout_flat.size(-1)
        )

        return gout_flat.view(ctx.original_shape), None


class MLIAPGhostExchangeModule(nn.Module):
    """
    LAMMPS ML-IAP ghost exchange.
    """

    def __init__(
        self,
        field: str = KEY.NODE_FEATURE,
    ):
        super().__init__()
        self.field = field

    def forward(
        self,
        data: AtomGraphDataType,
    ) -> AtomGraphDataType:
        """
        Perform LAMMPS ghost exchange with MPI communication.

        Requires:
            - data[KEY.LAMMPS_DATA]: LAMMPS object with forward_exchange method
            - data[KEY.MLIAP_NUM_LOCAL_GHOST]: torch.Tensor = [nlocal, nghost]
        """
        assert KEY.LAMMPS_DATA in data, (
            'LAMMPS_DATA required for MLIAPGhostExchangeModule'
        )

        lmp_data = data[KEY.LAMMPS_DATA]
        node_features = data[self.field]
        num_local_ghost = data[KEY.MLIAP_NUM_LOCAL_GHOST]
        nghost = num_local_ghost[1].item()

        # Assume node_features already exclude ghosts
        local_features = node_features

        # Prepare for LAMMPS exchange
        ghost_zeros = torch.zeros(
            (nghost,) + node_features.shape[1:],
            dtype=node_features.dtype,
            device=node_features.device,
        )

        prepared_features = torch.cat([local_features, ghost_zeros], dim=0)
        exchanged_features = LAMMPSMLIAPGhostExchangeOp.apply(
            prepared_features, lmp_data
        )

        data[self.field] = exchanged_features

        return data


class MLIAPGhostExchange(nn.Module):
    """
    Narrow x to the local atoms, then fill the ghost rows from their owners
    (forward_exchange; its backward sends the ghost gradients back).
    """

    def __init__(self):
        super().__init__()
        self.ghost_exchange = MLIAPGhostExchangeModule(field=KEY.NODE_FEATURE)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        nlocal = int(data[KEY.MLIAP_NUM_LOCAL_GHOST][0].item())
        data[KEY.NODE_FEATURE] = torch.narrow(data[KEY.NODE_FEATURE], 0, 0, nlocal)
        return self.ghost_exchange(data)


class MLIAPWrappedConvolution(nn.Module):
    """
    Convolution on local + ghost rows (exchanged by MLIAPGhostExchange after
    self_interaction_1), then node data narrowed back to the local atoms.
    extra_keys_to_narrow: per-node tensors of the layer computed on the ghost
    rows too (e.g. the attention output gate).
    """

    def __init__(self, conv, extra_keys_to_narrow: Sequence[str] = ()):
        super().__init__()

        self.conv = conv
        self._keys_to_narrow = (
            KEY.NODE_FEATURE,
            KEY.NODE_ATTR,
            KEY.ATOM_TYPE,
            KEY.ATOMIC_NUMBERS,
        ) + tuple(extra_keys_to_narrow)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        nlocal = int(data[KEY.MLIAP_NUM_LOCAL_GHOST][0].item())

        data = self.conv(data)

        for k in self._keys_to_narrow:
            data[k] = torch.narrow(data[k], 0, 0, nlocal)

        return data


class MLIAPWrappedIrrepsLinear(nn.Module):
    """
    onehot_to_feature_x: keep the types of all (local + ghost) atoms in
    KEY.ATOM_TYPE_ALL for per-edge type lookups, then narrow to local atoms.
    """

    def __init__(self, linear):
        super().__init__()

        self.linear = linear
        self._keys_to_narrow = (
            KEY.NODE_FEATURE,
            KEY.NODE_ATTR,
            KEY.ATOM_TYPE,
            KEY.ATOMIC_NUMBERS,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        nlocal = int(data[KEY.MLIAP_NUM_LOCAL_GHOST][0].item())

        data[KEY.ATOM_TYPE_ALL] = data[KEY.ATOM_TYPE]
        for k in self._keys_to_narrow:
            data[k] = torch.narrow(data[k], 0, 0, nlocal)

        data = self.linear(data)
        return data


_RE_SI1 = re.compile(r'(\d+)_self_interaction_1')
_RE_CONV = re.compile(r'(\d+)_convolution')


def patch_model_for_mliap(model: nn.Module) -> nn.Module:
    """
    Wrap a serial AtomGraphSequential (any backend) for LAMMPS ML-IAP, in place.
    Node-wise modules run on the local atoms only. The features are exchanged
    once per layer, right after {t}_self_interaction_1 (the convolution input),
    i.e. before every module of the layer that gathers source-node features
    (edge attention, node-conditioned radial MLP, convolution). Edge-level type
    lookups of ghost neighbours read KEY.ATOM_TYPE_ALL. Edge inputs computed
    once from edge_vec (edge_embedding, edge_pair_input) stay valid: ML-IAP
    back-propagates the whole model in one autograd graph.
    """
    modules = OrderedDict()
    for k, module in model._modules.items():
        m_conv = _RE_CONV.fullmatch(k)
        if m_conv:
            module = MLIAPWrappedConvolution(module)
        elif k == 'onehot_to_feature_x':
            module = MLIAPWrappedIrrepsLinear(module)
        modules[k] = module
        m_si1 = _RE_SI1.fullmatch(k)
        if m_si1:
            modules[f'{m_si1.group(1)}_ghost_exchange'] = MLIAPGhostExchange()
    model._modules.clear()
    model._modules.update(modules)
    return model


def mliap_compute_forces(
    model: nn.Module,
    lmp_data: Any,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> None:
    """
    One ML-IAP force call (SevenNetMLIAPWrapper.compute_forces) of a model
    wrapped by patch_model_for_mliap: atomic energies of the local atoms,
    total energy and pair forces dE/dr_ij are written to lmp_data.
    """
    nlocal = lmp_data.nlocal
    ntotal = lmp_data.ntotal
    no_pairs = lmp_data.npairs <= 1

    # edge_vectors should be f32 in 7net
    if no_pairs:
        edge_vectors = torch.zeros((0, 3), dtype=dtype, device=device)
        edge_index = torch.zeros((2, 0), dtype=torch.int64, device=device)
    else:
        edge_vectors = torch.as_tensor(lmp_data.rij, dtype=dtype, device=device)
        edge_index = torch.vstack(
            [
                torch.as_tensor(lmp_data.pair_i, dtype=torch.int64, device=device),
                torch.as_tensor(lmp_data.pair_j, dtype=torch.int64, device=device),
            ]
        )
    edge_vectors.requires_grad_(True)
    elems = torch.as_tensor(lmp_data.elems, dtype=torch.int64, device=device)
    num_atoms = torch.as_tensor(nlocal, dtype=torch.int64, device=device)
    mliap_num_local_ghost = torch.as_tensor(
        [nlocal, ntotal - nlocal], dtype=torch.int64, device=device
    )

    # data prep
    data = {
        KEY.EDGE_IDX: edge_index,
        KEY.EDGE_VEC: edge_vectors,
        KEY.ATOMIC_NUMBERS: elems,
        KEY.NUM_ATOMS: num_atoms,
        KEY.MLIAP_NUM_LOCAL_GHOST: mliap_num_local_ghost,
        KEY.LAMMPS_DATA: lmp_data,
        KEY.USE_MLIAP: torch.as_tensor(True, dtype=torch.bool),
    }

    # infer
    output = model(data)
    pred_atomic_energies = output[KEY.ATOMIC_ENERGY].view(-1)
    edge_forces = torch.autograd.grad(
        torch.sum(pred_atomic_energies),
        [edge_vectors],
        allow_unused=True,
    )[0]
    if edge_forces is None:
        edge_forces = torch.zeros_like(edge_vectors)
    if pred_atomic_energies.size(0) != nlocal:  # why this check is necessary?
        pred_atomic_energies = torch.narrow(pred_atomic_energies, 0, 0, nlocal)
    pred_total_energy = torch.sum(pred_atomic_energies)

    # update
    lmp_eatoms = torch.as_tensor(lmp_data.eatoms)
    lmp_eatoms.copy_(pred_atomic_energies)
    lmp_data.energy = pred_total_energy.detach()
    # upcasting edge_forces required for update_pair_forces_gpu
    lmp_data.update_pair_forces_gpu(edge_forces.to(torch.float64))
