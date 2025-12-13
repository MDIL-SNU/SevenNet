"""
Ghost Exchange modules for SevenNet (ported from NequIP)
"""

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
