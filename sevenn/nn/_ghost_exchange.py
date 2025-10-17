"""
Ghost Exchange modules for SevenNet (ported from NequIP)
Simplified version using tensor metadata only
"""
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


class GhostExchangeModule(nn.Module):
    """Base class for ghost atom exchange modules in SevenNet."""

    def __init__(
        self,
        field: str = KEY.NODE_FEATURE,
    ):
        super().__init__()
        self.field = field

    def forward(
        self,
        data: AtomGraphDataType,
        ghost_included: bool,
    ) -> AtomGraphDataType:
        raise NotImplementedError("Subclasses must implement forward method")


class LAMMPSMLIAPGhostExchangeOp(torch.autograd.Function):
    """Custom autograd function for LAMMPS ML-IAP ghost exchange with proper gradient routing."""
    
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
        
        This is critical for correct force calculation!
        """
        grad_output_flat = grad_output.view(grad_output.size(0), -1)
        gout_flat = torch.empty_like(grad_output_flat)

        # Reverse exchange: send ghost gradients to original atoms
        ctx.lmp_data.reverse_exchange(grad_output_flat, gout_flat, gout_flat.size(-1))

        return gout_flat.view(ctx.original_shape), None


class LAMMPSMLIAPGhostExchangeModule(GhostExchangeModule):
    """
    LAMMPS ML-IAP ghost exchange with actual MPI communication.
    This version still needs the LAMMPS object for forward/reverse exchange.
    """

    def __init__(
        self,
        field: str = KEY.NODE_FEATURE,
    ):
        super().__init__(field=field)

    def forward(
        self,
        data: AtomGraphDataType,
        ghost_included: bool = False,
    ) -> AtomGraphDataType:
        """
        Perform actual LAMMPS ghost exchange with MPI communication.
        
        Requires:
            - data[KEY.LAMMPS_DATA]: LAMMPS object with forward_exchange method
            - data[KEY.MLIAP_NUM_LOCAL_GHOST]: torch.Tensor[2, int64] = [nlocal, num_ghost]
        """
        assert KEY.LAMMPS_DATA in data, (
            "LAMMPS_DATA required for LAMMPSMLIAPGhostExchangeModule"
        )
        
        lmp_data = data[KEY.LAMMPS_DATA]
        node_features = data[self.field]
        num_local_ghost = data[KEY.MLIAP_NUM_LOCAL_GHOST]
        nlocal = num_local_ghost[0].item()
        num_ghost = num_local_ghost[1].item()
        ntotal = nlocal + num_ghost
        
        # Extract local features
        if ghost_included:
            local_features = node_features[:nlocal]
        else:
            local_features = node_features
        
        # Prepare for LAMMPS exchange
        ghost_zeros = torch.zeros(
            (num_ghost,) + node_features.shape[1:],
            dtype=node_features.dtype,
            device=node_features.device,
        )
        
        prepared_features = torch.cat([local_features, ghost_zeros], dim=0)
        exchanged_features = LAMMPSMLIAPGhostExchangeOp.apply(
            prepared_features, lmp_data
        )
        
        # # Flatten for LAMMPS exchange
        # original_shape = prepared_features.shape
        # features_flat = prepared_features.view(prepared_features.size(0), -1)
        # out_flat = torch.empty_like(features_flat)
        # 
        # # Perform LAMMPS MPI exchange
        # lmp_data.forward_exchange(features_flat, out_flat, out_flat.size(-1))
        # 
        # # Reshape back
        # exchanged_features = out_flat.view(original_shape)
        
        data[self.field] = exchanged_features
        
        # Store ghost features separately
        data[KEY.NODE_FEATURE_GHOST] = exchanged_features[nlocal:]
        
        return data

