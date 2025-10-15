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


class NoOpGhostExchangeModule(GhostExchangeModule):
    """No-op ghost exchange module (for training/non-LAMMPS usage)."""

    def forward(
        self,
        data: AtomGraphDataType,
        ghost_included: bool,
    ) -> AtomGraphDataType:
        return data


class SimpleGhostExchangeModule(GhostExchangeModule):
    """
    Simplified ghost exchange module for SevenNet.
    Uses only torch tensors for all metadata.
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
        Perform ghost exchange using tensor metadata.
        
        Args:
            data: AtomGraphDataType with metadata (all torch tensors):
                - KEY.USE_MLIAP (torch.Tensor[bool]): Whether LAMMPS ML-IAP mode is active
                - KEY.MLIAP_NUM_LOCAL_GHOST (torch.Tensor[2, int64]): [nlocal, num_ghost]
            ghost_included: If True, input features already include ghost atoms
                          If False, input features are local only
        
        Returns:
            data with ghost features appended/updated
        """
        # Check if MLIAP mode is active
        use_mliap = data.get(KEY.USE_MLIAP, torch.tensor(False, dtype=torch.bool))
        
        if not use_mliap.item():
            return data
        
        assert KEY.MLIAP_NUM_LOCAL_GHOST in data, "MLIAP_NUM_LOCAL_GHOST must be in data for ghost exchange"
        
        node_features = data[self.field]
        num_local_ghost = data[KEY.MLIAP_NUM_LOCAL_GHOST]  # [nlocal, num_ghost]
        nlocal = num_local_ghost[0].item()
        num_ghost = num_local_ghost[1].item()
        ntotal = nlocal + num_ghost
        
        # Extract local features
        if ghost_included:
            # Input already has ghosts, extract only local
            local_features = node_features[:nlocal]
        else:
            # Input is local only
            local_features = node_features
        
        # Prepare ghost features
        if num_ghost > 0:
            # Check if ghost features exist in data
            if KEY.NODE_FEATURE_GHOST in data:
                # Use existing ghost features (from previous exchange)
                ghost_features = data[KEY.NODE_FEATURE_GHOST]
            else:
                # Initialize ghost features as zeros
                ghost_features = torch.zeros(
                    (num_ghost,) + node_features.shape[1:],
                    dtype=node_features.dtype,
                    device=node_features.device,
                )
            
            # Concatenate local + ghost
            full_features = torch.cat([local_features, ghost_features], dim=0)
        else:
            # No ghosts needed
            full_features = local_features
        
        
        # Update data
        data[self.field] = full_features
        
        # Store ghost features separately for next layer
        if num_ghost > 0:
            data[KEY.NODE_FEATURE_GHOST] = ghost_features
        
        return data


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


def replace_ghost_exchange_modules(
    model: nn.Module,
    use_lammps_mpi: bool = False,
    field: str = KEY.NODE_FEATURE,
) -> nn.Module:
    """
    Replace NoOpGhostExchangeModule with active ghost exchange modules.
    
    Args:
        model: The model to modify
        use_lammps_mpi: If True, use LAMMPSMLIAPGhostExchangeModule (requires LAMMPS object)
                       If False, use SimpleGhostExchangeModule (metadata only)
        field: The field name for ghost exchange
    
    Returns:
        Modified model
    """
    def _recursive_replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, NoOpGhostExchangeModule):
                # Choose appropriate exchange module
                if use_lammps_mpi:
                    new_module = LAMMPSMLIAPGhostExchangeModule(field=field)
                else:
                    new_module = SimpleGhostExchangeModule(field=field)
                setattr(module, name, new_module)
            else:
                # Recursively process children
                _recursive_replace(child)
    
    _recursive_replace(model)
    return model


def prepare_mliap_data(lmp_data, device: torch.device) -> dict:
    """
    Helper function to convert LAMMPS data to SevenNet tensor format.
    
    Args:
        lmp_data: LAMMPS ML-IAP data object with nlocal and ntotal attributes
        device: torch device to place tensors on
    
    Returns:
        Dictionary with SevenNet-compatible tensor metadata
    """
    return {
        KEY.USE_MLIAP: torch.tensor(True, dtype=torch.bool, device=device),
        KEY.NUM_LOCAL_GHOST: torch.tensor(
            [lmp_data.nlocal, lmp_data.ntotal - lmp_data.nlocal],
            dtype=torch.int64,
            device=device
        ),
        KEY.LAMMPS_DATA: lmp_data,  # Keep for MPI exchange if needed
    }
