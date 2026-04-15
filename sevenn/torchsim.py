"""TorchSim wrapper for SevenNet models."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.loader.dataloader import Collater

import sevenn._keys as key
from sevenn.atom_graph_data import AtomGraphData
from sevenn.calculator import torch_script_type
from sevenn.util import load_checkpoint

try:
    import torch_sim as ts
    from torch_sim.elastic import voigt_6_to_full_3x3_stress
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
except ImportError as exc:
    raise ImportError('torch_sim required: pip install torch-sim-atomistic') from exc


if TYPE_CHECKING:
    from collections.abc import Callable

    from sevenn.nn.sequential import AtomGraphSequential


def _validate(model: AtomGraphSequential, modal: str) -> None:
    if not model.type_map:
        msg = 'type_map is missing'
        raise ValueError(msg)

    if model.cutoff == 0.0:
        msg = 'Model cutoff seems not initialized'
        raise ValueError(msg)

    modal_map = model.modal_map
    if modal_map:
        modal_ava = list(modal_map)
        if not modal:
            msg = f'modal argument missing (avail: {modal_ava})'
            raise ValueError(msg)
        if modal not in modal_ava:
            msg = f'unknown modal {modal} (not in {modal_ava})'
            raise ValueError(msg)
    elif not model.modal_map and modal:
        warnings.warn(
            f'modal={modal} is ignored as model has no modal_map',
            stacklevel=2,
        )


class SevenNetModel(ModelInterface):  # type: ignore[misc,valid-type]
    """Computes atomistic energies, forces and stresses using an SevenNet model.

    This class wraps an SevenNet model to compute energies, forces, and stresses for
    atomistic systems. It handles model initialization, configuration, and
    provides a forward pass that accepts a SimState object and returns model
    predictions.

    Examples:
        >>> model = SevenNetModel(model='7net-omni', modal='mpa')
        >>> results = model(state)

    """

    def __init__(
        self,
        model: AtomGraphSequential | str | Path,
        *,  # force remaining arguments to be keyword-only
        modal: str | None = None,
        neighbor_list_fn: Callable | None = None,
        enable_cueq: bool = False,
        enable_flash: bool = False,
        enable_oeq: bool = False,
        device: torch.device | str = 'auto',
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the SevenNetModel with specified configuration.

        Loads an SevenNet model from either a model object or a model path.
        Sets up the model parameters for subsequent use in energy and force
        calculations.

        Args:
            model (str | Path | AtomGraphSequential): The SevenNet model to wrap.
                Accepts either 1) a path to a checkpoint file, 2) a model instance,
                or 3) a pretrained model name.
            modal (str | None): modal (fidelity) if given model is multi-modal model.
                for 7net-mf-ompa, it should be one of 'mpa' (MPtrj + sAlex) or
                'omat24' (OMat24).
            enable_cueq (bool): Enable cuEquivariance backend.
            enable_flash (bool): Enable flashTP backend.
            enable_oeq (bool): Enable OpenEquivariance backend.
            neighbor_list_fn (Callable): Neighbor list function to use.
                Default is torch_nl_linked_cell.
            device (torch.device | str): Device to run the model on
            dtype (torch.dtype): Data type for computation

        Raises:
            ImportError: if torch_sim is not installed
            ValueError: the model doesn't have a cutoff
            ValueError: the model has a modal_map but modal is not given
            ValueError: the modal given is not in the modal_map
            ValueError: the model doesn't have a type_map

        """
        if neighbor_list_fn is None:
            neighbor_list_fn = torchsim_nl

        super().__init__()

        if isinstance(device, str):
            if device == 'auto':
                self._device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )
            else:
                self._device = torch.device(device)
        else:
            self._device = device

        if dtype is not torch.float32:
            raise ValueError(
                f'SevenNet currently only supports {torch.float32}, but '
                + f'received different dtype: {dtype}'
            )

        if isinstance(model, (str, Path)):
            cp = load_checkpoint(model)
            model = cp.build_model(
                enable_flash=enable_flash,
                enable_cueq=enable_cueq,
                enable_oeq=enable_oeq,
            )

        _validate(model, modal)

        model.eval_type_map = torch.tensor(data=True)

        self._dtype = dtype
        self._memory_scales_with = 'n_atoms_x_density'
        self._compute_stress = True
        self._compute_forces = True

        model.set_is_batch_data(True)
        model_loaded = model
        self.cutoff = torch.tensor(model.cutoff)
        self.neighbor_list_fn = neighbor_list_fn

        self.model = model_loaded
        self.modal = modal
        self.type_map = model.type_map

        self.model = model.to(self._device)
        self.model = self.model.eval()

        if self._dtype is not None:
            self.model = self.model.to(dtype=self._dtype)

        self.implemented_properties = ['energy', 'forces', 'stress']

    @property
    def device(self) -> torch.device:
        """Device the model is running on."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Data type for computation."""
        return self._dtype

    def forward(self, state: ts.SimState, **kwargs) -> dict[str, torch.Tensor]:
        """Perform forward pass to compute energies, forces, and other properties.

        Takes a simulation state and computes the properties implemented by
        the model, such as energy, forces, and stresses.

        Args:
            state (SimState): State object containing positions, cells,
                atomic numbers, and other system information.

        Returns:
            dict: Model predictions, which may include:
                - energy (torch.Tensor): Energy with shape [batch_size]
                - forces (torch.Tensor): Forces with shape [n_atoms, 3]
                - stress (torch.Tensor): Stress tensor with shape [batch_size, 3, 3],
                    if compute_stress is True

        Notes:
            The state is automatically transferred to the model's device if needed.
            All output tensors are detached from the computation graph.

        """
        if state.device != self._device:
            state = state.to(self._device)

        # Batched neighbor list using linked-cell algorithm with row-vector cell
        positions = state.positions
        n_systems = int(state.system_idx.max().item() + 1)
        edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
            positions,
            state.row_vector_cell,
            state.pbc,
            self.cutoff,
            state.system_idx,
        )

        # Build per-system SevenNet AtomGraphData by slicing the global NL
        n_atoms_per_system = state.system_idx.bincount()
        stride = torch.cat(
            (
                torch.tensor([0], device=self._device, dtype=torch.long),
                n_atoms_per_system.cumsum(0),
            ),
        )

        data_list = []
        for sys_idx in range(n_systems):
            sys_start = stride[sys_idx].item()
            sys_end = stride[sys_idx + 1].item()

            pos = positions[sys_start:sys_end]
            row_vector_cell = state.row_vector_cell[sys_idx]
            atomic_nums = state.atomic_numbers[sys_start:sys_end]

            mask = mapping_system == sys_idx
            edge_idx_sys_global = edge_index[:, mask]
            unit_shifts_sys = unit_shifts[mask]

            # Convert global indices to local indices
            edge_idx = edge_idx_sys_global - sys_start
            shifts = torch.mm(unit_shifts_sys, row_vector_cell)
            edge_vec = pos[edge_idx[1]] - pos[edge_idx[0]] + shifts
            vol = torch.det(row_vector_cell)

            data = {
                key.NODE_FEATURE: atomic_nums,
                key.ATOMIC_NUMBERS: atomic_nums.to(
                    dtype=torch.int64,
                    device=self._device,
                ),
                key.POS: pos,
                key.EDGE_IDX: edge_idx,
                key.EDGE_VEC: edge_vec,
                key.CELL: row_vector_cell,
                key.CELL_SHIFT: unit_shifts_sys,
                key.CELL_VOLUME: vol,
                key.NUM_ATOMS: torch.tensor(len(atomic_nums), device=self._device),
                key.DATA_MODALITY: self.modal,
            }
            data[key.INFO] = {}

            data = AtomGraphData(**data)
            data_list.append(data)

        batched_data = Collater([], follow_batch=None, exclude_keys=None)(data_list)
        batched_data.to(self._device)

        if isinstance(self.model, torch_script_type):
            batched_data[key.NODE_FEATURE] = torch.tensor(
                [self.type_map[z.item()] for z in batched_data[key.ATOMIC_NUMBERS]],
                dtype=torch.int64,
                device=self._device,
            )
            batched_data[key.POS].requires_grad_(
                requires_grad=True,
            )  # backward compatibility
            batched_data[key.EDGE_VEC].requires_grad_(requires_grad=True)
            batched_data = batched_data.to_dict()
            del batched_data['data_info']

        output = self.model(batched_data)

        results: dict[str, torch.Tensor] = {}
        energy = output[key.PRED_TOTAL_ENERGY]
        if energy is not None:
            results['energy'] = energy
        else:
            results['energy'] = torch.zeros(
                int(state.system_idx.max().item() + 1), device=self._device,
            )

        forces = output[key.PRED_FORCE]
        if forces is not None:
            results['forces'] = forces

        stress = output[key.PRED_STRESS]
        if stress is not None:
            results['stress'] = -voigt_6_to_full_3x3_stress(
                stress[..., [0, 1, 2, 4, 5, 3]],
            )

        results = {k: v.detach() for k, v in results.items()}

        return results


# TODO: replace this with torchsim_d3.SevenNetD3Model
class SevenNetD3Model(ModelInterface):  # type: ignore[misc,valid-type]
    """SevenNet + D3 dispersion composite model for TorchSim.

    Wraps SevenNetModel and D3Calculator, sums their E/F/S outputs.
    Interface-compatible with SevenNetModel (same forward signature).

    Args match SevenNetModel plus D3 parameters. Mirrors
    SevenNetD3Calculator's arguments for consistency.
    """

    def __init__(
        self,
        model: AtomGraphSequential | str | Path,
        *,
        modal: str | None = None,
        enable_cueq: bool = False,
        enable_flash: bool = False,
        enable_oeq: bool = False,
        neighbor_list_fn: Callable | None = None,
        device: torch.device | str = 'auto',
        dtype: torch.dtype = torch.float32,
        damping_type: str = 'damp_bj',
        functional_name: str = 'pbe',
        vdw_cutoff: float = 9000,
        cn_cutoff: float = 1600,
    ) -> None:
        super().__init__()

        # Check if the D3Calculator is available
        from sevenn.calculator import D3Calculator

        self.sevennet = SevenNetModel(
            model,
            modal=modal,
            enable_cueq=enable_cueq,
            enable_flash=enable_flash,
            enable_oeq=enable_oeq,
            neighbor_list_fn=neighbor_list_fn,
            device=device,
            dtype=dtype,
        )
        self.d3 = D3Calculator(
            damping_type=damping_type,
            functional_name=functional_name,
            vdw_cutoff=vdw_cutoff,
            cn_cutoff=cn_cutoff,
        )

        # Proxy ModelInterface attributes from inner SevenNetModel
        self._device = self.sevennet._device
        self._dtype = self.sevennet._dtype
        self._compute_stress = True
        self._compute_forces = True
        self._memory_scales_with = 'n_atoms_x_density'
        self.cutoff = self.sevennet.cutoff
        self.neighbor_list_fn = self.sevennet.neighbor_list_fn
        self.implemented_properties = ['energy', 'forces', 'stress']

    @property
    def device(self) -> torch.device:
        """Device the model is running on."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Data type for computation."""
        return self._dtype

    def forward(self, state: ts.SimState, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass: SevenNet (batched GPU) + D3 (per-system CUDA ctypes).

        Returns combined {"energy", "forces", "stress"} dict.
        """
        results = self.sevennet(state, **kwargs)

        # Make tensors writable (sevennet detaches but they may share storage)
        results = {k: v.clone() for k, v in results.items()}

        # D3 uses a separate CUDA kernel via ctypes; sync to avoid conflicts
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)

        # D3 wraps positions internally (load_atom_info applies floor()
        # in fractional coords), so no need to wrap here.

        n_per = state.n_atoms_per_system.tolist()
        offsets = [0]
        for n in n_per:
            offsets.append(offsets[-1] + n)

        for i, single in enumerate(state.split()):
            atoms = single.to_atoms()[0]
            self.d3.calculate(atoms)
            d3r = self.d3.results

            results['energy'][i] += d3r['energy']

            si, ei = offsets[i], offsets[i + 1]
            results['forces'][si:ei] += torch.from_numpy(
                np.ascontiguousarray(d3r['forces']),
            ).to(device=self._device, dtype=self._dtype)

            # D3 stress: Voigt-6 [xx,yy,zz,yz,xz,xy] in eV/A^3 (ASE sign)
            # SevenNet stress is also intensive (eV/A^3), so add directly
            d3_stress_3x3 = voigt_6_to_full_3x3_stress(
                torch.tensor(
                    d3r['stress'], device=self._device, dtype=self._dtype,
                ),
            )
            results['stress'][i] += d3_stress_3x3

        return results


class Float64Wrapper(ModelInterface):  # type: ignore[misc,valid-type]
    """Wraps a float32 model so torch-sim runs in float64 precision.

    Casts state tensors to float32 before calling the wrapped model, then
    casts outputs back to float64.  Reports ``dtype=float64`` to torch-sim
    so all optimizer / integrator arithmetic is done in double precision.

    This is needed because ``SumModel`` requires all children to share the
    same dtype, and ``D3DispersionModel`` defaults to float64.
    """

    def __init__(self, model: ModelInterface) -> None:
        super().__init__()
        self._model = model
        self._device = model.device
        self._dtype = torch.float64
        self._compute_stress = model.compute_stress
        self._compute_forces = model.compute_forces
        self._memory_scales_with = getattr(
            model, '_memory_scales_with', 'n_atoms_x_density',
        )
        # Expose cutoff / neighbor_list_fn if present
        if hasattr(model, 'cutoff'):
            self.cutoff = model.cutoff
        if hasattr(model, 'neighbor_list_fn'):
            self.neighbor_list_fn = model.neighbor_list_fn

    def forward(self, state: ts.SimState, **kwargs) -> dict[str, torch.Tensor]:
        state_f32 = state.to(dtype=torch.float32)
        output = self._model(state_f32, **kwargs)
        return {
            k: v.to(dtype=torch.float64) if isinstance(v, torch.Tensor) else v
            for k, v in output.items()
        }
