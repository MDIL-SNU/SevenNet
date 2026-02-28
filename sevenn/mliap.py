from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from ase.data import chemical_symbols

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified

except ModuleNotFoundError:
    raise ImportError(
        'LAMMPS package supporting ML-IAP should be installed. '
        'Please refer to the instruction in issue #246. '
        'https://github.com/MDIL-SNU/SevenNet/issues/246#issuecomment-3500546381'
    )

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType
from sevenn.nn._ghost_exchange import MLIAPGhostExchangeModule
from sevenn.util import load_checkpoint, pretrained_name_to_path


class MLIAPWrappedConvolution(nn.Module):
    def __init__(self, conv):
        super().__init__()

        self.conv = conv
        self.ghost_exchange = MLIAPGhostExchangeModule(field=KEY.NODE_FEATURE)
        self._keys_to_narrow = (
            KEY.NODE_FEATURE,
            KEY.NODE_ATTR,
            KEY.ATOM_TYPE,
            KEY.ATOMIC_NUMBERS,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        nlocal = int(data[KEY.MLIAP_NUM_LOCAL_GHOST][0].item())

        data[KEY.NODE_FEATURE] = torch.narrow(data[KEY.NODE_FEATURE], 0, 0, nlocal)
        data = self.ghost_exchange(data)

        data = self.conv(data)

        for k in self._keys_to_narrow:
            data[k] = torch.narrow(data[k], 0, 0, nlocal)

        return data


class MLIAPWrappedIrrepsLinear(nn.Module):
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

        for k in self._keys_to_narrow:
            data[k] = torch.narrow(data[k], 0, 0, nlocal)

        data = self.linear(data)
        return data


# Referred Nequip-MLIAP impl.
class SevenNetMLIAPWrapper(MLIAPUnified):
    """LAMMPS-MLIAP interface for SevenNet framework models."""

    def __init__(
        self,
        model_path: str,
        **kwargs: Any,
    ):
        """
        kwargs:
            element_types: list[str], e.g., ['H','O']  # MUST match pair_coeff order
            modal: Optional[str] = None
            use_cueq: bool = False
            use_flash: bool = False
        """

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load checkpoint
        self.model_path = model_path
        if os.path.isfile(self.model_path):
            checkpoint_path = self.model_path
        else:
            try:
                checkpoint_path = pretrained_name_to_path(self.model_path)
            except Exception:
                raise ValueError(
                    f'{self.model_path} is not a model path '
                    'and a pretrained model name.'
                )
        self.cp = load_checkpoint(checkpoint_path)
        print(f'[INFO] Loaded checkpoint from {checkpoint_path}', flush=True)
        self.model = None  # lazy init

        # calc_kwargs
        self.use_cueq = kwargs.get('use_cueq', False)
        self.use_flash = kwargs.get('use_flash', False)
        self.use_oeq = kwargs.get('use_oeq', False)
        self.modal = kwargs.get('modal', None)

        # extract configs
        config = self.cp.config
        if self.modal is None:
            assert config.get(KEY.MODAL_MAP, None) is None, (
                'Modal not given but model has modal_map: '
                f'{list(config[KEY.MODAL_MAP].keys())}'
            )
        else:
            assert self.modal in config[KEY.MODAL_MAP], (
                f'Modal {self.modal} not found in model.modal_map: '
                f'{list(config[KEY.MODAL_MAP].keys())}'
            )

        self.cutoff = float(config[KEY.CUTOFF])
        self.rcutfac = self.cutoff * 0.5

        chemical_species = config[KEY.CHEMICAL_SPECIES]  # must present
        syms = chemical_symbols.copy()
        for i, sym in enumerate(syms):
            if sym not in chemical_species:
                syms[i] = 'X'  # not supported
        self.element_types = syms

        # dummy
        self.ndescriptors = int(kwargs.get('ndescriptors', 1))
        self.nparams = int(kwargs.get('nparams', 1))

    """
    Lazy initialization of the model (called on first compute_forces).
    Since script models cannot be pickled, we delay building the model
    until the first compute_forces call.
    """

    def _ensure_model_initialized(self):
        if self.model is not None:
            return  # Already initialized
        print('[INFO] Lazy initializing SevenNet model...', flush=True)
        print(f'[INFO] cueq={self.use_cueq}, flashTP={self.use_flash}, oeq={self.use_oeq}', flush=True)
        model = self.cp.build_model(
            enable_cueq=self.use_cueq, enable_flash=self.use_flash, enable_oeq=self.use_oeq
        )

        for k, module in model._modules.items():
            if k.endswith('_convolution'):
                model._modules[k] = MLIAPWrappedConvolution(module)
            elif k == 'onehot_to_feature_x':
                model._modules[k] = MLIAPWrappedIrrepsLinear(module)

        model.set_is_batch_data(False)
        if self.modal is not None:
            model.eval_modal_map = False
            model.prepare_modal_deploy(self.modal)  # set the fidelity of input data

        model.delete_module_by_key('force_output')  # to autograd edge forces
        self.model = model
        self.model.to(self.device)
        self.model.eval()

    # -------- unified entrypoint --------
    def compute_forces(self, lmp_data):
        self._ensure_model_initialized()  # lazy init
        assert self.model, 'Model must be initialized'
        if lmp_data.nlocal == 0:
            return

        nlocal = lmp_data.nlocal
        ntotal = lmp_data.ntotal
        no_pairs = lmp_data.npairs <= 1

        # edge_vectors should be f32 in 7net
        if no_pairs:
            edge_vectors = torch.zeros(
                (0, 3), dtype=torch.float32, device=self.device
            )
            edge_index = torch.zeros(
                (2, 0), dtype=torch.int64, device=self.device
            )
        else:
            edge_vectors = torch.as_tensor(
                lmp_data.rij, dtype=torch.float32, device=self.device
            )
            edge_index = torch.vstack(
                [
                    torch.as_tensor(
                        lmp_data.pair_i, dtype=torch.int64, device=self.device
                    ),
                    torch.as_tensor(
                        lmp_data.pair_j, dtype=torch.int64, device=self.device
                    ),
                ]
            )
        edge_vectors.requires_grad_(True)
        elems = torch.as_tensor(
            lmp_data.elems, dtype=torch.int64, device=self.device
        )
        num_atoms = torch.as_tensor(nlocal, dtype=torch.int64, device=self.device)
        mliap_num_local_ghost = torch.as_tensor(
            [nlocal, ntotal - nlocal], dtype=torch.int64, device=self.device
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
        output = self.model(data)
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

    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
