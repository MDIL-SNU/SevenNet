# -*- coding: utf-8 -*-
"""
Minimal SevenNet ML-IAP wrapper

- __init__(model_path, tf32=False, **kwargs)
- Exposes only the unified attributes LAMMPS queries during connect():
- Implements compute_forces(self, lmp_data) with getter names.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from lammps.mliap.mliap_unified_abc import MLIAPUnified

import sevenn._keys as KEY
from sevenn.calculator import SevenNetCalculator
from sevenn.util import load_checkpoint, pretrained_name_to_path


# Referred Nequip-MLIAP impl.
class SevenNetLAMMPSMLIAPWrapper(MLIAPUnified):
    """LAMMPS-MLIAP interface for SevenNet framework models."""

    def __init__(
        self,
        model_path: str,
        **kwargs: Any,
    ):
        """
        kwargs:
            element_types: list[str], e.g., ['H','O']  # MUST match pair_coeff order
            cutoff: float (Angstrom)                   # required if not inferable
            modal: Optional[str] = None
            enable_cueq: bool = False
            enable_flash: bool = False
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
                raise ValueError(f'{self.model_path} is not a model path and a pretrained model name.')
        self.cp = load_checkpoint(checkpoint_path)
        print(f'[INFO] Loaded checkpoint from {checkpoint_path}', flush=True)
        self.model = None # lazy init

        # calc_kwargs
        self.use_cueq  = kwargs.get('enable_cueq', False)
        self.use_flash = kwargs.get('enable_flash', False)
        self.modal = kwargs.get('modal', None)

        # extract configs
        config = self.cp.config
        if self.modal is None:
            assert config.get(KEY.MODAL_MAP, None) is None, \
                f'Modal not given but model has modal_map: {list(config[KEY.MODAL_MAP].keys())}'
        else:
            assert self.modal in config[KEY.MODAL_MAP], \
                f'Modal {self.modal} not found in model.modal_map: {list(config[KEY.MODAL_MAP].keys())}'


        self.cutoff = float(config[KEY.CUTOFF])
        self.rcutfac = self.cutoff * 0.5
        self.element_types = list(config.get(KEY.CHEMICAL_SPECIES, None))
        print(f'[INFO] Initialized cutoff: {self.cutoff} rcutfac: {self.rcutfac}', flush=True)
        # dummy
        self.ndescriptors = int(kwargs.get('ndescriptors', 1))
        self.nparams = int(kwargs.get('nparams', 1))


    """
    Lazy initialization of the SevenNet model.
    Since script models cannot be pickled, we delay building the model
    until the first compute_forces call.
    """
    def _ensure_model_initialized(self):
        """Lazy initialization of the model (called on first compute_forces)."""

        if self.model is not None:
            return  # Already initialized
        print('[INFO] Lazy initializing SevenNet model...', flush=True)
        print(f'[INFO] cueq={self.use_cueq}, flashTP={self.use_flash}', flush=True)
        model = self.cp.build_model(
            enable_cueq=self.use_cueq, enable_flash=self.use_flash
        )
        model.set_is_batch_data(False)

        if self.modal is not None:
            model.eval_modal_map = False
            model.prepare_modal_deploy(self.modal) # set the fidelity of input data
            print(f'[INFO] channel = {self.modal}', flush=True)

        model.delete_module_by_key('force_output') # to autograd edge forces
        self.model = model
        self.model.to(self.device)
        self.model.eval()

    # -------- unified entrypoint --------
    def compute_forces(self, lmp_data):
        self._ensure_model_initialized() # lazy init
        if lmp_data.nlocal == 0 or lmp_data.npairs <= 1:
            return

        nlocal = lmp_data.nlocal
        ntotal = lmp_data.ntotal

        # edge_vectors should be f32 in 7net
        edge_vectors = torch.as_tensor(lmp_data.rij, dtype=torch.float32, device=self.device)
        edge_vectors.requires_grad_(True)
        edge_index = torch.vstack([
            torch.as_tensor(lmp_data.pair_i, dtype=torch.int64, device=self.device),
            torch.as_tensor(lmp_data.pair_j, dtype=torch.int64, device=self.device),
        ])
        elems = torch.as_tensor(lmp_data.elems, dtype=torch.int64, device=self.device)
        num_atoms = torch.as_tensor(nlocal, dtype=torch.int64, device=self.device)

        # data prep
        data = {
            KEY.EDGE_IDX       : edge_index,
            KEY.EDGE_VEC       : edge_vectors,
            KEY.ATOMIC_NUMBERS : elems,
            KEY.NUM_ATOMS      : num_atoms,

            KEY.USE_MLIAP      : torch.tensor(True, dtype=torch.bool),
            KEY.MLIAP_NUM_LOCAL_GHOST: torch.tensor([nlocal, ntotal-nlocal], dtype=torch.int64, device=self.device),
            KEY.LAMMPS_DATA    : lmp_data,
        }

        # infer
        output = self.model(data)
        pred_atomic_energies = output[KEY.ATOMIC_ENERGY].view(-1)
        edge_forces = torch.autograd.grad(
            torch.sum(pred_atomic_energies),
            [edge_vectors],
        )[0]
        if pred_atomic_energies.size(0) != nlocal:
            pred_atomic_energies = torch.narrow(pred_atomic_energies, 0, 0, nlocal)
        pred_total_energy = torch.sum(pred_atomic_energies)

        # update
        lmp_eatoms = torch.as_tensor(lmp_data.eatoms)
        lmp_eatoms.copy_(pred_atomic_energies)
        lmp_data.energy = pred_total_energy
        # upcasting edge_forces required for update_pair_forces_gpu
        lmp_data.update_pair_forces_gpu(edge_forces.to(torch.float64))


    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
