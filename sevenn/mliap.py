from __future__ import annotations

import os
from typing import Any

import torch
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
from sevenn.nn._ghost_exchange import (  # noqa: F401 (re-exported)
    MLIAPGhostExchange,
    MLIAPWrappedConvolution,
    MLIAPWrappedIrrepsLinear,
    mliap_compute_forces,
    patch_model_for_mliap,
)
from sevenn.util import load_checkpoint, pretrained_name_to_path


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
        self.use_lunatp = kwargs.get('use_lunatp', False)
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
        print(f'[INFO] cueq={self.use_cueq}, flashTP={self.use_flash}, oeq={self.use_oeq}, lunatp={self.use_lunatp}', flush=True)  # noqa: E501
        model = self.cp.build_model(
            enable_cueq=self.use_cueq,
            enable_flash=self.use_flash,
            enable_oeq=self.use_oeq,
            enable_lunatp=self.use_lunatp,
        )

        # exchange ghost features after every {t}_self_interaction_1
        patch_model_for_mliap(model)

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

        mliap_compute_forces(self.model, lmp_data, self.device)

    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
