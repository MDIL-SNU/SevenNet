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
            use_cuda_graph: bool = False  # one rank only; replays a captured
                CUDA graph of the model call (sevenn/nn/_mliap_cuda_graph.py)
            use_radial_table: bool = False  # radial MLPs as cubic tables in r
                (narrow layers) or with folded constants (wide layers), see
                sevenn/nn/radial_table.py optimize_radial
            use_fused: bool = False  # LunaTP, with the CUDA graph or the
                folded eager call (one rank): frozen-model kernels of
                sevenn/nn/lunatp_fused.py
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
        self.use_cuda_graph = kwargs.get('use_cuda_graph', False)
        self.use_radial_table = kwargs.get('use_radial_table', False)
        self.use_fused = kwargs.get('use_fused', False)
        self.graph = None  # lazy init

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

    def _build_model(self, ghost_exchange: bool):
        model = self.cp.build_model(
            enable_cueq=self.use_cueq,
            enable_flash=self.use_flash,
            enable_oeq=self.use_oeq,
            enable_lunatp=self.use_lunatp,
        )

        if ghost_exchange:
            # exchange ghost features after every {t}_self_interaction_1
            patch_model_for_mliap(model)

        model.set_is_batch_data(False)
        if self.modal is not None and ghost_exchange:
            model.eval_modal_map = False
            model.prepare_modal_deploy(self.modal)  # set the fidelity of input data

        model.delete_module_by_key('force_output')  # to autograd edge forces
        model.to(self.device)
        model.eval()
        if (
            not ghost_exchange
            and self.use_lunatp
            and (
                getattr(self, 'use_fused', False)
                or os.getenv('SEVENN_MLIAP_FUSED') == '1'
            )
            and os.getenv('SEVENN_MLIAP_FUSED') != '0'
        ):
            from sevenn.nn.lunatp_fused import fuse_radial_conv

            model.requires_grad_(False)
            names = fuse_radial_conv(model)
            print(f'[INFO] fused radial + conv: {" ".join(names)}', flush=True)
        if (
            getattr(self, 'use_radial_table', False)
            or os.getenv('SEVENN_MLIAP_RADIAL_TABLE') == '1'
        ) and os.getenv('SEVENN_MLIAP_RADIAL_TABLE') != '0':
            from sevenn.nn.radial_table import optimize_radial

            names = optimize_radial(
                model,
                int(os.getenv('SEVENN_RADIAL_TABLE_MAX_WIDTH', '1024')),
                kernel=self.device.type == 'cuda',
            )
            print(f'[INFO] radial MLPs: {" ".join(names)}', flush=True)
        return model

    def _ensure_model_initialized(self):
        if self.model is not None:
            return  # Already initialized
        print('[INFO] Lazy initializing SevenNet model...', flush=True)
        print(f'[INFO] cueq={self.use_cueq}, flashTP={self.use_flash}, oeq={self.use_oeq}, lunatp={self.use_lunatp}', flush=True)  # noqa: E501
        self.model = self._build_model(ghost_exchange=True)

    def _ensure_graph_initialized(self):
        if self.graph is not None:
            return
        from sevenn.nn._mliap_cuda_graph import MLIAPCudaGraph

        print('[INFO] SevenNet ML-IAP CUDA graph (one rank)', flush=True)
        print(f'[INFO] cueq={self.use_cueq}, flashTP={self.use_flash}, oeq={self.use_oeq}, lunatp={self.use_lunatp}', flush=True)  # noqa: E501
        # the plain serial model: ghosts are folded to their owners
        model = self._build_model(ghost_exchange=False)
        model.requires_grad_(False)  # the force backward skips dE/dW
        self.graph = MLIAPCudaGraph(
            model, self.device, self.cutoff, modal=self.modal
        )

    # -------- unified entrypoint --------
    def compute_forces(self, lmp_data):
        # getattr: potential files saved before use_cuda_graph existed
        use_graph = (
            getattr(self, 'use_cuda_graph', False)
            or os.getenv('SEVENN_MLIAP_CUDA_GRAPH') == '1'
        ) and self.device.type == 'cuda'
        if use_graph and os.getenv('SEVENN_MLIAP_CUDA_GRAPH') != '0':
            if getattr(self, 'graph', None) is None:
                self.graph = None
                self._ensure_graph_initialized()
            if self.graph.compute(lmp_data):
                return
        self._ensure_model_initialized()  # lazy init
        assert self.model, 'Model must be initialized'
        if lmp_data.nlocal == 0:
            return

        mliap_compute_forces(self.model, lmp_data, self.device)

    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
