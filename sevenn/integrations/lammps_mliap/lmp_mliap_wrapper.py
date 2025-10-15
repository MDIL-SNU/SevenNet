# -*- coding: utf-8 -*-
"""
Minimal SevenNet ML-IAP wrapper

- __init__(model_path, tf32=False, **kwargs)
- Exposes only the unified attributes LAMMPS queries during connect():
- Implements compute_forces(self, lmp_data) with getter names.
"""

from __future__ import annotations
from typing import Optional, Sequence, Dict, Any, Tuple, Union, List
import os
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from sevenn.calculator import SevenNetCalculator
except Exception:
    from sevenn.sevennet_calculator import SevenNetCalculator  # type: ignore


class SevenNetLAMMPSMLIAPWrapper:
    """LAMMPS-MLIAP interface for SevenNet framework models (minimal form)."""

    # required by some loaders (kept for parity with NequIP wrappers)
    model_bytes: bytes = b""
    model_filename: str = ""

    def __init__(
        self,
        model_path: str,
        tf32: bool = False,
        **kwargs: Any,
    ):
        """
        kwargs:
            device: 'cuda' | 'cpu' (default: auto)
            element_types: list[str], e.g., ['H','O']  # MUST match pair_coeff order
            cutoff: float (Angstrom)                   # required if not inferable
            rcutfac: float = 1.0
            ndescriptors: int = 0
            nparams: int = 0
            rmin0: float = 0.0
            rfac0: float = 1.0
            type_to_Z: list[int] (LAMMPS type->Z mapping in pair_coeff order)
            calculator_kwargs: dict (forwarded to SevenNetCalculator)
        """

        # --- device / dtype ---
        device = kwargs.get("device", None)
        if device is None:
            device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self.device = device
        # self.dtype = np.float32
        
        # --- SevenNet calculator ---
        self._model_spec = model_path
        self._tf32 = tf32
        self.calculator_kwargs = kwargs.get("calculator_kwargs", None) or {}

        ck = dict(device=self.device)
        ck.update(self.calculator_kwargs)
        self._model_spec = model_path
        self.calc = SevenNetCalculator(model=model_path, **ck)
        self.calc.model.delete_module_by_key('force_output') # to autograd edge forces
        self.calc.model.eval_modal_map = False
        if self.calc.modal:
            self._modal_idx = self.calc.model.modal_map[self.calc.modal]
            print(f"INIT: {self.calc.modal}, {self._modal_idx}")

        # cutoff: if not provided, try to infer from calculator
        cutoff = kwargs.get("cutoff", None)
        if cutoff is None:
            # SevenNetCalculator exposes `.cutoff` for checkpoint/torchscript/model instance
            if hasattr(self.calc, "cutoff"):
                cutoff = float(self.calc.cutoff)
        if cutoff is None or not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("Please provide cutoff=... (Angstrom); could not infer from SevenNet calculator.")
        self.cutoff = float(cutoff)

        # optional capability flags
        self.supports_energy = True
        self.supports_forces = True
        self.supports_virial = True

        # tiny cache
        self._atoms_cache = None
        self._natoms = None
        
        # Store these for later. (any fallback required?)
        self._init_element_types = kwargs.get("element_types", None)
        self._init_type_to_Z = kwargs.get("type_to_Z", None)
        self._init_cutoff = kwargs.get("cutoff", None)
        self._init_rcutfac = kwargs.get("rcutfac", 1.0) # ignored

        # These will be set in _ensure_model_initialized
        self.element_types = None
        self.type_to_Z = None
        self._init_cutoff = kwargs.get("cutoff", None)

        self.element_types = None
        self.type_to_Z = None
        self.cutoff = None
        self.rcutfac = None

        cfg = getattr(self.calc, "sevennet_config", None)
        syms = cfg.get("chemical_species", None)
        self.element_types = list(syms)
        self.ntypes = len(self.element_types)
        self.type_to_Z = np.array([chemical_symbols.index(sym) for sym in self.element_types], dtype=np.int64)
        print(f"[INFO] Pre-initialized element_types: {self.element_types}")
        print(f"[INFO] Pre-initialized ntypes: {self.ntypes}")
        print(f"[INFO] Pre-initialized type_to_Z: {self.type_to_Z}")

        # Pre-calculate cutoff and rcutfac if provided
        if self._init_cutoff is not None:
            self.cutoff = float(self._init_cutoff)
            self.rcutfac = self.cutoff * 0.5
            print(f"[INFO] Pre-initialized cutoff: {self.cutoff}, rcutfac: {self.rcutfac}")

        # --- unified metadata required by connect() ---
        self.ndescriptors = int(kwargs.get("ndescriptors", 1))  # NN potential → 0
        self.nparams      = int(kwargs.get("nparams", 1))       # NN potential → 0

        # # HACK
        # tmp_model = "/gpfs/hansw/jinmuyou/omni_speed/MLIAP_test/svn/omni_small.pth"
        # tmp_kwargs = {
        #         'modal': "mpa",
        #         'enable_cueq': True,
        #         }
        # tmp_calc = SevenNetCalculator(model=tmp_model, **tmp_kwargs)
        # tmp_cfg = getattr(tmp_calc, "sevennet_config", None)
        # syms = tmp_cfg.get("chemical_species", None)
        # self.element_types = list(syms)
        # self.cutoff = 6

    # why required?
    def _ensure_model_initialized(self):
        """Lazy initialization of the model (called on first compute_forces)."""
        if self.calc is not None:
            return  # Already initialized
        print("[INFO] Lazy initializing SevenNet model...")

        # Apply tf32
        if self._tf32 and _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # SevenNet Calculator
        ## HACK:
        # self._model_spec = "/gpfs/hansw/jinmuyou/omni_speed/MLIAP_test/svn/omni_small.pth"
        # self.calculator_kwargs = {
        #         'modal': "mpa",
        #         'enable_cueq': True,
        #         }

        ck = dict(device=self.device)
        ck.update(self.calculator_kwargs)
        self.calc = SevenNetCalculator(model=self._model_spec, **ck)
        self.calc.model.delete_module_by_key('force_output') # to autograd edge forces
        self.calc.model.eval_modal_map = False
        if self.calc.modal:
            self._modal_idx = self.calc.model.modal_map[self.calc.modal]
            print(f"MODEL INIT: {self.calc.modal}, {self._modal_idx}")

        # cutoff: if not provided, try to infer from calculator
        if self.cutoff is None:
            if hasattr(self.calc, "cutoff"):
                self.cutoff = float(self.calc.cutoff)
        if self.cutoff is None or not np.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("Please provide cutoff=... (Angstrom); could not infer from SevenNet calculator.")
        self.cutoff = float(self.cutoff)

        is_ts = isinstance(self.calc.model, torch.jit.ScriptModule)
        print(f"Model is torchscript: {is_ts}")

    
    def __getstate__(self):
        """Drop non-picklable runtime objects; keep only config/state for rebuild."""
        state = self.__dict__.copy()
        # Do not pickle live calculator or atoms cache
        state["calc"] = None
        state["_atoms_cache"] = None
        return state

    def __setstate__(self, state):
        """Rebuild runtime objects after unpickling."""
        self.__dict__.update(state)
        # Model will be lazily initialized on first use
        self.calc = None
        self._atoms_cache = None
        self._natoms = None

        # Ensure all lazy init parameters exist (for backward compatibility)
        if not hasattr(self, '_tf32'):
            self._tf32 = False
        if not hasattr(self, 'calculator_kwargs'):
            self.calculator_kwargs = {}
        if not hasattr(self, '_init_element_types'):
            self._init_element_types = None
        if not hasattr(self, '_init_type_to_Z'):
            self._init_type_to_Z = None
        if not hasattr(self, '_init_cutoff'):
            self._init_cutoff = None
        if not hasattr(self, '_init_rcutfac'):
            self._init_rcutfac = 1.0

        # ck = dict(device=self.device)
        # # If you passed calculator_kwargs in __init__, keep it somewhere (see below)
        # if hasattr(self, "calculator_kwargs") and self.calculator_kwargs:
        #     ck.update(self.calculator_kwargs)
        # # _model_spec must contain the model path/keyword
        # # DBG
        # print(f"[DEBUG] Rebuilding SevenNetCalculator with: {ck}, model={self._model_spec}")
        # self.calc = SevenNetCalculator(model=self._model_spec, **ck)
        # self.calc.model.eval_modal_map = False
        # print(ck)
        # if self.calc.modal:
        #     self._modal_idx = self.calc.model.modal_map[self.calc.modal]
        #     print(f"INIT: {self.calc.modal}, {self._modal_idx}")
        # self._atoms_cache = None
        # self._natoms = None
    
    @staticmethod
    def _dbg(name, arr, head=3):
        a = np.asarray(arr)
        flat = a.reshape(-1)
        print(f"[DBG:{name}] shape={a.shape} dtype={a.dtype} "
            f"min={float(a.min()):.6g} max={float(a.max()):.6g} "
            f"head={flat[:head]}")

    @staticmethod
    def debug_lmp_data_simple(lmp_data):
        print("="*60)
        print("LAMMPS lmp_data attributes:")
        print("="*60)
        
        # 모든 public 속성
        attrs = [a for a in dir(lmp_data) if not a.startswith('_')]
        
        for attr in sorted(attrs):
            try:
                val = getattr(lmp_data, attr)
                if callable(val):
                    print(f"  {attr:20s} <method>")
                else:
                    # 값의 타입과 shape 정보
                    if hasattr(val, 'shape'):
                        print(f"  {attr:20s} {type(val).__name__:15s} shape={val.shape}")
                    elif hasattr(val, '__len__') and not isinstance(val, str):
                        print(f"  {attr:20s} {type(val).__name__:15s} len={len(val)}")
                    else:
                        print(f"  {attr:20s} {type(val).__name__:15s} = {val}")
            except Exception as e:
                print(f"  {attr:20s} <error: {e}>")
        print("="*60)
        
    # -------- unified entrypoint --------
    def compute_forces(self, lmp_data):
        # Lazy initialization
        self._ensure_model_initialized()

        if lmp_data.nlocal == 0 or lmp_data.npairs <= 1:
            return
        
        import sevenn._keys as KEY
        
        nlocal = lmp_data.nlocal
        ntotal = lmp_data.ntotal
        
        edge_vectors = torch.as_tensor(lmp_data.rij, dtype=torch.float32, device=self.device) # should be f32 in 7net
        edge_vectors.requires_grad_(True)

        edge_index = torch.vstack([
            torch.as_tensor(lmp_data.pair_i, dtype=torch.int64, device=self.device),
            torch.as_tensor(lmp_data.pair_j, dtype=torch.int64, device=self.device),
        ])
        elems = torch.as_tensor(lmp_data.elems, dtype=torch.int64, device=self.device)
        Z = elems
        print(elems)

        num_atoms = torch.as_tensor(nlocal, dtype=torch.int64, device=self.device)

        data = {
            KEY.EDGE_IDX  : edge_index,
            KEY.EDGE_VEC: edge_vectors,
            KEY.ATOMIC_NUMBERS: Z,
            KEY.NUM_ATOMS: num_atoms,

            KEY.USE_MLIAP: torch.tensor(True, dtype=torch.bool),
            KEY.MLIAP_NUM_LOCAL_GHOST: torch.tensor([nlocal, ntotal-nlocal], dtype=torch.int64, device=self.device),
            KEY.LAMMPS_DATA: lmp_data,
        }

        if self.calc.modal:
            data[KEY.MODAL_TYPE] = torch.tensor(
                self._modal_idx,
                dtype=torch.int64,
                device=self.device,
            )

        output = self.calc.model(data)
        pred_atomic_energies = output[KEY.ATOMIC_ENERGY].view(-1)
        edge_forces = torch.autograd.grad(
            torch.sum(pred_atomic_energies),
            [edge_vectors],
        )[0]

        if pred_atomic_energies.size(0) != nlocal:
            pred_atomic_energies = torch.narrow(pred_atomic_energies, 0, 0, nlocal) 
        pred_total_energy = torch.sum(pred_atomic_energies) 

        lmp_eatoms = torch.as_tensor(lmp_data.eatoms)
        lmp_eatoms.copy_(pred_atomic_energies)
        lmp_data.energy = pred_total_energy
        lmp_data.update_pair_forces_gpu(edge_forces)    

        ## DEBUG
        self.debug_lmp_data_simple(lmp_data)
        save_data_in = {k: v for k, v in data.items() if k != KEY.LAMMPS_DATA}
        torch.save(save_data_in, "/gpfs/hansw/jinmuyou/omni_speed/MLIAP_test/svn_ompa/svn_data_in.pt")
        torch.save(pred_atomic_energies, "/gpfs/hansw/jinmuyou/omni_speed/MLIAP_test/svn_ompa/atomic_energies.pt")
        torch.save(edge_forces, "/gpfs/hansw/jinmuyou/omni_speed/MLIAP_test/svn_ompa/edge_forces.pt")
 
        
    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
