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
        self.dtype = np.float32
        
        self._model_spec = model_path
        calc_kwargs = kwargs.get("calculator_kwargs", None) or {}
        self.calculator_kwargs = calc_kwargs

        if tf32 and _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        
        # --- SevenNet calculator ---
        ck = dict(device=self.device)
        ck.update(self.calculator_kwargs)
        self._model_spec = model_path
        self.calc = SevenNetCalculator(model=model_path, **ck)

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
        
        # --- resolve element_types (universal fallback) ---
        element_types = kwargs.get("element_types", None)

        # 0) if user gave type_to_Z only, map it to symbols first
        if element_types is None:
            t2z_kw = kwargs.get("type_to_Z", None)
            if t2z_kw is not None:
                element_types = [chemical_symbols[int(z)] for z in t2z_kw]

        # 1) try to infer from SevenNet calculator config (preferred)
        if element_types is None:
            syms = None
            cfg = getattr(self.calc, "sevennet_config", None)
            if isinstance(cfg, dict):
                # e.g. ['Ac','Ag',...,'Zr'] (full species list from training)
                syms = cfg.get("chemical_species", None)
                if syms is not None and len(syms) == 0:
                    syms = None
            if syms is not None:
                element_types = list(syms)

        # 2) fallback: derive from calc.type_map (keys are atomic numbers)
        if element_types is None and hasattr(self.calc, "type_map"):
            zs = sorted(int(z) for z in self.calc.type_map.keys())  # ascending Z
            element_types = [chemical_symbols[z] for z in zs]

        if element_types is None:
            raise ValueError(
                "Please provide element_types=['H','O',...] (pair_coeff order) "
                "or type_to_Z=[1,8,...]; could not infer from SevenNet calculator."
            )

        self.element_types = list(element_types)
        
        # --- build LAMMPS type->Z mapping (must match element_types order) ---
        t2z_kw = kwargs.get("type_to_Z", None)
        if t2z_kw is None:
            # build from element_types (pair_coeff order)
            self.type_to_Z = np.array([chemical_symbols.index(sym) for sym in self.element_types], dtype=np.int64)
        else:
            self.type_to_Z = np.array(t2z_kw, dtype=np.int64)

        # sanity: type_to_Z must map back to the same element_types (order-sensitive)
        syms_back = [chemical_symbols[int(z)] for z in self.type_to_Z.tolist()]
        if syms_back != self.element_types:
            raise ValueError(f"type_to_Z {syms_back} does not match element_types {self.element_types}")

        # --- unified metadata required by connect() ---
        self.ndescriptors = int(kwargs.get("ndescriptors", 0))  # NN potential → 0
        self.nparams      = int(kwargs.get("nparams", 0))       # NN potential → 0
        self.rmin0        = float(kwargs.get("rmin0", 0.0))
        self.rfac0        = float(kwargs.get("rfac0", 1.0))
        self.rcutfac      = float(kwargs.get("rcutfac", 1.0))
        self.ntypes       = len(self.element_types)             # some builds probe this

    # -------- helpers --------
    def _maybe_types_to_Z(self, z_like: np.ndarray) -> np.ndarray:
        """Map LAMMPS types to atomic numbers Z. Handles 1-based and 0-based."""
        arr = np.asarray(z_like, dtype=np.int64)
        if self.type_to_Z is None or arr.size == 0:
            return arr
        zmap = self.type_to_Z  # shape (ntypes,)
        minv = int(arr.min())
        maxv = int(arr.max())
        # 1-based types: {1..ntypes}
        if minv >= 1 and maxv <= len(zmap):
            return zmap[arr - 1]
        # 0-based types: {0..ntypes-1}
        if minv >= 0 and maxv < len(zmap):
            return zmap[arr]
        # Otherwise assume it's already Z
        return arr

    def _ensure_atoms(self, pos: np.ndarray, z_like: np.ndarray,
                      cell: Optional[np.ndarray], pbc: Union[bool, Tuple[bool,bool,bool]]) -> Atoms:
        """Build or update an ASE Atoms with SevenNet calculator attached."""
        n = int(pos.shape[0])
        pos = np.asarray(pos, dtype=self.dtype)
        Z = self._maybe_types_to_Z(np.asarray(z_like, dtype=np.int64))
        pbc_tuple = (pbc, pbc, pbc) if isinstance(pbc, bool) else tuple(bool(x) for x in pbc)
        cell3 = None if cell is None else np.asarray(cell, dtype=self.dtype).reshape(3, 3)

        if self._atoms_cache is None or self._natoms != n:
            atoms = Atoms(numbers=Z.tolist(), positions=pos, cell=cell3, pbc=pbc_tuple)
            atoms.calc = self.calc
            self._atoms_cache, self._natoms = atoms, n
        else:
            atoms = self._atoms_cache
            atoms.positions[:] = pos
            if cell3 is not None:
                atoms.cell[:] = cell3
            atoms.pbc = pbc_tuple
        return atoms
    
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
        # Rebuild SevenNet calculator from stored config
        ck = dict(device=self.device)
        # If you passed calculator_kwargs in __init__, keep it somewhere (see below)
        if hasattr(self, "calculator_kwargs") and self.calculator_kwargs:
            ck.update(self.calculator_kwargs)
        # _model_spec must contain the model path/keyword
        self.calc = SevenNetCalculator(model=self._model_spec, **ck)
        self._atoms_cache = None
        self._natoms = None

    @staticmethod
    def _to_host_numpy(x, dtype=None):
        """Convert CuPy/Torch/Numpy to host NumPy array."""
        try:
            import cupy as cp
            if isinstance(x, cp.ndarray):   # CuPy → NumPy
                x = cp.asnumpy(x)
        except Exception:
            pass
        try:
            import torch
            if isinstance(x, torch.Tensor): # Torch → NumPy (CPU)
                x = x.detach().cpu().numpy()
        except Exception:
            pass
        a = np.asarray(x)
        return a.astype(dtype, copy=False) if dtype is not None else a
    
    @staticmethod
    def _dbg(name, arr, head=3):
        a = np.asarray(arr)
        flat = a.reshape(-1)
        print(f"[DBG:{name}] shape={a.shape} dtype={a.dtype} "
            f"min={float(a.min()):.6g} max={float(a.max()):.6g} "
            f"head={flat[:head]}")

    # -------- unified entrypoint --------
    def compute_forces(self, lmp_data):
        """
        getters expected on lmp_data:
            get_positions() -> (N,3) float64
            get_atomic_numbers() or get_types() -> (N,) int64
            get_edge_index() -> (2,E) int64
            get_edge_vectors() optional -> (E,3) float64
            get_cell() optional -> (3,3)
            get_pbc() optional -> bool or (3,)
        Updaters:
            update_pair_forces(E,3), update_pair_energy(E), update_atom_energy(float)
        """        
        # inputs
        pos = self._to_host_numpy(lmp_data.get_positions(), dtype=self.dtype)
        types= self._to_host_numpy(lmp_data.get_types(), dtype=np.int64)

        # edges (required for pairwise outputs)
        ei = self._to_host_numpy(lmp_data.get_edge_index(),dtype=np.int64)
        rij = self._to_host_numpy(lmp_data.get_edge_vectors(), dtype=self.dtype) 
        
        # print for debugging
        self._dbg("pos", pos); self._dbg("types", types); self._dbg("edge_index", ei); self._dbg("rij", rij)
        
        cell = self._to_host_numpy(lmp_data.get_cell(), dtype=self.dtype)  # (3,3)
        pbc  = tuple(bool(x) for x in lmp_data.get_pbc())
        nlocal = int(getattr(lmp_data, "nlocal"))
        tags   = self._to_host_numpy(lmp_data.get_tags(),  dtype=np.int64)
        
        pos_loc = pos[:nlocal]
        s_idx   = types[:nlocal].astype(np.int64, copy=False)
        
        is_ts = isinstance(self.calc.model, torch.jit.ScriptModule)
        if is_ts:
            x = s_idx
        else:
            Z = np.array([chemical_symbols.index(self.element_types[i]) for i in s_idx], dtype=np.int64)
        
        i_idx, j_idx = ei[0], ei[1]
        tag_to_local = {int(tags[i]): i for i in range(nlocal)}    # tag -> 0..nlocal-1
        
        valid = np.array([int(tags[j]) in tag_to_local for j in j_idx], dtype=bool)
        if not valid.all():
            i_idx = i_idx[valid]
            j_idx = j_idx[valid]
            rij   = rij[valid]

        j_mapped = np.array([tag_to_local[int(tags[j])] for j in j_idx ], dtype=np.int64)
        edge_index = np.stack([i_idx, j_mapped], axis=0).astype(np.int64)
        
        cell64      = cell.astype(np.float64, copy=False)
        dpos64      = (pos[j_idx] - pos[i_idx]).astype(np.float64, copy=False)
        shift_vec64 = (rij.astype(np.float64, copy=False) - dpos64)
        s64 = np.linalg.solve(cell64.T, shift_vec64.T).T
        
        s   = np.rint(s64).astype(np.float32, copy=False)

        # SevenNet
        dev = self.calc.device
        V   = float(np.dot(cell[0], np.cross(cell[1], cell[2])))
        data = {
            "pos": torch.as_tensor(pos_loc, device=dev, dtype=torch.float32).requires_grad_(True),
            "edge_index": torch.as_tensor(edge_index, device=dev, dtype=torch.long),
            "edge_vec":   torch.as_tensor(rij, device=dev, dtype=torch.float32).requires_grad_(True),
            "num_atoms": torch.as_tensor([nlocal], device=dev, dtype=torch.long),
            "cell_lattice_vectors": torch.as_tensor(cell, device=dev, dtype=torch.float32),
            "cell_volume": torch.as_tensor(V, device=dev, dtype=torch.float32),
            "pbc_shift": torch.as_tensor(s, device=dev, dtype=torch.float32),
        }
        if is_ts:
            data["x"] = torch.as_tensor(x, device=dev, dtype=torch.long)
        else:
            data["atomic_numbers"] = torch.as_tensor(Z, device=dev, dtype=torch.long)
        
        out = self.calc.model(data)

        E_total_t = out["inferred_total_energy"]
        F_t       = out["inferred_force"]
        S_t       = out["inferred_stress"]
        e_i_t     = out.get("atomic_energy", None)

        E_total = float(E_total_t.detach().cpu().item())
        F_loc   = F_t.detach().cpu().numpy()[:nlocal, :].astype(np.float64, copy=False)
        S_voigt = S_t.detach().cpu().numpy().astype(np.float64, copy=False)

        if e_i_t is not None: # Per-atom energy is available
            e_i = e_i_t.detach().cpu().numpy().reshape(-1)[:nlocal].astype(np.float64, copy=False)
            lmp_data.update_atom_energy(e_i)
        else: # Case of non-per atom energy : Directly use total E
            lmp_data.set_total_energy(E_total)
        
        
        self._dbg("edge_index_valid", edge_index); 
        self._dbg("rij_valid", rij)
        self._dbg("pbc_shift", s)
        print(f"[DBG:cell] H=\n{cell}\n[DBG:vol] V={V:.9g}")

            
        print(f"[DBG:E] E_total={E_total:.9g}")
        self._dbg("F_loc", F_loc)
        self._dbg("S_voigt(raw)", S_voigt)
        print(f"[DBG:sumF] ||sum_i F_i||={np.linalg.norm(F_loc.sum(axis=0)):.6g}")

        lmp_data.set_atom_forces(F_loc)
        lmp_data.add_global_stress(S_voigt)
        
    
    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
