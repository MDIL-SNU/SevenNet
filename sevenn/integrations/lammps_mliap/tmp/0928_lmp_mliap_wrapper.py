# -*- coding: utf-8 -*-
"""
Ultra-minimal SevenNet ML-IAP wrapper (NequIP-style init).

- __init__(model_path, tf32=False, **kwargs)
- Exposes only the unified attributes LAMMPS queries during connect():
- Implements compute_forces(self, lmp_data) with NequIP-like getter names.
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
        print("[save-probe] keys:", list(state.keys()))
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
        # --- tiny helper: call the first available method/attr name ---
        def _get(names):
            for n in names:
                if hasattr(lmp_data, n):
                    obj = getattr(lmp_data, n)
                    return obj() if callable(obj) else obj
            return None
        
        # inputs
        pos = self._to_host_numpy(lmp_data.get_positions(), dtype=self.dtype)

        # atomic numbers or types (recommended)
        z_like= self._to_host_numpy(lmp_data.get_types(), dtype=np.int64)

        # edges (required for pairwise outputs)
        ei = self._to_host_numpy(lmp_data.get_edge_index(),dtype=np.int64)
        i_idx, j_idx = ei[0], ei[1]
        E = ei.shape[1]

        # edge vectors (optional; else build from positions)
        rij = self._to_host_numpy(lmp_data.get_edge_vectors(), dtype=self.dtype) 

        # optional box/pbc
        nlocal = int(getattr(lmp_data, "nlocal"))
        # pos_loc = pos[:nlocal]
        # z_loc = z_like[:nlocal]
        
        # mins = pos_loc.min(axis=0)
        # maxs = pos_loc.max(axis=0)
        # box  = maxs - mins
        
        # pos_ase = pos_loc - mins
        # cell = np.diag(box.astype(self.dtype))
        # pbc = (True, True, True)
        cell = None
        pbc = (False, False, False)
        
        # --- guard against singular or missing cell when PBC is on ---
        def _is_singular_or_bad(M):
            try:
                return (M is None) or (M.shape != (3, 3)) or (not np.isfinite(M).all()) or (abs(np.linalg.det(M)) < 1e-12)
            except Exception:
                return True
        if any(bool(x) for x in (pbc if isinstance(pbc, (tuple, list)) else (pbc,))) and _is_singular_or_bad(cell):
            # Disable PBC if cell is unusable; SevenNet/matscipy requires a valid cell when PBC=True
            cell = None
            pbc = (False, False, False)
                

        # SevenNet
        atoms = self._ensure_atoms(pos, z_like, cell, pbc)
        total_E = float(atoms.get_potential_energy())
        F = self._to_host_numpy(atoms.get_forces(), dtype=self.dtype)
        
        per_atom_E = None
        epa = atoms.get_potential_energies()
        per_atom_E = self._to_host_numpy(epa, dtype=np.float64)
        
        if per_atom_E is not None:
            lmp_data.update_atom_energy(per_atom_E[:nlocal])

        # directions
        norm = np.linalg.norm(rij, axis=1, keepdims=True) + 1e-15
        ehat = rij / norm

        # pairwise forces via LSQ: Ei.T @ s ≈ Fi
        pair_force = np.zeros((E, 3), dtype=self.dtype)
        N = pos.shape[0]
        #N = nlocal
        edges_of_i = [[] for _ in range(N)]
        for e_id in range(E):
            ii = int(i_idx[e_id])
            if 0 <= ii < N:
                edges_of_i[i_idx[e_id]].append(e_id)
        for i in range(N):
            edges = edges_of_i[i]
            if not edges:
                continue
            Ei = ehat[edges, :]
            Fi = F[i, :]
            s, *_ = np.linalg.lstsq(Ei.T, Fi, rcond=None)
            pair_force[edges, :] = (s[:, None] * Ei)

        # pairwise energy distribution (if per-atom energies available)
        # if per_atom_E is not None:
        #     deg = np.zeros(N, dtype=np.int32)
        #     for e_id in range(E):
        #         deg[i_idx[e_id]] += 1
        #         deg[j_idx[e_id]] += 1
        #     deg_safe = np.where(deg > 0, deg, 1)
        #     contrib_i = per_atom_E[i_idx] / deg_safe[i_idx]
        #     contrib_j = per_atom_E[j_idx] / deg_safe[j_idx]
        #     pair_energy = 0.5 * (contrib_i + contrib_j)
        # else:
        #     pair_energy = np.zeros(E, dtype=self.dtype)

        # Try to pass device arrays first (for KOKKOS GPU); fallback to NumPy on failure
        pf64 = pair_force.astype(np.float64, copy=False)
        # pe64 = pair_energy.astype(np.float64, copy=False)

        pushed = False
        try:
            import cupy as cp
            # If CuPy is usable, move to device and hand its device pointer to LAMMPS
            pf_dev = cp.asarray(pf64)  # H2D if pf64 is NumPy; no copy if already CuPy
            lmp_data.update_pair_forces(pf_dev)
            # pe_dev = cp.asarray(pe64)
            # lmp_data.update_pair_energy(pe_dev)
            pushed = True
        except Exception:
            # CuPy not available or no CUDA device → fall back to CPU path
            pass

        if not pushed:
            lmp_data.update_pair_forces(pf64)
            # lmp_data.update_pair_energy(pe64)  
    
    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
