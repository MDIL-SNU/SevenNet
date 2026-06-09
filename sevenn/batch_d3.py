"""Batched D3 dispersion correction via a single CUDA kernel launch.

Wraps the ``pair_d3_batch`` shared library (``pair_d3_batch.cu``) so that the
D3 dispersion energy/forces/stress for every system in a batch is computed in
one CUDA kernel launch.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import torch

# ---- .so loading (mirrors calculator._load but for pair_d3_batch) ----


class _BatchPairD3(ctypes.Structure):
    """Opaque ctypes handle for BatchPairD3 C++ object."""
    pass


def _load_batch_d3() -> ctypes.CDLL:
    """Load (or compile) pair_d3_batch shared library."""
    from torch.utils.cpp_extension import LIB_EXT, _get_build_directory, load

    name = 'pair_d3_batch'
    package_dir = os.path.dirname(os.path.abspath(__file__))

    # Try pre-built locations first
    for search_dir in [
        package_dir,
        _get_build_directory(name, verbose=False),
    ]:
        try:
            return ctypes.CDLL(os.path.join(search_dir, f'{name}{LIB_EXT}'))
        except OSError:
            pass

    # Compile from source
    if os.access(package_dir, os.W_OK):
        compile_dir = package_dir
    else:
        compile_dir = _get_build_directory(name, verbose=False)

    major, minor = torch.cuda.get_device_capability()
    detected_arch = f'{major}.{minor}'

    if os.environ.get('TORCH_CUDA_ARCH_LIST'):
        base_archs = os.environ['TORCH_CUDA_ARCH_LIST']
        if detected_arch not in base_archs:
            print(
                f'Warning: TORCH_CUDA_ARCH_LIST={base_archs} does not include '
                f'detected GPU architecture {detected_arch}.'
            )
    else:
        base_archs = '6.1;7.0;7.5;8.0;8.6;8.9;9.0'
        if detected_arch not in base_archs:
            base_archs += f';{detected_arch}'
        os.environ['TORCH_CUDA_ARCH_LIST'] = base_archs

    load(
        name=name,
        sources=[os.path.join(package_dir, 'pair_e3gnn', 'pair_d3_batch.cu')],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-fmad=false'],
        build_directory=compile_dir,
        verbose=True,
        is_python_module=False,
    )

    return ctypes.CDLL(os.path.join(compile_dir, f'{name}{LIB_EXT}'))


# ---- Batch D3 wrapper ----

class BatchD3:
    """Python wrapper for batched D3 CUDA library (pair_d3_batch.so).

    Handles ctypes bindings, species tracking, and numpy<->C data marshalling.
    Computes D3 dispersion for all systems in a batch with a single CUDA
    kernel launch, replacing the serial per-system D3Calculator loop.
    """

    def __init__(
        self,
        damping_type: str = 'damp_bj',
        functional_name: str = 'pbe',
        vdw_cutoff: float = 9000,
        cn_cutoff: float = 1600,
    ) -> None:
        if not torch.cuda.is_available():
            raise NotImplementedError('CPU + D3 is not implemented yet')

        self.damp_name = damping_type.lower()
        self.func_name = functional_name.lower()
        if self.damp_name not in ['damp_bj', 'damp_zero']:
            raise ValueError(f'Invalid damping type: {self.damp_name}')

        self.rthr = vdw_cutoff
        self.cnthr = cn_cutoff

        self._lib = _load_batch_d3()
        self._setup_ctypes()

        self._obj = self._lib.batch_d3_init()
        self._lib.batch_d3_settings(
            self._obj,
            ctypes.c_double(self.rthr),
            ctypes.c_double(self.cnthr),
            self.damp_name.encode(),
            self.func_name.encode(),
        )

        # Track known species for coeff re-initialization
        self._known_species: list[int] = []

    def _setup_ctypes(self) -> None:
        lib = self._lib

        lib.batch_d3_init.restype = ctypes.POINTER(_BatchPairD3)
        lib.batch_d3_init.argtypes = []

        lib.batch_d3_settings.argtypes = [
            ctypes.POINTER(_BatchPairD3),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        lib.batch_d3_settings.restype = None

        lib.batch_d3_coeff.argtypes = [
            ctypes.POINTER(_BatchPairD3),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.batch_d3_coeff.restype = None

        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_double_p = ctypes.POINTER(ctypes.c_double)

        lib.batch_d3_compute.argtypes = [
            ctypes.POINTER(_BatchPairD3),
            ctypes.c_int,       # B
            c_int_p,            # natoms_each
            c_int_p,            # atomtype
            c_double_p,         # x_flat
            c_double_p,         # cells
            c_int_p,            # pbc
            c_double_p,         # energy_out
            c_double_p,         # forces_out
            c_double_p,         # stress_out
        ]
        lib.batch_d3_compute.restype = None

        lib.batch_d3_compute_async.argtypes = [
            ctypes.POINTER(_BatchPairD3),
            ctypes.c_int,
            c_int_p, c_int_p, c_double_p, c_double_p, c_int_p,
        ]
        lib.batch_d3_compute_async.restype = None

        lib.batch_d3_sync.argtypes = [
            ctypes.POINTER(_BatchPairD3),
            ctypes.c_int,
            c_int_p, c_double_p, c_double_p, c_double_p,
        ]
        lib.batch_d3_sync.restype = None

        lib.batch_d3_fin.argtypes = [ctypes.POINTER(_BatchPairD3)]
        lib.batch_d3_fin.restype = None

    def _ensure_coeff(self, atomic_numbers: np.ndarray) -> None:
        """Re-call coeff() if new species appear in the batch."""
        unique_Z = sorted(set(int(z) for z in atomic_numbers))
        if unique_Z != self._known_species:
            self._known_species = unique_Z
            ntypes = len(unique_Z)
            arr = (ctypes.c_int * ntypes)(*unique_Z)
            self._lib.batch_d3_coeff(self._obj, arr, ctypes.c_int(ntypes))

    def _build_atomtype(self, atomic_numbers: np.ndarray) -> np.ndarray:
        """Map atomic numbers to 1-indexed types matching coeff() ordering."""
        z_to_type = {z: i + 1 for i, z in enumerate(self._known_species)}
        return np.array([z_to_type[int(z)] for z in atomic_numbers], dtype=np.int32)

    def _prepare_arrays(
        self,
        B: int,
        natoms_each: np.ndarray,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        cells: np.ndarray,
        pbc: np.ndarray,
    ) -> None:
        """Prepare contiguous C arrays and store for later sync."""
        self._ensure_coeff(atomic_numbers)

        # Fake box for zero-cell systems (molecules without a cell).
        cells = cells.copy()
        max_cutoff = np.sqrt(max(self.rthr, self.cnthr)) * 0.52917726
        offsets = np.zeros(B + 1, dtype=np.int64)
        np.cumsum(natoms_each, out=offsets[1:])
        for s in range(B):
            if cells[s].sum() == 0.0:
                pos_s = positions[offsets[s]:offsets[s + 1]]
                lengths = (
                    pos_s.max(axis=0) - pos_s.min(axis=0) + max_cutoff + 1.0
                )
                cells[s] = np.diag(lengths)

        self._async_B = B
        self._async_N = int(natoms_each.sum())
        self._async_natoms = np.ascontiguousarray(natoms_each, dtype=np.int32)
        self._async_atomtype = np.ascontiguousarray(
            self._build_atomtype(atomic_numbers), dtype=np.int32,
        )
        self._async_x = np.ascontiguousarray(positions.reshape(-1), dtype=np.float64)
        self._async_cells = np.ascontiguousarray(cells.reshape(-1), dtype=np.float64)
        self._async_pbc = np.ascontiguousarray(pbc.reshape(-1), dtype=np.int32)

    def compute_async(
        self,
        B: int,
        natoms_each: np.ndarray,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        cells: np.ndarray,
        pbc: np.ndarray,
    ) -> None:
        """Launch D3 kernels asynchronously (returns before kernels finish)."""
        self._prepare_arrays(B, natoms_each, atomic_numbers, positions, cells, pbc)
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_double_p = ctypes.POINTER(ctypes.c_double)
        self._lib.batch_d3_compute_async(
            self._obj,
            ctypes.c_int(B),
            self._async_natoms.ctypes.data_as(c_int_p),
            self._async_atomtype.ctypes.data_as(c_int_p),
            self._async_x.ctypes.data_as(c_double_p),
            self._async_cells.ctypes.data_as(c_double_p),
            self._async_pbc.ctypes.data_as(c_int_p),
        )

    def sync_results(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wait for async D3 and return results."""
        B = self._async_B
        N_total = self._async_N

        energy_out = np.zeros(B, dtype=np.float64)
        forces_out = np.zeros(N_total * 3, dtype=np.float64)
        stress_out = np.zeros(B * 9, dtype=np.float64)

        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_double_p = ctypes.POINTER(ctypes.c_double)
        self._lib.batch_d3_sync(
            self._obj,
            ctypes.c_int(B),
            self._async_natoms.ctypes.data_as(c_int_p),
            energy_out.ctypes.data_as(c_double_p),
            forces_out.ctypes.data_as(c_double_p),
            stress_out.ctypes.data_as(c_double_p),
        )

        return (
            energy_out,
            forces_out.reshape(N_total, 3),
            stress_out.reshape(B, 3, 3),
        )

    def compute(
        self,
        B: int,
        natoms_each: np.ndarray,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        cells: np.ndarray,
        pbc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run batched D3 computation (synchronous)."""
        self.compute_async(B, natoms_each, atomic_numbers, positions, cells, pbc)
        return self.sync_results()

    def __del__(self) -> None:
        if hasattr(self, '_obj') and self._obj:
            self._lib.batch_d3_fin(self._obj)
            self._obj = None
