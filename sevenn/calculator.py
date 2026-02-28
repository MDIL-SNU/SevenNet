import ctypes
import os
import pathlib
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.jit
import torch.jit._script
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.data import chemical_symbols

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.train.dataload import unlabeled_atoms_to_graph

torch_script_type = torch.jit._script.RecursiveScriptModule


class SevenNetCalculator(Calculator):
    """Supporting properties:
    'free_energy', 'energy', 'forces', 'stress', 'energies'
    free_energy equals energy. 'energies' stores atomic energy.

    Multi-GPU acceleration is not supported with ASE calculator.
    You should use LAMMPS for the acceleration.
    """

    def __init__(
        self,
        model: Union[str, pathlib.PurePath, AtomGraphSequential] = '7net-0',
        file_type: str = 'checkpoint',
        device: Union[torch.device, str] = 'auto',
        modal: Optional[str] = None,
        enable_cueq: Optional[bool] = False,
        enable_flash: Optional[bool] = False,
        enable_oeq: Optional[bool] = False,
        sevennet_config: Optional[Dict] = None,  # Not used in logic, just meta info
        **kwargs,
    ) -> None:
        """Initialize SevenNetCalculator.

        Parameters
        ----------
        model: str | Path | AtomGraphSequential, default='7net-0'
            Name of pretrained models (7net-omni, 7net-mf-ompa, 7net-omat, 7net-l3i5,
            7net-0) or path to the checkpoint, deployed model or the model itself
        file_type: str, default='checkpoint'
            one of 'checkpoint' | 'torchscript' | 'model_instance'
        device: str | torch.device, default='auto'
            if not given, use CUDA if available
        modal: str | None, default=None
            modal (fidelity) if given model is multi-modal model. for 7net-mf-ompa,
            it should be one of 'mpa' (MPtrj + sAlex) or 'omat24' (OMat24)
            for 7net-omni, use one of 'mpa', 'omat24', 'matpes_pbe', 'matpes_r2scan',
            'mp_r2scan', 'oc20', 'oc22', 'odac23', 'omol25_low', 'omol25_high',
            'spice', 'qcml', 'pet_mad'
            case insensitive
        enable_cueq: bool, default=None (use the checkpoint's backend)
            if True, use cuEquivariant to accelerate inference.
        enable_flash: bool, default=None (use the checkpoint's backend)
            if True, use FlashTP to accelerate inference.
        enable_oeq: bool, default=False
            if True, use OpenEquivariance to accelerate inference.
        sevennet_config: dict | None, default=None
            Not used, but can be used to carry meta information of this calculator
        """
        super().__init__(**kwargs)
        self.sevennet_config = None

        if isinstance(model, pathlib.PurePath):
            model = str(model)

        allowed_file_types = ['checkpoint', 'torchscript', 'model_instance']
        file_type = file_type.lower()
        if file_type not in allowed_file_types:
            raise ValueError(f'file_type not in {allowed_file_types}')

        enable_cueq = os.getenv('SEVENNET_ENABLE_CUEQ') == '1' or enable_cueq
        enable_flash = os.getenv('SEVENNET_ENABLE_FLASH') == '1' or enable_flash
        enable_oeq = os.getenv('SEVENNET_ENABLE_OEQ') == '1' or enable_oeq

        if enable_cueq and file_type in ['model_instance', 'torchscript']:
            warnings.warn(
                'file_type should be checkpoint to enable cueq. cueq set to False'
            )
            enable_cueq = False

        # TODO: not verified this line
        if enable_oeq and file_type in ['model_instance', 'torchscript']:
            warnings.warn(
                'file_type should be checkpoint to enable oeq. oeq set to False'
            )
            enable_oeq = False

        if isinstance(device, str):  # TODO: do we really need this?
            if device == 'auto':
                self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        if file_type == 'checkpoint' and isinstance(model, str):
            cp = util.load_checkpoint(model)

            model_loaded = cp.build_model(
                enable_cueq=enable_cueq, enable_flash=enable_flash, enable_oeq=enable_oeq
            )
            model_loaded.set_is_batch_data(False)

            self.type_map = cp.config[KEY.TYPE_MAP]
            self.cutoff = cp.config[KEY.CUTOFF]
            self.sevennet_config = cp.config

        elif file_type == 'torchscript' and isinstance(model, str):
            if modal:
                raise NotImplementedError()
            extra_dict = {
                'chemical_symbols_to_index': b'',
                'cutoff': b'',
                'num_species': b'',
                'model_type': b'',
                'version': b'',
                'dtype': b'',
                'time': b'',
            }
            model_loaded = torch.jit.load(
                model, _extra_files=extra_dict, map_location=self.device
            )
            chem_symbols = extra_dict['chemical_symbols_to_index'].decode('utf-8')
            sym_to_num = {sym: n for n, sym in enumerate(chemical_symbols)}
            self.type_map = {
                sym_to_num[sym]: i for i, sym in enumerate(chem_symbols.split())
            }
            self.cutoff = float(extra_dict['cutoff'].decode('utf-8'))

        elif isinstance(model, AtomGraphSequential):
            if model.type_map is None:
                raise ValueError(
                    'Model must have the type_map to be used with calculator'
                )
            if model.cutoff == 0.0:
                raise ValueError('Model cutoff seems not initialized')
            model.eval_type_map = torch.tensor(True)  # ?
            model.set_is_batch_data(False)
            model_loaded = model
            self.type_map = model.type_map
            self.cutoff = model.cutoff
        else:
            raise ValueError('Unexpected input combinations')

        if self.sevennet_config is None and sevennet_config is not None:
            self.sevennet_config = sevennet_config

        self.model = model_loaded

        self.modal = None
        if isinstance(self.model, AtomGraphSequential):
            modal_map = self.model.modal_map
            if modal_map:
                modal_ava = list(modal_map.keys())
                if not modal:
                    raise ValueError(f'modal argument missing (avail: {modal_ava})')
                elif modal not in modal_ava:
                    raise ValueError(f'unknown modal {modal} (not in {modal_ava})')
                self.modal = modal
            elif not self.model.modal_map and modal:
                warnings.warn(f'modal={modal} is ignored as model has no modal_map')

        self.model.to(self.device)
        self.model.eval()
        self.implemented_properties = [
            'free_energy',
            'energy',
            'forces',
            'stress',
            'energies',
        ]

    def set_atoms(self, atoms: Atoms) -> None:
        # called by ase, when atoms.calc = calc
        zs = tuple(set(atoms.get_atomic_numbers()))
        for z in zs:
            if z not in self.type_map:
                sp = list(self.type_map.keys())
                raise ValueError(
                    f'Model do not know atomic number: {z}, (knows: {sp})'
                )

    def output_to_results(self, output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        energy = output[KEY.PRED_TOTAL_ENERGY].detach().cpu().item()
        num_atoms = output['num_atoms'].item()
        atomic_energies = output[KEY.ATOMIC_ENERGY].detach().cpu().numpy().flatten()
        forces = output[KEY.PRED_FORCE].detach().cpu().numpy()[:num_atoms, :]
        stress = np.array(
            (-output[KEY.PRED_STRESS])
            .detach()
            .cpu()
            .numpy()[[0, 1, 2, 4, 5, 3]]  # as voigt notation
        )
        # Store results
        return {
            'free_energy': energy,
            'energy': energy,
            'energies': atomic_energies,
            'forces': forces,
            'stress': stress,
            'num_edges': output[KEY.EDGE_IDX].shape[1],
        }

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        is_ts_type = isinstance(self.model, torch_script_type)

        # call parent class to set necessary atom attributes
        Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            raise ValueError('No atoms to evaluate')
        data = AtomGraphData.from_numpy_dict(
            unlabeled_atoms_to_graph(atoms, self.cutoff, with_shift=is_ts_type)
        )
        if self.modal:
            data[KEY.DATA_MODALITY] = self.modal

        data.to(self.device)  # type: ignore

        if is_ts_type:
            data[KEY.NODE_FEATURE] = torch.tensor(
                [self.type_map[z.item()] for z in data[KEY.NODE_FEATURE]],
                dtype=torch.int64,
                device=self.device,
            )
            data[KEY.POS].requires_grad_(True)  # backward compatibility
            data[KEY.EDGE_VEC].requires_grad_(True)  # backward compatibility
            data = data.to_dict()
            del data['data_info']

        self.results = self.output_to_results(self.model(data))


class SevenNetD3Calculator(SumCalculator):
    def __init__(
        self,
        model: Union[str, pathlib.PurePath, AtomGraphSequential] = '7net-0',
        file_type: str = 'checkpoint',
        device: Union[torch.device, str] = 'auto',
        sevennet_config: Optional[Any] = None,  # hold meta information
        damping_type: str = 'damp_bj',
        functional_name: str = 'pbe',
        vdw_cutoff: float = 9000,  # au^2, 0.52917726 angstrom = 1 au
        cn_cutoff: float = 1600,  # au^2, 0.52917726 angstrom = 1 au
        **kwargs,
    ) -> None:
        """Initialize SevenNetD3Calculator. CUDA required.

        Parameters
        ----------
        model: str | Path | AtomGraphSequential
            Name of pretrained models (7net-omni, 7net-mf-ompa, 7net-omat, 7net-l3i5,
            7net-0) or path to the checkpoint, deployed model or the model itself
        file_type: str, default='checkpoint'
            one of 'checkpoint' | 'torchscript' | 'model_instance'
        device: str | torch.device, default='auto'
            if not given, use CUDA if available
        modal: str | None, default=None
            modal (fidelity) if given model is multi-modal model. for 7net-mf-ompa,
            for 7net-omni, use one of 'mpa', 'omat24', 'matpes_pbe', 'matpes_r2scan',
            'mp_r2scan', 'oc20', 'oc22', 'odac23', 'omol25_low', 'omol25_high',
            'spice', 'qcml', 'pet_mad'
            it should be one of 'mpa' (MPtrj + sAlex) or 'omat24' (OMat24)
        enable_cueq: bool, default=False
            if True, use cuEquivariant to accelerate inference.
        damping_type: str, default='damp_bj'
            Damping type of D3, one of 'damp_bj' | 'damp_zero'
        functional_name: str, default='pbe'
            Target functional name of D3 parameters.
        vdw_cutoff: float, default=9000
            vdw cutoff of D3 calculator in au
        cn_cutoff: float, default=1600
            cn cutoff of D3 calculator in au
        """
        d3_calc = D3Calculator(
            damping_type=damping_type,
            functional_name=functional_name,
            vdw_cutoff=vdw_cutoff,
            cn_cutoff=cn_cutoff,
            **kwargs,
        )

        sevennet_calc = SevenNetCalculator(
            model=model,
            file_type=file_type,
            device=device,
            sevennet_config=sevennet_config,
            **kwargs,
        )

        super().__init__([sevennet_calc, d3_calc])


def _load(name: str) -> ctypes.CDLL:
    from torch.utils.cpp_extension import LIB_EXT, _get_build_directory, load

    # Load the library from the candidate locations

    package_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        return ctypes.CDLL(os.path.join(package_dir, f'{name}{LIB_EXT}'))
    except OSError:
        pass

    cache_dir = _get_build_directory(name, verbose=False)
    try:
        return ctypes.CDLL(os.path.join(cache_dir, f'{name}{LIB_EXT}'))
    except OSError:
        pass

    # Compile the library if it is not found

    if os.access(package_dir, os.W_OK):
        compile_dir = package_dir
    else:
        print('Warning: package directory is not writable. Using cache directory.')
        compile_dir = cache_dir

    # Matching CUDA architectures
    # If TORCH_CUDA_ARCH_LIST is set, we respect the user setting but warn if
    # it does not include the detected architecture.
    # If not set, we set it to a reasonable default including the detected arch.
    # We do not worry about CPU-only setup (not implemented yet).

    import torch
    major, minor = torch.cuda.get_device_capability()
    detected_arch = f'{major}.{minor}'

    if os.environ.get('TORCH_CUDA_ARCH_LIST'):
        base_archs = os.environ['TORCH_CUDA_ARCH_LIST']
        if detected_arch not in base_archs:
            print(
                f'Warning: TORCH_CUDA_ARCH_LIST={base_archs} does not include '
                f'detected GPU architecture {detected_arch}. '
                f'It may lead to compile error or NoKernelImageForDevice error.'
            )
    else:
        base_archs = '6.1;7.0;7.5;8.0;8.6;8.9;9.0'
        if detected_arch not in base_archs:
            base_archs += f';{detected_arch}'
        os.environ['TORCH_CUDA_ARCH_LIST'] = base_archs
        print(
            f'Info: TORCH_CUDA_ARCH_LIST is not set. '
            f'Detected GPU architecture {detected_arch}, '
            f'setting TORCH_CUDA_ARCH_LIST={base_archs} for compilation.'
        )

    load(
        name=name,
        sources=[os.path.join(package_dir, 'pair_e3gnn', 'pair_d3_for_ase.cu')],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-fmad=false'],
        build_directory=compile_dir,
        verbose=True,
        is_python_module=False,
    )

    return ctypes.CDLL(os.path.join(compile_dir, f'{name}{LIB_EXT}'))


class PairD3(ctypes.Structure):
    pass  # Opaque structure; only used as a pointer


class D3Calculator(Calculator):
    """ASE calculator for accelerated D3 van der Waals (vdW) correction.

    Example:
        from ase.calculators.mixing import SumCalculator
        calc_1 = SevenNetCalculator()
        calc_2 = D3Calculator()
        return SumCalculator([calc_1, calc_2])

    This calculator interfaces with the `libpaird3.so` library,
    which is compiled by nvcc during the package installation.
    If you encounter any errors, please verify
    the installation process and the compilation options in `setup.py`.
    Note: Multi-GPU parallel MD is not supported in this mode.
    Note: Cffi could be used, but it was avoided to reduce dependencies.
    """

    # Here, free_energy = energy
    implemented_properties = ['free_energy', 'energy', 'forces', 'stress']

    def __init__(
        self,
        damping_type: str = 'damp_bj',  # damp_bj, damp_zero
        functional_name: str = 'pbe',  # check the source code
        vdw_cutoff: float = 9000,  # au^2, 0.52917726 angstrom = 1 au
        cn_cutoff: float = 1600,  # au^2, 0.52917726 angstrom = 1 au
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if not torch.cuda.is_available():
            raise NotImplementedError('CPU + D3 is not implemented yet')

        self.rthr = vdw_cutoff
        self.cnthr = cn_cutoff
        self.damp_name = damping_type.lower()
        self.func_name = functional_name.lower()

        if self.damp_name not in ['damp_bj', 'damp_zero']:
            raise ValueError('Error: Invalid damping type.')

        self._lib = _load('pair_d3')

        self._lib.pair_init.restype = ctypes.POINTER(PairD3)
        self.pair = self._lib.pair_init()

        self._lib.pair_set_atom.argtypes = [
            ctypes.POINTER(PairD3),  # PairD3* pair
            ctypes.c_int,  # int natoms
            ctypes.c_int,  # int ntypes
            ctypes.POINTER(ctypes.c_int),  # int* types
            ctypes.POINTER(ctypes.c_double),  # double* x
        ]
        self._lib.pair_set_atom.restype = None

        self._lib.pair_set_domain.argtypes = [
            ctypes.POINTER(PairD3),  # PairD3* pair
            ctypes.c_int,  # int xperiodic
            ctypes.c_int,  # int yperiodic
            ctypes.c_int,  # int zperiodic
            ctypes.POINTER(ctypes.c_double),  # double* boxlo
            ctypes.POINTER(ctypes.c_double),  # double* boxhi
            ctypes.c_double,  # double xy
            ctypes.c_double,  # double xz
            ctypes.c_double,  # double yz
        ]
        self._lib.pair_set_domain.restype = None

        self._lib.pair_run_settings.argtypes = [
            ctypes.POINTER(PairD3),  # PairD3* pair
            ctypes.c_double,  # double rthr
            ctypes.c_double,  # double cnthr
            ctypes.c_char_p,  # const char* damp_name
            ctypes.c_char_p,  # const char* func_name
        ]
        self._lib.pair_run_settings.restype = None

        self._lib.pair_run_coeff.argtypes = [
            ctypes.POINTER(PairD3),  # PairD3* pair
            ctypes.POINTER(ctypes.c_int),  # int* atomic_numbers
        ]
        self._lib.pair_run_coeff.restype = None

        self._lib.pair_run_compute.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_run_compute.restype = None

        self._lib.pair_get_energy.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_get_energy.restype = ctypes.c_double

        self._lib.pair_get_force.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_get_force.restype = ctypes.POINTER(ctypes.c_double)

        self._lib.pair_get_stress.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_get_stress.restype = ctypes.POINTER(ctypes.c_double * 6)

        self._lib.pair_fin.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_fin.restype = None

    def _idx_to_numbers(self, Z_of_atoms):
        unique_numbers = list(dict.fromkeys(Z_of_atoms))
        return unique_numbers

    def _idx_to_types(self, Z_of_atoms):
        unique_numbers = list(dict.fromkeys(Z_of_atoms))
        mapping = {num: idx + 1 for idx, num in enumerate(unique_numbers)}
        atom_types = [mapping[num] for num in Z_of_atoms]
        return atom_types

    def _convert_domain_ase2lammps(self, cell):
        qtrans, ltrans = np.linalg.qr(cell.T, mode='complete')
        lammps_cell = ltrans.T
        signs = np.sign(np.diag(lammps_cell))
        lammps_cell = lammps_cell * signs
        qtrans = qtrans * signs
        lammps_cell = lammps_cell[(0, 1, 2, 1, 2, 2), (0, 1, 2, 0, 0, 1)]
        rotator = qtrans.T
        return lammps_cell, rotator

    def _stress2tensor(self, stress):
        tensor = np.array(
            [
                [stress[0], stress[3], stress[4]],
                [stress[3], stress[1], stress[5]],
                [stress[4], stress[5], stress[2]],
            ]
        )
        return tensor

    def _tensor2stress(self, tensor):
        stress = -np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[1, 2],
                tensor[0, 2],
                tensor[0, 1],
            ]
        )
        return stress

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            raise ValueError('No atoms to evaluate')

        if atoms.get_cell().sum() == 0:
            print(
                'Warning: D3Calculator requires a cell.\n'
                'Warning: An orthogonal cell large enough is generated.'
            )
            positions = atoms.get_positions()
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            max_cutoff = np.sqrt(max(self.rthr, self.cnthr)) * 0.52917726

            cell_lengths = max_pos - min_pos + max_cutoff + 1.0  # extra margin
            cell = np.eye(3) * cell_lengths

            atoms.set_cell(cell)
            atoms.set_pbc([True, True, True])  # for minus positions

        cell, rotator = self._convert_domain_ase2lammps(atoms.get_cell())

        Z_of_atoms = atoms.get_atomic_numbers()
        natoms = len(atoms)
        ntypes = len(set(Z_of_atoms))
        types = (ctypes.c_int * natoms)(*self._idx_to_types(Z_of_atoms))

        positions = atoms.get_positions() @ rotator.T
        x_flat = (ctypes.c_double * (natoms * 3))(*positions.flatten())

        atomic_numbers = (ctypes.c_int * ntypes)(*self._idx_to_numbers(Z_of_atoms))

        boxlo = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
        boxhi = (ctypes.c_double * 3)(cell[0], cell[1], cell[2])
        xy = cell[3]
        xz = cell[4]
        yz = cell[5]
        xperiodic, yperiodic, zperiodic = atoms.get_pbc()

        lib = self._lib
        assert lib is not None
        lib.pair_set_atom(self.pair, natoms, ntypes, types, x_flat)

        xperiodic = xperiodic.astype(int)
        yperiodic = yperiodic.astype(int)
        zperiodic = zperiodic.astype(int)
        lib.pair_set_domain(
            self.pair, xperiodic, yperiodic, zperiodic, boxlo, boxhi, xy, xz, yz
        )

        lib.pair_run_settings(
            self.pair,
            self.rthr,
            self.cnthr,
            self.damp_name.encode('utf-8'),
            self.func_name.encode('utf-8'),
        )

        lib.pair_run_coeff(self.pair, atomic_numbers)
        lib.pair_run_compute(self.pair)

        result_E = lib.pair_get_energy(self.pair)

        result_F_ptr = lib.pair_get_force(self.pair)
        result_F_size = natoms * 3
        result_F = np.ctypeslib.as_array(
            result_F_ptr, shape=(result_F_size,)
        ).reshape((natoms, 3))
        result_F = np.array(result_F)
        result_F = result_F @ rotator

        result_S = lib.pair_get_stress(self.pair)
        result_S = np.array(result_S.contents)
        result_S = (
            self._tensor2stress(rotator.T @ self._stress2tensor(result_S) @ rotator)
            / atoms.get_volume()
        )

        self.results = {
            'free_energy': result_E,
            'energy': result_E,
            'forces': result_F,
            'stress': result_S,
        }

    def __del__(self):
        if self._lib is not None:
            self._lib.pair_fin(self.pair)
            self._lib = None
            self.pair = None
