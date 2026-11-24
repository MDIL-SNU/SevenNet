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
from sevenn._const import (
    DEFAULT_RADIAL_TABLE_KNOTS,
    DEFAULT_RADIAL_TABLE_R_MIN,
    AtomGraphDataType,
)
from sevenn.nn._ghost_exchange import MLIAPGhostExchangeModule
from sevenn.nn.radial_table import supports_radial_table, tabulate_radial
from sevenn.util import load_checkpoint, pretrained_name_to_path


def _runtime_flag(value, env, default=False):
    override = os.getenv(env)
    if override in ('0', '1'):
        return override == '1'
    return default if value is None else value


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
        """Load a checkpoint for ML-IAP inference.

        On one rank, compatible models use radial interpolation. CUDA also
        enables replay and, for LunaTP, fused kernels by default.
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
        self.cp = load_checkpoint(checkpoint_path).compact_for_inference()
        print(f'[INFO] Loaded checkpoint from {checkpoint_path}', flush=True)
        self.model = None  # lazy init

        # calc_kwargs
        self.use_cueq = kwargs.get('use_cueq', False)
        self.use_flash = kwargs.get('use_flash', False)
        self.use_oeq = kwargs.get('use_oeq', False)
        self.use_lunatp = kwargs.get('use_lunatp', False)
        self.modal = kwargs.get('modal', None)
        self.use_cuda_graph = kwargs.get('use_cuda_graph')
        self.use_radial_table = kwargs.get('use_radial_table')
        self.use_fused = kwargs.get('use_fused')
        self.radial_knots = kwargs.get('radial_knots')
        if self.radial_knots is None:
            self.radial_knots = DEFAULT_RADIAL_TABLE_KNOTS
        self.radial_r_min = kwargs.get('radial_r_min')
        if self.radial_r_min is None:
            self.radial_r_min = DEFAULT_RADIAL_TABLE_R_MIN
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

    def _memory_options(self) -> None:
        """Set the per-process Torch memory limit and optional peak report."""
        if self.device.type != 'cuda' or getattr(self, '_memory_set', False):
            return
        self._memory_set = True
        fraction = os.getenv('SEVENN_CUDA_MEMORY_FRACTION')
        if fraction:
            torch.cuda.set_per_process_memory_fraction(float(fraction))
        if os.getenv('SEVENN_MLIAP_MEMORY_REPORT') == '1':
            import atexit

            def report():
                gb = torch.cuda.max_memory_allocated() / 2**30
                res = torch.cuda.max_memory_reserved() / 2**30
                print(
                    f'[SevenNet] peak GPU memory: {gb:.2f} GiB allocated, '
                    f'{res:.2f} GiB reserved', flush=True,
                )

            atexit.register(report)

    def _build_model(self, ghost_exchange: bool, fused: bool = False):
        self._memory_options()
        model = self.cp.build_model(
            enable_cueq=self.use_cueq,
            enable_flash=self.use_flash,
            enable_oeq=self.use_oeq,
            enable_lunatp=self.use_lunatp,
        )

        if ghost_exchange:
            for name, module in model._modules.items():
                if name.endswith('_convolution'):
                    model._modules[name] = MLIAPWrappedConvolution(module)
                elif name == 'onehot_to_feature_x':
                    model._modules[name] = MLIAPWrappedIrrepsLinear(module)

        model.set_is_batch_data(False)
        if self.modal is not None and ghost_exchange:
            model.eval_modal_map = False
            model.prepare_modal_deploy(self.modal)  # set the fidelity of input data

        model.delete_module_by_key('force_output')  # to autograd edge forces
        model.to(self.device)
        model.eval()
        radial_table = getattr(self, 'use_radial_table', False)
        knots = getattr(self, 'radial_knots', None)
        r_min = getattr(self, 'radial_r_min', DEFAULT_RADIAL_TABLE_R_MIN)
        if knots is None:  # ML-IAP potentials saved before the options split
            knots = int(
                os.getenv(
                    'SEVENN_MLIAP_FUSED_KNOTS',
                    getattr(self, 'fused_knots', DEFAULT_RADIAL_TABLE_KNOTS),
                )
            )
            r_min = float(getattr(self, 'fused_r_min', DEFAULT_RADIAL_TABLE_R_MIN))
            radial_table = radial_table or bool(fused and knots)
            knots = knots or DEFAULT_RADIAL_TABLE_KNOTS
        radial_table = _runtime_flag(
            radial_table, 'SEVENN_MLIAP_RADIAL_TABLE',
            default=self._single_rank and supports_radial_table(model),
        )
        if radial_table and knots < 2:
            raise ValueError('radial_knots must be at least 2')
        names = []
        if fused:
            from sevenn.nn.lunatp_fused import prepare_fused_inference

            names = prepare_fused_inference(
                model, self.cp.build_model, self.modal,
                knots if radial_table else 0, r_min,
                first=os.getenv('SEVENN_MLIAP_FUSED_FIRST', '1') != '0',
            )
            model.prune_species = bool(names)
            if names:
                print(f'[INFO] fused LunaTP: {" ".join(names)}', flush=True)
        if radial_table:
            tabulate_radial(
                model, knots, r_min,
                kernel=self.device.type == 'cuda', exclude=names,
            )
            print(f'[INFO] radial table: {knots} knots from {r_min} A', flush=True)
        return model

    def _ensure_model_initialized(self):
        if self.model is not None:
            return  # Already initialized
        print('[INFO] Lazy initializing SevenNet model...', flush=True)
        print(f'[INFO] cueq={self.use_cueq}, flashTP={self.use_flash}, oeq={self.use_oeq}, lunatp={self.use_lunatp}', flush=True)  # noqa: E501
        self.model = self._build_model(ghost_exchange=True)

    def _build_serial_runtime(self, replay: bool, fused: bool):
        from sevenn.nn._mliap_cuda_graph import MLIAPCudaGraph

        model = self._build_model(ghost_exchange=False, fused=fused)
        model.requires_grad_(False)
        runtime = MLIAPCudaGraph(
            model, self.device, self.cutoff, modal=self.modal, replay=replay,
        )
        mode = 'eager' if runtime.eager_fold else 'CUDA Graph replay'
        print(f'[INFO] SevenNet ML-IAP {mode} on one rank', flush=True)
        return runtime

    @staticmethod
    def _is_single_rank() -> bool:
        # Every rank must take the same ghost exchange path.
        return all(
            os.getenv(key, '1') == '1'
            for key in (
                'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE',
                'PMIX_UNIV_SIZE', 'SLURM_STEP_NUM_TASKS',
            )
        )

    def compute_forces(self, lmp_data):
        if not hasattr(self, '_cuda_single_rank'):
            elems = lmp_data.elems
            if elems is None:
                return
            self.device = torch.as_tensor(elems).device
            self._single_rank = self._is_single_rank()
            self._cuda_single_rank = self._single_rank and self.device.type == 'cuda'
            replay = _runtime_flag(
                getattr(self, 'use_cuda_graph', False), 'SEVENN_MLIAP_CUDA_GRAPH',
                default=True,
            )
            fused = self.use_lunatp and _runtime_flag(
                getattr(self, 'use_fused', False), 'SEVENN_MLIAP_FUSED',
                default=True,
            )
            if self._cuda_single_rank and (replay or fused):
                self.graph = self._build_serial_runtime(replay, fused)
        if (runtime := getattr(self, 'graph', None)) is not None:
            if runtime.compute(lmp_data):
                return
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
        edge_forces = edge_forces.to(torch.float64)
        if self.device.type == 'cuda':
            lmp_data.update_pair_forces_gpu(edge_forces)
        else:
            lmp_data.update_pair_forces(edge_forces.detach().numpy())

    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
