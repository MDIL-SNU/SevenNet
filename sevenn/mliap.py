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
            use_cuda_graph: bool = False  # CUDA Graph replay on one rank
            use_radial_table: bool = False  # tabulate or fold radial MLPs
            use_fused: bool = False  # fused LunaTP kernels for replay
            fused_knots: int = 2048  # 0 evaluates weights by distinct length
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
        self.use_cuda_graph = kwargs.get('use_cuda_graph', False)
        self.use_radial_table = kwargs.get('use_radial_table', False)
        self.use_fused = kwargs.get('use_fused', False)
        self.fused_knots = int(kwargs.get('fused_knots', 2048))
        self.fused_r_min = float(kwargs.get('fused_r_min', 0.25))
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
        """
        SEVENN_CUDA_MEMORY_FRACTION=f: this process may allocate at most that
        fraction of the GPU memory through torch (an out-of-memory error
        instead of exhausting a shared GPU / unified memory).
        SEVENN_MLIAP_MEMORY_REPORT=1: peak torch GPU memory printed at exit.
        """
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

    def _build_model(self, ghost_exchange: bool):
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
        if (
            not ghost_exchange
            and self.use_lunatp
            and (
                getattr(self, 'use_fused', False)
                or os.getenv('SEVENN_MLIAP_FUSED') == '1'
            )
            and os.getenv('SEVENN_MLIAP_FUSED') != '0'
        ):
            from sevenn.nn.inference_specialization import fold_modal
            from sevenn.nn.lunatp_fused import fuse_radial_conv

            model.requires_grad_(False)
            knots = int(
                os.getenv(
                    'SEVENN_MLIAP_FUSED_KNOTS', getattr(self, 'fused_knots', 2048)
                )
            )
            r_min = float(getattr(self, 'fused_r_min', 0.25))
            names = fuse_radial_conv(model, knots=knots, r_min=r_min)
            how = f' (weight table, {knots} knots from {r_min} A)' if knots else ''
            print(f'[INFO] fused radial + conv{how}: {" ".join(names)}', flush=True)
            if self.modal is not None and model.modal_map:
                names = fold_modal(model, self.modal)
                print(f'[INFO] modal folded: {" ".join(names)}', flush=True)
            if knots and os.getenv('SEVENN_MLIAP_FUSED_FIRST', '1') != '0':
                from sevenn.nn.lunatp_fused import collapse_first_layer

                reference = self.cp.build_model()
                reference.delete_module_by_key('force_output')
                names = collapse_first_layer(model, reference, self.modal)
                if names:
                    print(
                        f'[INFO] first layer as radial tables: {" ".join(names)}',
                        flush=True,
                    )
            model.prune_species = True  # by MLIAPCudaGraph, to the species seen
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
            if getattr(self, '_graph_single_rank', None) is None:
                # An extra owner exchange on some ranks would leave MPI peers
                # waiting in the model's feature exchange.
                self._graph_single_rank = all(
                    int(os.getenv(key, '1')) == 1
                    for key in (
                        'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'MV2_COMM_WORLD_SIZE'
                    )
                )
                if not self._graph_single_rank:
                    print(
                        '[INFO] SevenNet ML-IAP CUDA graph requires one rank',
                        flush=True,
                    )
            if self._graph_single_rank:
                if getattr(self, 'graph', None) is None:
                    self.graph = None
                    self._ensure_graph_initialized()
                if self.graph.compute(lmp_data):
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
        # upcasting edge_forces required for update_pair_forces_gpu
        lmp_data.update_pair_forces_gpu(edge_forces.to(torch.float64))

    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
