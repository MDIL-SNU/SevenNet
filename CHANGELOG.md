# Changelog
All notable changes to this project will be documented in this file.

## [0.13.1.dev]
### Added
- Training features used for SevenNet-Omni: batch training, `OrderedSampler`, `grad_clip`, `onecyclelr`
- Loss: MAE, L2MAE
- Dataset type: `aselmdb`, `custom`
- SevenNet-Nano checkpoints for `7net-nano-4.5`, `7net-nano-5.0`, `7net-nano-5.5`, and `7net-nano-6.0`
- Support LunaTP (`enable_lunatp`). LAMMPS through ML-IAP only.
- LAMMPS ML-IAP, one rank: `sevenn get_model --use_mliap --cuda_graph` (or `SEVENN_MLIAP_CUDA_GRAPH=1` with an existing potential file) replays a captured CUDA graph of the model call (LunaTP and e3nn backends). One capture serves an MD run: ghosts are folded to their owners (the plain serial model is called, no ghost exchange), the pair list is padded to a fixed number of edges with self-loops on fake atoms, and a changed pair list (LunaTP graph tables, with source segments padded by empty ones) is copied in. `SEVENN_MLIAP_FOLD_EAGER=1` runs the same folded model without a capture. Several ranks: turns itself off (eager ghost-exchange model)
- LAMMPS ML-IAP with LunaTP, one rank: `sevenn get_model --use_mliap --enable_lunatp --cuda_graph --fused` (or `SEVENN_MLIAP_FUSED=1` with an existing potential file) uses frozen-model kernels (`sevenn/nn/lunatp_fused.py`): the radial weights of a convolution are computed once per distinct edge length (the edges i-j and j-i of a full neighbor list share a row) and LunaTP's generated convolution kernels, patched as text, read them by row index; the backward kernel contracts the weight gradient with dw/dr (same GEMM, forward mode) so that the weight gradient array is never written and the radial MLP has no backward; the hidden layers of all radial MLPs come from one cubic table kernel; the species self connection is pruned to the species of the atoms (again, with a new capture, if another species appears); the modal one-hot input of the linear layers is a constant bias; the kernel blocks are launched in atom index order (with LAMMPS atom sorting, rows shared by neighboring atoms are reused from the 24 MB GPU cache; chosen by the coherence of the atom order). Same forces to fp32 round-off (6e-6 eV/A). 7net-omni, Si diamond, ms/step: 64 atoms 6.70 -> 3.73, 512 atoms 34.3 -> 18.0, 4096 atoms 304 -> 162 (from `--cuda_graph --radial_table`; original ML-IAP path 34.5 / 77.3 / 471). `SEVENN_MLIAP_FUSED_W16=1` (experimental, not the same function to round-off: 7e-4 eV/A) stores the weight rows as fp16: 3.37 / 13.1 / 126 ms
- `--fused`: the radial weights are cubic Hermite interpolated inside the LunaTP convolution kernels from a table in the edge length (`--fused_knots`, default 2048; `SEVENN_MLIAP_FUSED_KNOTS`), instead of a per-call weight array: no radial GEMM, and the kernels read the table mostly from the GPU cache. 7net-omni in LAMMPS, 4096 atoms: 162 -> 76 ms/step; forces within 5e-6 eV/A of the unmodified model. `--fused_knots 0` keeps the weight row per distinct edge length.
- `--fused`: tensor-product paths whose output irreps the following linear has no weights for are not computed (7net-omni: 34 of the 68 paths of the last-but-one convolution, half its per-edge weights, produce 0o/1e/2o/3e that the last layer cannot use; exact). The edge length and spherical harmonics come from one kernel (and one backward kernel) instead of ~180 elementwise kernels. Small graphs use shorter LunaTP source segments. 7net-omni in LAMMPS: 2.43 / 8.92 / 67.4 ms/step at 64 / 512 / 4096 atoms.
- `--fused`: the first interaction (self-interaction linear, convolution of the species embedding, linear) is one sum over neighbors of per-species radial tables times the spherical harmonics (`FusedFirstLayer`; exact, `SEVENN_MLIAP_FUSED_FIRST=0` turns it off). 7net-omni in LAMMPS: 2.41 / 8.75 / 66.3 ms/step at 64 / 512 / 4096 atoms.
- ML-IAP CUDA graph: the bookkeeping before each replay (ghost owners, padded edge index, change flags) is one kernel (`SEVENN_MLIAP_GRAPH_PREP=0`: the torch operations); graphs below 4096 edges run the fused backward with shorter source segments and a stream per group kernel. 7net-omni in LAMMPS: 2.29 / 8.44 / 64.7 ms/step at 64 / 512 / 4096 atoms (original ML-IAP path: 34.5 / 77.3 / 471).
- `sevenn.nn.radial_table.tabulate_radial`: the radial MLPs of an original SevenNet model (weights = a function of the edge length only) as cubic Hermite tables in r, one CUDA kernel per pass (cupy); `sevenn get_model --use_mliap --radial_table` or `SEVENN_MLIAP_RADIAL_TABLE=1`. Forces within fp32 round-off of the MLP (4e-6 eV/A)
- `cuda_graph` option of `SevenNetCalculator`: the model call is captured once and replayed for calls with the same atoms; a changed edge index is copied in and the edges are padded with self-loops at r_c (checked to change nothing), so one capture serves an MD run; a thrash guard stops captures that do not pay back

### Changed
- The default dtype of the rescale layer is changed to double precision. Use the `SEVENN_SHIFT_SCALE_DTYPE` environment variable for backward single-precision compatibility.
- The recommended LAMMPS ML-IAP version is bumped to `stable_22Jul2025_update4`.
- The model call makes no host synchronization (type map buffer, modal index fill, ModalWiseRescale index, force output atom count and volume clamp); same values
- `SevenNetCalculator` copies its inputs with one pinned host-to-device copy per dtype and its outputs with one device-to-host copy; same values
- `SevenNetCalculator` freezes the weights of a checkpoint model (`requires_grad_(False)`): the force backward skips dE/dW
- LunaTP: the LunaTP graph is cached on the input edge index; eager calculator calls put the edges in one order and reuse the previous call's device edge index when the topology is unchanged (no graph build); the model receives a plain dict
- Fewer kernels per model call with frozen weights (the calculator's model): the scaled weight matrices of the pair radial MLP and the edge gate are computed once, their input columns are taken by one split (one backward cat instead of zero-filled slice gradients), and in eval the gated modal one-hot is computed once per forward (same values)
- Pair radial MLP in eval: the normalize2mom constant of the activation is folded into the frozen weights of the next matmul (same function, fp32 rounding; model call deviation 3.7e-6 eV energy, 3.2e-6 eV/A force at most); `tanh(theta)` of the edge gate is computed once for frozen weights
- LunaTP: LunaEquivariance is pinned to GitHub commit a045a88 (0.2.0 with the CUDA 13 fix; the PyPI 0.2.0 wheel is the earlier release commit, whose graph kernel does not compile with CUDA 13)
- `sevenn lunatp_precompile` builds the forward-only kernel of a linear whose input and weights do not require grad (the calculator's atom-type embedding), which the calculator otherwise compiled at its first call: after a differentiable call, LunaEquivariance serves a no-grad linear call from the fused forward+backward kernel, so the precompile runs the no-grad combination first. `--train` also runs the linears on 65 rows (LunaEquivariance selects the second-derivative kernel of a small linear by the number of rows: 0.1.0 always, a045a88 in fp64). The convolution forward kernels, which a045a88 compiles when a convolution is built, are built in parallel (a share per worker). A failed precompile worker (CUDA out of memory on a shared GPU) runs once more.

### Fixed
- `D3Calculator()` segfault bug when reusing the calculator within different sized `Atoms`.
- LunaEquivariance 0.2.0 kernel libraries are named by their build digest (`<digest>/<name>_v<digest as integer>.so`) instead of torch's per-process extension version, which counts the libraries of the same name loaded before, so a process that loads the kernels in another order (a `sevenn lunatp_precompile` worker, a calculator after the precompile) no longer compiles an identical kernel again; libraries in a cache written by 0.2.0 without this change are compiled once more.


## [0.13.0]
### Added
- reEWC fine-tuning with forgetting prevention for single-modal models: optional experience replay (`rehearsal`, `load_memory_path`, `mem_batch_size`, `mem_ratio`) and an Elastic Weight Consolidation penalty from a precomputed Fisher matrix (`continue.fisher_information`, `continue.opt_params`, `continue.ewc_lambda`), plus a `cosineannealingwarmuplr` scheduler
- Support OpenEquivariance
- Per-atom stress (atomic virial) support in LAMMPS pair_e3gnn and ASE calculator
- `compute_atomic_virial` option in `SevenNetCalculator`
- Add Batch D3 and SevenNetD3Model in Torch-Sim interface

### Changed
- **[Breaking]** Rename optional dependency group `mliap` into `mliap12` (reflecting its CUDA 12.x dependency).
- Add `cueq13` and `mliap13` optional dependency groups for CUDA 13.x.
- **[Breaking]** `torchscript` file_type is no longer supported in `SevenNetCalculator`.
- LAMMPS pair_e3gnn refactored to use pair-wise force (dE/dr) instead of position-based gradient.
- Deploy no longer replaces force_output with ForceStressOutput; force/stress computed in LAMMPS C++ side.
- **[Breaking]** Accelerator None options (use the same setting as checkpoint) are removed and default is changed to False.

### Fixed
- Omni (flash TP trained checkpoint) loads w/o an error for flashTP disabled environment.
- Load FlashTP-saved checkpoints (e.g. SevenNet-Nano) when FlashTP is unavailable by falling back to the e3nn backend, so they work for inference and fine-tuning without FlashTP installed. An explicit `enable_flash=True` still fails loud.

## [0.12.1]
### Fixed
- FlashTP with LAMMPS parallel in torch
- Single-atom inference failure with ASE+flashTP, LAMMPS-Torch and LAMMPS ML-IAP
- CI/CD bugs
- inconsistent argument namings of flashTP

### Changed
- ninja dependency added for FlashTP compile
- all CLI argument name of enabling flashTP is now 'enable_flash', not 'enable_flashTP'

### Added
- TorchSim interface and docs
- SevenNet-Omni-i8, SevenNet-Omni-i12

## [0.12.0]
### Added
- Documentation moved to RTD
- LAMMPS-MLIAP integration with GhostExchangeOp
- SevenNet-Omni
- Example config for fine-tuning the SevenNet-MF-ompa model
- FlashTP support (https://github.com/SNU-ARC/flashTP)

### Changed
- ninja dependency added

### Fixed
- Multi modal model finetuning: Pass modal_map into config during multi-fidelity continual training #232
- omat deploy parallel fixed


## [0.11.2]
### Changed
- change cueq default config
- assign cell volume 0 instead of eps for non-pbc system
- remove redundant CELL and CELL_SHIFT

### Fixed
- Circular import in error_recorder


## [0.11.1]
From here, the version of 'main' branch has 'devX' after it diverges from the latest stable version

CLI interface changed in backward-compatible manner. Now `sevenn` has subcommands for
inference, train, etc

### Added
- subcommand with some aliases
- strict e3nn version requirement from __init__.py

### Changed
- pre-commit uses python3.11
- cuequivaraiance optional libraries
- some gitignores

### Fixed
- Circular import in sevenn.checkpoint (dev0)
- Fix typing issues
- Added missing typings (especially return type)


## [0.11.0]

Multi-fidelity learning implemented & New pretrained-models

### Added
- Build multi-fidelity model, SevenNet-MF, based on given modality in the yaml
- Modality support for sevenn_inference, sevenn_get_modal, and SevenNetCalculator
- sevenn_cp tool for checkpoint summary, input generation, multi-modal routines
- Modality append / assign using sevenn_cp
- Loss weighting for energy, force and stress for corresponding data label
- Ignore unlabelled data when calculating loss. (e.g. stress data for non-pbc structure)
- Dict style dataset input for multi-modal and data-weight
- (experimental) cuEquivariance support
- Downloading large checkpoints from url (7net-MF-ompa, 7net-omat)
- D3 wB97M param

### Changed
- Sort instructions of tensor product in convolution (+ fix flipped w3j coeff of old model)
- Lazy initialization for `IrrepsLinear` and `SelfConnection*`
- Checkpoint things using `sevenn/checkpoint.py`
- e3nn >= 0.5.0, to ensure changed CG coeff later on
- pandas as dependency
- old v1 presets are removed, liquid electrolyte fine-tune yaml is added

### Fixed
- More refactor for shift scale things + few bug fixes
- Correctly shuffle training set when distributed training is enabled
- D3 calculator system swap memory error fixed
- D3 compile uses $HOME/.cache if package directory is not writable


## [0.10.4]
### Added
- feats: D3 calculator
### Fixed
- bug: info dict sharing (therefore energy stress) when structure_list used
- torch >= 2.5.0 works
- numpy >= 2.0 works (need more testing)
### Changed
- sevennet_calculator.py => calculator
- fine tunine preset to use original loss function (Huber) and loss weights


## [0.10.3]
### Added
- SevenNet-l3i5, checkpoint, preset. (keywords: 7net-l3i5, sevennet-l3i5)
- SevenNet-l3i5 test
### Changed
- Now --help do not load unnecessary imports (fast!)
- README


## [0.10.2]
### Added
- Accelerated graph build routine if matscipy is installed  @hexagonerose
- matscipy vs. ase neighborlist unit test
- If valid set is not given but data_divide_ratio is given, validaset is created using random split. (shift, scale, and conv_denoiminator uses original whole statistics)
### Changed
- matscipy is included as dependency
- data_divide_ration defaults to 0.0 (not used)
### Fixed
- For torch version >= 2.4.0, Loading graph dataset no more raises warnings.
- Raise error when unknown element is found (SevenNetCalculator)

## [0.10.1]
### Added
- experimental `SevenNetAtomsDataset` which is memory efficient, can be enabled with `dataset_type='atoms'`
- Save meta data & statistics when the `SevenNetGraphDataset` saves its data.
### Changed
- Save checkpoint_0.pth (model before any training)
- `SevenNetGraphDataset._file_to_graph_list` -> `SevenNetGraphDataset.file_to_graph_list`
- Refactoring `SevenNetGraphDataset`, skips computing statistics if it is loaded, more detailed logging
- Prefer use .get when accessing config dict
### Fixed
- Fix error when loading `SevenNetGraphDataset` with other types of data (ex: extxyz) in one dataset


## [0.10.0]
SevenNet now have CI workflows using pytest and its coverage is 78%!
Substantial changes in cli apps and some outputs.

### Added
- [train_v2]: train_v2, with lots of refactoring + support `load_testset_path`. Original routine is accessible: `sevenn -m train_v1`.
- [train_v2]: `SevenNetGraphDataset` replaces old `AtomGrpahDataset`, which extends `InMemoryDataset` of PyG.
- [train_v2]: `sevenn_graph_build` for SevenNetGraphDataset. Previous .sevenn_data is accessible with --legacy option
- [train_v2]: Any number of additional datasets will be evaluated and recorded if it is given as 'load_{NAME}set_path' key (input.yaml).
- 'Univ' keyword for 'chemical_species'
- energy_key, force_key, stress_key options for `sevenn_graph_build`, @thangckt
- OpenMPI distributed training @thangckt
### Changed
- Read EFS of atoms from y_* keys of .info or .arrays dict, instead of caclculator results
- Now `type_map` and requires_grad is hidden inside `AtomGraphSequential`, and don't need to care about it.
- `log.sevenn` and `lc.csv` automatically find a safe filename (log0.sevenn, log1.sevenn, ...) to avoid overwriting.
- [train_v2]: train_v2 loads its training set via `load_trainset_path`, rather than previous `load_dataset_path`.
- [train_v2]: log.csv -> lc.csv, and columns have no units, (easier to postprocess with it) but still on `log.sevenn`.
### Fixed
- [e3gnn_serial]: can continue simulation even when atom tag becomes not consecutive (removing atom dynamically), @gasplant64
- [e3gnn_parallel]: undefined behavior when there is no atoms to send/recv (for non pbc system)
- [e3gnn_parallel]: incorrect force/stress in some edge cases (too small simulation cell & 2 process)
- [e3gnn_parallel]: revert commit 14851ef, now e3gnn_parallel is sane.
- [e3gnn_*]: += instead of = when saving virial stress and forces @gasplant64
- Now Logger correctly closes a file.
- ... and lots of small bugs I found during writing `pytest`.

## [0.9.5]
### Note
This version is not stable, but I tag it as v0.9.5 before making further changes.
LAMMPS `pair_e3gnn_parallel.*` should be re-compiled for the below changes regarding LAMMPS parallel.
This is the first changelog and may not reflect all the changes.
### Added
- Stress compute for LAMMPS sevennet parallel
- `sevenn_inference` now takes .extxyz input
- `sevenn_inference` gives MAE error
- Experimental `sevenn_inference` on the fly graph build option
### Changed
- **[Breaking]** Parallel LAMMPS model changed, old deployed parallel models will not work
- **[Breaking]** Parallel LAMMPS takes the directory of potentials as input. Accordingly, `sevenn_get_model -p` creates a folder with potentials.
- **[Breaking]** Except for serial LAMMPS models, force and stress are computed from gradients of edge vectors, not positions.
- Separate interaction block from model build
- Add typing for most of functions
- Remove clang pre-commit hook as it breaks lammps pair files
- `torch.load` with `weights_only=False`
- Line length limit 80 -> 85
- Refactor
### Fixed
- Correct batch size for SevenNet-0(11July2024)

## [0.9.4] - 2024-08-26
### Added
- D3 correction (contributed from dambi3613) for LAMMPS serial
