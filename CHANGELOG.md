# Changelog
All notable changes to this project will be documented in this file.

## [0.12.1]
### Fixed
- FlashTP with LAMMPS parallel in torch
- Single-atom inference failure with ASE+flashTP, LAMMPS-Torch and LAMMPS ML-IAP
- CI/CD bugs
- inconsistent argument namings of flashTP

### Changed
- ninja dependency added
- all CLI argument name of enabling flashTP is now 'enable_flash', not 'enable_flashTP'

### Added
- TorchSim interface
- Support OpenEquivariance

## [0.12.0]
### Added
- Documentation moved to RTD
- LAMMPS-MLIAP integration with GhostExchangeOp

### Added
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
