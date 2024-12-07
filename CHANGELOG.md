# Changelog
All notable changes to this project will be documented in this file.

## [0.10.2]
### Added
- Accelerated graph build routine if matscipy is installed  @hexagonerose
### Changed
- matscipy is included as dependency

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
