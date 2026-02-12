# Command line interface

SevenNet provides five commands for preprocessing, training, and deployment: `sevenn preset`, `sevenn graph_build`, `sevenn`, `sevenn inference`, and `sevenn get_model`.

(sevenn-preset)=
## `sevenn preset`

With the `sevenn preset` command, the input file setting the training parameters is generated automatically.
```bash
sevenn preset {preset keyword} > input.yaml
```

Available preset keywords are: `base`, `fine_tune`, `multi_modal`, `sevennet-0`, and `sevennet-l3i5`.
Check comments in the preset YAML files for explanations. For fine-tuning, be aware that most model hyperparameters cannot be modified unless explicitly indicated.
To reuse a preprocessed training set, you can specify `sevenn_data/${dataset_name}.pt` for the `load_trainset_path:` in the `input.yaml`.

(sevenn-graph-build)=
## `sevenn graph_build`

To obtain the preprocessed data, `sevenn_data/graph.pt`, `sevenn graph_build` command can be used.
The output files can be used for training (`sevenn`) or inference (`sevenn inference`) to skip the graph build stage.

```bash
sevenn graph_build {dataset path} {cutoff radius}
```

The output `sevenn_data/graph.yaml` contains statistics and meta information about the dataset.
These files must be located in the `sevenn_data` directory. If you move the dataset, move the entire `sevenn_data` directory without changing the contents.

See `sevenn graph_build --help` for more information.

(sevenn-train)=
## `sevenn train`

Given that `input.yaml` and `sevenn_data/graph.pt` are prepared, SevenNet can be trained by the following command:

```bash
sevenn train input.yaml -s
```

We support multi-GPU training using PyTorch DDP (distributed data parallel) with a single process (or a CPU core) per GPU.

```bash
torchrun --standalone --nnodes {number of nodes} --nproc_per_node {number of GPUs} --no_python sevenn input.yaml -d
```

Enable cuEquivariance or flashTP while training by adding the --enable_cueq or --enable_flash.

```bash
sevenn train input.yaml -s --enable_cueq # or --enable_flash
```

Please note that `batch_size` in `input.yaml` refers to the per-GPU batch size.

(sevenn-inference)=
## `sevenn inference`

Using the checkpoint after training, the properties such as energy, force, and stress can be inferred directly.

```bash
sevenn inference checkpoint_best.pth path_to_my_structures/*
```

This will create the `inference_results` directory, where CSV files contain predicted energy, force, stress, and their references (if available).
See `sevenn inference --help` for more information.


(sevenn-get-model)=
## `sevenn get_model`

The command is for LAMMPS integration of SevenNet. It deploys a model into LAMMPS-readable file(s).

See {doc}`lammps_torch` or {doc}`lammps_mliap` for installation.

See {doc}`accelerator` for installation of accelerators.

### Basic usage
```bash
sevenn get_model \
    {pretrained_name or checkpoint_path} \
    {--use_mliap} \  # For LAMMPS ML-IAP use.
    {--enable_flash | --enable_cueq} \  # For accelerators.
    {--modal {task_name}} \  # Required when using multi-fidelity model
    {--get_parallel}  # For parallel MD simulations
```

If `--use_mliap` is not set (TorchScript version), it will create `deployed_serial.pt` or a directory containing several `deployed_parallel_*.pt` files for parallel execution.
They can be used as a LAMMPS potential with the `e3gnn` or `e3gnn/parallel` pair_style in LAMMPS.
In this case, only `--enable_flash` is available.
Check {doc}`lammps_torch` for installation and lammps script in this use case.

If `--use_mliap` is set, it will create `deployed_serial_mliap.pt`.
The file can be used with the `mliap` pair_style in LAMMPS.
In this case, both `--enable_cueq` and `--enable_flash` are available, but you cannot specify both at the same time.
Parallel execution is not tested in this case.
Check {doc}`lammps_mliap` for installation and lammps script in this use case.

### Examples
```bash
# deyploy 7net-0 with flashTP enabled for LAMMPS PyTorch (TorchScript) interface
sevenn get_model 7net-0 --enable_flash

# deyploy 7net-0 with flashTP enabled for LAMMPS ML-IAP interface
sevenn get_model 7net-0 --use_mliap --enable_flash

# deyploy 7net-omni with cuEq enabled for LAMMPS ML-IAP interface, mpa modality
sevenn get_model 7net-Omni --use_mliap --enable_cueq --modal mpa

# deyploy 7net-mf-ompa with parallel for LAMMPS PyTorch interface omat24 modality
sevenn get_model 7net-mf-ompa --get_parallel --modal omat24
```

(sevenn-cp)=
## `sevenn cp`

This is an utility command. You can check model's complexity, metadata, and its modalities (or tasks) at glance.

```bash
sevenn cp 7net-mf-ompa

Sevennet version                                0.11.0
When                                  2025-03-10 23:32
Hash                  ce5b3f90adf546499e528174fbc72fce
Cutoff                                             6.0
Channel                                            128
Lmax                                                 3
Group (parity)                                      O3
Interaction layers                                   5
Self connection type                            nequip
Last epoch                                        None
Elements                                           119
cuEquivariance used                              False
FlashTP used                                     False
Modality                                   omat24, mpa
```
