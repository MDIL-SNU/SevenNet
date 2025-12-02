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

Enable cuEquivariance or flashTP while training by adding the --enable_cueq or --enable_flashTP.

```bash
sevenn train input.yaml -s --enable_cueq # or --enable_flashTP
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

The command is for LAMMPS integration of SevenNet. It deploys a model into a LAMMPS readable file(s).

See {doc}`../install/lammps_torch` or {doc}`../install/lammps_mliap` for installation.

The checkpoint can be deployed as LAMMPS potentials. The argument is either the path to the checkpoint or the name of a pretrained potential.

```bash
sevenn get_model 7net-0  # For pre-trained models
sevenn get_model {checkpoint path}  # For user-trained models
```

This will create `deployed_serial.pt`, which can be used as a LAMMPS potential with the `e3gnn` pair_style in LAMMPS.

The potential for parallel MD simulation can be obtained similarly.

```bash
sevenn get_model 7net-0 -p
sevenn get_model {checkpoint path} -p
```

This will create a directory with several `deployed_parallel_*.pt` files. The directory path itself is an argument for the LAMMPS script. Please do not modify or remove files in the directory.
These models can be used as LAMMPS potentials to run parallel MD simulations with a GNN potential across multiple GPUs.

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

