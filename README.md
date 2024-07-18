

<img src="SevenNet_logo.png" alt="Alt text" height="180">


# SevenNet

SevenNet (Scalable EquiVariance Enabled Neural Network) is a graph neural network interatomic potential package that supports parallel molecular dynamics simulations with [`LAMMPS`](https://github.com/lammps/lammps). Its underlying GNN model is based on [`nequip`](https://github.com/mir-group/nequip).

The project provides parallel molecular dynamics simulations using graph neural network interatomic potentials, which enable large-scale MD simulations or faster MD simulations.

The installation and usage of SevenNet are split into two parts: training (handled by PyTorch) and molecular dynamics (handled by [`LAMMPS`](https://github.com/lammps/lammps)). The model, once trained with PyTorch, is deployed using TorchScript and is later used to run molecular dynamics simulations via LAMMPS.

- [SevenNet](#sevennet)
  * [Installation](#installation)
  * [Usage](#usage)
    + [SevenNet-0](#sevennet-0)
    + [SevenNet Calculator for ASE](#sevennet-calculator-for-ase)
    + [Training sevenn](#training)
      - [Multi-GPU training](#multi-gpu-training)
    + [sevenn_graph build](#sevenn_graph_build)
    + [sevenn_inference](#sevenn_inference)
    + [sevenn_get_model](#sevenn_get_model)
  * [Installation for LAMMPS](#installation-for-lammps)
  * [Usage for LAMMPS](#usage-for-lammps)
    + [To check installation](#to-check-installation)
    + [For serial model](#for-serial-model)
    + [For parallel model](#for-parallel-model)
  * [Future Works](#future-works)
  * [Citation](#citation)

## Installation

* Python >= 3.8
* PyTorch >= 1.11
* [`TorchGeometric`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
* [`pytorch_scatter`](https://github.com/rusty1s/pytorch_scatter)

You can find the installation guides for these packages from the [`PyTorch official`](https://pytorch.org/get-started/locally/), [`TorchGeometric docs`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and [`pytorch_scatter`](https://github.com/rusty1s/pytorch_scatter). Remember that these packages have dependencies on your CUDA version.

**PLEASE NOTE:** You must install PyTorch, TorchGeometric, and pytorch_scatter before installing SevenNet. They are not marked as dependencies since they are coupled with the CUDA version.

```bash
pip install sevenn
```

## Usage

### SevenNet-0
SevenNet-0 is a general-purpose interatomic potential trained on the [`MPF dataset of M3GNet`](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599) or [`MPtrj dataset of CHGNet`](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842). You can try SevenNet-0 to your application without any training. If the accuracy is unsatisfactory, SevenNet-0 can be [fine-tuned](#training).

#### SevenNet-0 (11July2024)
This model was trained on [`MPtrj`](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842). We suggest starting with this model as we found that it performs better than the previous SevenNet-0 (22May2024). Check [`Matbench Discovery leaderborad`](https://matbench-discovery.materialsproject.org/) for this model's performance on materials discovery.

Whenever the checkpoint path is the input, this model can be loaded via `7net-0 | SevenNet-0 | 7net-0_11July2024 | SevenNet-0_11July2024` keywords.

#### SevenNet-0 (22May2024)
This model was trained on [`MPF.2021.2.8`](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599). This is the model used in [our paper](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00190).

Whenever the checkpoint path is the input, this model can be loaded via `7net-0_22May2024 | SevenNet-0_22May2024` keywords.

### SevenNet Calculator for ASE

[ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/) is a set of tools and Python modules for atomistic simulations. SevenNet-0 and SevenNet-trained potentials can be used with ASE for its use in python.

For pre-trained models,
```python
from sevenn.sevennet_calculator import SevenNetCalculator
sevenet_0_cal = SevenNetCalculator("7net-0", device='cpu')  # 7net-0, SevenNet-0, 7net-0_22May2024, 7net-0_11July2024 ...
```

For user trained models,
```python
from sevenn.sevennet_calculator import SevenNetCalculator
checkpoint_path = ### PATH TO CHECKPOINT ###
sevenet_cal = SevenNetCalculator(checkpoint_path, device='cpu')
```

### Training

```bash
sevenn_preset base > input.yaml
sevenn input.yaml -s
```

Other valid preset options are: `base`, `fine_tune`, and `sevennet-0`.
Check comments of `base` yaml for explanations.

To reuse a preprocessed training set, you can specify `${dataset_name}.sevenn_data` to the `load_dataset_path:` in the `input.yaml`.
Once you initiate training, `log.sevenn` will contain all parsed inputs from `input.yaml`. You can refer to the log to check the default inputs.

#### Multi-GPU training
We support multi-GPU training features using PyTorch DDP (distributed data parallel). We use one process (CPU core) per GPU.
```bash
torchrun --standalone --nnodes {number of nodes} --nproc_per_node {number of GPUs} --no_python sevenn input.yaml -d
```
Please note that `batch_size` in input.yaml indicates `batch_size` per GPU.

### sevenn_graph_build
```bash
sevenn_graph_build -f ase my_train_data.extxyz 5.0
```

You can preprocess the dataset with `sevenn_graph_build` to obtain `*.sevenn_data` files. The cutoff length should be provided.
See `sevenn_graph_build --help` for more information.

### sevenn_inference
```bash
sevenn_inference checkpoint_best.pt path_to_my_structures/*
```
This will create dir `sevenn_infer_result`. It includes .csv files that enumerate prediction/reference results of energy and force.
See `sevenn_inference --help` for more information.

### sevenn_get_model
This command is for deploying lammps potentials from checkpoints. The argument is either the path to checkpoint or the name of pre-trained potential.
```bash
sevenn_get_model 7net-0
```
This will create `deployed_serial.pt`, which can be used as lammps potential under `e3gnn` pair_style. 

The parallel model can be obtained in a similar way
```bash
sevenn_get_model 7net-0 -p
```
This will create multiple `deployed_parallel_*.pt` files. The number of deployed models equals the number of message-passing layers.
These models can be used as lammps potential to run parallel MD simulations with GNN potential using multiple GPU cards.

See `sevenn_inference --help` for more information.

## Installation for LAMMPS

* PyTorch (same version as used for training)
* LAMMPS version of 'stable_2Aug2023' [`LAMMPS`](https://github.com/lammps/lammps)
* (Optional) [`CUDA-aware OpenMPI`](https://www.open-mpi.org/faq/?category=buildcuda) for parallel MD

**PLEASE NOTE:** CUDA-aware OpenMPI is optional, but recommended for parallel MD. If it is not available, in parallel mode, GPUs will communicate via CPU. It is still faster than using only one GPU, but its efficiency is low.

**PLEASE NOTE:** CUDA-aware OpenMPI does not support NVIDIA Gaming GPUs. Given that the software is closely tied to hardware specifications, please consult with your server administrator if unavailable.

Ensure the LAMMPS version (stable_2Aug2023). You can easily switch the version using git.
```bash
git clone https://github.com/lammps/lammps.git lammps_dir
cd lammps_dir
git checkout stable_2Aug2023
```

Run sevenn_patch_lammps
```bash
sevenn_patch_lammps {path_to_lammps_dir}
```
Refer to `sevenn/pair_e3gnn/patch_lammps.sh` for the patch process.

Build LAMMPS with cmake (example):
```
$ cd {path_to_lammps_dir}
$ mkdir build
$ cd build
$ cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
$ make -j4
```

## Usage for LAMMPS

### To check installation

```bash
{lmp_binary} -help | grep e3gnn
```
You will see `e3gnn` and `e3gnn/parallel` as pair_style.

### For serial model

```
pair_style e3gnn
pair_coeff * * {path to serial model} {space separated chemical species}
```

### For parallel model

```
pair_style e3gnn/parallel
pair_coeff * * {number of segmented parallel models} {space separated paths of segmented parallel models} {space separated chemical species}
```

Check [sevenn_get_model](#sevenn_get_model) for deploying lammps models from checkpoint for both serial and parallel.

**PLEASE NOTE:** One GPU per MPI process is expected. If the available GPUs are fewer than the MPI processes, the simulation may run inefficiently.

**PLEASE NOTE:** Currently, the parallel version raises an error when there are no atoms in one of the subdomain cells. This issue can be addressed using the `processors` command and, more optimally, the `fix balance` command in LAMMPS. This will be patched in the future.

## Future Works

* Notebook examples and improved interface for non-command line usage
* Implementation of pressure output in parallel MD simulations.
* Development of support for a tiled communication style (also known as recursive coordinate bisection, RCB) in LAMMPS.
* Easy use of parallel models

## Citation
If you use SevenNet, please cite (1) parallel GNN-IP MD simulation by SevenNet or its pre-trained model SevenNet-0, (2) underlying GNN-IP architecture NequIP

(1) Y. Park, J. Kim, S. Hwang, and S. Han, "Scalable Parallel Algorithm for Graph Neural Network Interatomic Potentials in Molecular Dynamics Simulations". J. Chem. Theory Comput., 20(11), 4857 (2024) (https://pubs.acs.org/doi/10.1021/acs.jctc.4c00190)

(2) S. Batzner, A. Musaelian, L. Sun, M. Geiger, J. P. Mailoa, M. Kornbluth, N. Molinari, T. E. Smidt, and B. Kozinsky, "E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials". Nat. Commun., 13, 2453. (2022) (https://www.nature.com/articles/s41467-022-29939-5)
