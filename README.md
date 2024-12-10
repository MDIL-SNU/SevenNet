
<img src="SevenNet_logo.png" alt="Alt text" height="180">

# SevenNet

SevenNet (Scalable EquiVariance Enabled Neural Network) is a graph neural network (GNN) interatomic potential package that supports parallel molecular dynamics simulations with [`LAMMPS`](https://lammps.org). Its underlying GNN model is based on [`NequIP`](https://github.com/mir-group/nequip).

> [!CAUTION]
> SevenNet+LAMMPS parallel after the commit id of `14851ef (v0.9.3 ~ 0.9.5)` has a serious bug.
> It gives wrong forces when the number of mpi processes is greater than two. The corresponding pip version is yanked for this reason. The bug is fixed for the main branch since `v0.10.0`, and pip (`v0.9.3.post0`).


## Features
 - Pre-trained GNN interatomic potential, SevenNet-0 with fine-tuning interface
 - Python [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) calculator support
 - GPU-parallelized molecular dynamics with LAMMPS
 - CUDA-accelerated D3 (van der Waals) dispersion

## Pre-trained models
So far, we have released three pre-trained SevenNet models. Each model has various hyperparameters and training sets, resulting in different accuracy and speed. Please read the descriptions below carefully and choose the model that best suits your purpose.
In addition, as model performances, we provide the training MAEs (energy, force, and stress), F1 score for WBM dataset and $\kappa_{\mathrm{SRME}}$ from phonondb.

For detailed performance comparisons with other pre-trained models, please visit [Matbench Discovery](https://matbench-discovery.materialsproject.org/).
These models can be used as interatomic potential on LAMMPS, and also can be loaded through ASE calculator by calling the `keywords` of each model. Please refer [`ASE calculator`](#ase_calculator) to see the way to load a model through ASE calculator.

* **l3i5**

The model increases the maximum spherical harmonic degree ($l_{\mathrm{max}}$) to 3, compared to **SevenNet-0 (11Jul2024)** with $l_{\mathrm{max}}$ of 2.
While **l3i5** model provides significantly improved accuracy in range of systems, the inference speed is approximately four times slower than **SevenNet-0 (11Jul2024)** due to the increased number of parameters of 1.17 M.
For more information, see [here](sevenn/pretrained_potentials/SevenNet_l3i5).

    * MAE: 8.3 meV/atom (energy), 0.029 eV/Ang. (force), and 2.33 kbar (stress)
    * F1 score: 0.76, $\kappa_{\mathrm{SRME}}$: 0.560
    * Speed: 28m 38s / epoch (with 8 A100 GPU cards)

Keywords: `7net-l3i5`, `SevenNet-l3i5`

* **SevenNet-0 (11Jul2024)**

The training is changed from [`MPF.2021.2.8`](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599) to [`MPtrj`](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842), compared to **SevenNet-0 (22May2024)**
This model is default pre-trained model uploaded in ASE calculator.
For more information, click [here](sevenn/pretrained_potentials/SevenNet_0__11Jul2024).

    * MAE: 11.5 meV/atom (energy), 0.041 eV/Ang. (force), and 2.78 kbar (stress)
    * F1 score: 0.67, $\kappa_{\mathrm{SRME}}$: 0.767
    * Speed: 6m 41s / epoch (with 8 A100 GPU cards)

Keywords: `7net-0`, `SevenNet-0`, `7net-0_11Jul2024`, and `SevenNet-0_11Jul2024`

* **SevenNet-0 (22May2024)**

The model architecture is mainly line with [GNoME](https://github.com/google-deepmind/materials_discovery), a pretrained model that utilizes the NequIP architecture.
Five interaction blocks with node features that consist of 128 scalars (*l*=0), 64 vectors (*l*=1), and 32 tensors (*l*=2).
The convolutional filter employs a cutoff radius of 5 Angstrom and a tensor product of learnable radial functions from bases of 8 radial Bessel functions and $l_{\mathrm{max}}$ of 2, resulting in the number of parameters is 0.84 M.

The model was trained with [`MPF.2021.2.8`](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599) up to 600 epochs. For more information, please read the [paper](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00190) and visit [here](sevenn/pretrained_potentials/SevenNet_0__22May2024).

    * MAE: 16.3 meV/atom (energy), 0.037 eV/Ang. (force), and 2.96 kbar (stress)
    * F1 score: 0.65

Keywords: `7net-0_22May2024` and `SevenNet-0_22May2024`

**Acknowledgments**: The models trained on [`MPtrj`](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) were supported by the Neural Processing Research Center program of Samsung Advanced Institute of Technology, Samsung Electronics Co., Ltd. The computations for training models were carried out using the Samsung SSC-21 cluster.

## Contents
- [Installation](#installation)
- [Usage](#usage)
  - [ASE calculator](#ase-calculator)
  - [Training & inference](#training)
  - [MD simulation with LAMMPS](#md-simulation-with-lammps)
    - [Installation](#installation-for-lammps)
    - [Single-GPU MD](#single-gpu_md)
    - [Multi-GPU MD](#multi-gpu_md)
- [Future Works](#future-works)
- [Citation](#citation)

## Installation<a name="installation"></a>
### Requirements
- Python >= 3.8
- PyTorch >= 1.12.0, PyTorch < 2.5.0

Here are the recommended versions we've been using internally without any issues.
- PyTorch/2.2.2 + CUDA/12.1.0
- PyTorch/1.13.1 + CUDA/12.1.0
- PyTorch/1.12.0 + CUDA/11.6.2
Using the newer versions of CUDA with PyTorch is usually not a problem. For example, you can compile and use `PyTorch/1.13.1+cu117` with `CUDA/12.1.0`.

> [!IMPORTANT]
> Please install PyTorch manually depending on the hardware before installing the SevenNet.

Give that the PyTorch is successfully installed, please run the command below.
```bash
pip install sevenn
pip install https://github.com/MDIL-SNU/SevenNet.git # for the latest version
```
We strongly recommend checking `CHANGELOG.md` for new features and changes because the SevenNet is under active development.

## Usage<a name="usage"></a>
### ASE calculator<a name="ase_calculator"></a>

For a wider application in atomistic simulations, SevenNet provides the ASE interface through ASE calculator.
The model can be loaded through the following Python code.

```python
from sevenn.sevennet_calculator import SevenNetCalculator
calculator = SevenNetCalculator(model='7net-0', device='cpu')
```

Various pre-trained SevenNet models can be accessed by changing the `model` variable to any predefined keywords such as `7net-l3i5`, `7net-0_11Jul2024`, `7net-0_22May2024`, and so on. The default model is **SevenNet-0 (11Jul2024)**.

In addition, not only pre-trained models but also user-trained models can be applied in ASE calculator.
In this case, the path of checkpoint generated after training should be identified in `model` variable.

> [!TIP]
> When 'auto' is passed by `device`, SevenNet utilizes GPU acceleration if available.

### Training & inference

SevenNet provides five commands for training, deployment, or convenience: `sevenn_preset`, `sevenn_graph_build`, `sevenn`, `sevenn_inference`, `sevenn_get_model`.

#### 1. Input generation

With the `sevenn_preset` command, the input file that sets the training parameters is generated automatically.
```bash
sevenn_preset {preset keyword} > input.yaml
```

Available preset keywords are: `base`, `fine_tune`, and `sevennet-0`.
Check comments in the preset yaml files for explanations. For fine-tuning, note that most model hyperparameters cannot be modified unless explicitly indicated.
To reuse a preprocessed training set, you can specify `${dataset_name}.sevenn_data` to the `load_dataset_path:` in the `input.yaml`.

#### 2. Preprocess (optional)

You can preprocess the dataset with `sevenn_graph_build` to obtain `./sevenn_data/graph.pt` files. These files can be used for training (`sevenn`) or
inference (`sevenn_inference`), skipping the graph build stage.

```bash
sevenn_graph_build {dataset path} {cutoff radius}
```

The output `./sevenn_data/graph.yaml` contains statistics and meta information for the dataset.
These files must be located under the `sevenn_data`. If you move the dataset, move the entire `sevenn_data` directory without changing the contents.

See `sevenn_graph_build --help` for more information.

#### 3. Training

```bash
sevenn input.yaml -s
```

We support multi-GPU training features using PyTorch DDP (distributed data parallel). We use single process (or a CPU core) per GPU.

```bash
torchrun --standalone --nnodes {number of nodes} --nproc_per_node {number of GPUs} --no_python sevenn input.yaml -d
```

Please note that `batch_size` in input.yaml indicates `batch_size` per GPU.

#### 4. Inference

```bash
sevenn_inference checkpoint_best.pth path_to_my_structures/*
```

This will create dir `sevenn_infer_result`. It includes .csv files that enumerate prediction/reference results of energy and force.
See `sevenn_inference --help` for more information.

#### 5. Deployment<a name="deployment"></a>

This command is for deploying lammps potentials from checkpoints. The argument is either the path to checkpoint or the name of pre-trained potential.

```bash
sevenn_get_model 7net-0
```

This will create `deployed_serial.pt`, which can be used as lammps potential under `e3gnn` pair_style.

The parallel model can be obtained in a similar way

```bash
sevenn_get_model 7net-0 -p
```

This will create a directory with multiple `deployed_parallel_*.pt` files. The directory path itself is an argument for the lammps script. Please do not modify or remove files under the directory.
These models can be used as lammps potential to run parallel MD simulations with GNN potential using multiple GPU cards.

### MD simulation with LAMMPS

#### Installation

##### Requirements
- PyTorch < 2.5.0 (same version as used for training)
- LAMMPS version of 'stable_2Aug2023_update3' [`LAMMPS`](https://github.com/lammps/lammps)
- [`CUDA-aware OpenMPI`](https://www.open-mpi.org/faq/?category=buildcuda) for parallel MD (optional)
- MKL library

> [!IMPORTANT]
> CUDA-aware OpenMPI does not support NVIDIA Gaming GPUs. Given that the software is closely tied to hardware specifications, please consult with your server administrator if unavailable.

If your cluster supports the Intel MKL module (often included with Intel OneAPI, Intel Compiler, and other Intel-related modules), load the module. If it is unavailable, read the 'Note for MKL' section before running cmake.

CUDA-aware OpenMPI is optional but recommended for parallel MD. If it is not available, in parallel mode, GPUs will communicate via CPU. It is still faster than using only one GPU, but its efficiency is low.

Ensure the LAMMPS version (stable_2Aug2023_update3). You can easily switch the version using git. After switching the version, run `sevenn_patch_lammps` with the lammps directory path as an argument.

```bash
git clone https://github.com/lammps/lammps.git lammps_sevenn --branch stable_2Aug2023_update3 --depth=1
sevenn_patch_lammps ./lammps_sevenn {--d3}
```

> [!TIP]
> Add `--d3` option to install GPU accelerated [Grimme's D3 method](https://doi.org/10.1063/1.3382344) pair style (currently available in main branch only, not pip). For its usage and details, click [here](sevenn/pair_e3gnn).

You can refer to `sevenn/pair_e3gnn/patch_lammps.sh` for the detailed patch process.

Build LAMMPS with cmake (example):

```bash
cd ./lammps_sevenn
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
make -j4
```

You may encounter `MKL_INCLUDE_DIR NOT-FOUND` during cmake. This usually means the environment variable is not set correctly, or mkl-include is not present on your system.

Install mkl-include with:

```bash
conda install -c intel mkl-include
```

If you encounter an error, remove `-c intel`. This is a known bug in the recent Conda version.

Append the following to your cmake command:

```bash
-DMKL_INCLUDE_DIR=$CONDA_PREFIX/include
```

If you see hundreds of `undefined reference to XXX` errors with `libtorch_cpu.so` at the end of compilation, check your `$LD_LIBRARY_PATH`. PyTorch depends on MKL libraries (this is a default backend for torch+CPU), therefore you already have them. For example, if you installed PyTorch using Conda, you may find `libmkl_*.so` files under `$CONDA_PREFIX/lib`. Ensure that `$LD_LIBRARY_PATH` includes `$CONDA_PREFIX/lib`.

For other error cases, you might want to check [`pair-nequip`](https://github.com/mir-group/pair_nequip), as the `pair-nequip` and SevenNet+LAMMPS shares similar requirements: torch + LAMMPS.
If the compilation is successful, you will find the executable at `{path_to_lammps_dir}/build/lmp`. To use this binary easily, for example, create a soft link in your bin directory (which should be included in your `$PATH`).

```bash
ln -s {absolute_path_to_lammps_dir}/build/lmp $HOME/.local/bin/lmp
```

This will allow you to run the binary using `lmp -in my_lammps_script.lmp`.

```bash
{lammps_binary} -help | grep e3gnn
```

If the SevenNet is successfully installed in LAMMPS, you will see `e3gnn` and `e3gnn/parallel` as pair_style.

#### Single-GPU MD

```txt
units       metal
atom_style  atomic
pair_style  e3gnn
pair_coeff  * * {path to serial model} {space separated chemical species}
```

#### Multi-GPU MD

```txt
units       metal
atom_style  atomic
pair_style  e3gnn/parallel
pair_coeff  * * {number of message-passing layers} {path to the directory containing parallel model} {space separated chemical species}
```

For example,

```txt
pair_style e3gnn/parallel
pair_coeff * * 4 ./deployed_parallel Hf O
```
The number of message-passing layers is equal to the number of `*.pt` files in the `./deployed_parallel` directory.

Use [`sevenn_get_model`](#deployment) for deploying lammps models from checkpoint for both serial and parallel.

One GPU per MPI process is expected. The simulation may run inefficiently if the available GPUs are fewer than the MPI processes.

> [!CAUTION]
> Currently, the parallel version raises an error when there are no atoms in one of the subdomain cells. This issue can be addressed using the `processors` command and, more optimally, the `fix balance` command in LAMMPS. This will be patched in the future.

## Future Works<a name="fugure-works"></a>

- Notebook examples and improved interface for non-command line usage
- Development of a tiled communication style (also known as recursive coordinate bisection, RCB) in LAMMPS.

## Citation<a name="citation"></a>

If you use this code, please cite our paper:
```txt
@article{park_scalable_2024,
	title = {Scalable {Parallel} {Algorithm} for {Graph} {Neural} {Network} {Interatomic} {Potentials} in {Molecular} {Dynamics} {Simulations}},
	volume = {20},
	doi = {10.1021/acs.jctc.4c00190},
	number = {11},
	journal = {J. Chem. Theory Comput.},
	author = {Park, Yutack and Kim, Jaesun and Hwang, Seungwoo and Han, Seungwu},
	year = {2024},
	pages = {4857--4868},
}
```
