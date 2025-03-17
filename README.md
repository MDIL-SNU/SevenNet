
<img src="SevenNet_logo.png" alt="Alt text" height="180">

# SevenNet

SevenNet (Scalable EquiVariance Enabled Neural Network) is a graph neural network (GNN) interatomic potential package that supports parallel molecular dynamics simulations with [`LAMMPS`](https://lammps.org). Its underlying GNN model is based on [`NequIP`](https://github.com/mir-group/nequip).

> [!NOTE]
> We will soon release a CUDA-accelerated version of SevenNet, which will significantly increase the speed of our pre-trained models on [Matbench Discovery](https://matbench-discovery.materialsproject.org/).

## Features
 - Pre-trained GNN interatomic potential and fine-tuning interface.
 - Python [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) calculator support
 - GPU-parallelized molecular dynamics with LAMMPS
 - CUDA-accelerated D3 (van der Waals) dispersion
 - Multi-fidelity training for combining multiple database with different calculation settings. [Usage](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_MF_0/README.md).

## Pre-trained models
So far, we have released three pre-trained SevenNet models. Each model has various hyperparameters and training sets, resulting in different accuracy and speed. Please read the descriptions below carefully and choose the model that best suits your purpose.
We provide the training set MAEs (energy, force, and stress) F1 score, and RMSD for the WBM dataset, as well as $\kappa_{\mathrm{SRME}}$ from phonondb and CPS (Combined Performance Score). For details on these metrics and performance comparisons with other pre-trained models, please visit [Matbench Discovery](https://matbench-discovery.materialsproject.org/).

These models can be used as interatomic potential on LAMMPS, and also can be loaded through ASE calculator by calling the `keywords` of each model. Please refer [ASE calculator](#ase_calculator) to see the way to load a model through ASE calculator.
Additionally, `keywords` can be called in other parts of SevenNet, such as `sevenn_inference`, `sevenn_get_model`, and `checkpoint` in `input.yaml` for fine-tuning.

**Acknowledgments**: The models trained on [`MPtrj`](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) were supported by the Neural Processing Research Center program of Samsung Advanced Institute of Technology, Samsung Electronics Co., Ltd. The computations for training models were carried out using the Samsung SSC-21 cluster.

---

### **SevenNet-MF-ompa (17Mar2025)**
> Model keywords: `7net-mf-ompa` | `SevenNet-mf-ompa`

**This is our recommended pre-trained model**

This model leverages [multi-fidelity learning](https://pubs.acs.org/doi/10.1021/jacs.4c14455) to simultaneously train on the [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842), [sAlex](https://huggingface.co/datasets/fairchem/OMAT24), and [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) datasets. As of March 17, 2025, it has achieved state-of-the-art performance on the [Matbench Discovery](https://matbench-discovery.materialsproject.org/) in the CPS (Combined Performance Score). We have found that this model outperforms most tasks, except for isolated molecule energy, where it performs slightly worse than SevenNet-l3i5.

```python
from sevenn.calculator import SevenNetCalculator
# "mpa" refers to the MPtrj + sAlex modal, used for evaluating Matbench Discovery.
calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')  # Use modal='omat24' for OMat24-trained modal weights.
```
Theoretically, the `mpa` modal should produce PBE52 results, while the `omat24` modal yields PBE54 results.

When using the command-line interface of SevenNet, include the `--modal mpa` or `--modal omat24` option to select the desired modality.


#### **Matbench Discovery**
| CPS  | F1 | $\kappa_{\mathrm{SRME}}$ | RMSD |
|:---:|:---:|:---:|:---:|
|**0.883**|**0.901**|0.317| **0.0115** |

[Detailed instructions for multi-fidelity](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_MF_0/README.md)

[Link to the full-information checkpoint](https://figshare.com/articles/software/7net_MF_ompa/28590722?file=53029859)

---
### **SevenNet-omat (17Mar2025)**
> Model keywords: `7net-omat` | `SevenNet-omat`

 This model was trained solely on the [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) dataset. It achieves state-of-the-art (SOTA) performance in $\kappa_{\mathrm{SRME}}$ on [Matbench Discovery](https://matbench-discovery.materialsproject.org/); however, the F1 score was not available due to a difference in the POTCAR version. Similar to `SevenNet-MF-ompa`, this model outperforms `SevenNet-l3i5` in most tasks, except for isolated molecule energy.

[Link to the full-information checkpoint](https://figshare.com/articles/software/SevenNet_omat/28593938).

#### **Matbench Discovery**
* $\kappa_{\mathrm{SRME}}$: **0.221**
---
### **SevenNet-l3i5 (12Dec2024)**
> Model keywords: `7net-l3i5` | `SevenNet-l3i5`

The model increases the maximum spherical harmonic degree ($l_{\mathrm{max}}$) to 3, compared to `SevenNet-0` with $l_{\mathrm{max}}$ of 2. While **l3i5** offers improved accuracy across various systems compared to `SevenNet-0`, it is approximately four times slower. As of March 17, 2025, this model has achieved state-of-the-art (SOTA) performance on the CPS metric among compliant models, newly introduced in this [Matbench Discovery](https://matbench-discovery.materialsproject.org/).

#### **Matbench Discovery**
| CPS  | F1 | $\kappa_{\mathrm{SRME}}$ | RMSD |
|:---:|:---:|:---:|:---:|
|0.764 |0.76|0.55|0.0182|

---

### **SevenNet-0 (11Jul2024)**
> Model keywords:: `7net-0` | `SevenNet-0` | `7net-0_11Jul2024` | `SevenNet-0_11Jul2024`

The model architecture is mainly line with [GNoME](https://github.com/google-deepmind/materials_discovery), a pretrained model that utilizes the NequIP architecture.
Five interaction blocks with node features that consist of 128 scalars (*l*=0), 64 vectors (*l*=1), and 32 tensors (*l*=2).
The convolutional filter employs a cutoff radius of 5 Angstrom and a tensor product of learnable radial functions from bases of 8 radial Bessel functions and $l_{\mathrm{max}}$ of 2, resulting in the number of parameters is 0.84 M.
The model was trained with [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842).
This model is loaded as the default pre-trained model in ASE calculator.
For more information, click [here](sevenn/pretrained_potentials/SevenNet_0__11Jul2024).

#### **Matbench Discovery**
| F1 | $\kappa_{\mathrm{SRME}}$ |
|:---:|:---:|
|0.67|0.767|

---

In addition to these latest models, you can find our legacy models from [pretrained_potentials](./sevenn/pretrained_potentials).

## Contents
- [Installation](#installation)
- [Usage](#usage)
  - [ASE calculator](#ase-calculator)
  - [Training & inference](#training-and-inference)
  - [Notebook tutorials](#notebook-tutorial)
  - [MD simulation with LAMMPS](#md-simulation-with-lammps)
    - [Installation](#installation)
    - [Single-GPU MD](#single-gpu-md)
    - [Multi-GPU MD](#multi-gpu-md)
  - [Application of SevenNet-0](#application-of-sevennet-0)
- [Citation](#citation)

## Installation<a name="installation"></a>
### Requirements
- Python >= 3.8
- PyTorch >= 1.12.0

Here are the recommended versions we've been using internally without any issues.
- PyTorch/2.2.2 + CUDA/12.1.0
- PyTorch/1.13.1 + CUDA/12.1.0
- PyTorch/1.12.0 + CUDA/11.6.2
Using the newer versions of CUDA with PyTorch is usually not a problem. For example, you can compile and use `PyTorch/1.13.1+cu117` with `CUDA/12.1.0`.

> [!IMPORTANT]
> Please install PyTorch manually depending on the hardware before installing the SevenNet.

#### Optional requirements
- nvcc compiler

This should be available to use `SevenNetD3Calculator` or `D3Calculator`.

Give that the PyTorch is successfully installed, please run the command below.
```bash
pip install sevenn
pip install https://github.com/MDIL-SNU/SevenNet.git # for the latest version
```
We strongly recommend checking `CHANGELOG.md` for new features and changes because SevenNet is under active development.

## Usage<a name="usage"></a>
### ASE calculator<a name="ase_calculator"></a>

For a wider application in atomistic simulations, SevenNet provides the ASE interface through ASE calculator.
The model can be loaded through the following Python code.

```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator(model='7net-0', device='cpu')
```
SevenNet supports CUDA accelerated D3Calculator.
```python
from sevenn.calculator import SevenNetD3Calculator
calc = SevenNetD3Calculator(model='7net-0', device='cuda')
```
If you encounter `CUDA is not installed or nvcc is not available`, ensure the `nvcc` compiler is available. Currently, CPU + D3 is not supported.

Various pre-trained SevenNet models can be accessed by changing the `model` variable to any predefined keywords such as `7net-l3i5`, `7net-0_11Jul2024`, `7net-0_22May2024`, and so on. The default model is **SevenNet-0 (11Jul2024)**.

In addition, not only pre-trained models but also user-trained models can be applied in ASE calculator.
In this case, the path of checkpoint generated after training should be identified in `model` variable.

> [!TIP]
> When 'auto' is passed by `device`, SevenNet utilizes GPU acceleration if available.

### Training and inference

SevenNet provides five commands for preprocess, training, and deployment: `sevenn_preset`, `sevenn_graph_build`, `sevenn`, `sevenn_inference`, `sevenn_get_model`.

#### 1. Input generation

With the `sevenn_preset` command, the input file that sets the training parameters is generated automatically.
```bash
sevenn_preset {preset keyword} > input.yaml
```

Available preset keywords are: `base`, `fine_tune`, `multi_modal`, `sevennet-0`, and `sevennet-l3i5`.
Check comments in the preset yaml files for explanations. For fine-tuning, note that most model hyperparameters cannot be modified unless explicitly indicated.
To reuse a preprocessed training set, you can specify `sevenn_data/${dataset_name}.pt` to the `load_trainset_path:` in the `input.yaml`.

#### 2. Preprocess (optional)

To obtain the preprocessed data, `sevenn_data/graph.pt`, `sevenn_graph_build` command can be used.
The output files can be used for training (`sevenn`) or inference (`sevenn_inference`) to skip the graph build stage.

```bash
sevenn_graph_build {dataset path} {cutoff radius}
```

The output `sevenn_data/graph.yaml` contains statistics and meta information for the dataset.
These files must be located under the `sevenn_data`. If you move the dataset, move the entire `sevenn_data` directory without changing the contents.

See `sevenn_graph_build --help` for more information.

#### 3. Training

Given that `input.yaml` and `sevenn_data/graph.pt` are prepared, SevenNet can be trained by the following command:

```bash
sevenn input.yaml -s
```

We support multi-GPU training features using PyTorch DDP (distributed data parallel) with single process (or a CPU core) per GPU.

```bash
torchrun --standalone --nnodes {number of nodes} --nproc_per_node {number of GPUs} --no_python sevenn input.yaml -d
```

Please note that `batch_size` in input.yaml indicates `batch_size` per GPU.

#### 4. Inference

Using the checkpoint after the training, the properties such as energy, force, and stress can be inferred directly.

```bash
sevenn_inference checkpoint_best.pth path_to_my_structures/*
```

This will create the `sevenn_infer_result` directory, where csv files contain predicted energy, force, the stress, and their references (if available).
See `sevenn_inference --help` for more information.

#### 5. Deployment<a name="deployment"></a>

The checkpoint can be deployed as the LAMMPS potentials. The argument is either the path to checkpoint or the name of pre-trained potential.

```bash
sevenn_get_model 7net-0
sevenn_get_model {checkpoint path}
```

This will create `deployed_serial.pt`, which can be used as lammps potential under `e3gnn` pair_style.

The potential for parallel MD simulation can be obtained in a similar way.

```bash
sevenn_get_model 7net-0 -p
sevenn_get_model {checkpoint path} -p
```

This will create a directory with multiple `deployed_parallel_*.pt` files. The directory path itself is an argument for the lammps script. Please do not modify or remove files under the directory.
These models can be used as lammps potential to run parallel MD simulations with GNN potential using multiple GPU cards.

### Notebook tutorials<a name="notebook-tutorial"></a>

If you want to learn how to use the `sevenn` python library instead of the CLI command, please check out the notebook tutorials below.

| Notebooks | Google&nbsp;Colab | Descriptions |
|-----------|-------------------|--------------|
|[From scratch](https://github.com/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_python_tutorial.ipynb)|[![Open in Google Colab]](https://colab.research.google.com/github/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_python_tutorial.ipynb)|We can learn how to train the SevenNet from scratch, predict energy, forces, and stress using the trained model, perform structure relaxation, and draw EOS curves.|
|[Fine-tuning](https://github.com/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_finetune_tutorial.ipynb)|[![Open in Google Colab]](https://colab.research.google.com/github/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_finetune_tutorial.ipynb)|We can learn how to fine-tune the SevenNet and compare the results of the pretrained model with the fine-tuned model.|

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg

Sometimes, the Colab environment may crash due to memory issues. If you have good GPU resources in your local environment, it is recommended to download the tutorial from GitHub and run it locally.
```bash
git clone https://github.com/MDIL-SNU/sevennet_tutorial.git
```

### MD simulation with LAMMPS

#### Installation

##### Requirements
- PyTorch < 2.5.0 (same version as used for training)
- LAMMPS version of `stable_2Aug2023_update3`
- MKL library
- [`CUDA-aware OpenMPI`](https://www.open-mpi.org/faq/?category=buildcuda) for parallel MD (optional)

If your cluster supports the Intel MKL module (often included with Intel OneAPI, Intel Compiler, and other Intel-related modules), load the module.

CUDA-aware OpenMPI is optional but recommended for parallel MD. If it is not available, in parallel mode, GPUs will communicate via CPU. It is still faster than using only one GPU, but its efficiency is low.

> [!IMPORTANT]
> CUDA-aware OpenMPI does not support NVIDIA Gaming GPUs. Given that the software is closely tied to hardware specifications, please consult with your server administrator if unavailable.

1. Build LAMMPS with cmake.

Ensure the LAMMPS version (stable_2Aug2023_update3). You can easily switch the version using git. After switching the version, run `sevenn_patch_lammps` with the lammps directory path as an argument.

```bash
git clone https://github.com/lammps/lammps.git lammps_sevenn --branch stable_2Aug2023_update3 --depth=1
sevenn_patch_lammps ./lammps_sevenn {--d3}
```
You can refer to `sevenn/pair_e3gnn/patch_lammps.sh` for the detailed patch process.

> [!TIP]
> Add `--d3` option to install GPU accelerated [Grimme's D3 method](https://doi.org/10.1063/1.3382344) pair style. For its usage and details, click [here](sevenn/pair_e3gnn).

```bash
cd ./lammps_sevenn
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
make -j4
```

If the error `MKL_INCLUDE_DIR NOT-FOUND` occurs, please check the environment variable or read the `Possible solutions` below.
If compilation is done without any errors, please skip this.

<details>
<summary><b>Possible solutions</b></summary>

2. Install mkl-include via conda

```bash
conda install -c intel mkl-include
conda install mkl-include # if the above failed
```

3. Append `DMKL_INCLUDE_DIR` to the cmake command and repeat step 1

```bash
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DMKL_INCLUDE_DIR=$CONDA_PREFIX/include
```

If the `undefined reference to XXX` error with `libtorch_cpu.so` occurs, check the `$LD_LIBRARY_PATH`.
If PyTorch is installed using Conda, `libmkl_*.so` files can be found in `$CONDA_PREFIX/lib`.
Ensure that `$LD_LIBRARY_PATH` includes `$CONDA_PREFIX/lib`.

For other error cases, the solution can be found in [`pair-nequip`](https://github.com/mir-group/pair_nequip) repository as we share the architecture.

</details>

If the compilation is successful, the executable `lmp` can be found at `{path_to_lammps_dir}/build`.
To use this binary easily, create a soft link in your bin directory (which should be included in your `$PATH`).

```bash
ln -s {absolute_path_to_lammps_directory}/build/lmp $HOME/.local/bin/lmp
```

This will allow you to run the binary using `lmp -in my_lammps_script.lmp`.

#### Single-GPU MD

For single-GPU MD simulations, `e3gnn` pair_style should be used. The minimal input script is provided as follows:
```txt
units       metal
atom_style  atomic
pair_style  e3gnn
pair_coeff  * * {path to serial model} {space separated chemical species}
```

#### Multi-GPU MD

For multi-GPU MD simulations, `e3gnn/parallel` pair_style should be used. The minimal input script is provided as follows:
```txt
units       metal
atom_style  atomic
pair_style  e3gnn/parallel
pair_coeff  * * {number of message-passing layers} {directory of parallel model} {space separated chemical species}
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

### Application of SevenNet-0
If you are interested in the actual application of SevenNet, refer to [this paper](https://arxiv.org/abs/2501.05211) (data available at: [Zenodo](https://zenodo.org/records/14734414)).
In this study, SevenNet-0 was applied to the simulation of liquid electrolytes.

The fine-tuning procedure and relevant input files are provided in the links above, particularly in the `Fine-tuning.tar.xz` archive on Zenodo.

The yaml file used for fine-tuning is available via the command:
```bash
sevenn_preset fine_tune_le > input.yaml
```

## Citation<a name="citation"></a>

If you use this code, please cite our paper:
```txt
@article{park_scalable_2024,
	title = {Scalable Parallel Algorithm for Graph Neural Network Interatomic Potentials in Molecular Dynamics Simulations},
	volume = {20},
	doi = {10.1021/acs.jctc.4c00190},
	number = {11},
	journal = {J. Chem. Theory Comput.},
	author = {Park, Yutack and Kim, Jaesun and Hwang, Seungwoo and Han, Seungwu},
	year = {2024},
	pages = {4857--4868},
}
```

If you utilize the multi-fidelity feature of this code or the pretrained model SevenNet-MF-0, please cite the following paper:
```txt
@article{kim_sevennet_mf_2024,
	title = {Data-Efficient Multifidelity Training for High-Fidelity Machine Learning Interatomic Potentials},
	volume = {147},
	doi = {10.1021/jacs.4c14455},
	number = {1},
	journal = {J. Am. Chem. Soc.},
	author = {Kim, Jaesun and Kim, Jisu and Kim, Jaehoon and Lee, Jiho and Park, Yutack and Kang, Youngho and Han, Seungwu},
	year = {2024},
	pages = {1042--1054},
```
