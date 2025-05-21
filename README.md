
<img src="SevenNet_logo.png" alt="Alt text" height="180">

# SevenNet

SevenNet (Scalable EquiVariance-Enabled Neural Network) is a graph neural network (GNN)-based interatomic potential package that supports parallel molecular dynamics simulations using [`LAMMPS`](https://lammps.org). Its core model is based on [`NequIP`](https://github.com/mir-group/nequip).

> [!NOTE]
> We will soon release a CUDA-accelerated version of SevenNet, which will significantly increase the speed of our pretrained models on [Matbench Discovery](https://matbench-discovery.materialsproject.org/).

> [!TIP]
> SevenNet supports NVIDIA's [cuEquivariance](https://github.com/NVIDIA/cuEquivariance) for acceleration. In our benchmarks, we found that the cuEquivariance improves inference speed by a factor of three for the SevenNet-MF-ompa potential. See [Installation](#installation) for details.

## Features
 - Pretrained GNN interatomic potential and fine-tuning interface
 - Support for the Python [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) calculator
 - GPU-parallelized molecular dynamics with LAMMPS
 - CUDA-accelerated D3 (van der Waals) dispersion
 - Multi-fidelity training for combining multiple databases with different calculation settings ([Usage](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_MF_0/README.md))

## Pretrained models
So far, we have released multiple pretrained SevenNet models. Each model has various hyperparameters and training sets, leading to different levels of accuracy and speed. Please read the descriptions below carefully and choose the model that best suits your purpose.
We provide the F1 score, and RMSD for the WBM dataset, along with $\kappa_{\mathrm{SRME}}$ from phononDB and CPS (Combined Performance Score). For details on these metrics and performance comparisons with other pretrained models, please visit [Matbench Discovery](https://matbench-discovery.materialsproject.org/).

These models can be used as interatomic potentials in LAMMPS and loaded through the ASE calculator using each model’s keywords. Please refer to the [ASE calculator](#ase_calculator) section for instructions on loading a model via the ASE calculator.
Additionally, `keywords` can be used in other parts of SevenNet, such as `sevenn_inference`, `sevenn_get_model`, and the `checkpoint` section in `input.yaml` for fine-tuning.

**Acknowledgments**: The models trained on [`MPtrj`](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) were supported by the Neural Processing Research Center program at Samsung Advanced Institute of Technology, part of Samsung Electronics Co., Ltd. The computations for training models were carried out using the Samsung SSC-21 cluster.

---

### **SevenNet-MF-ompa (17Mar2025)**
> Model keywords: `7net-mf-ompa` | `SevenNet-mf-ompa`

**This is our recommended pretrained model**

This model leverages [multi-fidelity learning](https://pubs.acs.org/doi/10.1021/jacs.4c14455) to train simultaneously on the [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842), [sAlex](https://huggingface.co/datasets/fairchem/OMAT24), and [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) datasets. This model is the best among our pretrained models and achieves a high ranking on the [Matbench Discovery]((https://matbench-discovery.materialsproject.org/)) leaderboard. Our evaluations show that it outperforms other models on most tasks, except for the isolated molecule energy task, where it performs slightly worse than `SevenNet-l3i5`.

```python
from sevenn.calculator import SevenNetCalculator
# "mpa" refers to the MPtrj + sAlex modal, used for evaluating Matbench Discovery.
calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')  # Use modal='omat24' for OMat24-trained modal weights.
```
> [!NOTE]
> Each modal is expected to produce results that are more consistent with the DFT settings in the training datasets (e.g., `mpa`, trained on the combined [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) and [sAlex](https://huggingface.co/datasets/fairchem/OMAT24) datasets; `omat24`, trained on the [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) dataset). For detailed DFT settings, please refer to their papers.

When using the command-line interface of SevenNet, include the `--modal mpa` or `--modal omat24` option to select the desired modality.

#### **Matbench Discovery**
| CPS  | F1 | $\kappa_{\mathrm{SRME}}$ | RMSD |
|:---:|:---:|:---:|:---:|
|**0.845**|**0.901**|0.317| **0.064** |

[Detailed instructions for multi-fidelity learning](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_MF_0/README.md)

[Download link for fully detailed checkpoint](https://figshare.com/articles/software/7net_MF_ompa/28590722?file=53029859)

---
### **SevenNet-omat (17Mar2025)**
> Model keywords: `7net-omat` | `SevenNet-omat`

 This model was trained exclusively on the [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) dataset. It achieves high performance in $\kappa_{\mathrm{SRME}}$ on [Matbench Discovery](https://matbench-discovery.materialsproject.org/), but its F1 score is unavailable due to a difference in the POTCAR version. Like `SevenNet-MF-ompa`, this model outperforms `SevenNet-l3i5` on most tasks, except for the isolated molecule energy.

[Download link for fully detailed checkpoint](https://figshare.com/articles/software/SevenNet_omat/28593938).

#### **Matbench Discovery**
* $\kappa_{\mathrm{SRME}}$: **0.221**

---
### **SevenNet-l3i5 (12Dec2024)**
> Model keywords: `7net-l3i5` | `SevenNet-l3i5`

This model increases the maximum spherical harmonic degree ($l_{\mathrm{max}}$) to 3, compared to `SevenNet-0`, which has an $l_{\mathrm{max}}$ of 2. While **l3i5** offers improved accuracy for various systems, it is approximately four times slower than `SevenNet-0`.

#### **Matbench Discovery**
| CPS  | F1 | $\kappa_{\mathrm{SRME}}$ | RMSD |
|:---:|:---:|:---:|:---:|
|0.714 |0.760|0.550|0.085|

---

### **SevenNet-0 (11Jul2024)**
> Model keywords:: `7net-0` | `SevenNet-0` | `7net-0_11Jul2024` | `SevenNet-0_11Jul2024`

This model is one of our earliest pretrained models. Although we recommend using newer and more accurate models, it can still be useful in certain cases due to its shortest inference time. The model was trained on the [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) and is loaded as the default pretrained model in the ASE calculator.
For more information, click [here](sevenn/pretrained_potentials/SevenNet_0__11Jul2024).

#### **Matbench Discovery**
| F1 | $\kappa_{\mathrm{SRME}}$ |
|:---:|:---:|
|0.67|0.767|

---
You can find our legacy models in [pretrained_potentials](./sevenn/pretrained_potentials).

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
- PyTorch >= 2.0.0, PyTorch =< 2.5.2
- [Optional] cuEquivariance >= 0.4.0

For CUDA version, refer to PyTorch's compatibility matrix: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix

> [!IMPORTANT]
> Please install PyTorch manually based on your hardware before installing SevenNet.

Once PyTorch is successfully installed, please run the following command:
```bash
pip install sevenn
pip install git+https://github.com/MDIL-SNU/SevenNet.git # for the latest main branch
```

For cuEquivariance
```bash
pip install sevenn --extra cueq12  # cueq11 for CUDA version 11.*
```
The cuEquivariance can be enabled using `--enable_cueq` when training with `sevenn` via command line, and by setting `enable_cueq=True` in the `SevenNetCalculator`.
Note that you need Python version >= 3.10 to use cuEquivariance.

## Usage<a name="usage"></a>
### ASE calculator<a name="ase_calculator"></a>

SevenNet provides an ASE interface via the ASE calculator. Models can be loaded using the following Python code:
```python
from sevenn.calculator import SevenNetCalculator
# The 'modal' argument is required if the model is trained with multi-fidelity learning enabled.
calc_mf_ompa = SevenNetCalculator(model='7net-mf-ompa', modal='mpa')
```
SevenNet also supports CUDA-accelerated D3 calculations.
```python
from sevenn.calculator import SevenNetD3Calculator
calc = SevenNetD3Calculator(model='7net-0', device='cuda')
```
If you encounter the error `CUDA is not installed or nvcc is not available`, please ensure the `nvcc` compiler is available. Currently, CPU + D3 is not supported.

Various pretrained SevenNet models can be accessed by setting the model variable to predefined keywords like `7net-mf-ompa`, `7net-omat`, `7net-l3i5`, and `7net-0`.

The following table provides **approximate** maximum atom counts of **A100 GPU (80GB)** in a bulk system.
| Model | Max atoms |
|:---:|:---:|
|7net-0|~ 21,500|
|7net-l3i5|~ 9,300|
|7net-omat|~ 5,300|
|7net-mf-ompa|~ 3,300|

Note that this value depends on the target system. The limitation can be overcome by leveraging multi-GPUs with LAMMPS.

Additionally, user-trained models can be applied with the ASE calculator. In this case, the `model` parameter should be set to the checkpoint path from training.

> [!TIP]
> When 'auto' is passed to the `device` parameter (the default setting), SevenNet utilizes GPU acceleration if available.

### Training and inference

SevenNet provides five commands for preprocessing, training, and deployment: `sevenn_preset`, `sevenn_graph_build`, `sevenn`, `sevenn_inference`, and `sevenn_get_model`.

#### 1. Input generation

With the `sevenn_preset` command, the input file setting the training parameters is generated automatically.
```bash
sevenn_preset {preset keyword} > input.yaml
```

Available preset keywords are: `base`, `fine_tune`, `multi_modal`, `sevennet-0`, and `sevennet-l3i5`.
Check comments in the preset YAML files for explanations. For fine-tuning, be aware that most model hyperparameters cannot be modified unless explicitly indicated.
To reuse a preprocessed training set, you can specify `sevenn_data/${dataset_name}.pt` for the `load_trainset_path:` in the `input.yaml`.

#### 2. Preprocess (optional)

To obtain the preprocessed data, `sevenn_data/graph.pt`, `sevenn_graph_build` command can be used.
The output files can be used for training (`sevenn`) or inference (`sevenn_inference`) to skip the graph build stage.

```bash
sevenn_graph_build {dataset path} {cutoff radius}
```

The output `sevenn_data/graph.yaml` contains statistics and meta information about the dataset.
These files must be located in the `sevenn_data` directory. If you move the dataset, move the entire `sevenn_data` directory without changing the contents.

See `sevenn_graph_build --help` for more information.

#### 3. Training

Given that `input.yaml` and `sevenn_data/graph.pt` are prepared, SevenNet can be trained by the following command:

```bash
sevenn input.yaml -s
```

We support multi-GPU training using PyTorch DDP (distributed data parallel) with a single process (or a CPU core) per GPU.

```bash
torchrun --standalone --nnodes {number of nodes} --nproc_per_node {number of GPUs} --no_python sevenn input.yaml -d
```

Please note that `batch_size` in `input.yaml` refers to the per-GPU batch size.

#### 4. Inference

Using the checkpoint after training, the properties such as energy, force, and stress can be inferred directly.

```bash
sevenn_inference checkpoint_best.pth path_to_my_structures/*
```

This will create the `sevenn_infer_result` directory, where CSV files contain predicted energy, force, stress, and their references (if available).
See `sevenn_inference --help` for more information.

#### 5. Deployment<a name="deployment"></a>

The checkpoint can be deployed as LAMMPS potentials. The argument is either the path to the checkpoint or the name of a pretrained potential.

```bash
sevenn_get_model 7net-0  # For pre-trained models
sevenn_get_model {checkpoint path}  # For user-trained models
```

This will create `deployed_serial.pt`, which can be used as a LAMMPS potential with the `e3gnn` pair_style in LAMMPS.

The potential for parallel MD simulation can be obtained similarly.

```bash
sevenn_get_model 7net-0 -p
sevenn_get_model {checkpoint path} -p
```

This will create a directory with several `deployed_parallel_*.pt` files. The directory path itself is an argument for the LAMMPS script. Please do not modify or remove files in the directory.
These models can be used as LAMMPS potentials to run parallel MD simulations with a GNN potential across multiple GPUs.

### Notebook tutorials<a name="notebook-tutorial"></a>

If you want to learn how to use the `sevenn` Python library instead of the CLI command, please check out the notebook tutorials below.

| Notebooks | Google&nbsp;Colab | Descriptions |
|-----------|-------------------|--------------|
|[From scratch](https://github.com/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_python_tutorial.ipynb)|[![Open in Google Colab]](https://colab.research.google.com/github/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_python_tutorial.ipynb)|We can learn how to train the SevenNet from scratch, predict energy, forces, and stress using the trained model, perform structure relaxation, and draw EOS curves.|
|[Fine-tuning](https://github.com/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_finetune_tutorial.ipynb)|[![Open in Google Colab]](https://colab.research.google.com/github/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_finetune_tutorial.ipynb)|We can learn how to fine-tune the SevenNet and compare the results of the pretrained model with the fine-tuned model.|

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg

Sometimes, the Colab environment may crash due to memory issues. If you have sufficient GPU resources in your local environment, we recommend downloading the tutorials from GitHub and running them on your machine.
```bash
git clone https://github.com/MDIL-SNU/sevennet_tutorial.git
```

### MD simulation with LAMMPS

#### Installation

##### Requirements
- PyTorch (it is recommended to use the same version as used during training)
- LAMMPS version of `stable_2Aug2023_update3`
- MKL library
- [`CUDA-aware OpenMPI`](https://www.open-mpi.org/faq/?category=buildcuda) for parallel MD (optional)

If your cluster supports the Intel MKL module (often included with Intel OneAPI, Intel Compiler, and other Intel-related modules), load that module.

CUDA-aware OpenMPI is optional but recommended for parallel MD. If it is not available, GPUs will communicate via the CPU when running in parallel mode. It is still faster than using only one GPU, but its efficiency is lower.

> [!IMPORTANT]
> CUDA-aware OpenMPI does not support NVIDIA gaming GPUs. Since the software is closely tied to hardware specifications, please consult your server administrator if CUDA-aware OpenMPI is unavailable.

###### 1. Build LAMMPS with cmake.

Ensure the LAMMPS version is `stable_2Aug2023_update3`. You can easily switch the version using Git. After switching the version, run `sevenn_patch_lammps` with the LAMMPS directory path as an argument.

```bash
git clone https://github.com/lammps/lammps.git lammps_sevenn --branch stable_2Aug2023_update3 --depth=1
sevenn_patch_lammps ./lammps_sevenn {--d3}
```
You can refer to `sevenn/pair_e3gnn/patch_lammps.sh` for details of the patch process.

> [!TIP]
> Add `--d3` option to install GPU-accelerated [Grimme's D3 method](https://doi.org/10.1063/1.3382344) pair style. For its usage and details, click [here](sevenn/pair_e3gnn).

```bash
cd ./lammps_sevenn
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
make -j4
```

If the error `MKL_INCLUDE_DIR NOT-FOUND` occurs, please check the environment variable or read the `Possible solutions` section below.
If compilation completes without any errors, please skip this.

<details>
<summary><b>Possible solutions</b></summary>

###### 2. Install mkl-include via conda

```bash
conda install -c intel mkl-include
conda install mkl-include # if the above failed
```

###### 3. Append `DMKL_INCLUDE_DIR` to the cmake command and repeat step 1

```bash
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DMKL_INCLUDE_DIR=$CONDA_PREFIX/include
```

If the `undefined reference to XXX` error with `libtorch_cpu.so` occurs, check the `$LD_LIBRARY_PATH`.
If PyTorch is installed using Conda, `libmkl_*.so` files can be found in `$CONDA_PREFIX/lib`.
Ensure that `$LD_LIBRARY_PATH` includes `$CONDA_PREFIX/lib`.

For other error cases, solution can be found in the [`pair-nequip`](https://github.com/mir-group/pair_nequip) repository, as we share the same architecture.

</details>

If the compilation is successful, the executable `lmp` can be found at `{path_to_lammps_dir}/build`.
To use this binary easily, create a soft link to your bin directory, which should be included in your `$PATH`.

```bash
ln -s {absolute_path_to_lammps_directory}/build/lmp $HOME/.local/bin/lmp
```

This allows you to run the binary using `lmp -in my_lammps_script.lmp`.

#### Single-GPU MD

For single-GPU MD simulations, the `e3gnn` pair_style should be used. A minimal input script is provided below:
```txt
units       metal
atom_style  atomic
pair_style  e3gnn
pair_coeff  * * {path to serial model} {space separated chemical species}
```

#### Multi-GPU MD

For multi-GPU MD simulations, the `e3gnn/parallel` pair_style should be used. A minimal input script is provided below:
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
The number of message-passing layers corresponds to the number of `*.pt` files in the `./deployed_parallel` directory.

To deploy LAMMPS models from checkpoints for both serial and parallel execution, use [`sevenn_get_model`](#deployment).

It is expected that there is one GPU per MPI process. If the number of available GPUs is less than the number of MPI processes, the simulation may run inefficiently.

> [!CAUTION]
> Currently, the parallel version encounters an error when one of the subdomain cells contains no atoms. This issue can be addressed using the `processors` command and, more effectively, the `fix balance` command in LAMMPS. A patch for this issue will be released in a future update.

### Application of SevenNet-0
If you are interested in practical applications of SevenNet, you may want to check [this paper](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00025d) (data available on [Zenodo](https://doi.org/10.5281/zenodo.15205477)).
This study utilized SevenNet-0 for simulating liquid electrolytes.

The fine-tuning procedure and associated input files are accessible through the links above, specifically within the `Fine-tuning.tar.xz` archive on Zenodo.

The YAML file used for fine-tuning can be obtained using the following command:
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

If you utilize the multi-fidelity feature of this code or the pretrained model SevenNet-MF-ompa, please cite the following paper:
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
