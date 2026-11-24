# Accelerators

This document describes available accelerator integrations in SevenNet and their installation guide.

:::{caution}
We do not support CuEquivariance and LunaTP for [LAMMPS: Torch](./lammps_torch.md). You must use [LAMMPS: ML-IAP](./lammps_mliap.md) for CuEquivariance and LunaTP.
:::

[FlashTP](https://openreview.net/forum?id=wiQe95BPaB), [CuEquivariance](https://github.com/NVIDIA/cuEquivariance), and [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance) provide acceleration for both SevenNet training and inference.

:::{tip}
For small systems, FlashTP with [LAMMPS: Torch](./lammps_torch.md) shows performance advantage over cuEquivariance with [LAMMPS: ML-IAP](./lammps_mliap.md).
A performance crossover occurs at around 10³ atoms, beyond which cuEquivariance becomes more efficient.  (For more information, check section 2.7 of the [SevenNet-Omni paper](https://doi.org/10.1038/s41467-026-70195-8))

FlashTP with [LAMMPS: Torch](./lammps_mliap.md) is generally faster than FlashTP with [LAMMPS: ML-IAP](./lammps_mliap.md).
:::

## [CuEquivariance](https://github.com/NVIDIA/cuEquivariance)

CuEquivariance is an NVIDIA Python library designed to facilitate the construction of high-performance geometric neural networks using segmented polynomials and triangular operations. CuEquivariance accelerates SevenNet during training, inference with ASE and LAMMPS via ML-IAP.

### Requirements
- Python >= 3.10
- cuEquivariance >= 0.6.1

### Installation
After installation of SevenNet, install cuEquivariance via the pip extras:

```bash
pip install sevenn[cueq12]  # For CUDA 12.x
pip install sevenn[cueq13]  # For CUDA 13.x
```

This will install `cuequivariance`, `cuequivariance-torch`, and the corresponding `cuequivariance-ops-torch-cu{12,13}`.

Alternatively, you can install cuEquivariance manually following [their guideline](https://github.com/NVIDIA/cuEquivariance).

:::{note}
Some GeForce GPUs do not support `pynvml`,
causing `pynvml.NVMLError_NotSupported: Not Supported`.
Then try a lower cuEquivariance version, such as 0.6.1.
:::

Check your installation:
```bash
python -c 'from sevenn.nn.cue_helper import is_cue_available; print(is_cue_available())'
True
```

## [FlashTP](https://github.com/SNU-ARC/flashTP)

FlashTP, presented in [FlashTP: Fused, Sparsity-Aware Tensor Product for Machine Learning Interatomic Potentials](https://openreview.net/forum?id=wiQe95BPaB), is a high-performance Tensor-Product library for Machine Learning Interatomic Potentials (MLIPs).

FlashTP accelerates SevenNet during both training and inference, achieving up to ~4× speedup.

### Requirements
- Python >= 3.10
- flashTP >= 0.1.0
- CUDA toolkit >= 12.0

### Installation
Choose `CUDA_ARCH_LIST` for your GPU(s) (see [compute compatibility](https://developer.nvidia.com/cuda/gpus))

```bash
git clone https://github.com/SNU-ARC/flashTP.git
cd flashTP
pip install -r requirements.txt
CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation
```

:::{note}
During installation of FlashTP,
`subprocess.CalledProcessError: ninja ... exit status 137`
typically indicates out-of-memory during compilation.
Try reducing the build parallelism:
```bash
export MAX_JOBS=1
```
:::

Check your installation:
```bash
python -c 'from sevenn.nn.flash_helper import is_flash_available; print(is_flash_available())'
True
```

For more information, see [FlashTP](https://github.com/SNU-ARC/flashTP).

## [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance)

OpenEquivariance (oEq) is a library for acceleration of equivariant tensor products. Unlike cuEquivariance, its kernels are compiled once per irreps configuration, eliminating runtime recompilation overhead during training on diverse datasets.

### Requirements
- Python >= 3.10
- openequivariance

### Installation
```bash
pip install sevenn[oeq]
```

Check your installation:
```bash
python -c 'from sevenn.nn.oeq_helper import is_oeq_available; print(is_oeq_available())'
True
```

For more information, see [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance).

## [LunaTP (experimental)](https://github.com/HPC-AI-Team/LunaEquivariance)

:::{caution}
LunaTP support in SevenNet is currently experimental.
:::

LunaTP (LunaEquivariance package) replaces the tensor product convolution with a fused kernel that accumulates the messages of each destination atom in registers before one store to global memory. SevenNet also replaces the linear layers, the self-connection tensor product, and the gate nonlinearity with their LunaEquivariance counterparts. The kernels are compiled at the first use of each irreps configuration and cached on disk.

### Requirements
- Python >= 3.10, PyTorch >= 2.8
- NVIDIA GPU with compute capability 8.0 or later, CUDA toolkit with `nvcc`
- LunaEquivariance 0.3.0

### Installation
```bash
pip install sevenn[lunatp]
```

Check your installation:
```bash
python -c 'from sevenn.nn.lunatp_helper import is_lunatp_available; print(is_lunatp_available())'
True
```

### Kernel compilation
Building a LunaTP model compiles its kernels in parallel processes (default: min(8, available CPUs)) before the first calculation. The compiled kernels are cached on disk, and later builds of the same model load them. The cache depends on the Python environment, `LD_LIBRARY_PATH`, and the GPU architecture.

`SEVENNET_LUNATP_PRECOMPILE_WORKERS` sets the number of processes. Each process loads the model on the GPU, and `0` turns the parallel compilation off, so that each kernel compiles at its first use. `sevenn train` also compiles the kernels for training with force and stress losses.

`sevenn lunatp_precompile` compiles the kernels of a checkpoint ahead of time, for example on a node other than the one of the later calculations:
```bash
sevenn lunatp_precompile 7net-omni           # kernels for inference (energy, forces, stress)
sevenn lunatp_precompile 7net-omni --train   # also the kernels for training with force and stress losses
```
`--max_workers` sets the number of parallel processes.

### Frozen-model kernels (`fused`)

For frozen models with a plain radial MLP, `fused` evaluates cubic Hermite
tables inside the LunaTP convolution kernels. These tables approximate the
original radial MLP. Fused inference can also specialize the first interaction
to the species present.

```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator('7net-omni', modal='mpa', enable_lunatp=True, fused=True)
```
For LAMMPS, see the `--fused` option of {ref}`sevenn get_model<sevenn-get-model>` with `--use_mliap --enable_lunatp --cuda_graph` (one MPI rank). The kernels are built at the first call and cached, as the other LunaTP kernels.

For more information, see [LunaEquivariance](https://github.com/HPC-AI-Team/LunaEquivariance).

## Usage
After the installation, you can leverage the accelerator with appropriate flag (`--enable_cueq`), (`--enable_flash`), (`--enable_oeq`), or (`--enable_lunatp`) options

- [Training](./cli.md#sevenn-train)
- [ASE Calculator](./ase_calculator.md)
- {ref}`LAMMPS<sevenn-get-model>`
