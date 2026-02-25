# Accelerators

This document describes available accelerator integrations in SevenNet and their installation guide.

:::{caution}
We do not support CuEquivariance for [LAMMPS: Torch](./lammps_mliap.md). You must use [LAMMPS: ML-IAP](./lammps_torch.md) for CuEquivariance.
:::

[CuEquivariance](https://github.com/NVIDIA/cuEquivariance) and [FlashTP](https://openreview.net/forum?id=wiQe95BPaB) provide acceleration for both SevenNet training and inference. (For speed, check the section 2.7 of [SevenNet-Omni paper](https://arxiv.org/abs/2510.11241))

:::{tip}
For small systems, FlashTP with [LAMMPS: Torch](./lammps_mliap.md) shows performance advantage over cuEquivariance with [LAMMPS: ML-IAP](./lammps_torch.md).
A performance crossover occurs at around 10³ atoms, beyond which cuEquivariance becomes more efficient.

FlashTP with [LAMMPS: Torch](./lammps_mliap.md) is generally faster than FlashTP with [LAMMPS: ML-IAP](./lammps_mliap.md).
:::

## [CuEquivariance](https://github.com/NVIDIA/cuEquivariance)

CuEquivariance is an NVIDIA Python library designed to facilitate the construction of high-performance geometric neural networks using segmented polynomials and triangular operations. CuEquivariance accelerates SevenNet during training, inference with ASE and LAMMPS via ML-IAP.

### Requirements
- Python >= 3.10
- cuEquivariance >= 0.6.1

### Installation
After installation of SevenNet, install cuEquivariance following their guideline.
To use cuEquivariance with SevenNet, you need to install `cuequivariance`, `cuequivariance-torch`, and `cuequivariance-ops-torch-cu{12, 13}` (depending on your CUDA version)
[cuEquivariance](https://github.com/NVIDIA/cuEquivariance).

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

## Usage
After the installation, you can leverage the accelerator with appropriate flag (`--enable_cueq`) or (`--enable_flash`) options

- [Training](./cli.md#sevenn-train)
- [ASE Calculator](./ase_calculator.md)
- [LAMMPS](./cli.md#sevenn-get-model)
