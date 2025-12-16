# Accelerators

CuEquivariance and flashTP provide acceleration for both SevenNet training and inference. For inference speed benchmark results, check the section [2.7 Inference speed](https://arxiv.org/abs/2510.11241) of our paper.

## CuEquivariance

CuEquivariance is an NVIDIA Python library designed to facilitate the construction of high-performance geometric neural networks using segmented polynomials and triangular operations. CuEquivariance accelerates SevenNet during training, inference with ASE and LAMMPS via ML-IAP. For more information, refer to [cuEquivariance](https://github.com/NVIDIA/cuEquivariance).

### Requirements
- Python >= 3.10
- cuEquivariance >= 0.6.1

### Installation
Install via:

```bash
pip install sevenn[cueq12]  # cueq11 for CUDA version 11.*
```

:::{note}
Some GeForce GPUs do not support `pynvml`,
causing `pynvml.NVMLError_NotSupported: Not Supported`.
Then try a lower cuEquivariance version, such as 0.6.1.
:::




:::{note}
If `pip install sevenn[cueq12]` fails to install the latest version of SevenNet, try installing the base package instead:
```
pip install sevenn
```

If this successfully installs the latest version, the issue is likely related to **cuEquivariance compatibility**.  
You can verify this by installing cuEquivariance manually:
```
pip install cuequivariance-ops-torch-cu12
pip install cuequivariance-torch
```

For more details, see the [cuEquivariance documentation](https://github.com/NVIDIA/cuEquivariance).
:::


## FlashTP

FlashTP, presented in [FlashTP: Fused, Sparsity-Aware Tensor Product for Machine Learning Interatomic Potentials](https://openreview.net/forum?id=wiQe95BPaB), is a high-performance Tensor-Product library for Machine Learning Interatomic Potentials (MLIPs). FlashTP accelerates SevenNet during training and inference with ASE, LAMMPS (single & multi-GPU), LAMMPS via ML-IAP, providing up to ~4× speedup. For more information, refer to [flashTP](https://github.com/SNU-ARC/flashTP).

### Requirements
- Python >= 3.10
- flashTP >= 0.1.0
- CUDA toolkit >= 12

### Installation
Install via:
```bash
git clone https://github.com/SNU-ARC/flashTP.git
cd flashTP
pip install -r requirements.txt
CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation
```
Customize CUDA_ARCH_LIST to match the [compute compatibility](https://developer.nvidia.com/cuda/gpus) of the GPU.


:::{note}
During installation of flashTP,
`subprocess.CalledProcessError: ninja ... exit status 137`
typically indicates **out-of-memory** during compilation.
Try reducing the build parallelism:
```bash
export MAX_JOBS=1
```
:::

## CuEquivariance, flashTP usage

CuEquivariance and FlashTP can be used with:

| Feature | Training | ASE | LAMMPS via pytorch | LAMMPS via ML-IAP |
|--------------------|----------|-----|--------|--------|
| **cuEq** | [Training](./cli.md#sevenn-train) | [ASE](./ase_calculator.md) | — | [ML-IAP](./lammps_mliap.md#potential-deployment) |
| **flashTP** | [Training](./cli.md#sevenn-train) | [ASE](./ase_calculator.md) | [LAMMPS (single & multi-GPU)](./lammps_torch.md#build) | [ML-IAP](./lammps_mliap.md#potential-deployment) |

:::{caution}
Currently, among the available accelerators, only **flashTP without [d3](./d3.md)** supports multi-GPU LAMMPS.
:::
:::{note}
For small systems, flashTP with PyTorch shows a clear and consistent performance advantage.
A performance crossover occurs at around O(10³) atoms, beyond which cuEquivariance with ML-IAP becomes more efficient.

FlashTP with PyTorch is generally faster than flashTP with ML-IAP.
:::
