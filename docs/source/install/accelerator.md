# Accelerators

CuEquivariance and flashTP provide acceleration for both SevenNet training and inference. For Benchmark results, follow [here](https://arxiv.org/abs/2510.11241)

## CuEquivariance

CuEquivariance is an NVIDIA Python library designed to facilitate the construction of high-performance geometric neural networks using segmented polynomials and triangular operations. For more information, refer to [cuEquivariance](https://github.com/NVIDIA/cuEquivariance).

### Requirements
- Python >= 3.10
- cuEquivariance >= 0.6.1

Install via:

```bash
pip install sevenn[cueq12]  # cueq11 for CUDA version 11.*
```

:::{note}
Some GeForce GPUs do not support `pynvml`,
causing `pynvml.NVMLError_NotSupported: Not Supported`.
Then try a lower cuEquivariance version, such as 0.6.1.
:::

## FlashTP

FlashTP is a high-performance Tensor-Product library for Machine Learning Interatomic Potentials (MLIPs). For more information and the installation guide, refer to [flashTP](https://github.com/SNU-ARC/flashTP).

### Requirements
- Python >= 3.10
- flashTP >= 0.1.0

:::{note}
During installation of flashTP,
`subprocess.CalledProcessError: ninja ... exit status 137`
typically indicates **out-of-memory** during compilation.
Try reducing the build parallelism:
```bash
export MAX_JOBS=1
```
:::
