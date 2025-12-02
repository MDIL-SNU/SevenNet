# Accelerators

## CuEquivariance

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

### Requirements
- Python >= 3.10
- flashTP >= 0.1.0

Follow the installation guide of [flashTP](https://github.com/SNU-ARC/flashTP).

:::{note}
During installation of flashTP,
`subprocess.CalledProcessError: ninja ... exit status 137`
typically indicates **out-of-memory** during compilation.
Try reducing the build parallelism:
```bash
export MAX_JOBS=1
```
:::
