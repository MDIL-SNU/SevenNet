# Accelerators

## CuEquivariance

### Requirements
- Python >= 3.10
- cuEquivariance >= 0.6.1

Install via:

```bash
pip install sevenn[cueq12]  # cueq11 for CUDA version 11.*
```

**Notes:**
* Some GeForce GPUs do not support `pynvml`, causing
  `pynvml.NVMLError_NotSupported: Not Supported`
  Try a lower cuEquivariance version, such as 0.6.1.


## FlashTP

### Requirements
- Python >= 3.10
- flashTP >= 0.1.0

Follow the installation guide of flashTP:

[https://github.com/SNU-ARC/flashTP/](https://github.com/SNU-ARC/flashTP)

**Notes:**
* during installation of flashTP
  `subprocess.CalledProcessError: ninja ... exit status 137`
  typically indicaets **out-of-memory** during comlation.
  Try reducing the build parallelsim.
```bash
export MAX_JOBS=1
```
