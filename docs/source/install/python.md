## Python
### Requirements
- Python >= 3.8
- PyTorch >= 2.0.0, PyTorch =< 2.7.0
- [Optional] cuEquivariance >= 0.7.0

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
pip install sevenn[cueq12]  # cueq11 for CUDA version 11.*
```
The cuEquivariance can be enabled using `--enable_cueq` when training with `sevenn` via command line, and by setting `enable_cueq=True` in the `SevenNetCalculator`.
Note that you need Python version >= 3.10 to use cuEquivariance.

