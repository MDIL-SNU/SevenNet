## LAMMPS (Torch)
### Requirements
- PyTorch (it is recommended to use the same version as used during training)
- LAMMPS version of `stable_2Aug2023_update3`
- MKL library
- [`CUDA-aware OpenMPI`](https://www.open-mpi.org/faq/?category=buildcuda) for parallel MD (optional)
- cmake

If your cluster supports the Intel MKL module (often included with Intel OneAPI, Intel Compiler, and other Intel-related modules), load that module.

CUDA-aware OpenMPI is optional but recommended for parallel MD. If it is not available, GPUs will communicate via the CPU when running in parallel mode. It is still faster than using only one GPU, but its efficiency is lower.

> [!IMPORTANT]
> CUDA-aware OpenMPI does not support NVIDIA gaming GPUs. Since the software is closely tied to hardware specifications, please consult your server administrator if CUDA-aware OpenMPI is unavailable.

### Build

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

If the error `MKL_INCLUDE_DIR NOT-FOUND` occurs, you can set it to some existing path:
```bash
-DMKL_INCLUDE_DIR=/tmp 
```

If the compilation is successful, the executable `lmp` can be found at `{path_to_lammps_dir}/build`.
To use this binary easily, create a soft link to your bin directory, which should be included in your `$PATH`.

```bash
ln -s {absolute_path_to_lammps_directory}/build/lmp $HOME/.local/bin/lmp
```

This allows you to run the binary using `lmp -in my_lammps_script.lmp`.
