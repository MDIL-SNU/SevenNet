# LAMMPS/Torch

## Requirements
- LAMMPS version of `stable_2Aug2023_update3`
- [`CUDA-aware OpenMPI`](https://www.open-mpi.org/faq/?category=buildcuda) for faster parallel MD (optional)

CUDA-aware OpenMPI is optional but recommended for parallel MD. If it is not available, GPUs will communicate via the CPU when running in parallel mode. It is still faster than using only one GPU, but its efficiency is lower.

:::{important}
CUDA-aware OpenMPI does not support NVIDIA gaming GPUs. Since the software is closely tied to hardware specifications, please consult your server administrator if CUDA-aware OpenMPI is unavailable.
:::

## Build

Ensure the LAMMPS version is `stable_2Aug2023_update3`. You can easily switch the version using Git. After switching the version, run `sevenn patch_lammps` with the LAMMPS directory path as an argument.

```bash
git clone https://github.com/lammps/lammps.git lammps_sevenn --branch stable_2Aug2023_update3 --depth=1
sevenn patch_lammps ./lammps_sevenn {--d3}
```
You can refer to `sevenn/pair_e3gnn/patch_lammps.sh` for details of the patch process.

:::{tip}
Add `--d3` option to install GPU-accelerated [Grimme's D3 method](https://doi.org/10.1063/1.3382344) pair style. For its usage and details, see the {doc}`../user_guide/d3`.
:::

:::{tip}
Add `--flashTP` option to install SevenNet with flashTP for LAMMPS. You must preinstall [flashTP(Optional)](../install/accelerator.md#flashtp) before building LAMMPS with flashTP.
:::

```bash
cd ./lammps_sevenn
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
make -j4
```

If the error `MKL_INCLUDE_DIR NOT-FOUND` occurs, you can use a dummy directory.
```bash
-DMKL_INCLUDE_DIR=/tmp
```

“Undefined reference” errors can be caused by missing linked libraries, incorrect link order, ABI mismatches, or missing dependent shared libraries.
If the missing symbol lives in a shared library, ensure the library’s location is included in $LD_LIBRARY_PATH or CMake RPATH settings.

For other error cases, solution can be found in the [`pair-nequip`](https://github.com/mir-group/pair_nequip) repository, as we share the same architecture.

If the compilation is successful, the executable `lmp` can be found at `{path_to_lammps_dir}/build`.
To use this binary easily, create a soft link to your bin directory, which should be included in your `$PATH`.

```bash
ln -s {absolute_path_to_lammps_directory}/build/lmp $HOME/.local/bin/lmp
```

This allows you to run the binary using `lmp -in my_lammps_script.lmp`.

## Usage

### Single-GPU MD

For single-GPU MD simulations, the `e3gnn` pair_style should be used. A minimal input script is provided below:
```text
units       metal
atom_style  atomic
pair_style  e3gnn
pair_coeff  * * {path to serial model} {space separated chemical species}
```

### Multi-GPU MD

For multi-GPU MD simulations, the `e3gnn/parallel` pair_style should be used. A minimal input script is provided below:

```text
units       metal
atom_style  atomic
pair_style  e3gnn/parallel
pair_coeff  * * {number of message-passing layers} {directory of parallel model} {space separated chemical species}
```

For example,

```text
pair_style e3gnn/parallel
pair_coeff * * 4 ./deployed_parallel Hf O
```
The number of message-passing layers corresponds to the number of `*.pt` files in the `./deployed_parallel` directory.

To deploy LAMMPS models from checkpoints for both serial and parallel execution, use {ref}`sevenn get_model<sevenn-get-model>`.

It is expected that there is one GPU per MPI process. If the number of available GPUs is less than the number of MPI processes, the simulation may run inefficiently.

:::{caution}
Currently, the parallel version encounters an error when one of the subdomain cells contains no atoms. This issue can be addressed using the `processors` command and, more effectively, the `fix balance` command in LAMMPS. A patch for this issue will be released in a future update.
:::

:::{caution}
Currently, our D3 algorithm is not supported by multi-GPU.
:::