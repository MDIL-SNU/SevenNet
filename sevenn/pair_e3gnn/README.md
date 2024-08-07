We support the LAMMPS pair style `d3` of the Grimme's D3 dispersion (van der Waals) correction scheme accelerated with CUDA, which can be used within LAMMPS in conjunction with SevenNet.

**PLEASE NOTE:** Currently, this D3 code does not support GPU parallelism yet. So it can only be run on a single GPU.

# About Grimme's D3 code accelerated with CUDA 

This is LAMMPS implementation of [Grimme's D3 method](https://doi.org/10.1063/1.3382344). We have ported the code from the [original fortran code](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3) to a LAMMPS pair style written in C++. The D3 method is semi-empirical and significantly faster than DFT, but it runs slower compared to SevenNet. To address this, we have adopted CUDA and single precision (FP32) operations to accelerate the code.

# Installation for LAMMPS

We support the installation of D3 code with SevenNet via the `sevenn_patch_lammps` command. You can simply add the `--d3` argument:

```bash
sevenn_patch_lammps ./lammps_sevenn --d3
```

You can follow the remaining installation steps in the SevenNet documentation. For detailed installation options, refer to `sevenn/pair_e3gnn/patch_lammps.sh`.

Also, this code requires a GPU with a compute capability of **at least 6.0**. If you try to compile it with version 5.0, you may encounter an `atomicAdd` error. The target compute capability of this code follows the setting of LibTorch in SevenNet, except for version 5.0. You can manually select the target capability using the environment variable. For example, you can use: `export TORCH_CUDA_ARCH_LIST="60;71;80;86;89;90"`.

# Usage for LAMMPS

You can use the D3 dispersion correction in LAMMPS with SevenNet through the `pair/hybrid` command:

```txt
pair_style hybrid/overlay e3gnn d3 {cutoff_d3} {cutoff_d3_CN} {type_of_damping}
pair_coeff e3gnn * * {path_to_serial_model} {space_separated_chemical_species}
pair_coeff d3 * * {path_of_r0ab.csv} {path_of_d3_pars.csv} {name_of_functional} {space_separated_chemical_species}

# Adhoc solution for the invalid pressure calculation. (VIRIAL_PAIR <- VIRIAL_FDOTR by calling compute pressure)
compute {name_of_your_compute} all pressure NULL virial pair/hybrid d3
# Adhoc solution for the issue of messed atom order on 3Aug2023. 
atom_modify sort 0 0 
```

for example,

```txt
pair_style hybrid/overlay e3gnn d3 9000 1600 d3_damp_bj
pair_coeff e3gnn * * ./deployed_serial.pt C H O
pair_coeff d3 * * ./r0ab.csv ./d3_pars.csv pbe C H O
compute vp_d3 all pressure NULL virial pair/hybrid d3
atom_modify sort 0 0
```

You can find `r0ab.csv` and `d3_pars.csv` files in the `pair_e3gnn` directory. These files are necessary to calculate D3 interactions.

`cutoff_d3` and `cutoff_d3_CN` are square of cutoff radii for energy/force and coordination number, respectively. Units are Bohr radius: 1 (Bohr radius) = 0.52917721 (Å). Default values are `9000` and `1600`, respectively. this is also the default values used in VASP.

Available `type_of_damping` are as follows:
- `d3_damp_zero`: Zero damping
- `d3_damp_bj`: Becke-Johnson damping

Available `name_of_functional` options are the same as in the original Fortran code. SevenNet-0 is trained on the 'PBE' functional, so you should specify 'pbe' in the script when using it.

## Note: Default cutoff parameters
On the [VASP DFT-D3](https://www.vasp.at/wiki/index.php/DFT-D3) page, the `VDW_RADIUS` and `VDW_CNRADIUS` are `50.2` and `20.0`, respectively (units are Å). However, when running VASP 6.3.2 with D3 using zero damping (BJ does not provide such a log), the default values in the OUTCAR file are `50.2022` and `21.1671`. These values are the same as our defaults.

# Features
- Selective(or no) periodic boundary condition: implemented, But only PBC/noPBC can be checked through original FORTRAN code; selective PBC cannot
- 3-body term, n > 8 term: not implemented (as to VASP)
- Modified versions of zero and bj damping

# Cautions
- It can be slower than the CPU with a small number of atoms.
- The maximum number of atoms that can be calculated is 46,340 (overflow issue).
- There can be occured small amounts of numerical error
  - The introduction of some FP32 operations can lead to minor numerical errors, particularly in pressure calculations, but these are generally smaller than those seen with SevenNet.
  - If the error is too large, ensure that the `fmad=false` option in `patch_lammps.sh` is correctly applied during build.

# To do
- Remove atom_modify / compute virial dependency.
- Add support for ASE as calculator interface.
- Add support for multi GPUs (with `e3gnn/parallel`).
- Implement without Unified Memory.
- Unfix the `threadsPerBlock=128`.
- Unroll the repetition loop `k` (for small number of atoms).

OpenACC does not support 

# Contributors
- Hyungmin An: Ported the original Fortran D3 code to C++ with OpenMP and MPI.
- Gijin Kim: Accelerated the C++ D3 code with OpenACC[^1] and CUDA, and currently maintains it.

[^1] Since OpenACC is not compatible with libtorch, we chose to use the CUDA.