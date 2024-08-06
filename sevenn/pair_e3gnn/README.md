We support the use of the Grimme's D3 dispersion (van der Waals) correction scheme accelerated by CUDA, which can be used within LAMMPS in conjunction with SevenNet.

**PLEASE NOTE:** Currently, this D3 code does not support GPU parallelism yet. So it can only be run on a single GPU.

# About Grimme's D3 code accelerated with CUDA 

This is LAMMPS implementation of [Grimme's D3 method](https://doi.org/10.1063/1.3382344). We have ported the code from the [original fortran code](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3) to a LAMMPS pair style written in C++. The D3 method is semi-empirical and significantly faster than DFT, but it runs slower when combined with SevenNet. Therefore, we have adopted CUDA to accelerate the code.

We tested various implementations using MPI, OpenMP, OpenACC, and CUDA to accelerate the code. Among these, OpenACC and CUDA proved to be the most efficient. However, since OpenACC is not compatible with libtorch, we chose to use the CUDA version.

Initially, we implemented the code using double precision (FP64), similar to the original, but to further speed up the calculations, we transitioned some operations to single precision (FP32). Consequently, this code may introduce minor numerical errors, particularly noticeable in stress calculations. However, these errors are significantly smaller than the numerical errors inherent in SevenNet itself, so we have decided to use FP32-based operations.

# Installation of D3 code for LAMMPS

We support the installation of D3 code with SevenNet via the `sevenn_patch_lammps` command. You can simply add the `--d3` argument:

```bash
sevenn_patch_lammps ./lammps_sevenn --d3
```

You can follow the remaining installation steps in the SevenNet documentation. For detailed installation options, refer to `sevenn/pair_e3gnn/patch_lammps.sh`.

**PLEASE NOTE**: Currently, this installation code sets the target compute capability using `CMAKE_CUDA_ARCHITECTURES="50;80;86;89;90"`, which is the default setting for CUDA 12.1 as used by libtorch. While libtorch can automatically detect and configure your system's compute capability, our D3 code installation code does not support automatic detection. Therefore, if you want to specify a different compute capability for D3 code to optimize performance, you can do so by adding `-D CMAKE_CUDA_ARCHITECTURES="{ARCH}"` at the end of your cmake commands. For your information, you can specify the compute capability for libtorch by setting the environment variable `TORCH_CUDA_ARCH_LIST`.

**PLEASE NOTE**: Setting `fmad=false` for the NVCC compiler is essential to obtain precise results. This is the default option as described in patch_lammps.sh.

# Usage for LAMMPS

You can use the D3 dispersion correction in LAMMPS with SevenNet by using the `pair/hybrid` command:

```txt
pair_style pair/hybrid e3gnn d3 {cutoff_d3} {cutoff_d3_CN} {type_of_damping}
pair_coeff e3gnn * * {path_to_serial_model} {space_separated_chemical_species}
pair_coeff d3 * * {path_of_r0ab.csv} {path_of_d3_pars.csv} {name_of_functional} {space_separated_chemical_species}
compute {name_of_your_compute} all pressure NULL virial pair/hybrid
atom_modify sort 0 0 # Adhoc solution for the issue on 3Aug2023, which will be resolved soon
```

for example,

```txt
pair_style pair/hybrid e3gnn d3 9000 1600 d3_damp_bj
pair_coeff e3gnn * * ./deployed_serial.pt C H O
pair_coeff d3 * * ./r0ab.csv ./d3_pars.csv pbe C H O
compute vp_d3 all pressure NULL virial pair/hybrid d3
atom_modify sort 0 0
```

You can find `r0ab.csv` and `d3_pars.csv` files in the `pair_e3gnn` directory. These files are necessary to calculate D3 interactions.

`cutoff_d3` and `cutoff_d3_CN` are *square* of cutoff radii for energy/force and coordination number, respectively. Units are Bohr radius: 1 (Bohr radius) = 0.52917721 (Å). Default values are `9000` and `1600`, respectively. this is also the default values used in VASP. check the [note](#note-the-default-values-of-cutoff-parameters) for more informaiton.

Available `type_of_damping` are as follows:
- `d3_damp_zero`: Zero damping
- `d3_damp_bj`: Becke-Johnson damping

`name_of_functional` should match the exchange-correlation functional of the DFT calculations in the training set for the SevenNet potentials. For instance, SevenNet-0 was trained using the PBE functional, so you should specify `pbe` when using it.

`compute {name_of_your_compute} all pressure NULL virial pair/hybrid` is necesarry option for some simualtions. check the [note](#note-to-use-pair_d3-with-lammps-hybrid-hybridoverlay)

## Note: Defaults values of cutoff parameters
In [VASP DFT-D3](https://www.vasp.at/wiki/index.php/DFT-D3) page, `VDW_RADIUS` and `VDW_CNRADIUS` are `50.2` and `20.0`, respectively (units are Å). But you can check the default value of these in OUTCAR: `50.2022` and `21.1671`, which is same to default values of this code. To check this by yourself, run VASP with D3 using zero damping (BJ does not give such log).

## Note: To use `pair_d3` with LAMMPS `hybrid`, `hybrid/overlay`
In case you are doing calculation where pressure also affects simulation (e.g., NPT molecular dynamics simulation, geometry optimization with cell relax): ***you must add `compute (name_of_your_compute) all pressure NULL virial pair/hybrid d3` to your lammps input script.***

In D3, the result of computation (energy, force, stress) will be updated to actual variables in `update` function:
```cpp
void PairD3::update(int eflag, int vflag) {
    int n = atom->natoms;
    // Energy update
    if (eflag) { eng_vdwl += disp_total * AU_TO_EV; }

    double** f_local = atom->f;       // Local force of atoms
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            f_local[i][j] += f[i][j] * AU_TO_EV / AU_TO_ANG;
        }
    }

    // Stress update
    if (vflag) {
        virial[0] += sigma[0][0] * AU_TO_EV;
        virial[1] += sigma[1][1] * AU_TO_EV;
        virial[2] += sigma[2][2] * AU_TO_EV;
        virial[3] += sigma[0][1] * AU_TO_EV;
        virial[4] += sigma[0][2] * AU_TO_EV;
        virial[5] += sigma[1][2] * AU_TO_EV;
    }
}
```
In this code, virial stresses are updated only when `vflag` turned on.
However, the `pair_hybrid.cpp` in `lammps/src` explains how virial stresses are calculated in hybrid style:
```cpp
/* ----------------------------------------------------------------------
  call each sub-style's compute() or compute_outer() function
  accumulate sub-style global/peratom energy/virial in hybrid
  for global vflag = VIRIAL_PAIR:
    each sub-style computes own virial[6]
    sum sub-style virial[6] to hybrid's virial[6]
  for global vflag = VIRIAL_FDOTR:
    call sub-style with adjusted vflag to prevent it calling
      virial_fdotr_compute()
    hybrid calls virial_fdotr_compute() on final accumulated f
------------------------------------------------------------------------- */
```
Here, `compute pressure` with `pair/hybrid` will switch on the `VIRIAL_PAIR` style, so the virial stress will accumulated to `hybrid/overlay`.
Otherwise, `VIRIAL_FDOTR` may be turned on (which may skip the `vflag` part in `update` function; is it default?) and will give some errorneous value (computed from accumulated forces).   
The `pair_style` after `compute pressure` can be any pair_style; only the `VIRIAL_PAIR` matters in this case.

# To do
- Remove atom_modify dependency.
- Add support for ASE as calculator interface.
- Add support for multi GPUs (with `e3gnn/parallel`).
- Implement without Unified Memory.
- Unfix the `threadsPerBlock=128`.
- Unroll the repetition loop `k` (for small number of atoms).

# Features
- Selective or no periodic boundary condition: implemented (But only PBC/noPBC can be checked through original FORTRAN code; selective PBC cannot)
- 3-body term, n > 8 term: not implemented (Same condition to VASP)
- Modified versions of zero and bj damping

# Cautions
- It can be slower than the CPU with a small number of atoms.
- The CUDA math library differs from C, which can lead to numerical errors.
- The maximum number of atoms that can be calculated is 46,340 (overflow issue).

# Contributors
- Hyungmin An: Ported original Fortran D3 code to C++.
- Gijin Kim: Ported C++ D3 code to CUDA and currently maintains it.
