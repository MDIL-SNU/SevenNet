# LAMMPS: ML-IAP

:::{caution}
Currently the parallel implementation of LAMMPS/ML-IAP is not tested.
:::

## Requirements
- cython == 3.0.11
- cupy-cuda12x
- flashTP (optional, follow [here](accelerator.md#flashtp))
- cuEquivariance (optional, follow [here](accelerator.md#cuequivariance))

Install via:
```bash
pip install sevenn[mliap]
```

## Build
Get LAMMPS source code:
```bash
git clone https://github.com/lammps/lammps lammps-mliap
cd lammps-mliap
git checkout ccca772
```
:::{note}
We found that some of the latest versions of LAMMPS produce inconsistent energies. Therefore, we highly recommend using this specific commit. This restriction will be relaxed once consistency checks are completed.
:::


Configure the LAMMPS build with the necessary options:
```bash
cd ./lammps-mliap
cmake \
    -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=$(pwd)/lib/kokkos/bin/nvcc_wrapper \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D BUILD_MPI=ON \
    -D PKG_ML-IAP=ON \
    -D PKG_ML-SNAP=ON \
    -D MLIAP_ENABLE_PYTHON=ON \
    -D PKG_PYTHON=ON \
    -D BUILD_SHARED_LIBS=ON \
    cmake
```

Build LAMMPS and install the Python bindings:
```bash
cmake --build build -j 8
cd build
make install-python
```
:::{note}
If the compilation fails, consider specifying the GPU architecture of the node you are building on in KOKKOS.
For example, add the following flag to your `cmake` command: `-D KOKKOS_ARCH_AMPERE86=ON` when you are using GPU with Ampere 86 architecture like RTX A5000.
:::

### (Optional) Build with GPU-D3 pair style
To build LAMMPS with the GPU-accelerated [Grimme's D3 method](https://doi.org/10.1063/1.3382344) (see {doc}`../user_guide/d3`), follow the steps below:
1. Clone the SevenNet source repository.
```bash
git clone https://github.com/MDIL-SNU/SevenNet.git
```

2. Copy the `pair_d3`-related scripts into `lammps/src`.
```bash
cp {sevenn_src_path}/sevenn/pair_e3gnn/{pair_d3.cu,pair_d3.h,pair_d3_pars.h} {lammps_src_path}/src/
```

3. Modify `{lammps_src_path}/cmake/CMakeLists.txt`. Lines starting with `-` indicate the original content, and lines starting with `+` indicate the modified or added content.
```
# Declare CUDA at the header. This part depends to LAMMPS version.
 project(lammps
         DESCRIPTION "The LAMMPS Molecular Dynamics Simulator"
         HOMEPAGE_URL "https://www.lammps.org"
-        LANGUAGES CXX C)
+        LANGUAGES CXX C CUDA)

...

# Compile pair_d3.cu
-file(GLOB ALL_SOURCES CONFIGURE_DEPENDS ${LAMMPS_SOURCE_DIR}/[^.]*.cpp)
+file(GLOB ALL_SOURCES CONFIGURE_DEPENDS ${LAMMPS_SOURCE_DIR}/[^.]*.cpp  ${LAMMPS_SOURCE_DIR}/[^.]*.cu)

...

# At the end of the CMakeLists.txt, add cuda-related flags
+find_package(CUDA)
+set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fmad=false -O3 --expt-relaxed-constexpr")
+target_link_libraries(lammps PUBLIC ${CUDA_LIBRARIES} cuda)
```

4. Add the target CUDA architecture in the `cmake` configuration. An example configuration is shown below.
```bash
cmake \
    -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=$(pwd)/lib/kokkos/bin/nvcc_wrapper \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ARCH_AMPERE89=ON \  # for ML-IAP Kokkos implementation
    -D BUILD_MPI=ON \
    -D PKG_ML-IAP=ON \
    -D PKG_ML-SNAP=ON \
    -D MLIAP_ENABLE_PYTHON=ON \
    -D PKG_PYTHON=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_CUDA_ARCHITECTURES="89" \  # for CUDA D3 pair style
    cmake
```

5. Build LAMMPS and install the Python bindings as above.

## Usage
### Potential deployment
Please check [sevenn graph_build](./cli.md#sevenn-graph-build) for detail.

An ML-IAP potential checkpoint can be deployed using ``sevenn get_model`` command with ``--use_mliap`` flag.
- By default, output file name will be ``deployed_serial_mliap.pt``.
  (You can customize the output file name using ``--output_prefix`` flag.)
- You can accelerate the inference with ``--enable_cueq`` or ``--enable_flashTP`` flag:
```bash
sevenn get_model \
    {pretrained_name or checkpoint_path} \
    --use_mliap \
    --modal {task_name}  # Required when using multi-fidelity model
```


### Single-GPU MD
Below is an example snippet of a LAMMPS script for using SevenNet models with ML-IAP:
```text
    # ---------- Initialization ----------
    units           metal
    boundary        p p p
    atom_style      atomic
    atom_modify     map yes
    newton          on

    read_data       {your_system_file_path}

    # ----- ML-IAP potential settings ----
    pair_style      mliap unified {your_potential_file_path.pt} 0
    pair_coeff      * * {space separated chemical species}
```

:::{note}
If the LAMMPS is built with GPU-D3 pair style, you can combine SevenNet with D3 through the `pair/hybrid` command as the example below. For detailed instruction about parameters, supporting functionals and damping types, refer to {doc}`../user_guide/d3`.
```
pair_style hybrid/overlay mliap unified {your_potential_file_path.pt} 0 d3 9000 1600 damp_bj pbe
pair_coeff      * * mliap {space separated chemical species}
pair_coeff      * * d3    {space separated chemical species}
```
:::

Example command to run a LAMMPS simulation with ML-IAP:
```bash
/path/to/lammps-mliap/build/lmp -in lammps.in -k on g 1 -sf kk -pk kokkos newton on neigh half
```
