## ML-IAP
### Requirements
- LAMMPS with ML-IAP integration is only available with Kokkos GPU support
- You must use the latest version of LAMMPS  
  **(WARNING: previous `stable_2Aug2023_update3` version LAMMPS does not support ML-IAP integration)**


### Build
First, install the required Python dependencies:
```bash
pip install cython==3.0.11
pip install cupy-cuda12x
```

Clone the latest version of LAMMPS:
```bash
git clone https://github.com/lammps/lammps lammps-mliap --depth=1
```

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


### Potential deployment
An ML-IAP potential checkpoint can be deployed using ``sevenn_get_model`` command with ``--use_mliap`` flag.
- By default, output file name will be ``deployed_serial_mliap.pt``.  
  (You can customize the output file name using ``--output_prefix`` flag.)
- You can accelerate the process with ``enable_cueq`` or ``enable_flashTP`` flag:
```bash
sevenn_get_model \
    {pretrained_name or checkpoint_path} \
    --use_mliap \
    --modal {modal_name}                    # Optional (Depends on your model selection)
```


### LAMMPS ML-IAP Script Example
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
    pair_coeff      * * {space seperated chemical species}
```

Example command to run a LAMMPS simulation with ML-IAP:
```bash
/path/to/lammps-mliap/build/lmp -in lammps.in -k on g 1 -sf kk -pk kokkos newton on neigh half
```
