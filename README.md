# SEVENNet

SEVENNet (Scalable EquiVariance Enabled Neural Network) is a graph nueral network interatomic potential pacakage that supports parallel moleculary dyanmics simulation on LAMMPS. The underlying GNN model is same as [`nequip`](https://github.com/mir-group/nequip).

Note that SEVENNet is under active development and not stable.

## REQUIREMENTS - training

* Python >= 3.8
* Pytorch >= 1.11
* TorchGeometric 
* [`e3nn`](https://github.com/e3nn/e3nn)

## INTSTALLATION - training

```
git clone https://github.com/MDIL-SNU/SEVENN.git
cd SEVENN
pip install . 
```

## USAGE - training

### start training:
```
sevenn input.yaml
```

You can find example of `input.yaml` from `SEVENN/example_inputs`
structure_list file is to select VASP OUTCARs for training.


### Get parallel models
After the training, you can find `deployed_model_best.pt` as serial model for MD simulation.
or, you can have parallel model from checkpoint.

```
sevenn_get_parallel checkpoint_best.pt
```

It will generate segemented parallel models with same number of message passing layers in model. You need all of them to run parallel MD.

## REQUIREMENTS - MD

Latest stable version of [`LAMMPS`](https://github.com/lammps/lammps)
CUDA-aware OpenMPI for parallel MD

You can check whether your OpenMPI awrae of CUDA or not by ompi_info.

```
ompi_info --all | grep btl_openib_have_cuda_gdr
  > MCA btl openib: informational "btl_openib_have_cuda_gdr" (current value: "true", data source: default, level: 5 tuner/detail, type: bool)
```

## INSTALLATION - MD

Note that following command will overwrite `comm_brickc.cpp` and `comm_brick.h` in original `LAMMPS`. It does not affects the original capability of `LAMMPS`, but if you're not sure, backup the two files from source.

```
cp SEVENN/pair_e3gnn/* path_to_lammps/src/
```

If you correctly installed CUDA-aware OpenMPI, the remaining process is exactly same as [`pair-nequip`](https://github.com/mir-group/pair_nequip).

Following modifications to lammps/cmake/CMakeLists.txt:

Change set(CMAKE_CXX_STANDARD 11) to set(CMAKE_CXX_STANDARD 14)
Append the following lines:

```
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
```

cmake & build lammps
```
cd lammps
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

## USAGE - MD

You can find example MD input scripts for `LAMMPS` from `SEVENN/example_inputs`. If you correctly installed the `LAMMPS`, there are two additional pair style avaialble, `e3gnn` and `e3gnn/parallel`

In `pair_coeff` of lammps script, you need path of serial or parallel deployed models from training. And for parallel model, you should specify how many segemented models will be given.

### serial model:
```
pair_style e3gnn
pair_coeff * * {path to serial model} {chemical species}
```

### parallel model:
```
pair_style e3gnn/parallel
pair_coeff * * {number of segemented parallel models} {space seperated paths of segemented parallel models} {chemical species}
```

### Execute LAMMPS with MPI
```
mpirun -np {# of GPUs want to use} {path to lammps binary} -in {lammps input scripts}
```

If `CUDA-aware OpenMPI` is not found (it detacts automatically in code), `e3gnn/parallel` will not utilize GPUs even if they are found. You can check whether the `OpenMPI` is found or not from the standard output of the `LAMMPS` simulation. Basically, the one gpu per one mpi process is expected. If the accessible GPUs are less than MPI process, a simulation runs inefficiently or fail. You can select sepcific GPUs by setting `CUDA_VISIBLE_DEVIECES` environment variable.

