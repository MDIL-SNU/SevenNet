# SEVENNet

SEVENNet (Scalable EquiVariance Enabled Neural Network) is a graph neural network interatomic potential package that supports parallel molecular dynamics simulations on [`LAMMPS`](https://github.com/lammps/lammps). Its underlying GNN model is the same as that found in [`nequip`](https://github.com/mir-group/nequip). 

The project is inspired by [`nequip`](https://github.com/mir-group/nequip) and provides a solution for enabling parallel molecular dynamics simulations using graph neural network interatomic potentials, which was not possible despite their superior performance.

**PLEASE NOTE:** We are currently preparing a paper that provides a detailed description of the algorithms implemented in this project. In addition, SEVENNet is currently under active development and may not be fully stable.

The installation and usage of SEVENNet are split into two parts: training (handled by PyTorch) and inference (handled by [`LAMMPS`](https://github.com/lammps/lammps)). The model, once trained with PyTorch, is deployed using TorchScript and is later utilized to run molecular dynamics simulations via LAMMPS. To ensure smooth operation between training and inference, it is important to use consistent versions of CUDA and PyTorch.


## Requirements for Training

* Python >= 3.8
* PyTorch >= 1.11
* [`TorchGeometric`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
* [`pytorch_scatter`](https://github.com/rusty1s/pytorch_scatter)

You can find their installation guide from [`PyTorch official`](https://pytorch.org/get-started/locally/), [`TorchGeometric docs`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html`) and [`pytorch_scatter`](https://github.com/rusty1s/pytorch_scatter). Remember that these pacakages has dependency on your CUDA version.

## Installation for Training

```
git clone https://github.com/MDIL-SNU/SEVENNet.git
cd SEVENNet
pip install . 
```

It will automatically install additional python packages (that is not dependent on your CUDA) such as e3nn or ase.

## Usage for Training

### To start training:

```
cd example_inputs/training
sevenn input.yaml
```

Examples of `input.yaml` can be found under `SEVENN/example_inputs`. Use the `structure_list` file to select VASP OUTCARs for training. You can reuse preprocessed training set by specifying `${dataset_name}.sevenn_data` to `load_dataset_path:` of input.yaml. Both `structure_list` and `load_dataset_path` can be a list to help you easily augment training sets.

Once you initiate training, `log.sevenn` will contain all parsed inputs from `input.yaml`, or it will use default values if none are specified. You can refer to this log to understand the default inputs when they're not specified, allowing you to modify them in your next usage for improved results.
Currently, explanations of model hyperparameters can be found at [`nequip`](https://github.com/mir-group/nequip), as our dedicated documentation is still under preparation.

### To generate parallel models:

After the training, you will find `deployed_model_best.pt`, a serial model for MD simulation. Alternatively, you can generate a parallel model from the checkpoint using the following command:

```
sevenn_get_parallel checkpoint_best.pt
```

This will generate segmented parallel models with the same number of message passing layers as the model. You need all of these to run parallel MD.

## Requirements for Molecular Dynamics (MD)

* Same version of PyTorch used to training
* Latest stable version of [`LAMMPS`](https://github.com/lammps/lammps)
* [`CUDA-aware OpenMPI`](https://www.open-mpi.org/faq/?category=buildcuda) for parallel MD 

**PLEASE NOTE:** CUDA-aware OpenMPI may not support NVIDIA Gaming GPUs. Given that the software is closely tied to hardware specifications, it would be advisable to consult with your server administrator rather than attempting to compile it yourself. This approach can save you time.

You can check whether your OpenMPI is CUDA-aware or not by using `ompi_info`:

```
ompi_info --all | grep btl_openib_have_cuda_gdr
  > MCA btl openib: informational "btl_openib_have_cuda_gdr" (current value: "true", data source: default, level: 5 tuner/detail, type: bool)
```

## Installation for MD

Please note that the following command will overwrite `comm_brickc.cpp` and `comm_brick.h` in the original `LAMMPS`. While it does not affect the original functionality of `LAMMPS`, you may want to backup these files from the original source if you're unsure.

```
cp SEVENN/pair_e3gnn/* path_to_lammps/src/
```

If you have correctly installed CUDA-aware OpenMPI, the remaining process is identical to [`pair-nequip`](https://github.com/mir-group/pair_nequip).

Please make the following modifications to lammps/cmake/CMakeLists.txt:

Change `set(CMAKE_CXX_STANDARD 11)` to `set(CMAKE_CXX_STANDARD 14)`.

Then append the following lines:

```
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
```

To build lammps with cmake:

```
cd lammps
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

## Usage for MD

### To check installation:

For serial MD,
```
$ cd ${path_to_SEVENNet}/example_inputs/md_serial_example
$ {lammps_binary} -in in.lmp

###lammps outputs for 5 MD steps###

$ grep PairE3GNN log.lammps
PairE3GNN using device : CUDA
```

For parallel MD
```
$ cd ../md_parallel_example 
$ mpirun -np {# of GPUs you want to use} {lammps_binary} -in in.lmp

###lammps outputs for 5 MD steps###

$ grep PairE3GNN log.lammps
PairE3GNNParallel using device : CUDA
PairE3GNNParallel cuda-aware mpi : True
```

Example MD input scripts for `LAMMPS` can be found under `SEVENN/example_inputs`. If you've correctly installed `LAMMPS`, there are two additional pair styles available: `e3gnn` and `e3gnn/parallel`.

In the `pair_coeff` of the lammps script, you need to provide the path of the trained models (either serial or parallel). For parallel models, you should specify how many segmented models will be used.

### For serial model:

```
pair_style e3gnn
pair_coeff * * {path to serial model} {chemical species}
```

### For parallel model:

```
pair_style e3gnn/parallel
pair_coeff * * {number of segmented parallel models} {space separated paths of segmented parallel models} {chemical species}
```

### To execute LAMMPS with MPI:

```
mpirun -np {# of GPUs you want to use} {path to lammps binary} -in {lammps input scripts}
```

If a CUDA-aware OpenMPI is not found (it detects automatically in the code), `e3gnn/parallel` will not utilize GPUs even if they are available. You can check whether `OpenMPI` is found or not from the standard output of the `LAMMPS` simulation. Ideally, one GPU per MPI process is expected. If the available GPUs are fewer than the MPI processes, the simulation may run inefficiently or fail. You can select specific GPUs by setting the `CUDA_VISIBLE_DEVICES` environment variable.

## Known Bugs

* When parsing VASP `OUTCARs` with `structure_list`, if the folder contains a `POSCAR` with selective dynamics, it does not read the `OUTCAR` correctly.
* When parsing VASP `OUTCARs` with `structure_list`, spin polarized calculations are not yet supported.
* Models with a large number of parameters (>=5 message passing layers, >= 64 channels) show initial RMSEs of energy and force significantly larger than the original `nequip`. Since the difference should be marginal, we are actively investigating this issue.
* The calculated stress on `LAMMPS` is incorrect.

