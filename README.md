

<img src="SevenNet_logo.png" alt="Alt text" height="180">


# SEVENNet

SEVENNet (Scalable EquiVariance Enabled Neural Network) is a graph neural network interatomic potential package that supports parallel molecular dynamics simulations with [`LAMMPS`](https://github.com/lammps/lammps). Its underlying GNN model is based on [`nequip`](https://github.com/mir-group/nequip).

The project provides parallel molecular dynamics simulations using graph neural network interatomic potentials, which was not possible despite their superior performance.

**PLEASE NOTE:** SevenNet is under active development and may not be fully stable.

The installation and usage of SEVENNet are split into two parts: training (handled by PyTorch) and molecular dynamics (handled by [`LAMMPS`](https://github.com/lammps/lammps)). The model, once trained with PyTorch, is deployed using TorchScript and is later used to run molecular dynamics simulations via LAMMPS.

## Known issues

* The pressure of the parallel version in LAMMPS is not supported yet.
* When using parallel MD, if the simulation cell is too small (one of cell dimension < cutoff radius), the calculated force is incorrect.

However the second issue rarely matters since you can not fully utilize a GPU in this condition. In this case, using only a single GPU gives almost same speed as multiple GPUs.
Even though, we're looking for the solution.

## Requirements for Training

* Python >= 3.8
* PyTorch >= 1.11
* [`TorchGeometric`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
* [`pytorch_scatter`](https://github.com/rusty1s/pytorch_scatter)

You can find the installation guides for these packages from the [`PyTorch official`](https://pytorch.org/get-started/locally/), [`TorchGeometric docs`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and [`pytorch_scatter`](https://github.com/rusty1s/pytorch_scatter). Remember that these packages have dependencies on your CUDA version.

## Installation for Training

```
git clone https://github.com/MDIL-SNU/SEVENNet.git
cd SEVENNet
pip install .
```

## Usage for Training

### To start training using 'sevenn'

```
cd example_inputs/training
sevenn input_full.yaml -s
```

Examples of `input_full.yaml` can be found under `SEVENN/example_inputs`. The `structure_list` file is used to select VASP OUTCARs for training.
To reuse a preprocessed training set, you can specify `${dataset_name}.sevenn_data` to the `load_dataset_path:` in the `input.yaml`.

Once you initiate training, `log.sevenn` will contain all parsed inputs from `input.yaml`. Any parameters not specified in the input will be automatically assigned as their default values. You can refer to the log to check the default inputs.
Currently, detailed explanations of model hyperparameters can be found at [`nequip`](https://github.com/mir-group/nequip), or inside input_full.yaml.

### Multi-GPU training

We support multi-GPU training feature using PyTorch DDP (distributed data parallel). We use one process (CPU core) per GPU.
```
torchrun --standalone --nnodes={# of nodes} --nproc_per_node {# of GPUs} --no_python sevenn input.yaml -d
```
Please note that `batch_size` in input.yaml indicates `batch_size` per GPU.

### Check model quality using 'sevenn_inference'

Assuming that you've done temporal training of 10 epochs by above "To start training using 'sevenn'", try below at the same directory
```
sevenn_inference checkpoint_best.pt ../data/label_1/*
```

This will create dir 'sevenn_infer_result'. It includes .csv files that enumerate prediction/reference results of energy and force on OUTCARs in data/label_1 directory.
You can try 'sevenn_inference --help' for more information of this command.

### To deploy models from checkpoint using 'sevenn_get_model'

Assuming that you've done temporal training of 10 epochs by above "To start training using 'sevenn'", try below at the same directory
```
sevenn_get_model checkpoint_best.pt
```

This will create 'deployed_serial.pt', which can be used as lammps potential under `e3gnn` pair_style. Check the lammps installation process below.

The parallel model can be obtained in a similar way
```
sevenn_get_model checkpoint_best.pt -p
```

This will create multiple deployed_parallel_*.pt' files. The number of deployed models equals the number of message-passing layers.
These models can be used as lammps potential to run parallel MD simulations with GNN potential using multiple GPU cards.

## Requirements for Molecular Dynamics (MD)

* PyTorch (same version as used for training)
* LAMMPS version of '23 June 2022' [`LAMMPS`](https://github.com/lammps/lammps)
* [`CUDA-aware OpenMPI`](https://www.open-mpi.org/faq/?category=buildcuda) for parallel MD

**PLEASE NOTE:** CUDA-aware OpenMPI may not support NVIDIA Gaming GPUs. Given that the software is closely tied to hardware specifications, it would be advisable to consult with your server administrator rather than attempting to compile it yourself.

You can check whether your OpenMPI is CUDA-aware by using `ompi_info` command:

```
$ ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
mca:mpi:base:param:mpi_built_with_cuda_support:value:true
```

We're currently developing other options (other than CUDA-aware OpenMPI) to leverage parallel MD. Please let us know other inter GPU communication backends you want for the SevenNet.

## Installation for MD

Note that the following commands will overwrite `comm_brick.cpp` and `comm_brick.h` in the original `LAMMPS`. While it does not affect the original functionality of `LAMMPS`, you may want to back up these files from the source if you're unsure.

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

### To check installation

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
$ cd ${path_to_SEVENNet}/example_inputs/md_serial_example
$ mpirun -np {# of GPUs you want to use} {lammps_binary} -in in.lmp

###lammps outputs for 5 MD steps###

$ grep PairE3GNN log.lammps
PairE3GNNParallel using device : CUDA
PairE3GNNParallel cuda-aware mpi : True
```

Example MD input scripts for `LAMMPS` can be found under `SEVENN/example_inputs`. If you've correctly installed `LAMMPS`, there are two additional pair styles available: `e3gnn` and `e3gnn/parallel`.

In the `pair_coeff` of the lammps script, you need to provide the path of the trained models (either serial or parallel). For parallel models, you should specify how many segmented models will be used.

### For serial model

```
pair_style e3gnn
pair_coeff * * {path to serial model} {chemical species}
```

### For parallel model

```
pair_style e3gnn/parallel
pair_coeff * * {number of segmented parallel models} {space separated paths of segmented parallel models} {chemical species}
```

### To execute parallel MD simulation with LAMMPS

```
mpirun -np {# of MPI rank to use} {path to lammps binary} -in {lammps input script}
```

If a CUDA-aware OpenMPI is not found (it detects automatically in the code), `e3gnn/parallel` will not utilize GPUs even if they are available. You can check whether `OpenMPI` is found or not from the standard output of the `LAMMPS` simulation. Ideally, one GPU per MPI process is expected. If the available GPUs are fewer than the MPI processes, the simulation may run inefficiently or fail. You can select specific GPUs by setting the `CUDA_VISIBLE_DEVICES` environment variable.
