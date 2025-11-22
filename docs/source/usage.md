# Usage

SevenNet supports various interface and acceleartions to use and train model efficiently. 

(ase_calculator)=
## ASE calculator

SevenNet provides an ASE interface via the ASE calculator. Models can be loaded using the following Python code:
```python
from sevenn.calculator import SevenNetCalculator
# The 'modal' argument is required if the model is trained with multi-fidelity learning enabled.
calc_mf_ompa = SevenNetCalculator(model='7net-mf-ompa', modal='mpa')
```
SevenNet also supports CUDA-accelerated D3 calculations.
```python
from sevenn.calculator import SevenNetD3Calculator
calc = SevenNetD3Calculator(model='7net-0', device='cuda')
```
If you encounter the error `CUDA is not installed or nvcc is not available`, please ensure the `nvcc` compiler is available. Currently, CPU + D3 is not supported.

Various pretrained SevenNet models can be accessed by setting the model variable to predefined keywords like `7net-mf-ompa`, `7net-omat`, `7net-l3i5`, and `7net-0`.

The following table provides **approximate** maximum atom counts of **A100 GPU (80GB)** in a bulk system.
| Model | Max atoms |
|:---:|:---:|
|7net-0|~ 21,500|
|7net-l3i5|~ 9,300|
|7net-omat|~ 5,300|
|7net-mf-ompa|~ 3,300|

Note: These limits vary depending on the target system. To handle larger systems, multi-GPU parallelization using LAMMPS can be employed.

Additionally, user-trained models can be applied with the ASE calculator. In this case, the `model` parameter should be set to the checkpoint path from training.

> [!TIP]
> When 'auto' is passed to the `device` parameter (the default setting), SevenNet utilizes GPU acceleration if available.

## Command line interface: training and inference

SevenNet provides five commands for preprocessing, training, and deployment: `sevenn_preset`, `sevenn_graph_build`, `sevenn`, `sevenn_inference`, and `sevenn_get_model`.

### 1. Input generation

With the `sevenn_preset` command, the input file setting the training parameters is generated automatically.
```bash
sevenn_preset {preset keyword} > input.yaml
```

Available preset keywords are: `base`, `fine_tune`, `multi_modal`, `sevennet-0`, and `sevennet-l3i5`.
Check comments in the preset YAML files for explanations. For fine-tuning, be aware that most model hyperparameters cannot be modified unless explicitly indicated.
To reuse a preprocessed training set, you can specify `sevenn_data/${dataset_name}.pt` for the `load_trainset_path:` in the `input.yaml`.

### 2. Preprocess (optional)

To obtain the preprocessed data, `sevenn_data/graph.pt`, `sevenn_graph_build` command can be used.
The output files can be used for training (`sevenn`) or inference (`sevenn_inference`) to skip the graph build stage.

```bash
sevenn_graph_build {dataset path} {cutoff radius}
```

The output `sevenn_data/graph.yaml` contains statistics and meta information about the dataset.
These files must be located in the `sevenn_data` directory. If you move the dataset, move the entire `sevenn_data` directory without changing the contents.

See `sevenn_graph_build --help` for more information.

### 3. Training

Given that `input.yaml` and `sevenn_data/graph.pt` are prepared, SevenNet can be trained by the following command:

```bash
sevenn input.yaml -s
```

We support multi-GPU training using PyTorch DDP (distributed data parallel) with a single process (or a CPU core) per GPU.

```bash
torchrun --standalone --nnodes {number of nodes} --nproc_per_node {number of GPUs} --no_python sevenn input.yaml -d
```

Please note that `batch_size` in `input.yaml` refers to the per-GPU batch size.

### 4. Inference

Using the checkpoint after training, the properties such as energy, force, and stress can be inferred directly.

```bash
sevenn_inference checkpoint_best.pth path_to_my_structures/*
```

This will create the `sevenn_infer_result` directory, where CSV files contain predicted energy, force, stress, and their references (if available).
See `sevenn_inference --help` for more information.

(deployment)=
### 5. Deployment

The checkpoint can be deployed as LAMMPS potentials. The argument is either the path to the checkpoint or the name of a pretrained potential.

```bash
sevenn_get_model 7net-0  # For pre-trained models
sevenn_get_model {checkpoint path}  # For user-trained models
```

This will create `deployed_serial.pt`, which can be used as a LAMMPS potential with the `e3gnn` pair_style in LAMMPS.

The potential for parallel MD simulation can be obtained similarly.

```bash
sevenn_get_model 7net-0 -p
sevenn_get_model {checkpoint path} -p
```

This will create a directory with several `deployed_parallel_*.pt` files. The directory path itself is an argument for the LAMMPS script. Please do not modify or remove files in the directory.
These models can be used as LAMMPS potentials to run parallel MD simulations with a GNN potential across multiple GPUs.

(notebook-tutorial)=
## Notebook tutorials

If you want to learn how to use the `sevenn` Python library instead of the CLI command, please check out the notebook tutorials below.

| Notebooks | Google&nbsp;Colab | Descriptions |
|-----------|-------------------|--------------|
|[From scratch](https://github.com/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_python_tutorial.ipynb)|[![Open in Google Colab]](https://colab.research.google.com/github/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_python_tutorial.ipynb)|We can learn how to train the SevenNet from scratch, predict energy, forces, and stress using the trained model, perform structure relaxation, and draw EOS curves.|
|[Fine-tuning](https://github.com/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_finetune_tutorial.ipynb)|[![Open in Google Colab]](https://colab.research.google.com/github/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_finetune_tutorial.ipynb)|We can learn how to fine-tune the SevenNet and compare the results of the pretrained model with the fine-tuned model.|

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg

Sometimes, the Colab environment may crash due to memory issues. If you have sufficient GPU resources in your local environment, we recommend downloading the tutorials from GitHub and running them on your machine.
```bash
git clone https://github.com/MDIL-SNU/sevennet_tutorial.git
```

## LAMMPS

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

To deploy LAMMPS models from checkpoints for both serial and parallel execution, use [`sevenn_get_model`](#deployment).

It is expected that there is one GPU per MPI process. If the number of available GPUs is less than the number of MPI processes, the simulation may run inefficiently.

> [!CAUTION]
> Currently, the parallel version encounters an error when one of the subdomain cells contains no atoms. This issue can be addressed using the `processors` command and, more effectively, the `fix balance` command in LAMMPS. A patch for this issue will be released in a future update.


