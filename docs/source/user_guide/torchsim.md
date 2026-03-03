(torchsim)=
# TorchSim

[TorchSim](https://github.com/TorchSim/torch-sim) is a GPU-native atomistic simulation engine built on PyTorch. It provides batched MD, relaxation, and more with significant speedups over ASE. See the [TorchSim documentation](https://torchsim.github.io/torch-sim/) for full details.

SevenNet provides its own `SevenNetModel` wrapper for TorchSim, located in `sevenn.torchsim`.

## Installation

TorchSim is an optional dependency of SevenNet (requires Python >= 3.12). Install it via:
```bash
pip install sevenn[torchsim]
```

## Usage

### Loading a model

`SevenNetModel` accepts the same model specifiers as `SevenNetCalculator`: a pretrained model name, a checkpoint path, or a model object.

```python
from sevenn.torchsim import SevenNetModel

model = SevenNetModel(model="7net-omni", modal="mpa")
```

The `device` parameter defaults to `'auto'` (CUDA if available, otherwise CPU).

### Batched MD

```python
import torch_sim as ts
from ase.build import bulk

atoms = bulk("Cu", "fcc", a=3.58, cubic=True).repeat((2, 2, 2))

final_state = ts.integrate(
    system=[atoms] * 10,
    model=model,
    n_steps=100,
    timestep=0.002,
    temperature=300,
    integrator=ts.Integrator.nvt_langevin,
)
```

### Relaxation

```python
relaxed_state = ts.optimize(
    system=atoms,
    model=model,
    optimizer=ts.Optimizer.fire,
)
```

### Multi-modal models

For multi-fidelity models such as `7net-mf-ompa`, pass the `modal` argument:
```python
model = SevenNetModel(model="7net-mf-ompa", modal="omat24")
```

Available modals for `7net-mf-ompa` are `'mpa'` and `'omat24'`.

For further TorchSim features (trajectory reporting, autobatching, cell filters, etc.), refer to the [TorchSim tutorials](https://torchsim.github.io/torch-sim/user/overview.html).
