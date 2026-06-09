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

You can enable accelerators (cuEquivariance, flashTP, or OpenEquivariance) via the corresponding flags. For more information about accelerators, follow [here](./accelerator.md).
```python
model = SevenNetModel(model="7net-omni", modal="mpa", enable_oeq=True)
# or enable_cueq=True or enable_flash=True
```

The `device` parameter defaults to `auto` (CUDA if available, otherwise CPU).

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
    system=[atoms] * 10,
    model=model,
    optimizer=ts.Optimizer.fire,
)
```

## SevenNet + D3 dispersion

`SevenNetD3Model` adds the CUDA-accelerated Grimme D3 dispersion correction on top of `SevenNetModel`. It is a drop-in replacement: it accepts the same model specifiers and accelerator flags, and plugs into `ts.integrate` / `ts.optimize` the same way.

```python
from sevenn.torchsim import SevenNetD3Model

model = SevenNetD3Model(model="7net-omni", modal="mpa", **d3_kwargs)

final_state = ts.integrate(
    system=[atoms] * 10,
    model=model,
    n_steps=100,
    timestep=0.002,
    temperature=300,
    integrator=ts.Integrator.nvt_langevin,
)
```

It requires a CUDA GPU (the D3 backends are GPU-only) and accepts the same D3 parameters as the ASE calculators (`damping_type`, `functional_name`, `vdw_cutoff`, `cn_cutoff`); see {doc}`./d3` and the [`D3Calculator` documentation](./ase_calculator.md#d3-dispersion-correction).

### Choosing the D3 backend

D3 can be evaluated two ways, selected by `d3_mode`:

- `serial`: a per-system loop over the ASE-style `D3Calculator`.
- `batch`: a single batched CUDA kernel launch that computes D3 for all systems at once.
- `auto` (default): use `batch` when the number of systems in the batch exceeds `d3_batch_threshold` (default `4`), otherwise `serial`.

The `auto` heuristic exists because the two paths have different trade-offs: the batched kernel amortizes its per-call overhead across many systems, so it wins for large batches, while the serial loop is cheaper for the few-system case. Tune the cutoff with `d3_batch_threshold`, or force a single backend with `d3_mode`:

```python
# Always use the batched CUDA kernel
model = SevenNetD3Model(model="7net-omni", modal="mpa", d3_mode="batch")

# Auto, but switch to batched D3 only above 8 systems
model = SevenNetD3Model(
    model="7net-omni", modal="mpa", d3_mode="auto", d3_batch_threshold=8,
)
```
