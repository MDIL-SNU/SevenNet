# Apple GPU (MPS)

On a Mac with Apple silicon, SevenNet runs on the GPU through PyTorch's MPS backend. Moving a
model to `mps` switches the e3nn layers to fused Metal kernels and pads edge lists so that the
GPU sees few tensor shapes. Nothing else changes in the inputs or the commands.

## Requirements

- macOS on Apple silicon, PyTorch 2.7 or later (2.6 runs the reference e3nn code on the GPU).
- float32: MPS has no float64. Shift and scale stored in float64 are converted when the model
  moves to the GPU, so results agree with the CPU to float32 precision, not bit for bit.

## Usage

```python
from sevenn.calculator import SevenNetCalculator

calc = SevenNetCalculator('7net-0', device='mps')
```

```bash
sevenn inference --device mps 7net-0 structures.extxyz
sevenn train input.yaml        # with device: mps under train
```

`Trainer`, `sevenn.scripts.inference.inference` and user scripts behave the same way: the
switch happens in `model.to('mps')`, and `model.to('cpu')` undoes it; parameters converted to
float32 stay float32. A model instance given to the calculator is changed in place.

## Selection and opt-out

| Request | Result |
|---|---|
| `device='cpu'` | CPU; the Apple GPU code is not imported |
| `device='mps'` | Apple GPU, fused kernels where supported; edge padding when every convolution is fused |
| `device='mps'` and `SEVENN_DISABLE_MPS=1` | Apple GPU, reference e3nn code, no padding |
| model built with cuEquivariance, FlashTP, OpenEquivariance or LunaTP, moved to `mps` | error |

`SEVENN_DISABLE_MPS` accepts `1`, `true` or `yes`. Modules the kernels do not cover keep the
e3nn code on the GPU with a warning; nothing moves to the CPU, unless `PYTORCH_ENABLE_MPS_FALLBACK=1`
is set, which lets PyTorch run operations on the CPU silently (SevenNet warns about it).

## LAMMPS

TorchScript cannot hold the Metal kernels, so the Apple GPU pair style runs the model in
Python and loads the checkpoint itself: there is no `sevenn get_model` step. It is serial
(one MPI rank) and takes `pair_style e3gnn` with
`pair_coeff * * checkpoint.pth [modal NAME] element1 element2 ...`.

Patched build: on a Mac with an Apple GPU, `sevenn patch_lammps` installs this pair style in
place of the TorchScript ones, also for a CPU build (`SEVENN_DISABLE_MPS=1` keeps the TorchScript
pair styles, which take deployed `.pt` models). The binary uses the Python environment that ran
the patch.

```bash
sevenn patch_lammps ./lammps
cd lammps && mkdir build && cd build
cmake ../cmake -DBUILD_MPI=no -DBUILD_OMP=no && make -j 8
```

Plugin for `pip install lammps`, from an installed SevenNet:

```bash
SRC="$(python -c "import sevenn.mps.lammps.patch as p; print(p.SRC)")"
bash "$SRC/build_plugin.sh" python
export LAMMPS_PLUGIN_PATH="$SRC"
```

The plugin is built next to its sources, so the installed package must be writable. The pair
style refuses what it cannot honour: several MPI ranks, units other than `metal`,
`neigh_modify exclude` and `include`, `pair_style hybrid` and `hybrid/overlay`, deployed `.pt`
files; per-atom stress is not available.

## Performance

ZrO2, energy, forces and stress per ASE call (ms): the median of 10 calls after 3 warm-up calls (on the
M5 Pro, the median of three such runs where three were made). MacBook Pro M5 Pro (48 GB, PyTorch
2.14.1) and Mac mini M2 (8 GB, PyTorch 2.15.0):

| Model, atoms | M5 Pro, fused | M5 Pro, e3nn on GPU | M5 Pro, CPU | M2, fused | M2, e3nn on GPU | M2, CPU |
|---|---|---|---|---|---|---|
| 7net-0, 96 | 14.4 | 144 | 305 | 42.8 | 540 | 446 |
| 7net-0, 768 | 64.9 | 1,004 | 2,311 | 278 | | |
| 7net-l3i5, 96 | 21.0 | 378 | 775 | 65.6 | > 5 GB | 1,171 |
| 7net-l3i5, 768 | 104.5 | > 20 GB | | > 5 GB | | |

"> N GB": stopped at the memory cap of the measurement; empty: not measured. The fused results agree
with the reference e3nn code on the GPU and with the CPU to float32 precision.

## Limits

- Eval mode: linear, self-connection and gate layers give first derivatives only, and linear and
  self-connection layers treat their weights as constants (forces, stress and training are complete; for other parameter
  gradients use train mode or the CPU).
- 32-bit indices: a graph above about 2^31 elements per kernel buffer is refused.
- Autocast keeps the reference code.
- Radial tables (`radial_table`) are not available: off by default on `mps`, and `radial_table=True`
  is an error there.
