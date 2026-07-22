(ase_calculator)=
# ASE calculator

### `SevenNetCalculator`

SevenNet provides an ASE interface via the ASE calculator. Models can be loaded using the following Python code:
```python
from sevenn.calculator import SevenNetCalculator
# The 'modal' argument is required if the model is trained with multi-fidelity learning enabled.
calc_omni = SevenNetCalculator(model='7net-omni', modal='mpa')
```

## D3 dispersion correction

SevenNet also supports a CUDA-accelerated Grimme D3 van der Waals corrections through ASE calculators.
For the method, units, supported functionals/damping types, and limitations, see {doc}`./d3`.

### `SevenNetD3Calculator` (SevenNet + D3)

It sums a `SevenNetCalculator` and a `D3Calculator`, returning combined energy, forces, and stress.

```python
# To obtain 7net-0 + D3(BJ) results
from sevenn.calculator import SevenNetD3Calculator
calc = SevenNetD3Calculator(
           model='7net-0',
           device='cuda',
           functional_name='pbe',  # 7net-0 is trained with PBE results
           damping_type='damp_bj',
       )
```

It accepts every `SevenNetCalculator` argument (`model`, `modal`, `enable_cueq`/`enable_flash`/`enable_oeq`, ...) plus the D3 parameters listed below.

### `D3Calculator` (D3 only)

`D3Calculator` computes only the D3 dispersion term. Use it directly when you want the dispersion contribution on its own, or combine it with another calculator via ASE's `SumCalculator` (this is exactly what `SevenNetD3Calculator` does internally):

```python
from ase.calculators.mixing import SumCalculator
from sevenn.calculator import SevenNetCalculator, D3Calculator

calc = SumCalculator([SevenNetCalculator(model='7net-0'), D3Calculator()])
```

### D3 parameters

Both calculators share the same D3 knobs:

| Parameter | Default | Description |
| --- | --- | --- |
| `damping_type` | `'damp_bj'` | Becke-Johnson damping |
| `functional_name` | `'pbe'` | XC functional where the D3 parameters are fitted to |
| `vdw_cutoff` | `9000` | Squared energy/force cutoff in Bohr² (≈ 50.20 Å) |
| `cn_cutoff` | `1600` | Squared coordination-number cutoff in Bohr² (≈ 21.17 Å) |

See {doc}`./d3` for the full explanation of each option and the list of supported functionals.

:::{caution}
The D3 calculators require a CUDA GPU (compute capability ≥ 6.0); CPU + D3 is not supported. There is also a hard limit of 46,340 atoms per call. See the Cautions section of {doc}`./d3`.
:::

Use enable_cueq, enable_flash, or enable_oeq to use cuEquivariance, flashTP, or OpenEquivariance for faster inference.
For more information about accelerators, follow [here](./accelerator.md)
```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator(model='7net-0', enable_cueq=True) # or enable_flash=True or enable_oeq=True
```

If you encounter the error `CUDA is not installed or nvcc is not available`, please ensure the `nvcc` compiler is available. Currently, CPU + D3 is not supported.

Various pretrained SevenNet models can be accessed by setting the model variable to predefined keywords like `7net-omni`, `7net-nano-5.5`, `7net-mf-ompa`, `7net-omat`, `7net-l3i5`, and `7net-0`.

User-trained models can be applied with the ASE calculator. In this case, the `model` parameter should be set to the checkpoint path from training.

:::{tip}
When 'auto' is passed to the `device` parameter (the default setting), SevenNet utilizes GPU acceleration if available.
:::
