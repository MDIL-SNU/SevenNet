# ASE calculator

SevenNet provides an ASE interface via the ASE calculator. Models can be loaded using the following Python code:
```python
from sevenn.calculator import SevenNetCalculator
# The 'modal' argument is required if the model is trained with multi-fidelity learning enabled.
calc_omni = SevenNetCalculator(model='7net-omni', modal='mpa')
```
SevenNet also supports CUDA-accelerated D3 calculations.
For more information about D3, follow [here](../user_guide/d3.md))
```python
from sevenn.calculator import SevenNetD3Calculator
calc = SevenNetD3Calculator(model='7net-0', device='cuda')
```

Use enable_cueq or enable_flashTP to use cuEquivariance or flashTP for faster inference.
For more information about cuEq and flashTP, follow [here](../install/accelerator.md)
```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator(model='7net-0', enable_cueq=True) # or enable_flashTP=True
```

If you encounter the error `CUDA is not installed or nvcc is not available`, please ensure the `nvcc` compiler is available. Currently, CPU + D3 is not supported.

Various pretrained SevenNet models can be accessed by setting the model variable to predefined keywords like `7net-omni`, `7net-mf-ompa`, `7net-omat`, `7net-l3i5`, and `7net-0`.

User-trained models can be applied with the ASE calculator. In this case, the `model` parameter should be set to the checkpoint path from training.

:::{tip}
When 'auto' is passed to the `device` parameter (the default setting), SevenNet utilizes GPU acceleration if available.
:::
