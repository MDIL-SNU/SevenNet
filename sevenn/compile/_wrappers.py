from typing import List, Optional, Sequence

import torch
import torch.nn

from sevenn.nn.sequential import AtomGraphSequential

_INPUTS_LIST = sorted(
    [
        # "atom_onehot_index",
        # "x",
        # "node_attr",
        'atom_type',
        'cell_volume',
        'num_atoms',
        'edge_vec',
        'edge_index',
    ]
)


_OUTPUTS_LIST = sorted(
    [
        'inferred_force',
        'atomic_energy',
        'inferred_stress',
        'inferred_total_energy',
    ]
)


def inputs_list() -> List[str]:
    return _INPUTS_LIST.copy()


def outputs_list() -> List[str]:
    return _OUTPUTS_LIST.copy()


class DictInputOutputWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.input_keys = input_keys or inputs_list()
        self.output_keys = output_keys or outputs_list()

    def forward(self, data):
        inputs = [data[key] for key in self.input_keys]
        with torch.inference_mode():
            outputs = self.model(inputs)
        return {key: arg for key, arg in zip(self.output_keys, outputs)}


class ListInputOutputWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Sequential,  # 7net sequential
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.input_keys = input_keys or inputs_list()
        self.output_keys = output_keys or outputs_list()

    def forward(self, *args: torch.Tensor) -> List[torch.Tensor]:
        inputs = {key: arg for key, arg in zip(self.input_keys, args)}
        for module in self.model:
            inputs = module(inputs)
        outputs = inputs
        return [outputs[key] for key in self.output_keys]
