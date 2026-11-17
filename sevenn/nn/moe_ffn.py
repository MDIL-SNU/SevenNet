"""
Element-routed mixture-of-experts FFN for the metaformer block (config `ffn_moe`).

Each node goes through a shared FFN plus ONE expert FFN chosen by its element group
(s-block, p-block non-metals, p-block metals, 3d, 4d/5d, f-block), each with half of
the dense hidden width, so the active FLOPs per node equal the dense FFN. Routing
depends only on the element, never on positions, so energy, forces and the Hessian
stay smooth; there is no router to train and no load-balancing loss.

Sources (ideas only): element-group experts + shared expert as in TACE/TECE
(arXiv:2509.14961, arXiv:2607.10664; github.com/xvzemin/tace, MIT); element-routed
MoE with a shared expert, DPA3-MoE (arXiv:2603.07977; deepmd-kit, LGPL-3.0, no code
reused); shared experts in LLM MoE (DeepSeekMoE, arXiv:2401.06066).
"""
from typing import Dict, List

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

_S_BLOCK = {1, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88}
_NONMETAL = {2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 33, 34, 35, 36, 52, 53, 54,
             85, 86}
_P_METAL = {13, 31, 32, 49, 50, 51, 81, 82, 83, 84}
NUM_ELEMENT_GROUPS = 6


def element_group(z: int) -> int:
    if z in _S_BLOCK:
        return 0
    if z in _NONMETAL:
        return 1
    if z in _P_METAL:
        return 2
    if 21 <= z <= 30:
        return 3
    if 57 <= z <= 71 or 89 <= z <= 103:
        return 5
    return 4   # 4d/5d transition metals (and anything heavier)


def type_to_expert_table(type_map: Dict[int, int], num_species: int) -> torch.Tensor:
    table = torch.zeros(num_species, dtype=torch.long)
    for z, idx in type_map.items():
        table[idx] = element_group(int(z))
    return table


@compile_mode('script')
class MoEDispatch(nn.Module):
    """write x[nodes of expert k] and their indices to per-expert data keys"""

    def __init__(self, type_to_expert: torch.Tensor, layer: int,
                 data_key_x: str = KEY.NODE_FEATURE) -> None:
        super().__init__()
        self.key_x = data_key_x
        self.register_buffer('type_to_expert', type_to_expert.clone())
        n = int(type_to_expert.max().item()) + 1
        self.keys_x: List[str] = [f'{layer}_moe_x{k}' for k in range(n)]
        self.keys_idx: List[str] = [f'{layer}_moe_idx{k}' for k in range(n)]

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        expert = self.type_to_expert.index_select(0, data[KEY.ATOM_TYPE])
        for k in range(len(self.keys_x)):
            idx = torch.nonzero(expert == k).squeeze(-1)
            data[self.keys_idx[k]] = idx
            data[self.keys_x[k]] = x.index_select(0, idx)
        return data


@compile_mode('script')
class MoECombine(nn.Module):
    """x (shared FFN output) += scatter of every expert's output to its nodes"""

    def __init__(self, num_experts: int, layer: int,
                 data_key_x: str = KEY.NODE_FEATURE) -> None:
        super().__init__()
        self.key_x = data_key_x
        self.keys_x: List[str] = [f'{layer}_moe_x{k}' for k in range(num_experts)]
        self.keys_idx: List[str] = [
            f'{layer}_moe_idx{k}' for k in range(num_experts)
        ]

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        for k in range(len(self.keys_x)):
            x = x.index_add(0, data[self.keys_idx[k]], data[self.keys_x[k]])
        data[self.key_x] = x
        return data
