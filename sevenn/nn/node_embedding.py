from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional
from ase.symbols import symbols2numbers
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class OnehotEmbedding(nn.Module):
    """
    convenient module for md simulation
    input : tensor of shape (N, 1)
    output : tensor of shape (N, num_classes)
    ex) [0 1 1 0] -> [[1, 0] [0, 1] [0, 1] [1, 0]] (num_classes = 2)
    """
    def __init__(
        self,
        num_classes: int,
        data_key_in: str = KEY.NODE_FEATURE,
        data_key_out: str = None,
        data_key_additional: str = KEY.NODE_ATTR,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.KEY_INPUT = data_key_in
        if data_key_out is None:
            self.KEY_OUTPUT = data_key_in
        else:
            self.KEY_OUTPUT = data_key_out
        self.KEY_ADDITIONAL = data_key_additional

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        inp = data[self.KEY_INPUT]
        embd = torch.nn.functional.one_hot(inp, self.num_classes)
        data[self.KEY_OUTPUT] = embd
        if self.KEY_ADDITIONAL is not None:
            data[self.KEY_ADDITIONAL] = embd
        return data


def get_type_mapper_from_specie(specie_list: List[str]):
    """
    from ['Hf', 'O']
    return {72: 0, 16: 1}
    """
    specie_list = sorted(specie_list)
    type_map = {}
    unique_counter = 0
    for specie in specie_list:
        atomic_num = symbols2numbers(specie)[0]
        if atomic_num in type_map:
            continue
        type_map[atomic_num] = unique_counter
        unique_counter += 1
    return type_map


def one_hot_atom_embedding(atomic_numbers: List[int], type_map: Dict[int, int]):
    """
    atomic numbers from ase.get_atomic_numbers
    type_map from get_type_mapper_from_specie()
    """
    num_classes = len(type_map)
    type_numbers = torch.LongTensor([type_map[num] for num in atomic_numbers])
    embd = torch.nn.functional.one_hot(type_numbers, num_classes)
    embd = embd.type(torch.FloatTensor)

    return embd


def main():
    _ = 1
    """
    with open('raw_data_from_parse_structure_list_tmp.pickle', 'rb') as f:
        res = pickle.load(f)
    tmp = res["96atom"][3]

    atomic_numbers, edge_src, edge_dst, \
        edge_vec, shift, pos, cell, E, F = ASE_atoms_to_data(tmp, 4.0)

    type_map = get_type_mapper_from_specie(['Hf', 'O'])
    embd = one_hot_atom_embedding(atomic_numbers, type_map)
    """


if __name__ == "__main__":
    main()

