from typing import Any, Dict, List, Optional, Sequence

import torch
from ase.atoms import Atoms
from torch_geometric.loader.dataloader import Collater

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData

from .dataload import atoms_to_graph


class AtomsToGraphCollater(Collater):

    def __init__(
        self,
        dataset: Sequence[Atoms],
        cutoff: float,
        type_map: Dict[int, int],  # Z -> node onehot
        requires_grad_key: str = KEY.EDGE_VEC,
        key_x: str = KEY.NODE_FEATURE,
        transfer_info: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        y_from_calc: bool = True,
    ):
        # quite original collator's type mismatch with []
        super().__init__([], follow_batch, exclude_keys)
        self.dataset = dataset
        self.cutoff = cutoff
        self.type_map = type_map
        self.requires_grad_key = requires_grad_key
        self.transfer_info = transfer_info
        self.y_from_calc = y_from_calc
        self.key_x = key_x

    def _Z_to_onehot(self, Z):
        return torch.LongTensor(
            [self.type_map[z.item()] for z in Z]
        )

    def __call__(self, batch: List[Any]) -> Any:
        # build list of graph
        graph_list = []
        for stct in batch:
            graph = atoms_to_graph(
                stct,
                self.cutoff,
                transfer_info=self.transfer_info,
                y_from_calc=self.y_from_calc,
            )
            graph = AtomGraphData.from_numpy_dict(graph)
            graph[self.key_x] = self._Z_to_onehot(graph[self.key_x])
            # graph[self.requires_grad_key].requires_grad_(True)
            graph_list.append(graph)
        return super().__call__(graph_list)
