from typing import Any, List, Optional, Sequence

from ase.atoms import Atoms
from torch_geometric.loader.dataloader import Collater

from sevenn.atom_graph_data import AtomGraphData

from .dataload import atoms_to_graph


class AtomsToGraphCollater(Collater):

    def __init__(
        self,
        dataset: Sequence[Atoms],
        cutoff: float,
        transfer_info: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        y_from_calc: bool = True,
    ):
        # quite original collator's type mismatch with []
        super().__init__([], follow_batch, exclude_keys)
        self.dataset = dataset
        self.cutoff = cutoff
        self.transfer_info = transfer_info
        self.y_from_calc = y_from_calc

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
            graph_list.append(graph)
        return super().__call__(graph_list)
