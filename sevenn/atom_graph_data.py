from typing import List, Dict

import torch_geometric

import sevenn._keys as KEY


class AtomGraphData(torch_geometric.data.Data):
    """
    Args:
        x (Tensor, optional): atomic numbers with shape :obj:`[num_nodes,
            atomic_numbers]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y_energy: scalar # unit of eV (VASP raw)
        y_force: [num_nodes, 3] # unit of eV/A (VASP raw)
        y_stress: [6]  # [xx, yy, zz, xy, yz, zx] # unit of eV/A^3 (VASP raw)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.

    x, y_force, pos should be aligned with each other.
    """
    def __init__(
        self,
        x,
        edge_index,
        pos,
        edge_attr=None,
        y_energy=None,
        y_force=None,
        y_stress=None,
        edge_vec=None,
        init_node_attr=True,
    ):
        super(AtomGraphData, self).__init__(
            x,
            edge_index,
            edge_attr,
            pos=pos
        )
        self[KEY.ENERGY] = y_energy
        self[KEY.FORCE] = y_force
        self[KEY.STRESS] = y_stress
        self[KEY.EDGE_VEC] = edge_vec
        if init_node_attr:
            self[KEY.NODE_ATTR] = x

    def to_dict(self):
        return {k: v for k, v in self.items()}
