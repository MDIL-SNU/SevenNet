import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, SphericalHarmonics
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class EdgePreprocess(nn.Module):
    """
    preprocessing pos to edge vectors and edge lengths
    currently used in sevenn/scripts/deploy for lammps serial model
    """

    def __init__(self, is_stress: bool) -> None:
        super().__init__()
        # controlled by 'AtomGraphSequential'
        self.is_stress = is_stress
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            cell = data[KEY.CELL].view(-1, 3, 3)
        else:
            cell = data[KEY.CELL].view(3, 3)
        cell_shift = data[KEY.CELL_SHIFT]
        pos = data[KEY.POS]

        batch = data[KEY.BATCH]  # for deploy, must be defined first
        if self.is_stress:
            if self._is_batch_data:
                num_batch = int(batch.max().cpu().item()) + 1
                strain = torch.zeros(
                    (num_batch, 3, 3),
                    dtype=pos.dtype,
                    device=pos.device,
                )
                strain.requires_grad_(True)
                data['_strain'] = strain

                sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
                pos = pos + torch.bmm(
                    pos.unsqueeze(-2), sym_strain[batch]
                ).squeeze(-2)
                cell = cell + torch.bmm(cell, sym_strain)
            else:
                strain = torch.zeros(
                    (3, 3),
                    dtype=pos.dtype,
                    device=pos.device,
                )
                strain.requires_grad_(True)
                data['_strain'] = strain

                sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
                pos = pos + torch.mm(pos, sym_strain)
                cell = cell + torch.mm(cell, sym_strain)

        idx_src = data[KEY.EDGE_IDX][0]
        idx_dst = data[KEY.EDGE_IDX][1]

        edge_vec = pos[idx_dst] - pos[idx_src]

        if self._is_batch_data:
            edge_vec = edge_vec + torch.einsum(
                'ni,nij->nj', cell_shift, cell[batch[idx_src]]
            )
        else:
            edge_vec = edge_vec + torch.einsum(
                'ni,ij->nj', cell_shift, cell.squeeze(0)
            )
        data[KEY.EDGE_VEC] = edge_vec
        data[KEY.EDGE_LENGTH] = torch.linalg.norm(edge_vec, dim=-1)
        return data


class BesselBasis(nn.Module):
    """
    f : (*, 1) -> (*, bessel_basis_num)
    """

    def __init__(
        self,
        cutoff_length: float,
        bessel_basis_num: int = 8,
        trainable_coeff: bool = True,
    ) -> None:
        super().__init__()
        self.num_basis = bessel_basis_num
        self.prefactor = 2.0 / cutoff_length
        coeffs = torch.FloatTensor([
            n * math.pi / cutoff_length for n in range(1, bessel_basis_num + 1)
        ])
        if trainable_coeff:
            self.coeffs = nn.Parameter(coeffs)
        else:
            # buffer (not in state_dict) so that .to(device) moves it
            self.register_buffer('coeffs', coeffs, persistent=False)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        ur = r.unsqueeze(-1)  # to fit dimension
        return self.prefactor * torch.sin(self.coeffs * ur) / ur


class PolynomialCutoff(nn.Module):
    """
    f : (*, 1) -> (*, 1)
    https://arxiv.org/pdf/2003.03123.pdf
    """

    def __init__(
        self,
        cutoff_length: float,
        poly_cut_p_value: int = 6,
    ) -> None:
        super().__init__()
        p = poly_cut_p_value
        self.cutoff_length = cutoff_length
        self.p = p
        self.coeff_p0 = (p + 1.0) * (p + 2.0) / 2.0
        self.coeff_p1 = p * (p + 2.0)
        self.coeff_p2 = p * (p + 1.0) / 2.0

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r / self.cutoff_length
        return (
            1
            - self.coeff_p0 * torch.pow(r, self.p)
            + self.coeff_p1 * torch.pow(r, self.p + 1.0)
            - self.coeff_p2 * torch.pow(r, self.p + 2.0)
        )


class XPLORCutoff(nn.Module):
    """
    https://hoomd-blue.readthedocs.io/en/latest/module-md-pair.html
    """

    def __init__(
        self,
        cutoff_length: float,
        cutoff_on: float,
    ) -> None:
        super().__init__()
        self.r_on = cutoff_on
        self.r_cut = cutoff_length
        assert self.r_on < self.r_cut

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r_sq = r * r
        r_on_sq = self.r_on * self.r_on
        r_cut_sq = self.r_cut * self.r_cut
        return torch.where(
            r < self.r_on,
            1.0,
            (r_cut_sq - r_sq) ** 2
            * (r_cut_sq + 2 * r_sq - 3 * r_on_sq)
            / (r_cut_sq - r_on_sq) ** 3,
        )


@compile_mode('script')
class SphericalEncoding(nn.Module):
    def __init__(
        self,
        lmax: int,
        parity: int = -1,
        normalization: str = 'component',
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.normalization = normalization
        self.irreps_in = Irreps('1x1o') if parity == -1 else Irreps('1x1e')
        self.irreps_out = Irreps.spherical_harmonics(lmax, parity)
        self.sph = SphericalHarmonics(
            self.irreps_out,
            normalize=normalize,
            normalization=normalization,
            irreps_in=self.irreps_in,
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.sph(r)


@compile_mode('script')
class AgnesiTransform(nn.Module):
    """
    Agnesi distance transform applied before the radial basis:
        x' = 1 / (1 + a (r/r0)^q / (1 + (r/r0)^(q - p))),
    r0 = mean covalent radius of the edge's two elements (ASE table).
    From ACEpotentials.jl (Witt et al., J. Chem. Phys. 159, 164101 (2023),
    doi:10.1063/5.0158783), as used by MACE (mace/modules/radial.py
    AgnesiTransform; MACE-MH-1 config distance_transform: Agnesi).
    The cutoff envelope stays a function of r (smooth at the cutoff).
    """

    def __init__(
        self,
        type_map: Dict[int, int],
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
    ) -> None:
        super().__init__()
        from ase.data import covalent_radii

        n_type = max(type_map.values()) + 1
        radius = torch.zeros(n_type)
        for z, idx in type_map.items():
            radius[idx] = float(covalent_radii[z])
        self.register_buffer('radius', radius)
        self.q, self.p, self.a = q, p, a

    def forward(
        self, r: torch.Tensor, edge_idx: torch.Tensor, node_type: torch.Tensor
    ) -> torch.Tensor:
        r0 = 0.5 * (self.radius[node_type[edge_idx[0]]]
                    + self.radius[node_type[edge_idx[1]]])
        x = r / r0
        return 1.0 / (1.0 + self.a * x.pow(self.q) / (1.0 + x.pow(self.q - self.p)))


@compile_mode('script')
class EdgeEmbedding(nn.Module):
    """
    embedding layer of |r| by
    RadialBasis(|r|)*CutOff(|r|)
    f : (N_edge) -> (N_edge, basis_num)
    """

    def __init__(
        self,
        basis_module: nn.Module,
        cutoff_module: nn.Module,
        spherical_module: nn.Module,
        save_components: bool = False,
        distance_transform: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.basis_function = basis_module
        self.cutoff_function = cutoff_module
        self.spherical = spherical_module
        self.save_components = save_components  # for PairRadialMLP
        self.distance_transform = distance_transform
        # node type indices for distance_transform: the node features before
        # one-hot (serial, ML-IAP); KEY.ATOM_TYPE_ALL (local + ghost) in the
        # e3gnn/parallel segments, where x is local only / already features
        self.key_node_type = KEY.NODE_FEATURE

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        rvec = data[KEY.EDGE_VEC]
        r = torch.linalg.norm(data[KEY.EDGE_VEC], dim=-1)
        data[KEY.EDGE_LENGTH] = r

        env = self.cutoff_function(r)
        if self.distance_transform is not None:
            # node features still hold the type indices (before one-hot)
            r_in = self.distance_transform(
                r, data[KEY.EDGE_IDX], data[self.key_node_type]
            )
            rbf = self.basis_function(r_in)
        else:
            rbf = self.basis_function(r)
        data[KEY.EDGE_EMBEDDING] = rbf * env.unsqueeze(-1)
        if self.save_components:
            data[KEY.EDGE_RBF] = rbf
            data[KEY.EDGE_ENVELOPE] = env
        data[KEY.EDGE_ATTR] = self.spherical(rvec)

        return data
