"""
Learned pair repulsion outside the normalized feature stream (config
pair_repulsion). Added to the atomic energy after the shift/scale (eV):

    E_rep = sum_{i<j} phi_ij(r_ij)
    phi_ij(r) = u(r / r_c,ij) / r * sum_k A_ij,k exp(-kappa_ij,k r)
    A_ij,k = exp(lambda_i,k + lambda_j,k)         (eV A)
    kappa_ij,k = softplus(theta_i,k) + softplus(theta_j,k)   (1/A)
    r_c,ij = min(s (R_i + R_j), r_max)

lambda, theta: learned per-element embeddings (num_species x K; stored
divided by param_scale, see below); R: Cordero
covalent radii (ase.data.covalent_radii, B. Cordero et al., Dalton Trans.
2008, 2832, doi:10.1039/b801115j); u: DimeNet polynomial envelope (J.
Gasteiger et al., arXiv:2003.03123, p = 6: u, u', u'' vanish at x = 1; u is
non-increasing on [0, 1] since u'(x) = -p(p+1)(p+2)/2 x^(p-1) (1 - x)^2),
evaluated in the factored form u(x) = (1 - x)^3 sum_{n<p} C(n+2, 2) x^n
(same polynomial; positive coefficients, so u > 0 for x < 1 also in fp32,
where 1 - c0 x^p + ... cancels to rounding noise near x = 1).

Why this form
- Every term is positive and strictly decreasing in r, the envelope is
  positive and non-increasing, so phi_ij > 0 and dphi/dr < 0 on (0, r_c) for
  ANY parameter values; with the 1/r factor phi diverges as r -> 0, so it
  eventually dominates the bounded energy of the pre-norm network (whose
  normalized features and Bessel radial inputs stay bounded as r -> 0,
  dev/diatomics-eval/short_range_coverage.txt section 4).
- It is the screened-Coulomb family (N. Bohr, Mat. Fys. Medd. Dan. Vid.
  Selsk. 18, 8 (1948): Z1 Z2 e^2/r exp(-r/a); G. Moliere, Z. Naturforsch. A 2,
  133 (1947); J. F. Ziegler, J. P. Biersack, U. Littmark, The Stopping and
  Range of Ions in Solids (Pergamon, 1985), universal ZBL screening function:
  four such terms) with ZBL's own combining rules: amplitudes Z_i Z_j are
  log-additive and the inverse universal screening length (Z_i^0.23 +
  Z_j^0.23)/0.8854 a0 is additive, so the universal ZBL potential is one
  point of this family (K = 4). Nothing is fixed to it: amplitudes and decay
  constants are learned from the data (OMat24 rattled / AIMD frames reach
  ~0.45-0.5 r_cov,sum with forces up to ~1000 eV/A). In the data range the
  terms act as Born-Mayer overlap repulsion A exp(-r/rho) (M. Born, J. E.
  Mayer, Z. Phys. 75, 1 (1932); log-additive amplitudes as in the
  Huggins-Mayer form, M. L. Huggins, J. E. Mayer, J. Chem. Phys. 1, 643
  (1933)).
- Element embeddings + combining rules transfer to pairs without short-range
  data (e.g. Ln-Ln, data only >= ~2 A): lambda_Ln and theta_Ln are learned
  from every pair that contains Ln (Ln-H, Ln-O, ... down to ~0.6 A).
- Cutoff at the covalent-radii sum with the p = 6 envelope, as MACE's ZBL
  pair term (mace/modules/radial.py ZBLBasis, github.com/ACEsuit/mace @
  7215fad; used in MACE-MH-1, arXiv:2510.25380); also NequIP's ZBL pair
  potential (github.com/mir-group/nequip, nequip/nn/pair_potential.py). The
  term only acts where covalent spheres overlap (compressed contacts), its
  value, force and force constant vanish at r_c,ij, and nearest neighbours at
  equilibrium (~r_cov,sum) are barely touched. Our variant learns all
  parameters per element (MACE: fixed ZBL, optionally 2 global scalars).
- Tiny init (A_ij = amp_init, default 1e-4 eV A; < 2 meV at 0.1 A): the model
  starts as the model without it. Log-parameterized amplitudes give relative
  (multiplicative) updates, so Adam moves them at the learning rate from a
  tiny start (exactly zero would give no gradient).
- param_scale c (default 8): lambda and theta are stored divided by c and
  multiplied by c in the forward (the "equalized learning rate" trick, T.
  Karras et al., arXiv:1710.10196), so Adam, which moves every stored
  parameter by ~lr per step, moves the log-amplitudes and decay constants
  c times faster; the function and the init are unchanged (c = 8 is a power
  of two: lambda / c * c is exact). Why: with c = 1 the network learns the
  repulsion of the data first and the tiny pair term stays unused (synthetic
  check: pair share of the wall at the data floor 0.00); with c = 4 / 8 / 16
  it carries 0.6 / 0.77 / 0.75 of it and extrapolates far better
  (dev/short-range-arch/notes.txt section 6).

Two modules: PairRepulsion computes the per-atom term (KEY.PAIR_REPULSION_
ENERGY) right after the convolution of layer 1, PairRepulsionToEnergy adds it
to the atomic energy after the rescale. In the middle of the forward its ~30
small kernels are dispatched while the GPU still works on the convolutions
(and their backward runs late in the backward pass), which hides most of
their launch cost (paired login-GPU timing vs a08, 2k atoms: +1-3 % instead
of +4-5 % with the computation at the end; dev/short-range-arch/notes.txt).

Parallel MD: per edge from the segment's own KEY.EDGE_LENGTH and the types of
both atoms (KEY.ATOM_TYPE_ALL in e3gnn/parallel and ML-IAP, where sources can
be ghosts). In e3gnn/parallel (per-segment backward) PairRepulsion moves to
the head of the last segment, after that segment's edge_embedding
(model_build._to_parallel_model), so the term and its force come from that
segment's edge vectors. Half of phi goes to the centre atom of every directed
edge (edge_index[0], always local); the row count is that of the (local)
node features, so each rank sums only its local atoms and no communication
is added.
"""
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


def _softplus_inv(y: float) -> float:
    return y + math.log(-math.expm1(-y))


@compile_mode('script')
class PairRepulsion(nn.Module):
    """
    Per-edge positive, monotonically decreasing pair repulsion (see module
    docstring): per-atom energy (N,) in KEY.PAIR_REPULSION_ENERGY, N = rows
    of KEY.NODE_FEATURE (local atoms in ML-IAP / e3gnn/parallel).
    """

    def __init__(
        self,
        type_map: Dict[int, int],
        num_species: int,
        num_terms: int = 2,
        cutoff_scale: float = 1.0,
        max_cutoff: float = 6.0,
        amp_init: float = 1e-4,
        decay_init: Optional[List[float]] = None,
        poly_p: int = 6,
        param_scale: float = 8.0,
        data_key_atom_type: str = KEY.ATOM_TYPE,
    ) -> None:
        super().__init__()
        from ase.data import covalent_radii

        if decay_init is None:  # pair inverse decay lengths, 2 .. 5 1/A
            decay_init = [
                2.0 * (2.5 ** (k / max(num_terms - 1, 1)))
                for k in range(num_terms)
            ]
        assert len(decay_init) == num_terms, 'decay_init: one value per term'
        assert num_terms > 0 and cutoff_scale > 0 and max_cutoff > 0
        assert amp_init > 0 and all(d > 0 for d in decay_init)
        assert param_scale > 0
        self.num_terms = int(num_terms)
        self.param_scale = float(param_scale)
        self.cutoff_scale = float(cutoff_scale)
        self.max_cutoff = float(max_cutoff)
        self.key_atom_type = data_key_atom_type

        radius = torch.zeros(num_species)
        for z, idx in type_map.items():
            radius[idx] = float(covalent_radii[z])
        self.register_buffer('radius', radius)

        # per element: lambda (log amplitude, half of the pair's) and theta
        # (softplus^-1 of half the pair's inverse decay length), stored / c
        c = self.param_scale
        lam = 0.5 * math.log(amp_init)
        self.log_amp = nn.Parameter(
            torch.full((num_species, num_terms), lam) / c
        )
        theta = torch.tensor([_softplus_inv(0.5 * d) for d in decay_init])
        self.decay_raw = nn.Parameter(theta.repeat(num_species, 1) / c)

        # envelope u(x) = (1 - x)^3 q(x), q(x) = sum_{n<p} C(n+2, 2) x^n
        assert int(poly_p) == poly_p and poly_p >= 1
        n = torch.arange(int(poly_p), dtype=torch.get_default_dtype())
        self.register_buffer('q_powers', n, persistent=False)
        self.register_buffer(
            'q_coeffs', (n + 1.0) * (n + 2.0) / 2.0, persistent=False
        )

    def pair_table(self) -> torch.Tensor:
        """
        [r_c | log A_k | kappa_k] of every ordered type pair, (S * S, 1 + 2K)
        (one gather per edge instead of six)
        """
        ns = self.radius.shape[0]
        radius = self.radius
        rc = torch.clamp(
            self.cutoff_scale * (radius.unsqueeze(1) + radius.unsqueeze(0)),
            max=self.max_cutoff,
        )
        lam = self.log_amp * self.param_scale  # effective lambda, theta
        log_amp = lam.unsqueeze(1) + lam.unsqueeze(0)
        kappa = F.softplus(self.decay_raw * self.param_scale)
        kappa = kappa.unsqueeze(1) + kappa.unsqueeze(0)
        table = torch.cat([rc.unsqueeze(-1).to(log_amp.dtype), log_amp, kappa], -1)
        return table.reshape(ns * ns, 1 + 2 * self.num_terms)

    def pair_energy(
        self, r: torch.Tensor, ti: torch.Tensor, tj: torch.Tensor
    ) -> torch.Tensor:
        """phi_ij(r) per edge (r: (E,), ti / tj: type indices (E,))"""
        k = self.num_terms
        tab = self.pair_table()[ti * self.radius.shape[0] + tj].to(r.dtype)
        # beyond r_c: x = 1, where u(1) = 0 exactly (integer coefficients) and
        # no gradient flows (clamp), so phi and its derivatives are exactly 0
        x = torch.clamp(r / tab[:, 0], max=1.0)
        q = torch.pow(x.unsqueeze(-1), self.q_powers) @ self.q_coeffs
        env = torch.pow(1.0 - x, 3) * q
        terms = torch.exp(tab[:, 1 : 1 + k] - tab[:, 1 + k :] * r.unsqueeze(-1))
        return terms.sum(-1) * env / r

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        r = data[KEY.EDGE_LENGTH]
        edge_idx = data[KEY.EDGE_IDX]
        types = data[self.key_atom_type]
        phi = self.pair_energy(r, types[edge_idx[0]], types[edge_idx[1]])
        data[KEY.PAIR_REPULSION_ENERGY] = torch.zeros(
            data[KEY.NODE_FEATURE].shape[0], dtype=phi.dtype, device=phi.device
        ).index_add(0, edge_idx[0], 0.5 * phi)
        return data


@compile_mode('script')
class PairRepulsionToEnergy(nn.Module):
    """atomic energy (after the shift/scale, eV) += pair repulsion per atom"""

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        energy = data[KEY.ATOMIC_ENERGY]
        e_rep = data[KEY.PAIR_REPULSION_ENERGY].unsqueeze(-1).to(energy.dtype)
        data[KEY.ATOMIC_ENERGY] = energy + e_rep
        return data
