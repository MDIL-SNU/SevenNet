"""
Isolated-atom anchor (config anchor_isolated_atoms: true):

    E_i = shift_{Z_i,m} + scale_{Z_i,m} * [ f(h_i) - f(h0_{Z_i,m}) ]

f(h_i) = the network's scaled atomic energy of atom i (KEY.SCALED_ATOMIC_ENERGY:
the readout of the final node features, per-layer readouts included) and
h0_{Z,m} = the final node state that the SAME network gives an atom of
species Z in modality m without any neighbour (every message zero). An atom
with no neighbour within the cutoff therefore gets E = shift_{Z,m} exactly,
the fixed DFT single-atom energy, for any weights; for every other atom only a
constant per (species, modality) is removed from f, so the fit capacity of
the bulk is unchanged. Pair repulsion (added after the shift/scale) is a sum
over edges and vanishes for an isolated atom as well.

Why a fixed shift alone is not enough: the shift is the isolated-atom energy
only if the network's contribution of an atom with no neighbour is zero.
That holds by construction when the node energy is built from messages only
(Allegro: edge energies; MACE: no residual of the species embedding in the
first layer and a symmetric contraction without constant term), but not
with node-local paths: here the species embedding passes the residual
stream, the (modal-aware) self-interaction linears, FFNs, gates, norms and
the readout, so f(h0_Z) = c_Z != 0. No training structure contains an
isolated atom, so c_Z is unconstrained (measured: up to -1.4 eV for Cs, more
than its cohesive energy: bcc Cs had no energy well, the dissociated state
was lower). The same issue and fix exist in descriptor models: DeePMD-kit's
fitting option atom_ener (energy of an atom in vacuum per type) subtracts
the fitting net evaluated at the zero descriptor of an isolated atom; here
the zero-neighbour state is not a zero input but the network's own h0_Z, so
it is computed by the network itself.

Training (AnchorVirtualNodes first, AnchorSubtract before the shift/scale):
one virtual isolated node per (species, modality) present in the batch is
appended to the batch (no edges, each in its own dummy graph that carries
its modality); the same forward computes f for real and virtual nodes in the
same kernels, AnchorSubtract subtracts f of the matching virtual node from
every real atom (in float64) and drops the virtual rows. The constant thus
carries gradients every step. Real atoms are unaffected by the virtual nodes
(no edge reaches them, every op of the model is node-local or per edge);
forces / stress only see real edges. Cost: + (#species x modalities in the
batch) node rows, no edge work, one torch.unique.

Inference: c_{Z,m} = scale_{Z,m} f(h0_{Z,m}) is a constant once the weights
are fixed. fold_isolated_anchor() computes f(h0) with one edge-less forward
of all (species, modality) pairs (isolated_scaled_energy), subtracts
scale * f(h0) from the shift and removes the two modules: the folded model is
a plain SevenNet model (serial, TorchScript deploy, e3gnn/parallel, ML-IAP pay
nothing). SevenNetCheckpoint.build_model never folds (continue training,
append_modal, ... need the DFT shift); the inference entry points fold
(calculator, deploy, deploy_parallel / build_parallel_model_list, ML-IAP,
torchsim, inference). An unfolded model is still correct everywhere except
in e3gnn/parallel segments (never built with the modules; folded by
build_parallel_model_list) and ML-IAP (patch_model_for_mliap folds first).

Sources (ideas, no code copied): per-element energy reference E0 + learned
interaction energy, I. Batatia et al., MACE, arXiv:2206.07697 (code
github.com/ACEsuit/mace, mace/modules/models.py atomic_energies_fn + E0s);
DeePMD-kit, J. Zeng et al., J. Chem. Phys. 159, 054801 (2023),
arXiv:2304.09409 (fitting net bias_atom_e / atom_ener, deepmd/tf/fit/ener.py);
edge-energy decomposition, A. Musaelian et al., Allegro, Nat. Commun. 14,
579 (2023), arXiv:2204.05249.
"""
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType

from .node_embedding import OnehotEmbedding
from .scale import ModalWiseRescale, Rescale, SpeciesWiseRescale

# per-node tensors that may be read after AnchorSubtract (shift/scale, pair
# repulsion, reduce, user outputs); other intermediate per-node tensors keep
# their virtual rows (nothing reads them afterwards)
ANCHOR_NODE_KEYS = (
    KEY.NODE_FEATURE,
    KEY.NODE_ATTR,
    KEY.ATOM_TYPE,
    KEY.PAIR_REPULSION_ENERGY,
    KEY.SELF_CONNECTION_TEMP,
    KEY.LAYER_ENERGY,
    KEY.NODE_COORD_LOG1P,
)
ANCHOR_MODULE_KEYS = ('anchor_virtual_nodes', 'anchor_subtract')


@compile_mode('script')
class AnchorVirtualNodes(nn.Module):
    """
    First module of an anchored model: appends one virtual isolated node per
    (species, modality) present in the data, after the real nodes, as type
    indices in KEY.NODE_FEATURE (before the one-hot embedding). No edge is
    added (edge_index is untouched: virtual rows are never a source or a
    receiver). Batched data: every virtual node gets its own dummy graph id
    after the real graphs; with modalities the dummy graph carries the virtual
    node's modality (KEY.MODAL_TYPE extended), so modal-aware linears and
    shift/scale see it. Non-batched data has one modality for all nodes.
    KEY.ANCHOR_INDEX: row offset (into the virtual rows) of every real atom.
    """

    def __init__(self, num_species: int, use_modality: bool) -> None:
        super().__init__()
        self.num_species = int(num_species)
        self.use_modality = bool(use_modality)
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        types = data[KEY.NODE_FEATURE]
        modal_batch = self.use_modality and self._is_batch_data
        if modal_batch:
            modal = data[KEY.MODAL_TYPE]
            key = modal.index_select(0, data[KEY.BATCH]) * self.num_species
            key = key + types
        else:
            key = types
        uniq, inv = torch.unique(key, sorted=True, return_inverse=True)
        data[KEY.ANCHOR_INDEX] = inv
        data[KEY.NODE_FEATURE] = torch.cat(
            [types, torch.remainder(uniq, self.num_species)]
        )
        if self._is_batch_data:
            batch = data[KEY.BATCH]
            data[KEY.ANCHOR_BATCH] = batch
            ids = torch.arange(uniq.shape[0], dtype=batch.dtype, device=batch.device)
            if modal_batch:
                modal = data[KEY.MODAL_TYPE]
                data[KEY.ANCHOR_MODAL] = modal
                data[KEY.MODAL_TYPE] = torch.cat(
                    [modal, torch.div(uniq, self.num_species, rounding_mode='floor')]
                )
                v_batch = ids + modal.shape[0]
            else:
                v_batch = ids + (batch[-1:] + 1)
            data[KEY.BATCH] = torch.cat([batch, v_batch])
        return data


@compile_mode('script')
class AnchorSubtract(nn.Module):
    """
    Right before the shift/scale of an anchored model: scaled atomic energy of
    every real atom minus that of its virtual isolated twin (float64), then
    the virtual rows are dropped from the per-node tensors (ANCHOR_NODE_KEYS)
    and the real batch / per-graph modality restored.
    """

    def __init__(self, use_modality: bool) -> None:
        super().__init__()
        self.use_modality = bool(use_modality)
        self.node_keys: List[str] = list(ANCHOR_NODE_KEYS)
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        inv = data[KEY.ANCHOR_INDEX]
        n = inv.shape[0]
        x = data[KEY.SCALED_ATOMIC_ENERGY].to(torch.float64)
        n_total = x.shape[0]
        data[KEY.SCALED_ATOMIC_ENERGY] = x[:n] - x[n:].index_select(0, inv)
        for k in self.node_keys:
            if k in data:
                t = data[k]
                if t.dim() > 0 and t.shape[0] == n_total:
                    data[k] = t[:n]
        if self._is_batch_data:
            data[KEY.BATCH] = data[KEY.ANCHOR_BATCH]
            if self.use_modality:
                modal = data[KEY.ANCHOR_MODAL]
                data[KEY.MODAL_TYPE] = modal
                if KEY.MODAL_ATTR in data:
                    data[KEY.MODAL_ATTR] = data[KEY.MODAL_ATTR][: modal.shape[0]]
        return data


def is_anchored(model: nn.Module) -> bool:
    """model (AtomGraphSequential) has the unfolded anchor modules"""
    return any(
        isinstance(m, (AnchorVirtualNodes, AnchorSubtract))
        for m in model._modules.values()
    )


def _compute_dtype(model: nn.Module) -> torch.dtype:
    for name, p in model.named_parameters():
        if p.is_floating_point() and not name.startswith('rescale_atomic_energy'):
            return p.dtype
    return torch.get_default_dtype()


@torch.no_grad()
def isolated_scaled_energy(
    model: nn.Module,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """
    f(h0_{Z,m}): scaled atomic energy (before the shift/scale) of an isolated
    atom of every species Z and modality m, (num_modalities or 1, num_species),
    float64, from one edge-less batched forward of the model's own modules
    (backend and dtype as they are; the anchor modules and a modal-deploy
    input module are skipped, so it also works on a plain or folded model).
    """
    from .sequential import _ModalInputPrepare

    modules = model._modules
    num_species = modules['onehot_idx_to_onehot'].num_classes
    modal_embed = modules.get('one_hot_modality', None)
    num_modal = modal_embed.num_classes if modal_embed is not None else 1
    if device is None:
        device = next(model.parameters()).device
    dtype = _compute_dtype(model)
    n = num_modal * num_species
    data: Dict[str, torch.Tensor] = {
        KEY.NODE_FEATURE: torch.arange(num_species, device=device).repeat(num_modal),
        KEY.BATCH: torch.arange(n, device=device),
        KEY.NUM_ATOMS: torch.ones(n, dtype=torch.int64, device=device),
        KEY.EDGE_IDX: torch.zeros(2, 0, dtype=torch.int64, device=device),
        KEY.EDGE_VEC: torch.zeros(0, 3, dtype=dtype, device=device),
    }
    if modal_embed is not None:
        data[KEY.MODAL_TYPE] = torch.arange(
            num_modal, device=device
        ).repeat_interleave(num_species)
    was_batch = getattr(model, 'is_batch_data', True)
    model.set_is_batch_data(True)  # type: ignore
    try:
        for name, module in modules.items():
            if isinstance(module, (AnchorVirtualNodes, _ModalInputPrepare)):
                continue
            if isinstance(module, AnchorSubtract) or name == 'rescale_atomic_energy':
                break
            data = module(data)
            if isinstance(module, OnehotEmbedding):  # always float32 one-hot
                for k in (module.key_output, module.key_additional_output):
                    if k is not None and data[k].is_floating_point():
                        data[k] = data[k].to(dtype)
    finally:
        model.set_is_batch_data(was_batch)  # type: ignore
    return data[KEY.SCALED_ATOMIC_ENERGY].reshape(num_modal, num_species).to(
        torch.float64
    )


def fold_into_rescale(rescale: nn.Module, f0: torch.Tensor) -> nn.Module:
    """
    shift <- shift - scale * f0 (f0: (num_modalities or 1, num_species)
    scaled isolated-atom energies). Returns the updated module (in place), or
    a new SpeciesWiseRescale for a scalar Rescale. A modal-wise f0 turns a
    species-wise shift of ModalWiseRescale into a modal-wise one.
    """
    f0 = f0.detach()
    if isinstance(rescale, ModalWiseRescale):
        shift, scale = rescale.shift.detach(), rescale.scale.detach()
        f0 = f0.to(device=shift.device, dtype=torch.float64)
        num_modal, num_species = f0.shape
        sc = scale.to(torch.float64)
        if not rescale.use_modal_wise_scale:
            sc = sc.unsqueeze(0).expand(num_modal, num_species)
        sh = shift.to(torch.float64)
        if not rescale.use_modal_wise_shift:
            sh = sh.unsqueeze(0).expand(num_modal, num_species)
        assert sh.shape == f0.shape, (sh.shape, f0.shape)
        rescale.shift = nn.Parameter(
            (sh - sc * f0).to(shift.dtype).contiguous(),
            requires_grad=rescale.shift.requires_grad,
        )
        rescale.use_modal_wise_shift = True
        return rescale
    if f0.shape[0] != 1:
        raise ValueError('modal-wise isolated energies need ModalWiseRescale')
    if isinstance(rescale, SpeciesWiseRescale):
        shift, scale = rescale.shift.detach(), rescale.scale.detach()
        f0 = f0[0].to(device=shift.device, dtype=torch.float64)
        new = shift.to(torch.float64) - scale.to(torch.float64) * f0
        rescale.shift = nn.Parameter(
            new.to(shift.dtype), requires_grad=rescale.shift.requires_grad
        )
        return rescale
    if isinstance(rescale, Rescale):
        shift, scale = rescale.shift.detach(), rescale.scale.detach()
        f0 = f0[0].to(device=shift.device, dtype=torch.float64)
        sh = shift.to(torch.float64).expand_as(f0)
        sc = scale.to(torch.float64).expand_as(f0)
        new = SpeciesWiseRescale(
            (sh - sc * f0).tolist(),
            sc.tolist(),
            data_key_in=rescale.key_input,
            data_key_out=rescale.key_output,
        )
        return new.to(device=shift.device, dtype=shift.dtype)
    raise ValueError(f'unknown shift/scale module {type(rescale).__name__}')


def fold_isolated_anchor(
    model: nn.Module, f0: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Fold the anchor of an anchored model (in place) for inference: the shift
    becomes shift - scale * f(h0) and the two anchor modules are removed, so
    the result is a plain model with the same energies / forces / stress.
    No-op for a model without the anchor modules (not anchored, or already
    folded). f0: isolated_scaled_energy of the same weights (computed if None).
    Do not save the folded model's state dict as a checkpoint of the anchored
    config (the shift would be subtracted twice).
    """
    if not is_anchored(model):
        return model
    if f0 is None:
        f0 = isolated_scaled_energy(model)
    rescale = model._modules['rescale_atomic_energy']
    folded = fold_into_rescale(rescale, f0)
    if folded is not rescale:
        model._modules['rescale_atomic_energy'] = folded
    for key in ANCHOR_MODULE_KEYS:
        if key in model._modules:
            del model._modules[key]
    assert not is_anchored(model)
    return model
