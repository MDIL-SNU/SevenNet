"""
Parallel MD gives the same energy, forces and stress as the serial model
(float64, CPU), for every architecture option. SevenNet has two parallel MD
interfaces; both are emulated exactly in Python here:

1. LAMMPS e3gnn/parallel (sevenn/pair_e3gnn/pair_e3gnn_parallel.cpp): the
   segments of sevenn/scripts/deploy.py (build_parallel_model_list, the same
   code deploy_parallel saves) run as PairE3GNNParallel::compute does on every
   rank of a spatial decomposition: graph nodes = local atoms + ghost atoms
   within the cutoff deduplicated by tag (periodic images of a rank's own
   atoms map to the local node), segment-0 inputs (x = local types,
   x_ghost = ghost types, edge_index, edge_vec, num_atoms, nlocal),
   forward_comm of x between segments (ghost rows = the owner's local rows),
   per-segment torch.autograd.grad w.r.t. (edge_vec, x, self_cont_tmp,
   x_ghost) with the C++ wrt-tensor bookkeeping, reverse_comm of the ghost
   gradients, dE/dr -> forces and virial. Once with the eager segments, and
   once with e3nn.util.jit.script + torch.jit.freeze (as deploy_parallel),
   with the graph executor optimization off as in the pair style. One test
   goes through a real checkpoint file, deploy_parallel and torch.jit.load.
2. LAMMPS ML-IAP (sevenn/mliap.py): the serial model wrapped by
   patch_model_for_mliap runs mliap_compute_forces on an emulated lmp_data
   (forward_exchange: ghost rows <- owner's rows; reverse_exchange:
   scatter-add of the ghost gradients to the owners), with one rank (all
   ghosts are periodic images of local atoms) and with two ranks in lockstep
   (threads + barrier). Ghosts are LAMMPS atoms (one per periodic image) and
   include extra ghosts beyond the cutoff (skin) that no edge uses. One test
   goes through SevenNetMLIAPWrapper with a stub lammps module.

Standard check for a new op: add its config to CONFIGS below (next to
tests/unit_tests/test_physics_consistency.py) and run this file. A new module
that transforms the node features before the first convolution needs a ghost
twin (model_build._ghost_twin); one that gathers source-node features or
types per edge must read x_ghost / KEY.ATOM_TYPE_ALL in parallel (see
EdgeAttention, NodeCondRadialInput); a position-dependent tensor must be
recomputed in every e3gnn/parallel segment (see edge_pair_input).
"""
import contextlib
import copy
import sys
import threading
import types

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.neighborlist import primitive_neighbor_list
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType
from sevenn.nn._ghost_exchange import mliap_compute_forces, patch_model_for_mliap
from sevenn.nn.node_embedding import OnehotEmbedding
from sevenn.scripts.deploy import build_parallel_model_list
from tests.unit_tests.test_arch_options import _metaformer, _partial
from tests.unit_tests.test_model import cutoff, get_model, get_model_config
from tests.unit_tests.test_physics_consistency import _graph

_xplor = {'cutoff_function': {'cutoff_function_name': 'XPLOR', 'cutoff_on': 3.5}}
_modal = {  # as our runs and 7net-omni (deployed with a modal)
    'use_modality': True,
    KEY.NUM_MODALITIES: 2,
    KEY.MODAL_MAP: {'mpa': 0, 'omat24': 1},
    'use_modal_node_embedding': False,
    'use_modal_self_inter_intro': True,
    'use_modal_self_inter_outro': True,
    'use_modal_output_block': True,
    'use_modal_wise_shift': True,
    'use_modal_wise_scale': False,
    'shift': [-1.5, -2.5],
    'scale': 1.0,
}
_A0 = {**_partial, **_xplor, **_modal}  # nequip block, linear residual
_A2 = {**_metaformer, **_xplor, **_modal}  # metaformer, identity residual
_A3 = {**_A2, 'radial_pair_embedding': True}
_A4 = {**_A3, 'readout_activation': 'silu'}
CONFIGS = {
    # nequip block (original SevenNet path)
    'nequip_sc': {},
    'A0_nequip_linear_sc': _A0,
    # metaformer block, node-local FFN / gate / residual variants
    'A2_metaformer': _A2,
    'swiglu': {**_A2, 'ffn_type': 'swiglu'},
    'layer_scale': {**_A2, 'layer_scale_init': 0.1},
    'ffn_hidden_scale_4': {**_A2, 'ffn_hidden_scale': 4},
    'final_norm_no_ffn': {**_A2, 'use_final_norm': True, 'use_ffn': False},
    'linear_residual': {**_A2, 'self_connection_type': 'linear'},
    # element-pair radial MLP and the edge-embedding options it reads
    'A3_pair_radial': _A3,
    'A4_silu_readout': _A4,
    'poly_envelope': {**_A4, 'cutoff_function': {
        'cutoff_function_name': 'poly_cut', 'poly_cut_p_value': 6}},
    'frozen_bessel': {**_A4, 'radial_basis': {
        'radial_basis_name': 'bessel', 'bessel_basis_num': 8,
        'trainable_coeff': False}},
}
# subsets for the slower tests (scripted segments, checkpoint files, GPU)
SCRIPT_CONFIGS = [n for n in ('A0_nequip_linear_sc', 'a06_radial_node_cond',
                              'kitchen_sink') if n in CONFIGS]
FILE_CONFIG = [n for n in ('kitchen_sink', 'A0_nequip_linear_sc')
               if n in CONFIGS][0]
GPU_CONFIGS = [n for n in ('A0_nequip_linear_sc', 'moe', 'bilinear_gate',
                           'kitchen_sink', 's2_bilinear') if n in CONFIGS]
MODAL = 'mpa'

# (supercell, grid): 2 ranks along a (a ghost tag is reached through both
# periodic sides -> dedup), 3 ranks along a, 2x2 ranks; every rank also sees
# periodic images of its own atoms along the undivided axes
E3GNN_DOMAINS = [((2, 2, 2), (2, 1, 1)), ((3, 2, 2), (3, 1, 1)),
                 ((2, 2, 2), (2, 2, 1))]


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@contextlib.contextmanager
def _default_dtype(dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


@compile_mode('script')
class _OnehotEmbedding64(OnehotEmbedding):
    """one-hot in float64 (OnehotEmbedding always gives float32)"""

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        inp = data[self.key_x]
        embd = torch.nn.functional.one_hot(inp, self.num_classes).to(torch.float64)
        data[self.key_output] = embd
        if self.key_additional_output is not None:
            data[self.key_additional_output] = embd
        if self.key_save is not None:
            data[self.key_save] = inp
        return data


def _double(model):
    model.double()
    for k, m in list(model._modules.items()):
        if isinstance(m, OnehotEmbedding):
            model._modules[k] = _OnehotEmbedding64(
                m.num_classes, m.key_x, m.key_output, m.key_save,
                m.key_additional_output,
            )
    return model


def _randomize(model):
    """perturb constant-initialized parameters (zero pair tables, attention
    biases, node-conditioning projections, FiLM, norm affine, ...): a wrong
    ghost type or feature would hide behind them"""
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if any(s in n for s in ('denominator', 'rescale', 'coeffs')):
                continue
            if p.numel() > 0 and bool(torch.all(p == p.reshape(-1)[0])):
                p.add_(0.3 * torch.randn(p.shape, generator=gen, dtype=p.dtype))


def _config(cf):
    config = get_model_config()
    config.update(copy.deepcopy(cf))
    return config


def _serial(cf):
    """randomized float64 serial model (modal 'mpa' if multimodal)"""
    torch.manual_seed(0)
    model = _double(get_model(copy.deepcopy(cf)))
    _randomize(model)
    model.set_is_batch_data(False)
    if model.modal_map:
        model.prepare_modal_deploy(MODAL)
    model.eval()
    return model


def _segments(model, cf, script=False):
    """e3gnn/parallel segments built as deploy_parallel does (float64)"""
    sd = {k: v for k, v in model.state_dict().items()}
    modal = MODAL if model.modal_map else None
    segs = build_parallel_model_list(sd, _config(cf), modal)
    out = []
    for seg in segs:
        seg = _double(seg)
        seg.set_is_batch_data(False)
        seg.eval()
        if script:
            import e3nn.util.jit

            seg = torch.jit.freeze(e3nn.util.jit.script(seg))
        out.append(seg)
    return out


def _atoms(supercell, seed=0):
    """rattled, strained NaCl supercell with some H and O (4 species)"""
    atoms = bulk('NaCl', 'rocksalt', a=5.63, cubic=True) * supercell
    rng = np.random.default_rng(seed)
    atoms.set_cell(atoms.cell @ (np.eye(3) + 0.02 * rng.normal(size=(3, 3))),
                   scale_atoms=True)
    atoms.rattle(0.05, seed=seed + 1)
    sym = atoms.get_chemical_symbols()
    pick = rng.choice(len(atoms), size=len(atoms) // 6, replace=False)
    for n, i in enumerate(pick):
        sym[i] = 'O' if n % 2 == 0 else 'H'
    atoms.set_chemical_symbols(sym)
    atoms.wrap()
    return atoms


def _serial_efs(model, atoms):
    out = model(_graph(atoms))
    return (
        out[KEY.PRED_TOTAL_ENERGY].item(),
        out[KEY.PRED_FORCE].detach().numpy(),
        out[KEY.PRED_STRESS].detach().numpy(),
    )


def _owner(atoms, grid):
    frac = atoms.get_scaled_positions(wrap=True)
    idx = [np.minimum((frac[:, d] * grid[d]).astype(int), grid[d] - 1)
           for d in range(3)]
    return np.ravel_multi_index(idx, grid)


def _types(atoms, config):
    tm = config[KEY.TYPE_MAP]
    return np.array([tm[z] for z in atoms.get_atomic_numbers()])


def _efs_from_edges(atoms, ranks, energies, dE_dr):
    """forces and stress (sevenn order, = -virial / V) as the pair styles"""
    forces = np.zeros((len(atoms), 3))
    virial = np.zeros(6)
    for rk, f in zip(ranks, dE_dr):
        f = f.detach().cpu().double().numpy()
        np.add.at(forces, rk['tag_i'], f)
        np.add.at(forces, rk['tag_j'], -f)
        v = rk['vec']
        voigt = np.concatenate(
            [v * f, (v[:, 0] * f[:, 1])[:, None], (v[:, 1] * f[:, 2])[:, None],
             (v[:, 2] * f[:, 0])[:, None]], axis=1,
        )
        virial -= voigt.sum(0)
    return sum(energies), forces, virial / atoms.get_volume()


# ------------------------------------------------------------------ e3gnn/parallel
def _e3gnn_domains(atoms, grid, rc=cutoff):
    """per-rank graph of pair_e3gnn_parallel.cpp: local atoms, ghost nodes by tag"""
    owner = _owner(atoms, grid)
    i, j, S = primitive_neighbor_list(
        'ijS', atoms.pbc, atoms.cell, atoms.positions, rc
    )
    vec = atoms.positions[j] - atoms.positions[i] + S @ np.asarray(atoms.cell)
    ranks = []
    for r in range(int(np.prod(grid))):
        local = np.nonzero(owner == r)[0]
        graph_idx = {int(t): n for n, t in enumerate(local)}
        ghost, src, dst, ev, images = [], [], [], [], set()
        for ii, a in enumerate(local):  # ilist order, then neighbor order
            for e in np.nonzero(i == a)[0]:
                b = int(j[e])
                if b not in graph_idx:
                    graph_idx[b] = len(graph_idx)
                    ghost.append(b)
                src.append(ii)
                dst.append(graph_idx[b])
                ev.append(vec[e])
                images.add((b, tuple(S[e])))
        nodes = np.concatenate([local, np.array(ghost, dtype=int)])
        src, dst = np.array(src), np.array(dst)
        ranks.append({
            'local': local, 'ghost': np.array(ghost, dtype=int), 'nodes': nodes,
            'src': src, 'dst': dst, 'vec': np.array(ev),
            'tag_i': nodes[src], 'tag_j': nodes[dst],
            'n_self_image': sum(1 for b, s in images
                                if any(s) and owner[b] == r),
            'n_dup_ghost': len({(b, s) for b, s in images if owner[b] != r})
            - len(ghost),
        })
    return owner, ranks


def _ghost_sources(ranks, device):
    """per rank: {owner rank: (ghost rows, owner's local rows)}"""
    local_row = {int(t): (r, n) for r, rk in enumerate(ranks)
                 for n, t in enumerate(rk['local'])}
    for rk in ranks:
        src = {}
        for g, t in enumerate(rk['ghost']):
            ro, no = local_row[int(t)]
            src.setdefault(ro, ([], []))
            src[ro][0].append(g)
            src[ro][1].append(no)
        rk['ghost_src'] = {
            ro: (torch.tensor(a, device=device), torch.tensor(b, device=device))
            for ro, (a, b) in src.items()
        }


def _e3gnn_parallel(
    segments, atoms, config, grid, dtype=torch.float64, device='cpu'
):
    """Python replica of PairE3GNNParallel::compute (all ranks in lockstep)"""
    owner, ranks = _e3gnn_domains(atoms, grid, config[KEY.CUTOFF])
    tp = _types(atoms, config)
    _ghost_sources(ranks, device)

    def forward_comm(x_local, rk):  # x_ghost: the owners' local rows
        x_ghost = x_local[0].new_zeros(len(rk['ghost']), x_local[0].shape[1])
        for ro, (g, no) in rk['ghost_src'].items():
            x_ghost[g] = x_local[ro][no]
        return x_ghost

    def reverse_comm(x_local_grad, x_ghost_grad):  # ghost grads -> owners
        full = [g.clone() for g in x_local_grad]
        for rk, gg in zip(ranks, x_ghost_grad):
            for ro, (g, no) in rk['ghost_src'].items():
                full[ro].index_add_(0, no, gg[g])
        return full

    def dev(a):
        return torch.as_tensor(a, device=device)

    out, wrt = [], []
    for rk in ranks:
        nloc = len(rk['local'])
        inp = {
            'x': dev(tp[rk['local']]),
            'x_ghost': dev(tp[rk['ghost']]),
            'edge_index': dev(np.stack([rk['src'], rk['dst']])),
            'edge_vec': dev(rk['vec']).to(dtype).requires_grad_(True),
            'num_atoms': dev([nloc]),
            'nlocal': torch.tensor([nloc]),  # on cpu
        }
        wrt.append([[inp['edge_vec']]])
        out.append(segments[0](inp))
    x_dims = []  # per exchange (forward_comm)
    for seg in segments[1:]:
        x_local = [o['x'].detach() for o in out]
        x_dims.append(x_local[0].shape[1])
        for r, rk in enumerate(ranks):
            o = out[r]
            x_ghost = forward_comm(x_local, rk)
            x_ghost.requires_grad_(True)
            o['x_ghost'] = x_ghost
            o['edge_vec'] = o['edge_vec'].clone()
            wrt[r].append([o['edge_vec'], o['x'], o['self_cont_tmp'], o['x_ghost']])
            out[r] = seg(o)
    energy = [o['inferred_total_energy'].squeeze() for o in out]

    n_rank, n_seg = len(ranks), len(segments)
    dE_dr = [torch.zeros(len(rk['src']), 3, dtype=dtype, device=device)
             for rk in ranks]
    of = [None] * n_rank
    x_full = [None] * n_rank
    sc_grad = [None] * n_rank
    x_loc_grad = [None] * n_rank
    x_gh_grad = [None] * n_rank
    for k in reversed(range(n_seg)):
        for r in range(n_rank):
            if k == n_seg - 1:
                grads = torch.autograd.grad([energy[r]], wrt[r][k])
            else:
                grads = torch.autograd.grad(
                    of[r], wrt[r][k], [x_full[r], sc_grad[r]]
                )
            dE_dr[r] = dE_dr[r] + grads[0]
            if k == 0:
                continue
            of[r] = [wrt[r][k][1], wrt[r][k][2]]  # x, self_cont_tmp
            x_loc_grad[r] = grads[1].detach()
            sc_grad[r] = grads[2]  # no communication
            x_gh_grad[r] = grads[3].detach()
        if k > 0:
            x_full = reverse_comm(x_loc_grad, x_gh_grad)
    energies = [e.item() for e in energy]
    for rk in ranks:
        rk['x_dims'] = x_dims
    return _efs_from_edges(atoms, ranks, energies, dE_dr), ranks


def _assert_close(par, ser, tag):
    (e, f, s), (e0, f0, s0) = par, ser
    assert abs(e - e0) <= 1e-10 * max(1.0, abs(e0)), (tag, e, e0)
    fmax = max(1.0, np.abs(f0).max())
    assert np.abs(f - f0).max() <= 1e-8 * fmax, (tag, np.abs(f - f0).max())
    smax = max(1.0, np.abs(s0).max())
    assert np.abs(s - s0).max() <= 1e-8 * smax, (tag, np.abs(s - s0).max())
    return abs(e - e0), np.abs(f - f0).max(), np.abs(s - s0).max()


@pytest.mark.parametrize('name', list(CONFIGS))
def test_e3gnn_parallel(name):
    cf = CONFIGS[name]
    model = _serial(cf)
    segs = _segments(model, cf)
    for supercell, grid in E3GNN_DOMAINS:
        atoms = _atoms(supercell)
        ser = _serial_efs(model, atoms)
        assert np.abs(ser[1]).max() > 1e-3
        par, ranks = _e3gnn_parallel(segs, atoms, _config(cf), grid)
        # the decomposition exercises ghosts, self images and ghost dedup
        assert all(len(rk['ghost']) > 0 for rk in ranks)
        assert all(rk['n_self_image'] > 0 for rk in ranks)
        if grid == (2, 1, 1):
            assert all(rk['n_dup_ghost'] > 0 for rk in ranks)
        _assert_close(par, ser, (name, grid))


@pytest.mark.parametrize('name', SCRIPT_CONFIGS)
def test_e3gnn_parallel_torchscript(name):
    """segments scripted and frozen as deploy_parallel, executor as LAMMPS"""
    cf = CONFIGS[name]
    model = _serial(cf)
    segs = _segments(model, cf, script=True)
    supercell, grid = E3GNN_DOMAINS[0]
    atoms = _atoms(supercell)
    with torch.jit.optimized_execution(False):
        par, _ = _e3gnn_parallel(segs, atoms, _config(cf), grid)
    _assert_close(par, _serial_efs(model, atoms), name)


@pytest.mark.parametrize('name', list(CONFIGS))
def test_parallel_segments_structure(name):
    """ghost twins, per-segment edge inputs, ghost-aware edge modules"""
    from sevenn.nn.normalization import SelfConnectionResidualIntro
    from sevenn.nn.radial_pair import EdgePairInput

    cf = CONFIGS[name]
    model = _serial(cf)
    segs = _segments(model, cf)
    names0 = list(segs[0]._modules)
    # layer 0: every node-wise module up to the first convolution has a ghost
    # twin (same weights) right after it; residual intros act on local atoms
    i0 = names0.index('onehot_to_feature_x')
    i1 = names0.index('0_self_interaction_1')
    for k in names0[i0: i1 + 1]:
        if k.startswith('ghost_') or k == 'one_hot_ghost':
            continue
        if k.endswith('_self_connection_intro'):
            assert f'ghost_{k}' not in names0
            continue
        assert names0.index(f'ghost_{k}') > names0.index(k)
        sd = segs[0]._modules[k].state_dict()
        sd_g = segs[0]._modules[f'ghost_{k}'].state_dict()
        assert sd.keys() == sd_g.keys()
        assert all(torch.equal(sd[p], sd_g[p]) for p in sd)
    # every segment rebuilds its position-dependent edge inputs
    pair = any(isinstance(m, EdgePairInput) for m in model.modules())
    head = ['edge_embedding'] + (['edge_pair_input'] if pair else [])
    for seg in segs[1:]:
        assert list(seg._modules)[: len(head)] == head
    need_types = False
    for seg in segs:
        for k, m in seg._modules.items():
            if isinstance(m, EdgePairInput):
                assert m.key_atom_type == KEY.ATOM_TYPE_ALL
                need_types = True
            if isinstance(m, SelfConnectionResidualIntro):
                assert m.clone_identity == k.endswith('_self_connection_intro')
    assert ('atom_type_all' in names0) == need_types
    # the serial model is untouched
    for m in model.modules():
        assert not getattr(m, 'is_parallel', False)
        assert not getattr(m, 'clone_identity', False)
        assert getattr(m, 'key_atom_type', KEY.ATOM_TYPE) == KEY.ATOM_TYPE


def _save_checkpoint(model_cf, path, dtype=torch.float32):
    import sevenn

    torch.manual_seed(0)
    with _default_dtype(dtype):
        model = get_model(copy.deepcopy(model_cf))
        _randomize(model)
    cfg = _config(model_cf)
    cfg['version'] = sevenn.__version__
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, path)
    return path


def _serial_f32(cp_path, atoms):
    from sevenn.util import model_from_checkpoint

    with _default_dtype(torch.float32):
        model, _ = model_from_checkpoint(str(cp_path))
        model.set_is_batch_data(False)
        if model.modal_map:
            model.prepare_modal_deploy(MODAL)
        model.eval()
        g = _graph(atoms)
        for k, v in g.items():
            if torch.is_tensor(v) and v.is_floating_point():
                g[k] = v.float()
        out = model(g)
    return (
        out[KEY.PRED_TOTAL_ENERGY].item(),
        out[KEY.PRED_FORCE].detach().numpy().astype(np.float64),
        out[KEY.PRED_STRESS].detach().numpy().astype(np.float64),
    )


def _assert_close_f32(par, ser, tag):
    (e, f, s), (e0, f0, s0) = par, ser
    assert abs(e - e0) <= 1e-5 * max(1.0, abs(e0)), (tag, e, e0)
    assert np.abs(f - f0).max() <= 1e-4 * max(1.0, np.abs(f0).max()), tag
    assert np.abs(s - s0).max() <= 1e-4 * max(1.0, np.abs(s0).max()), tag


def test_deploy_parallel_files(tmp_path):
    """checkpoint -> deploy_parallel -> torch.jit.load -> emulated LAMMPS (fp32)"""
    from sevenn.scripts.deploy import deploy_parallel

    cf = CONFIGS[FILE_CONFIG]
    cp = _save_checkpoint(cf, tmp_path / 'cp.pth')
    out_dir = tmp_path / 'deployed'
    with _default_dtype(torch.float32):  # as the CLI
        deploy_parallel(str(cp), str(out_dir), modal=MODAL)
    segs, comm = [], []
    for t in range(_config(cf)[KEY.NUM_CONVOLUTION]):
        extra = {'comm_size': '', 'cutoff': ''}
        segs.append(torch.jit.load(
            str(out_dir / f'deployed_parallel_{t}.pt'), _extra_files=extra
        ))
        comm.append(int(extra['comm_size']))
        assert float(extra['cutoff']) == cutoff
    atoms = _atoms((2, 2, 2))
    with _default_dtype(torch.float32), torch.jit.optimized_execution(False):
        par, ranks = _e3gnn_parallel(
            segs, atoms, _config(cf), (2, 1, 1), dtype=torch.float32
        )
    _assert_close_f32(par, _serial_f32(cp, atoms), 'deploy_parallel')
    # the comm buffer (meta comm_size) holds the largest exchanged x
    assert set(comm) == {max(ranks[0]['x_dims'])}, (comm, ranks[0]['x_dims'])


# --------------------------------------------------------------------- ML-IAP
def _mliap_domains(atoms, grid, skin=1.0, rc=cutoff, device='cpu'):
    """per-rank LAMMPS atoms: locals, one ghost per periodic image within
    cutoff + skin of a local atom, pairs (local i, any j) within the cutoff"""
    owner = _owner(atoms, grid)
    i, j, S, d = primitive_neighbor_list(
        'ijSd', atoms.pbc, atoms.cell, atoms.positions, rc + skin
    )
    vec = atoms.positions[j] - atoms.positions[i] + S @ np.asarray(atoms.cell)
    ranks = []
    for r in range(int(np.prod(grid))):
        local = np.nonzero(owner == r)[0]
        nloc = len(local)
        row = {int(t): n for n, t in enumerate(local)}
        ghosts = {}  # (tag, image) -> ghost index
        pi, pj, rij = [], [], []
        for ii, a in enumerate(local):
            for e in np.nonzero(i == a)[0]:
                b, s = int(j[e]), tuple(int(v) for v in S[e])
                if not any(s) and owner[b] == r:
                    jj = row[b]
                else:
                    jj = nloc + ghosts.setdefault((b, s), len(ghosts))
                if d[e] < rc:
                    pi.append(ii)
                    pj.append(jj)
                    rij.append(vec[e])
        gtag = np.array([b for (b, _), _ in sorted(ghosts.items(),
                                                   key=lambda kv: kv[1])],
                        dtype=int)
        nodes = np.concatenate([local, gtag])
        pi, pj = np.array(pi), np.array(pj)
        ranks.append({
            'local': local, 'ghost': gtag, 'nodes': nodes, 'pi': pi, 'pj': pj,
            'vec': np.array(rij), 'tag_i': nodes[pi], 'tag_j': nodes[pj],
            'n_unused_ghost': len(gtag) - len(np.unique(pj[pj >= nloc])),
        })
    _ghost_sources(ranks, device)
    return ranks


class _World:
    def __init__(self, ranks):
        self.ranks = ranks
        self.barrier = threading.Barrier(len(ranks), timeout=600)
        self.buf = [None] * len(ranks)


class _FakeLmpData:
    """the part of LAMMPS ML-IAP's lmp_data used by sevenn (MPI emulated)"""

    def __init__(self, world, rank, atoms):
        rk = world.ranks[rank]
        self.world, self.rank, self.rk = world, rank, rk
        self.nlocal = len(rk['local'])
        self.ntotal = len(rk['nodes'])
        self.npairs = len(rk['pi'])
        self.pair_i, self.pair_j, self.rij = rk['pi'], rk['pj'], rk['vec']
        self.elems = atoms.get_atomic_numbers()[rk['nodes']]
        self.eatoms = np.zeros(self.nlocal)
        self.energy = None
        self.pair_forces = None
        self.n_exchange = 0

    def forward_exchange(self, feats, out, vec_len):
        w, n = self.world, self.nlocal
        assert feats.shape == (self.ntotal, vec_len)
        w.buf[self.rank] = feats[:n].clone()
        w.barrier.wait()
        out.copy_(feats)
        for ro, (g, no) in self.rk['ghost_src'].items():
            out[n + g] = w.buf[ro][no]
        w.barrier.wait()
        self.n_exchange += 1

    def reverse_exchange(self, grads, out, vec_len):
        w, n = self.world, self.nlocal
        w.buf[self.rank] = grads[n:].clone()
        w.barrier.wait()
        out.copy_(grads)
        out[n:] = 0.0
        for s, rk in enumerate(w.ranks):
            if self.rank in rk['ghost_src']:
                g, no = rk['ghost_src'][self.rank]
                out[:n].index_add_(0, no, w.buf[s][g])
        w.barrier.wait()

    def update_pair_forces_gpu(self, fij):
        self.pair_forces = fij


def _mliap_model(model):
    model = copy.deepcopy(model)
    patch_model_for_mliap(model)
    model.delete_module_by_key('force_output')
    model.eval()
    return model


def _run_mliap(models, atoms, grid, dtype=torch.float64, compute=None,
               rc=cutoff, device='cpu'):
    ranks = _mliap_domains(atoms, grid, rc=rc, device=device)
    world = _World(ranks)
    lmps = [_FakeLmpData(world, r, atoms) for r in range(len(ranks))]
    compute = compute or (
        lambda m, lmp: mliap_compute_forces(m, lmp, torch.device(device), dtype)
    )
    errors = []

    def run(r):
        try:
            compute(models[r], lmps[r])
        except BaseException as exc:  # noqa: B036
            errors.append(exc)
            world.barrier.abort()

    if len(ranks) == 1:
        run(0)
    else:
        threads = [threading.Thread(target=run, args=(r,))
                   for r in range(len(ranks))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    if errors:
        raise errors[0]
    assert all(np.isclose(lmp.eatoms.sum(), float(lmp.energy)) for lmp in lmps)
    res = _efs_from_edges(
        atoms, ranks, [float(lmp.energy) for lmp in lmps],
        [lmp.pair_forces for lmp in lmps],
    )
    return res, ranks, lmps


@pytest.mark.parametrize('grid', [(1, 1, 1), (2, 1, 1)])
@pytest.mark.parametrize('name', list(CONFIGS))
def test_mliap(name, grid):
    cf = CONFIGS[name]
    model = _serial(cf)
    atoms = _atoms((2, 2, 2))
    ser = _serial_efs(model, atoms)
    n_rank = int(np.prod(grid))
    models = [_mliap_model(model) for _ in range(n_rank)]
    par, ranks, lmps = _run_mliap(models, atoms, grid)
    assert all(rk['n_unused_ghost'] > 0 for rk in ranks)  # skin ghosts
    n_layer = _config(cf)[KEY.NUM_CONVOLUTION]
    assert all(lmp.n_exchange == n_layer for lmp in lmps)  # one per layer
    _assert_close(par, ser, (name, grid))


def test_mliap_wrapper_class(tmp_path, monkeypatch):
    """SevenNetMLIAPWrapper (stub lammps module) on a checkpoint, fp32"""
    stub = types.ModuleType('lammps.mliap.mliap_unified_abc')

    class MLIAPUnified:
        def __init__(self, *args, **kwargs):
            pass

    stub.MLIAPUnified = MLIAPUnified  # type: ignore
    for name in ('lammps', 'lammps.mliap'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, 'lammps.mliap.mliap_unified_abc', stub)
    monkeypatch.delitem(sys.modules, 'sevenn.mliap', raising=False)
    from sevenn.mliap import SevenNetMLIAPWrapper

    cf = CONFIGS[FILE_CONFIG]
    cp = _save_checkpoint(cf, tmp_path / 'cp.pth')
    atoms = _atoms((2, 2, 2))
    with _default_dtype(torch.float32):
        wrappers = [SevenNetMLIAPWrapper(str(cp), modal=MODAL) for _ in range(2)]
        par, _, _ = _run_mliap(
            wrappers, atoms, (2, 1, 1), compute=lambda w, lmp: w.compute_forces(lmp)
        )
    assert all('_ghost_exchange' in ''.join(w.model._modules) for w in wrappers)
    _assert_close_f32(par, _serial_f32(cp, atoms), 'mliap wrapper')


# ------------------------------------------------- GPU backends (skipped on CPU)
def _backend_available(backend):
    if not torch.cuda.is_available():
        return False
    if backend == 'lunatp':
        from sevenn.nn.lunatp_helper import is_lunatp_available

        return is_lunatp_available()
    from sevenn.nn.oeq_helper import is_oeq_available

    return is_oeq_available()


@pytest.mark.parametrize('backend', ['lunatp', 'oeq'])
@pytest.mark.parametrize('name', GPU_CONFIGS)
def test_mliap_gpu_backend(tmp_path, name, backend):
    """ML-IAP wrapping of accelerated models (LunaTP: fused RMSNorm, gates,
    linears, bilinear gate; oeq: convolution), fp32 on GPU, 1 rank"""
    if not _backend_available(backend):
        pytest.skip(f'{backend} not available')
    from sevenn.util import model_from_checkpoint

    cf = CONFIGS[name]
    cp = _save_checkpoint(cf, tmp_path / 'cp.pth')
    atoms = _atoms((2, 2, 2))
    with _default_dtype(torch.float32):
        model, _ = model_from_checkpoint(
            str(cp), **{f'enable_{backend}': True}
        )
        model.set_is_batch_data(False)
        patch_model_for_mliap(model)
        if model.modal_map:
            model.prepare_modal_deploy(MODAL)
        model.delete_module_by_key('force_output')
        model.to('cuda').eval()
        par, _, lmps = _run_mliap(
            [model], atoms, (1, 1, 1), dtype=torch.float32, device='cuda'
        )
    assert lmps[0].n_exchange == _config(cf)[KEY.NUM_CONVOLUTION]
    _assert_close_f32(par, _serial_f32(cp, atoms), (name, backend))


@pytest.mark.parametrize('name', [SCRIPT_CONFIGS[0], FILE_CONFIG])
def test_e3gnn_parallel_oeq(tmp_path, name):
    """deploy_parallel's oeq segments (fp32, GPU) through the emulator"""
    if not _backend_available('oeq'):
        pytest.skip('oeq not available')
    from sevenn.util import model_from_checkpoint

    cf = CONFIGS[name]
    cp = _save_checkpoint(cf, tmp_path / 'cp.pth')
    atoms = _atoms((2, 2, 2))
    with _default_dtype(torch.float32):
        model, config = model_from_checkpoint(str(cp), enable_oeq=True)
        segs = build_parallel_model_list(
            model.state_dict(), config, MODAL, use_oeq=True
        )
        for s in segs:
            s.to('cuda').set_is_batch_data(False)
            s.eval()
        par, _ = _e3gnn_parallel(
            segs, atoms, config, (2, 1, 1), dtype=torch.float32, device='cuda'
        )
    _assert_close_f32(par, _serial_f32(cp, atoms), (name, 'oeq'))
