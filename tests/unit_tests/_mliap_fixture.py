import numpy as np
import torch
from ase.build import bulk
from ase.neighborlist import neighbor_list
from test_model import cutoff, get_model_config

import sevenn
from sevenn.calculator import SevenNetCalculator
from sevenn.model_build import build_E3_equivariant_model
from sevenn.util import chemical_species_preprocess, model_from_checkpoint

MODAL = 'mpa'


def atoms_with_ghosts():
    atoms = bulk('NaCl', 'rocksalt', a=5.63, cubic=True) * (2, 2, 2)
    atoms.symbols[:4] = ['H', 'O', 'Na', 'Cl']
    atoms.rattle(0.05, seed=1)
    atoms.wrap()
    return atoms


def save_checkpoint(path, self_connection='nequip', modal=False, **model_options):
    config = get_model_config()
    config.update(
        chemical_species_preprocess(['H', 'O', 'Na', 'Cl']),
        version=sevenn.__version__,
        self_connection_type=self_connection,
    )
    if modal:
        config.update(
            use_modality=True,
            _number_of_modalities=2,
            _modal_map={'pbe': 0, MODAL: 1},
            use_modal_node_embedding=False,
            use_modal_self_inter_intro=True,
            use_modal_self_inter_outro=True,
            use_modal_output_block=True,
            use_modal_wise_shift=False,
            use_modal_wise_scale=False,
        )
    config.update(model_options)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = build_E3_equivariant_model(config, parallel=False)
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, path)
    return path


def reference(path, modal=None):
    return SevenNetCalculator(
        str(path), device='cpu', modal=modal, radial_table=False,
    )


def evaluate(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()


def assert_efs(actual, expected):
    for value, ref, atol in zip(actual, expected, (5e-5, 1e-4, 1e-5)):
        np.testing.assert_allclose(value, ref, rtol=1e-5, atol=atol)


def graph_model(path, enable_lunatp=False):
    model, _ = model_from_checkpoint(str(path), enable_lunatp=enable_lunatp)
    model.set_is_batch_data(False)
    model.delete_module_by_key('force_output')
    return model.requires_grad_(False).cuda().eval()


def compute_graph(graph, atoms, index_dtype=np.int32):
    lmp = FakeLmpData(atoms, index_dtype)
    assert graph.compute(lmp)
    return lmp.efs(atoms), lmp.npairs


def moved(atoms, step):
    shifted = atoms.copy()
    shifted.rattle(0.03 * step, seed=10 + step)
    return shifted


class FakeLmpData:
    def __init__(self, atoms, index_dtype=np.int32, foreign=False):
        i, j, image, self.rij = neighbor_list('ijSD', atoms, cutoff)
        self.nlocal = len(atoms)
        ghosts, slots = np.unique(
            np.column_stack((j, image)), axis=0, return_inverse=True
        )
        is_ghost = ghosts[:, 1:].any(axis=1)
        self.ghost_owner = ghosts[is_ghost, 0]
        rows = ghosts[:, 0].copy()
        rows[is_ghost] = np.arange(len(self.ghost_owner)) + self.nlocal
        self.pair_i = i.astype(index_dtype)
        self.pair_j = rows[slots].astype(index_dtype)
        self.elems = np.concatenate(
            (atoms.numbers, atoms.numbers[self.ghost_owner])
        ).astype(index_dtype)
        self.ntotal = len(self.elems)
        self.npairs = len(i)
        self.eatoms = np.zeros(self.nlocal)
        self.energy = None
        self.foreign = foreign

    def forward_exchange(self, send, recv, width):
        assert send.shape == (self.ntotal, width)
        recv.copy_(send)
        recv[self.nlocal:] = send[self.ghost_owner] if not self.foreign else -7.

    def update_pair_forces_gpu(self, pair_forces):
        self.pair_forces = pair_forces[:self.npairs].detach().cpu().numpy()

    def efs(self, atoms):
        i, j = self.pair_i, self.pair_j.astype(np.int64)
        owners = np.concatenate((np.arange(self.nlocal), self.ghost_owner))
        forces = np.zeros((self.nlocal, 3))
        np.add.at(forces, i, self.pair_forces)
        np.add.at(forces, owners[j], -self.pair_forces)
        virial = self.rij.T @ self.pair_forces
        stress = virial[[0, 1, 2, 1, 2, 0], [0, 1, 2, 2, 0, 1]]
        total = float(self.energy)
        np.testing.assert_allclose(total, self.eatoms.sum(), rtol=1e-6)
        return total, forces, stress / atoms.get_volume()
