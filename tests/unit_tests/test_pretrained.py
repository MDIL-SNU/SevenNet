# test_pretrained: output consistency for pretrained models

import pytest
import torch
from ase.build import bulk, molecule

from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import model_from_checkpoint, pretrained_name_to_path


def acl(a, b, atol=1e-6):
    return torch.allclose(a, b, atol=atol)


@pytest.fixture
def atoms_pbc():
    atoms1 = bulk('NaCl', 'rocksalt', a=5.63)
    atoms1.set_cell([[1.0, 2.815, 2.815], [2.815, 0.0, 2.815], [2.815, 2.815, 0.0]])
    atoms1.set_positions([[0.0, 0.0, 0.0], [2.815, 0.0, 0.0]])
    return atoms1


@pytest.fixture
def atoms_mol():
    atoms2 = molecule('H2O')
    atoms2.set_positions([[0.0, 0.2, 0.12], [0.0, 0.76, -0.48], [0.0, -0.76, -0.48]])
    return atoms2


def test_7net0_22May2024(atoms_pbc, atoms_mol):
    """
    Reference from v0.9.3.post1 with SevenNetCalculator
    """
    cp_path = pretrained_name_to_path('7net-0_22May2024')
    model, config = model_from_checkpoint(cp_path)
    cutoff = config['cutoff']

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    g1_ref_e = torch.tensor([-3.4140868186950684])
    g1_ref_f = torch.tensor(
        [
            [1.2628037e01, 7.5093508e-03, 1.3480943e-02],
            [-1.2628037e01, -7.5093508e-03, -1.3480917e-02],
        ]
    )
    g1_ref_s = -1 * torch.tensor(
        [-0.65014917, -0.01990843, -0.02000658, 0.03286226, 0.00589222, 0.03291973]
    )

    g2_ref_e = torch.tensor([-12.808363914489746])
    g2_ref_f = torch.tensor(
        [
            [9.31322575e-10, -1.30241165e01, 6.93116236e00],
            [-1.39698386e-09, 9.28001022e00, -9.51867390e00],
            [5.23868948e-10, 3.74410582e00, 2.58751225e00],
        ]
    )

    assert acl(g1.inferred_total_energy, g1_ref_e)
    assert acl(g1.inferred_force, g1_ref_f)
    assert acl(g1.inferred_stress, g1_ref_s)

    assert acl(g2.inferred_total_energy, g2_ref_e)
    assert acl(g2.inferred_force, g2_ref_f)


def test_7net0_11July2024(atoms_pbc, atoms_mol):
    """
    Reference from v0.9.3.post1 with SevenNetCalculator
    """
    cp_path = pretrained_name_to_path('7net-0_11July2024')
    model, config = model_from_checkpoint(cp_path)
    cutoff = config['cutoff']

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    model.set_is_batch_data(True)

    g1_ref_e = torch.tensor([-3.779199])
    g1_ref_f = torch.tensor(
        [
            [12.666697, 0.04726403, 0.04775861],
            [-12.666697, -0.04726403, -0.04775861],
        ]
    )
    g1_ref_s = -1 * torch.tensor(
        # xx, yy, zz, xy, yz, zx
        [-0.6439122, -0.03643947, -0.03643981, 0.04543639, 0.00599139, 0.04544507]
    )

    g2_ref_e = torch.tensor([-12.782808303833008])
    g2_ref_f = torch.tensor(
        [
            [0.0, -1.3619621e01, 7.5937047e00],
            [0.0, 9.3918495e00, -1.0172190e01],
            [0.0, 4.2277718e00, 2.5784855e00],
        ]
    )

    assert acl(g1.inferred_total_energy, g1_ref_e)
    assert acl(g1.inferred_force, g1_ref_f)
    assert acl(g1.inferred_stress, g1_ref_s)

    assert acl(g2.inferred_total_energy, g2_ref_e)
    assert acl(g2.inferred_force, g2_ref_f)


def test_7net_l3i5(atoms_pbc, atoms_mol):
    """
    Reference from v0.9.3.post1 with SevenNetCalculator
    """
    cp_path = pretrained_name_to_path('7net-l3i5')
    model, config = model_from_checkpoint(cp_path)
    cutoff = config['cutoff']

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    model.set_is_batch_data(True)

    g1_ref_e = torch.tensor([-3.611131191253662])
    g1_ref_f = torch.tensor(
        [
            [13.430887, 0.08655541, 0.08754013],
            [-13.430886, -0.08655544, -0.08754011],
        ]
    )
    g1_ref_s = -1 * torch.tensor(
        # xx, yy, zz, xy, yz, zx
        [-0.6818918, -0.04104544, -0.04107663, 0.04794561, 0.00565416, 0.04793138]
    )

    g2_ref_e = torch.tensor([-12.700481414794922])
    g2_ref_f = torch.tensor(
        [
            [0.0, -1.4547814e01, 8.1347866],
            [0.0, 1.0308369e01, -1.0880318e01],
            [0.0, 4.2394452, 2.7455316],
        ]
    )

    assert acl(g1.inferred_total_energy, g1_ref_e)
    assert acl(g1.inferred_force, g1_ref_f, 1e-5)
    assert acl(g1.inferred_stress, g1_ref_s, 1e-5)

    assert acl(g2.inferred_total_energy, g2_ref_e)
    assert acl(g2.inferred_force, g2_ref_f, 1e-5)
