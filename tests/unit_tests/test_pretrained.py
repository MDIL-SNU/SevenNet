# test_pretrained: output consistency for pretrained models

import pytest
import torch
from ase.build import bulk, molecule

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import model_from_checkpoint, pretrained_name_to_path


def acl(a, b, atol=1e-6):
    return torch.allclose(a.float(), b.float(), atol=atol)


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
    assert acl(g2.inferred_force, g2_ref_f)


def test_7net_mf_0(atoms_pbc, atoms_mol):
    cp_path = pretrained_name_to_path('7net-mf-0')
    model, config = model_from_checkpoint(cp_path)
    cutoff = config['cutoff']

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    g1[KEY.DATA_MODALITY] = 'R2SCAN'
    g2[KEY.DATA_MODALITY] = 'R2SCAN'

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    model.set_is_batch_data(True)

    g1_ref_e = torch.tensor([-11.607587814331055])
    g1_ref_f = torch.tensor(
        [
            [8.512259, 0.07307914, 0.06676716],
            [-8.512257, -0.07307915, -0.06676716],
        ]
    )
    g1_ref_s = -1 * torch.tensor(
        # xx, yy, zz, xy, yz, zx
        [-0.4516204, -0.02483013, -0.02485001, 0.03247492, 0.00259375, 0.03250402]
    )

    g2_ref_e = torch.tensor([-14.172412872314453])
    g2_ref_f = torch.tensor(
        [
            [4.6566129e-10, -1.3429364e01, 6.9344816e00],
            [2.3283064e-09, 8.9132404e00, -9.6807365e00],
            [-2.7939677e-09, 4.5161238e00, 2.7462559e00],
        ]
    )

    assert acl(g1.inferred_total_energy, g1_ref_e)
    assert acl(g1.inferred_force, g1_ref_f)
    assert acl(g1.inferred_stress, g1_ref_s)

    assert acl(g2.inferred_total_energy, g2_ref_e)
    assert acl(g2.inferred_force, g2_ref_f)


def test_7net_mf_ompa_mpa(atoms_pbc, atoms_mol):
    cp_path = pretrained_name_to_path('7net-mf-ompa')
    model, config = model_from_checkpoint(cp_path)
    cutoff = config['cutoff']

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    # mpa
    g1[KEY.DATA_MODALITY] = 'mpa'
    g2[KEY.DATA_MODALITY] = 'mpa'

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    model.set_is_batch_data(True)

    g1_ref_e = torch.tensor([-3.490943193435669])
    g1_ref_f = torch.tensor(
        [
            [1.2680445e01, -2.7985498e-04, -2.7979910e-04],
            [-1.2680446e01, 2.7984008e-04, 2.7981028e-04],
        ]
    )
    g1_ref_s = -1 * torch.tensor(
        # xx, yy, zz, xy, yz, zx
        [-0.6481662, -0.02462837, -0.02462837, 0.02693467, 0.00459635, 0.02693467]
    )

    g2_ref_e = torch.tensor([-12.597525596618652])
    g2_ref_f = torch.tensor(
        [
            [0.0, -12.245223, 7.26795],
            [0.0, 8.816763, -9.423925],
            [0.0, 3.4284601, 2.1559749],
        ]
    )
    assert acl(g1.inferred_total_energy, g1_ref_e)
    assert acl(g1.inferred_force, g1_ref_f)
    assert acl(g1.inferred_stress, g1_ref_s)

    assert acl(g2.inferred_total_energy, g2_ref_e)
    assert acl(g2.inferred_force, g2_ref_f)


def test_7net_mf_ompa_omat(atoms_pbc, atoms_mol):
    cp_path = pretrained_name_to_path('7net-mf-ompa')
    model, config = model_from_checkpoint(cp_path)
    cutoff = config['cutoff']

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    # mpa
    g1[KEY.DATA_MODALITY] = 'omat24'
    g2[KEY.DATA_MODALITY] = 'omat24'

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    model.set_is_batch_data(True)

    g1_ref_e = torch.tensor([-3.5094668865203857])
    g1_ref_f = torch.tensor(
        [
            [1.2562084e01, -1.4219694e-03, -1.4219843e-03],
            [-1.2562084e01, 1.4219508e-03, 1.4219955e-03],
        ]
    )
    g1_ref_s = -1 * torch.tensor(
        # xx, yy, zz, xy, yz, zx
        [-0.6430905, -0.0254128, -0.02541281, 0.0268343, 0.00460021, 0.0268343]
    )

    g2_ref_e = torch.tensor([-12.6202974319458])
    g2_ref_f = torch.tensor(
        [
            [0.0, -12.205926, 7.2050343],
            [0.0, 8.790399, -9.368677],
            [0.0, 3.4155273, 2.163643],
        ]
    )
    assert acl(g1.inferred_total_energy, g1_ref_e)
    assert acl(g1.inferred_force, g1_ref_f)
    assert acl(g1.inferred_stress, g1_ref_s)

    assert acl(g2.inferred_total_energy, g2_ref_e)
    assert acl(g2.inferred_force, g2_ref_f)


def test_7net_omat(atoms_pbc, atoms_mol):
    cp_path = pretrained_name_to_path('7net-omat')
    model, config = model_from_checkpoint(cp_path)
    cutoff = config['cutoff']

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    model.set_is_batch_data(True)

    g1_ref_e = torch.tensor([-3.5033323764801025])
    g1_ref_f = torch.tensor(
        [
            [12.533154, 0.02358698, 0.02358694],
            [-12.533153, -0.02358699, -0.02358697],
        ]
    )
    g1_ref_s = -1 * torch.tensor(
        # xx, yy, zz, xy, yz, zx
        [-0.6420925, -0.02781446, -0.02781446, 0.02575445, 0.00381664, 0.02575445]
    )

    g2_ref_e = torch.tensor([-12.403768539428711])
    g2_ref_f = torch.tensor(
        [
            [0, -12.848297, 7.11432],
            [0.0, 9.265477, -9.564951],
            [0.0, 3.58282, 2.4506311],
        ]
    )
    assert acl(g1.inferred_total_energy, g1_ref_e)
    assert acl(g1.inferred_force, g1_ref_f)
    assert acl(g1.inferred_stress, g1_ref_s)

    assert acl(g2.inferred_total_energy, g2_ref_e)
    assert acl(g2.inferred_force, g2_ref_f)


def test_7net_omni_mpa(atoms_pbc, atoms_mol):
    cp_path = pretrained_name_to_path('7net-omni')
    model, config = model_from_checkpoint(
        cp_path, enable_flash=False, enable_cueq=False
    )  # to test in cpu, require e3nn
    cutoff = config['cutoff']

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    # mpa
    g1[KEY.DATA_MODALITY] = 'mpa'
    g2[KEY.DATA_MODALITY] = 'mpa'

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    model.set_is_batch_data(True)

    g1_ref_e = torch.tensor([-3.483455181121826])
    g1_ref_f = torch.tensor(
        [
            [12.707214, 0.01643602, 0.01643606],
            [-12.707215, -0.01643603, -0.01643603],
        ]
    )
    g1_ref_s = -1 * torch.tensor(
        # xx, yy, zz, xy, yz, zx
        [-0.6500675, -0.0290563, -0.0290563, 0.02576996, 0.00374571, 0.02576996]
    )

    g2_ref_e = torch.tensor([-12.918253898620605])
    g2_ref_f = torch.tensor(
        [
            [0.0, -13.32638, 7.1434574],
            [0.0, 9.442289, -9.77207],
            [0.0, 3.8840904, 2.6286132],
        ]
    )
    assert acl(g1.inferred_total_energy, g1_ref_e)
    assert acl(g1.inferred_force, g1_ref_f)
    assert acl(g1.inferred_stress, g1_ref_s)

    assert acl(g2.inferred_total_energy, g2_ref_e)
    assert acl(g2.inferred_force, g2_ref_f)


def test_7net_omni_i8_mpa(atoms_pbc, atoms_mol):
    cp_path = pretrained_name_to_path('7net-omni-i8')
    model, config = model_from_checkpoint(
        cp_path, enable_flash=False, enable_cueq=False
    )  # to test in cpu, require e3nn
    cutoff = config['cutoff']

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    # mpa
    g1[KEY.DATA_MODALITY] = 'mpa'
    g2[KEY.DATA_MODALITY] = 'mpa'

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    model.set_is_batch_data(True)

    g1_ref_e = torch.tensor([-3.4679641723632812])
    g1_ref_f = torch.tensor(
        [
            [12.718483, -0.013534063, -0.013534037],
            [-12.718483, 0.013534017, 0.013534039],
        ]
    )
    g1_ref_s = -1 * torch.tensor(
        # xx, yy, zz, xy, yz, zx
        [-0.6499892, -0.02532190, -0.02532190, 0.02772916, 0.00378853, 0.02772916]
    )

    g2_ref_e = torch.tensor([-12.922063827514648])
    g2_ref_f = torch.tensor(
        [
            [0.0, -13.452224, 7.3066516],
            [0.0, 9.5646286, -9.9248161],
            [0.0, 3.8875942, 2.6181641],
        ]
    )
    assert acl(g1.inferred_total_energy, g1_ref_e)
    assert acl(g1.inferred_force, g1_ref_f)
    assert acl(g1.inferred_stress, g1_ref_s)

    assert acl(g2.inferred_total_energy, g2_ref_e)
    assert acl(g2.inferred_force, g2_ref_f)


def test_7net_omni_i12_mpa(atoms_pbc, atoms_mol):
    cp_path = pretrained_name_to_path('7net-omni-i12')
    model, config = model_from_checkpoint(
        cp_path, enable_flash=False, enable_cueq=False
    )  # to test in cpu, require e3nn
    cutoff = config['cutoff']

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    # mpa
    g1[KEY.DATA_MODALITY] = 'mpa'
    g2[KEY.DATA_MODALITY] = 'mpa'

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    model.set_is_batch_data(True)

    g1_ref_e = torch.tensor([-3.503857135772705])
    g1_ref_f = torch.tensor(
        [
            [12.539756, 0.027810518, 0.027810508],
            [-12.539756, -0.027810529, -0.027810507],
        ]
    )
    g1_ref_s = -1 * torch.tensor(
        # xx, yy, zz, xy, yz, zx
        [-0.6424894, -0.02873472, -0.02873472, 0.02597278, 0.00331444, 0.02597278]
    )

    g2_ref_e = torch.tensor([-12.92691421508789])
    g2_ref_f = torch.tensor(
        [
            [0.0, -13.374825, 7.3012400],
            [0.0, 9.5462170, -9.8908463],
            [0.0, 3.8286080, 2.5896058],
        ]
    )
    assert acl(g1.inferred_total_energy, g1_ref_e)
    assert acl(g1.inferred_force, g1_ref_f)
    assert acl(g1.inferred_stress, g1_ref_s)

    assert acl(g2.inferred_total_energy, g2_ref_e)
    assert acl(g2.inferred_force, g2_ref_f)


@pytest.mark.parametrize(
    'name, cutoff_ref, g1_ref_e, g1_ref_f, g1_ref_s, g2_ref_e, g2_ref_f',
    [
        (
            '7net-nano-4.5',
            4.5,
            -3.5564712945510024,
            [
                [11.843839645385742, -0.06617936491966248, -0.06617937982082367],
                [-11.843839645385742, 0.06617936491966248, 0.0661793053150177],
            ],
            [
                0.605344831943512,
                0.02144481986761093,
                0.021444812417030334,
                -0.03438035026192665,
                -0.003832500660791993,
                -0.03438035771250725,
            ],
            -12.919640449266296,
            [
                [0.0, -13.341249465942383, 7.200963020324707],
                [0.0, 9.452506065368652, -9.814258575439453],
                [0.0, 3.8887438774108887, 2.6132965087890625],
            ],
        ),
        (
            '7net-nano-5.0',
            5.0,
            -3.4316864360099615,
            [
                [12.497757911682129, -0.009577874094247818, -0.009577933698892593],
                [-12.497756958007812, 0.009577878750860691, 0.009577957913279533],
            ],
            [
                0.6648563146591187,
                0.040433332324028015,
                0.04043332114815712,
                -0.029892124235630035,
                -0.003740792628377676,
                -0.029892126098275185,
            ],
            -12.910758047958524,
            [
                [0.0, -13.562267303466797, 7.167722702026367],
                [0.0, 9.584957122802734, -9.880509376525879],
                [0.0, 3.9773099422454834, 2.7127861976623535],
            ],
        ),
        (
            '7net-nano-5.5',
            5.5,
            -3.532371955806397,
            [
                [12.454666137695312, -0.034893378615379333, -0.0348934531211853],
                [-12.454666137695312, 0.034893378615379333, 0.03489343076944351],
            ],
            [
                0.6540243029594421,
                0.021435830742120743,
                0.021435843780636787,
                -0.03487098217010498,
                -0.00511842779815197,
                -0.03487098589539528,
            ],
            -12.922036098234884,
            [
                [0.0, -13.198136329650879, 7.108638763427734],
                [0.0, 9.389389038085938, -9.699457168579102],
                [0.0, 3.8087470531463623, 2.5908186435699463],
            ],
        ),
        (
            '7net-nano-6.0',
            6.0,
            -3.484466470155766,
            [
                [12.169546127319336, -0.02828906662762165, -0.028289003297686577],
                [-12.169546127319336, 0.028289061039686203, 0.02828901819884777],
            ],
            [
                0.6406453251838684,
                0.026448842138051987,
                0.026448845863342285,
                -0.026171253994107246,
                -0.003374285064637661,
                -0.026171250268816948,
            ],
            -12.934674727250979,
            [
                [0.0, -13.23235034942627, 7.185378074645996],
                [0.0, 9.451717376708984, -9.761429786682129],
                [0.0, 3.780632972717285, 2.576051712036133],
            ],
        ),
    ],
    ids=['7net-nano-4.5', '7net-nano-5.0', '7net-nano-5.5', '7net-nano-6.0'],
)
def test_7net_nano(
    name,
    cutoff_ref,
    g1_ref_e,
    g1_ref_f,
    g1_ref_s,
    g2_ref_e,
    g2_ref_f,
    atoms_pbc,
    atoms_mol,
):
    cp_path = pretrained_name_to_path(name)
    cp_dict = torch.load(cp_path, map_location='cpu', weights_only=False)
    assert sorted(cp_dict) == ['config', 'hash', 'model_state_dict', 'time']

    model, config = model_from_checkpoint(
        cp_path, enable_flash=False, enable_cueq=False
    )
    cutoff = config['cutoff']
    assert cutoff == cutoff_ref

    g1 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_pbc, cutoff))
    g2 = AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms_mol, cutoff))

    model.set_is_batch_data(False)
    g1 = model(g1)
    g2 = model(g2)

    assert acl(g1.inferred_total_energy, torch.tensor(g1_ref_e))
    assert acl(g1.inferred_force, torch.tensor(g1_ref_f))
    assert acl(g1.inferred_stress, torch.tensor(g1_ref_s))

    assert acl(g2.inferred_total_energy, torch.tensor(g2_ref_e))
    assert acl(g2.inferred_force, torch.tensor(g2_ref_f))
