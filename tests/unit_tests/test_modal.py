# # deploy is test on lammps
# test append modality
#    from no modality model to modality yes model
#    from modality model to more modality model
#    different shift scale settings
# test modality options (check num param)
# calculators with modality

import copy
# + modal checkpoint continue and test_train
# + sevenn_cp test things in test_cli
import pathlib

import pytest
from ase.build import bulk

import sevenn.train.graph_dataset as graph_ds
import sevenn.util as util
from sevenn.calculator import SevenNetCalculator
from sevenn.model_build import build_E3_equivariant_model

cutoff = 5.0
data_root = (pathlib.Path(__file__).parent.parent / 'data').resolve()
hfo2_path = str(data_root / 'systems' / 'hfo2.extxyz')
sevennet_0_path = util.pretrained_name_to_path('7net-0_11July2024')


@pytest.fixture(scope='module')
def graph_dataset_path(tmp_path_factory):
    gd_path = tmp_path_factory.mktemp('gd')
    ds = graph_ds.SevenNetGraphDataset(
        cutoff=cutoff, root=str(gd_path), files=[hfo2_path], processed_name='tmp.pt'
    )
    return ds.processed_paths[0]


_modal_cfg = {
    'use_modal_node_embedding': False,
    'use_modal_self_inter_intro': True,
    'use_modal_self_inter_outro': True,
    'use_modal_output_block': True,
    'use_modality': True,
    'use_modal_wise_shift': True,  # T/F should be tested
    'use_modal_wise_scale': False,  # T/F should be tested
    'load_trainset_path': [
        {
            'data_modality': 'modal_new',
            'file_list': [{'file': hfo2_path}],
        }
    ],
}


@pytest.fixture(scope='module')
def snet_0_cp():
    return util.load_checkpoint(sevennet_0_path)


@pytest.fixture(scope='module')
def snet_0_calc():
    return SevenNetCalculator()


@pytest.fixture()
def bulk_atoms():
    atoms = bulk('Si') * 3
    atoms.rattle()
    return atoms


def assert_atoms(atoms1, atoms2, rtol=1e-5, atol=1e-6):
    import numpy as np

    def acl(a, b, rtol=rtol, atol=atol):
        return np.allclose(a, b, rtol=rtol, atol=atol)

    assert len(atoms1) == len(atoms2)
    assert acl(atoms1.get_cell(), atoms2.get_cell())
    assert acl(atoms1.get_potential_energy(), atoms2.get_potential_energy())
    assert acl(atoms1.get_forces(), atoms2.get_forces(), rtol * 10, atol * 10)
    assert acl(
        atoms1.get_stress(voigt=False),
        atoms2.get_stress(voigt=False),
        rtol * 10,
        atol * 10,
    )
    # assert acl(atoms1.get_potential_energies(), atoms2.get_potential_energies())


def get_modal_cfg(overwrite=None):
    modal_cfg = copy.deepcopy(_modal_cfg).copy()
    if overwrite:
        modal_cfg.update(overwrite)
    return modal_cfg


@pytest.mark.parametrize(
    'cfg_overwrite',
    [
        ({}),
        ({'use_modal_wise_scale': True}),
        ({'use_modal_wise_shift': False}),
        ({'use_modal_self_inter_intro': False}),
    ],
)
def test_append_modal_sevennet_0(
    cfg_overwrite,
    snet_0_cp,
    snet_0_calc,
    bulk_atoms,
    graph_dataset_path,
    tmp_path,
):
    modal_cfg = snet_0_cp.config
    modal_cfg.pop('load_dataset_path')
    modal_cfg.pop('load_validset_path')
    modal_cfg.update(get_modal_cfg(cfg_overwrite))
    modal_cfg['shift'] = 'elemwise_reference_energies'
    modal_cfg['scale'] = 'per_atom_energy_std'
    modal_cfg['load_trainset_path'][0]['file_list'] = [{'file': graph_dataset_path}]

    new_state_dict = snet_0_cp.append_modal(
        modal_cfg, original_modal_name='pbe', working_dir=tmp_path
    )
    sevennet_0_w_modal = build_E3_equivariant_model(modal_cfg)
    sevennet_0_w_modal.load_state_dict(new_state_dict, strict=True)

    atoms1 = bulk_atoms
    atoms2 = copy.deepcopy(atoms1)

    atoms1.calc = snet_0_calc
    atoms2.calc = SevenNetCalculator(
        model=sevennet_0_w_modal, file_type='model_instance', modal='pbe'
    )

    assert_atoms(atoms1, atoms2)
