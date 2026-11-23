"""Deployed models list their species in the model's index order."""

from pathlib import Path

import pytest
import torch

import sevenn._keys as KEY
from sevenn.scripts.deploy import deploy, deploy_parallel

CP_0 = Path(__file__).parent.parent / 'data' / 'checkpoints' / 'cp_0.pth'


@pytest.fixture
def reordered_checkpoint(tmp_path):
    """cp_0 with the same type map (Hf: 0, O: 1), its keys stored as O, Hf."""
    cp = torch.load(CP_0, map_location='cpu', weights_only=False)
    type_map = cp['config'][KEY.TYPE_MAP]
    assert list(type_map.items()) == [(72, 0), (8, 1)]
    cp['config'][KEY.TYPE_MAP] = dict(reversed(list(type_map.items())))
    path = str(tmp_path / 'reordered.pth')
    torch.save(cp, path)
    return path


def test_deploy_serial_keeps_species_of_reordered_type_map(
    tmp_path, reordered_checkpoint
):
    """LAMMPS reads the index of a species off its position in the stored list,
    so key order there would swap the species."""
    fname = str(tmp_path / 'serial.pt')
    deploy(reordered_checkpoint, fname)

    extra = {'chemical_symbols_to_index': b''}
    torch.jit.load(fname, map_location='cpu', _extra_files=extra)
    assert extra['chemical_symbols_to_index'].decode().split() == ['Hf', 'O']


def test_deploy_parallel_keeps_species_of_reordered_type_map(
    tmp_path, reordered_checkpoint
):
    folder = tmp_path / 'parallel'
    deploy_parallel(reordered_checkpoint, str(folder))

    segments = sorted(folder.glob('*.pt'))
    assert segments
    for segment in segments:
        extra = {'chemical_symbols_to_index': b''}
        torch.jit.load(str(segment), map_location='cpu', _extra_files=extra)
        assert extra['chemical_symbols_to_index'].decode().split() == ['Hf', 'O']
