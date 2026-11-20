"""
Frozen-model LunaTP path (sevenn/nn/lunatp_fused.py) in the LAMMPS ML-IAP
CUDA graph, on the emulated lmp_data of test_parallel_consistency.py: fused
radial convolutions (weight rows per distinct edge length, dE/dr contracted
in the backward kernel, hidden layers from a table), species self connection
pruned to the species present, modal input folded. Energy, forces and stress
of the serial model along a sequence of structures, from one capture; fp16
weight rows are close but not equal.
"""
import pytest
import torch

from sevenn.nn._mliap_cuda_graph import MLIAPCudaGraph
from tests.unit_tests.test_arch_options import ARCHS
from tests.unit_tests.test_mliap_cuda_graph import _compute, _moved
from tests.unit_tests.test_model import cutoff
from tests.unit_tests.test_parallel_consistency import (
    GPU_CONFIGS,
    MODAL,
    _assert_close_f32,
    _atoms,
    _backend_available,
    _cf,
    _default_dtype,
    _run_mliap,
    _save_checkpoint,
    _serial_f32,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _backend_available('lunatp')),
    reason='needs CUDA and LunaTP',
)


def _fused_graph(cp, half_weights=False):
    from sevenn.nn.lunatp_fused import fold_modal, fuse_radial_conv
    from sevenn.util import model_from_checkpoint

    model, _ = model_from_checkpoint(str(cp), enable_lunatp=True)
    model.set_is_batch_data(False)
    model.delete_module_by_key('force_output')
    model.requires_grad_(False)
    model.to('cuda').eval()
    names = fuse_radial_conv(model, half_weights=half_weights)
    modal = MODAL if model.modal_map else None
    if modal is not None:
        fold_modal(model, modal)
    model.prune_species = True
    return MLIAPCudaGraph(model, torch.device('cuda'), cutoff, modal=modal), names


@pytest.mark.parametrize('name', GPU_CONFIGS)
def test_fused_mliap_cuda_graph(tmp_path, name):
    if name == 'moe':  # no capture (torch.nonzero per expert)
        pytest.skip('ffn_moe is not captured')
    cp = _save_checkpoint(_cf(name, ARCHS[0]), tmp_path / 'cp.pth')
    atoms = _atoms((2, 2, 2))
    with _default_dtype(torch.float32):
        graph, names = _fused_graph(cp)
        if not names:
            pytest.skip('no convolution of this config is fused')
        for step in range(4):
            moved = _moved(atoms, step)
            par, _, _ = _run_mliap(
                [graph], moved, (1, 1, 1), compute=_compute, device='cuda'
            )
            _assert_close_f32(par, _serial_f32(cp, moved), (name, step))
    assert graph.enabled
    assert graph.n_captures == 1 and graph.n_replays == 3


def test_fused_half_weights(tmp_path):
    """fp16 weight rows: the forces of the serial model to ~1e-3"""
    cp = _save_checkpoint(_cf(GPU_CONFIGS[0], ARCHS[0]), tmp_path / 'cp.pth')
    atoms = _moved(_atoms((2, 2, 2)), 1)
    with _default_dtype(torch.float32):
        graph, names = _fused_graph(cp, half_weights=True)
        if not names:
            pytest.skip('no convolution of this config is fused')
        par, _, _ = _run_mliap(
            [graph], atoms, (1, 1, 1), compute=_compute, device='cuda'
        )
        ref = _serial_f32(cp, atoms)
    err = float(abs(par[1] - ref[1]).max())
    assert 0.0 < err < 2e-2 * float(abs(ref[1]).max())
