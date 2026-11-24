import torch
from _mliap_fixture import assert_efs, atoms_with_ghosts, evaluate
from test_model import get_model_config

import sevenn
import sevenn._keys as KEY
from sevenn.calculator import SevenNetCalculator
from sevenn.model_build import build_E3_equivariant_model
from sevenn.util import chemical_species_preprocess, load_checkpoint


def test_compact_inference_checkpoint(tmp_path):
    config = get_model_config()
    config.update(
        chemical_species_preprocess(['H', 'O', 'Na', 'Cl']),
        version=sevenn.__version__,
        irreps_manual=['4x0e', '4x0e + 4x2e', '4x0e + 4x2e', '4x0e'],
        use_bias_in_linear=True,
    )
    checkpoint_path = tmp_path / 'original.pth'
    model = build_E3_equivariant_model(config)
    torch.save(
        {'config': config, 'model_state_dict': model.state_dict()}, checkpoint_path
    )
    checkpoint = load_checkpoint(checkpoint_path)
    compact_path = tmp_path / 'inference.pth'
    checkpoint.save_inference_checkpoint(compact_path)
    compact = load_checkpoint(compact_path)

    assert not checkpoint.config[KEY.DROP_UNUSED_TP_IRREPS]
    assert compact.config[KEY.DROP_UNUSED_TP_IRREPS]
    assert compact.compact_for_inference() is compact
    weight = '0_convolution.weight_nn.layer2.weight'
    assert (
        compact.model_state_dict[weight].shape[1]
        < checkpoint.model_state_dict[weight].shape[1]
    )
    atoms = atoms_with_ghosts()
    reference = evaluate(SevenNetCalculator(model, device='cpu'), atoms)
    for path in (checkpoint_path, compact_path):
        assert_efs(
            evaluate(SevenNetCalculator(str(path), device='cpu'), atoms), reference
        )
