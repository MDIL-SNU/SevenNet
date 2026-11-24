from sevenn.calculator import SevenNetCalculator
from sevenn.nn.inference_specialization import fold_modal, prune_self_connection
from sevenn.util import load_checkpoint
from tests.unit_tests._mliap_fixture import (
    MODAL,
    assert_efs,
    atoms_with_ghosts,
    evaluate,
    reference,
    save_checkpoint,
)


def test_species_and_modal_specialization_without_lunatp(tmp_path):
    path = save_checkpoint(
        tmp_path / 'original.pth', modal=True, use_bias_in_linear=True
    )
    original = reference(path, MODAL)
    model = load_checkpoint(path).build_model()
    assert fold_modal(model, MODAL)
    specialized = SevenNetCalculator(model, device='cpu', modal=MODAL)
    atoms = atoms_with_ghosts()
    for numbers in ({8, 11, 17}, {1, 8, 11, 17}):
        columns = [model.type_map[z] for z in numbers]
        assert prune_self_connection(model, columns)
        subset = atoms[[z in numbers for z in atoms.numbers]]
        assert_efs(evaluate(specialized, subset), evaluate(original, subset))
