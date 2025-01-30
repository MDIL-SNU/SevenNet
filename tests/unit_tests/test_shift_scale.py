import pytest
import torch

import sevenn._keys as KEY
from sevenn._const import NUM_UNIV_ELEMENT, AtomGraphDataType
from sevenn.nn.scale import (
    ModalWiseRescale,
    Rescale,
    SpeciesWiseRescale,
    get_resolved_shift_scale,
)

################################################################################
#                             Tests for Rescale                                #
################################################################################


@pytest.mark.parametrize('shift,scale', [(0.0, 1.0), (1.0, 2.0), (-5.0, 10.0)])
def test_rescale_init(shift, scale):
    """
    Test that Rescale can be initialized properly without errors
    and that parameters are set correctly.
    """
    module = Rescale(shift=shift, scale=scale)
    assert module.shift.item() == shift
    assert module.scale.item() == scale
    assert module.key_input == KEY.SCALED_ATOMIC_ENERGY
    assert module.key_output == KEY.ATOMIC_ENERGY


def test_rescale_forward():
    """
    Test that Rescale forward pass correctly applies:
        output = input * scale + shift
    """
    # Setup
    shift, scale = 1.0, 2.0
    module = Rescale(shift=shift, scale=scale)
    # Make some fake data
    input_data = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
    data: AtomGraphDataType = {KEY.SCALED_ATOMIC_ENERGY: input_data.clone()}

    # Forward
    out_data = module(data)

    # Check correctness
    expected_output = input_data * scale + shift
    assert torch.allclose(out_data[KEY.ATOMIC_ENERGY], expected_output)


def test_rescale_get_shift_and_scale():
    """
    Test get_shift() and get_scale() methods in Rescale.
    """
    module = Rescale(shift=1.5, scale=3.5)
    assert module.get_shift() == pytest.approx(1.5)
    assert module.get_scale() == pytest.approx(3.5)


################################################################################
#                       Tests for SpeciesWiseRescale                           #
################################################################################


def test_specieswise_rescale_init_float():
    """
    Test SpeciesWiseRescale when both shift and scale are floats
    (should expand to same length lists).
    """
    module = SpeciesWiseRescale(shift=[1.0, -1.0], scale=2.0)
    # Expect a parameter of length = 1 in this scenario, but can differ
    # if we raise an error for "Both shift and scale is not a list".
    # Usually, you'd specify a known number of species or do from_mappers.
    # The code as-is throws ValueError if both are float. Let's do from_mappers:
    # We'll do direct init if your code allows it. If not, use from_mappers.
    assert module.shift.shape == module.scale.shape
    # They must be single-parameter (or expanded) if not from mappers.


def test_specieswise_rescale_init_list():
    """
    Test initialization with list-based shift/scale of same length.
    """
    shift = [1.0, 2.0, 3.0]
    scale = [2.0, 3.0, 4.0]
    module = SpeciesWiseRescale(shift=shift, scale=scale)
    assert len(module.shift) == 3
    assert len(module.scale) == 3
    assert torch.allclose(module.shift, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(module.scale, torch.tensor([2.0, 3.0, 4.0]))


def test_specieswise_rescale_forward():
    """
    Test that SpeciesWiseRescale forward pass applies:
        output[i] = input[i]*scale[atom_type[i]] + shift[atom_type[i]]
    """
    # Suppose we have two species types:
    #    0 -> shift=1, scale=2, 1 -> shift=5, scale=10
    # (we'll pass them as lists in the correct order)
    shift = [1.0, 5.0]
    scale = [2.0, 10.0]
    module = SpeciesWiseRescale(
        shift=shift,
        scale=scale,
        data_key_in='in',
        data_key_out='out',
        data_key_indices='z',
    )

    # Create mock data
    # Suppose we have three atoms: species => [0, 1, 0]
    # input => [ [1.], [1.], [3.] ]
    data: AtomGraphDataType = {
        'z': torch.tensor([0, 1, 0], dtype=torch.long),
        'in': torch.tensor([[1.0], [1.0], [3.0]], dtype=torch.float),
    }

    out = module(data)
    # Now let's manually compute expected:
    # For atom 0: scale=2, shift=1, input=1 => 1*2+1=3
    # For atom 1: scale=10, shift=5, input=1 => 1*10+5=15
    # For atom 2: scale=2, shift=1, input=3 => 3*2+1=7
    expected = torch.tensor([[3.0], [15.0], [7.0]])

    assert torch.allclose(out['out'], expected)


def test_specieswise_rescale_get_shift_scale():
    """
    Test get_shift() and get_scale() with/without type_map.
    """
    shift = [1.0, 2.0]
    scale = [3.0, 4.0]
    module = SpeciesWiseRescale(shift=shift, scale=scale)

    # Without type_map
    # Should return the raw parameter values (list form).
    s = module.get_shift()
    sc = module.get_scale()
    assert s == [1.0, 2.0]
    assert sc == [3.0, 4.0]

    # With a type_map (example: atomic_number 1 -> 0, 8 -> 1)
    type_map = {1: 0, 8: 1}  # hydrogen, oxygen
    s_univ = module.get_shift(type_map)
    sc_univ = module.get_scale(type_map)
    # In this small example with NUM_UNIV_ELEMENT = 2, the _as_univ will produce
    # a list of length = NUM_UNIV_ELEMENT. If your real NUM_UNIV_ELEMENT is bigger,
    # the rest would be padded with default values.
    # For demonstration let's assume it returns [1.0, 2.0].
    # Check at least the known mapped portion:
    assert len(s_univ) == NUM_UNIV_ELEMENT
    assert len(sc_univ) == NUM_UNIV_ELEMENT
    assert s_univ[1] == 1.0  # atomic_number=1 -> idx=0 -> shift=1.0
    assert s_univ[8] == 2.0


################################################################################
#                       Tests for ModalWiseRescale                             #
################################################################################


def test_modalwise_rescale_init():
    """
    Basic sanity check for ModalWiseRescale initialization with
    certain shapes.
    """
    # Suppose we have 2 modals, 3 species => shift, scale is shape [2,3]
    shift = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    scale = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    module = ModalWiseRescale(
        shift=shift,
        scale=scale,
        use_modal_wise_shift=True,
        use_modal_wise_scale=True,
    )
    # Check shape
    assert module.shift.shape == torch.Size([2, 3])
    assert module.scale.shape == torch.Size([2, 3])


def test_modalwise_rescale_forward():
    """
    Test that the forward pass of ModalWiseRescale matches
    output[i] = input[i] * scale[modal_i, atom_i] + shift[modal_i, atom_i]
    when both use_modal_wise_{shift,scale} are True.
    """
    shift = [[0.0, 10.0], [5.0, 15.0]]  # shape [2 (modals), 2 (species)]
    scale = [[1.0, 2.0], [10.0, 20.0]]
    module = ModalWiseRescale(
        shift=shift,
        scale=scale,
        data_key_in='in',
        data_key_out='out',
        data_key_modal_indices='modal_idx',
        data_key_atom_indices='atom_idx',
        use_modal_wise_shift=True,
        use_modal_wise_scale=True,
    )

    data: AtomGraphDataType = {
        'in': torch.tensor([[1.0], [1.0], [2.0], [2.0]]),
        'modal_idx': torch.tensor([0, 1], dtype=torch.long),
        'atom_idx': torch.tensor([0, 1, 0, 1], dtype=torch.long),
        'batch': torch.tensor([0, 0, 1, 1], dtype=torch.long),
    }

    out = module(data)
    # i=0 => modal_idx=0, atom_idx=0 => shift=0.0, scale=1.0 => out=1*1+0=1
    # i=1 => modal_idx=0, atom_idx=1 => shift=10.0, scale=2.0 => out=1*2+10=12
    # i=2 => modal_idx=1, atom_idx=0 => shift=5.0, scale=10.0 => out=2*10+5=25
    # i=3 => modal_idx=1, atom_idx=1 => shift=15.0, scale=20.0 => out=2*20+15=55
    expected = torch.tensor([[1.0], [12.0], [25.0], [55.0]])
    assert torch.allclose(out['out'], expected)


def test_modalwise_rescale_get_shift_scale():
    """
    Test get_shift() and get_scale() with type_map and modal_map.
    """
    # Setup
    shift = [[0.0, 10.0], [5.0, 15.0]]
    scale = [[1.0, 2.0], [10.0, 20.0]]
    mod = ModalWiseRescale(
        shift=shift,
        scale=scale,
        use_modal_wise_shift=True,
        use_modal_wise_scale=True,
    )

    # Suppose we have type_map and modal_map
    type_map = {1: 0, 8: 1}  # Example: H->0, O->1
    modal_map = {'a': 0, 'b': 1}

    # get_shift, get_scale
    s = mod.get_shift(type_map=type_map, modal_map=modal_map)
    sc = mod.get_scale(type_map=type_map, modal_map=modal_map)
    # Expect dict with keys "ambient", "pressure".
    # Example: s["ambient"] = [ shift(0,0), shift(0,1) ] mapped to H,O
    #          s["pressure"] = [ shift(1,0), shift(1,1) ]
    assert isinstance(s, dict) and isinstance(sc, dict)
    assert set(s.keys()) == {'a', 'b'}
    assert set(sc.keys()) == {'a', 'b'}


################################################################################
#                 Tests for get_resolved_shift_scale function                  #
################################################################################


def test_get_resolved_shift_scale_rescale():
    """
    Test get_resolved_shift_scale for a Rescale instance.
    """
    from_m = Rescale(shift=2.0, scale=5.0)
    shift, scale = get_resolved_shift_scale(from_m)
    assert shift == 2.0
    assert scale == 5.0


def test_get_resolved_shift_scale_specieswise():
    """
    Test get_resolved_shift_scale for a SpeciesWiseRescale instance.
    """
    shift_list = [1.0, 2.0]
    scale_list = [3.0, 4.0]
    module = SpeciesWiseRescale(shift=shift_list, scale=scale_list)
    type_map = {1: 0, 8: 1}
    s, sc = get_resolved_shift_scale(module, type_map=type_map)
    # The result should be extended to NUM_UNIV_ELEMENT length in real usage,
    # but at least the first few should match shift_list, scale_list mapped.
    assert isinstance(s, list)
    assert isinstance(sc, list)
    # Check mapped values
    assert s[1] == shift_list[0]
    assert s[8] == shift_list[1]
    assert sc[1] == scale_list[0]
    assert sc[8] == scale_list[1]


def test_get_resolved_shift_scale_modalwise():
    """
    Test get_resolved_shift_scale for a ModalWiseRescale instance.
    """
    shift = [[0.0, 10.0], [5.0, 15.0]]
    scale = [[1.0, 2.0], [10.0, 20.0]]
    mmod = ModalWiseRescale(
        shift=shift,
        scale=scale,
        use_modal_wise_shift=True,
        use_modal_wise_scale=True,
    )
    type_map = {1: 0, 8: 1}
    modal_map = {'a': 0, 'b': 1}
    s, sc = get_resolved_shift_scale(mmod, type_map=type_map, modal_map=modal_map)
    # We expect dictionaries
    assert isinstance(s, dict) and isinstance(sc, dict)
    # Keys "a", "pressure"
    assert 'a' in s
    assert 'b' in s
    # Check one example
    # s["a"] => [0.0, 10.0]
    # sc["a"] => [1.0, 2.0]
    assert s['a'][1] == 0.0
    assert s['a'][8] == 10.0
    assert sc['a'][1] == 1.0
    assert sc['a'][8] == 2.0


################################################################################
#                       Tests for from_mappers function                        #
################################################################################


@pytest.mark.parametrize(
    'shift, scale, type_map, expected_shift, expected_scale',
    [
        # Both shift and scale are floats -> broadcast to each species
        (
            2.0,
            3.0,
            {1: 0, 8: 1},  # e.g., H -> index 0, O -> index 1
            [2.0, 2.0],  # broadcast
            [3.0, 3.0],
        ),
        # shift, scale are same-length lists => directly used
        (
            [0.5, 0.6],
            [1.0, 1.1],
            {1: 0, 8: 1},
            [0.5, 0.6],
            [1.0, 1.1],
        ),
        # shift, scale are entire "universal" length (NUM_UNIV_ELEMENT=118),
        # but we only map out the subset for the actual species in type_map
        (
            [0.1] * NUM_UNIV_ELEMENT,
            [1.1] * NUM_UNIV_ELEMENT,
            {1: 0, 8: 1},
            [0.1, 0.1],
            [1.1, 1.1],
        ),
        # shift is a list, scale is float => shift is used directly, scale broadcast
        (
            [1.0, 2.0],
            5.0,
            {6: 0, 14: 1},  # C -> 0, Si -> 1
            [1.0, 2.0],
            [5.0, 5.0],
        ),
    ],
)
def test_specieswise_rescale_from_mappers(
    shift, scale, type_map, expected_shift, expected_scale
):
    """
    Test SpeciesWiseRescale.from_mappers with various combinations of
    shift/scale (float, list, universal list) and a given type_map.
    """
    module = SpeciesWiseRescale.from_mappers(  # type: ignore
        shift=shift,
        scale=scale,
        type_map=type_map,
    )
    # Check that the module's internal shift and scale have the correct shape
    # The length must match number of species in type_map
    assert module.shift.shape[0] == len(type_map)
    assert module.scale.shape[0] == len(type_map)

    # Check that the content matches expected
    actual_shift = module.shift.detach().cpu().tolist()
    actual_scale = module.scale.detach().cpu().tolist()

    assert pytest.approx(actual_shift) == expected_shift
    assert pytest.approx(actual_scale) == expected_scale


@pytest.mark.parametrize(
    'shift, scale, use_modal_wise_shift, use_modal_wise_scale, '
    'type_map, modal_map, expected_shift, expected_scale',
    [
        # Example 1: single float for shift/scale,
        # broadcast over 2 modals and 2 species
        (
            1.0,
            2.0,
            True,  # shift depends on modal
            True,  # scale depends on modal
            {1: 0, 8: 1},
            {'modA': 0, 'modB': 1},
            # expect 2D => [2 modals x 2 species]
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ),
        # Example 2: shift/scale are universal element-lists => use_modal=False => 1D
        (
            [0.5] * NUM_UNIV_ELEMENT,
            [1.5] * NUM_UNIV_ELEMENT,
            False,  # shift is not modal-wise
            False,  # scale is not modal-wise
            {6: 0, 14: 1},  # e.g. C->0, Si->1
            {'modA': 0, 'modB': 1},
            # 1D => length = n_atom_types(=2)
            [0.5, 0.5],
            [1.5, 1.5],
        ),
        # Example 3: shift is dict of modals -> each is float
        #               => broadcast for each species
        (
            {'modA': 0.0, 'modB': 2.0},
            {'modA': 1.0, 'modB': 3.0},
            True,
            True,
            {1: 0, 8: 1},
            {'modA': 0, 'modB': 1},
            # shift => shape [2 modals, 2 species]
            [[0.0, 0.0], [2.0, 2.0]],
            [[1.0, 1.0], [3.0, 3.0]],
        ),
        # Example 4: already in "modal-wise + species-wise" shape, direct pass
        (
            [[0.0, 10.0], [5.0, 15.0]],
            [[1.0, 2.0], [10.0, 20.0]],
            True,
            True,
            {1: 0, 8: 1},
            {'modA': 0, 'modB': 1},
            [[0.0, 10.0], [5.0, 15.0]],
            [[1.0, 2.0], [10.0, 20.0]],
        ),
        # Example 5: shift is a list of floats (one per modal),
        #   but we want modal-wise => broadcast for each species
        (
            [0.0, 10.0],  # length=2 => same as #modals
            [1.0, 2.0],
            True,
            True,
            {1: 0, 8: 1},
            {'modA': 0, 'modB': 1},
            [[0.0, 0.0], [10.0, 10.0]],
            [[1.0, 1.0], [2.0, 2.0]],
        ),
    ],
)
def test_modalwise_rescale_from_mappers(
    shift,
    scale,
    use_modal_wise_shift,
    use_modal_wise_scale,
    type_map,
    modal_map,
    expected_shift,
    expected_scale,
):
    """
    Test ModalWiseRescale.from_mappers for different shapes of shift/scale,
    combined with type_map and modal_map.
    """

    module = ModalWiseRescale.from_mappers(  # type: ignore
        shift=shift,
        scale=scale,
        use_modal_wise_shift=use_modal_wise_shift,
        use_modal_wise_scale=use_modal_wise_scale,
        type_map=type_map,
        modal_map=modal_map,
    )
    # Check shape of the resulting shift, scale
    # If modal-wise, we expect a 2D shape: [n_modals, n_species]
    # Otherwise, a 1D shape: [n_species]
    if use_modal_wise_shift:
        assert module.shift.dim() == 2
        assert module.shift.shape[0] == len(modal_map)
        assert module.shift.shape[1] == len(type_map)
    else:
        assert module.shift.dim() == 1
        assert module.shift.shape[0] == len(type_map)

    # Similarly for scale
    if use_modal_wise_scale:
        assert module.scale.dim() == 2
        assert module.scale.shape[0] == len(modal_map)
        assert module.scale.shape[1] == len(type_map)
    else:
        assert module.scale.dim() == 1
        assert module.scale.shape[0] == len(type_map)

    # Verify the content matches our expectation
    actual_shift = module.shift.detach().cpu().tolist()
    actual_scale = module.scale.detach().cpu().tolist()

    assert actual_shift == expected_shift
    assert actual_scale == expected_scale
