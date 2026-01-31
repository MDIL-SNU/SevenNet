"""Tests for SevenNetModel torchsim integration.

This test file is structured to work in the torchsim repository's test directory.
It uses the factory function pattern from torchsim's test infrastructure.
"""

import pytest
import torch

try:
    from torch_sim.models.interface import validate_model_outputs
    from torch_sim.testing import (
        SIMSTATE_BULK_GENERATORS,
        assert_model_calculator_consistency,
    )

    TORCH_SIM_AVAILABLE = True
except ImportError:
    TORCH_SIM_AVAILABLE = False  # pyright: ignore[reportConstantRedefinition]
    SIMSTATE_BULK_GENERATORS = {}  # pyright: ignore[reportConstantRedefinition]

    def validate_model_outputs(*args, **kwargs):
        return None

    def assert_model_calculator_consistency(*args, **kwargs):
        return None

if not TORCH_SIM_AVAILABLE:
    pytest.skip(
        'torch_sim not installed. Install torch-sim-atomistic separately if needed.',
        allow_module_level=True,
    )

import sevenn.util
from sevenn.calculator import SevenNetCalculator
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.torchsim import SevenNetModel

# Test configuration
model_name = 'sevennet-mf-ompa'
modal_name = 'mpa'
DTYPE = torch.float32
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@pytest.fixture
def pretrained_sevenn_model() -> AtomGraphSequential:
    """Load a pretrained SevenNet model for testing."""
    cp = sevenn.util.load_checkpoint(model_name)
    model_loaded = cp.build_model()
    model_loaded.set_is_batch_data(True)  # pyright: ignore[reportCallIssue]
    return model_loaded.to(DEVICE)


@pytest.fixture
def sevenn_model(pretrained_sevenn_model: AtomGraphSequential) -> SevenNetModel:
    """Create a SevenNetModel wrapper for the pretrained model."""
    return SevenNetModel(
        model=pretrained_sevenn_model, modal=modal_name, device=DEVICE, dtype=DTYPE
    )


@pytest.fixture
def sevenn_calculator() -> SevenNetCalculator:
    """Create a SevenNetCalculator for consistency testing."""
    return SevenNetCalculator(model_name, modal=modal_name, device=str(DEVICE))


def test_sevenn_model_output_validation(sevenn_model: SevenNetModel) -> None:
    """Test that a model implementation follows the ModelInterface contract."""
    validate_model_outputs(sevenn_model, DEVICE, DTYPE)


def test_sevenn_model_dtype_validation(
    pretrained_sevenn_model: AtomGraphSequential
) -> None:
    """Test that SevenNetModel raises an error if dtype is not float32."""
    with pytest.raises(
        ValueError,
        match='SevenNetModel currently only supports torch.float32, but received '
        + 'different dtype: torch.float64',
    ):
        _ = SevenNetModel(
            model=pretrained_sevenn_model,
            modal=modal_name,
            device=DEVICE,
            dtype=torch.float64,
        )


@pytest.mark.parametrize('sim_state_name', SIMSTATE_BULK_GENERATORS)
def test_sevenn_model_consistency(
    sim_state_name: str,
    sevenn_model: SevenNetModel,
    sevenn_calculator: SevenNetCalculator,
) -> None:
    """Test consistency between SevenNetModel and SevenNetCalculator.

    NOTE: sevenn is broken for the benzene simstate is ase comparison."""
    sim_state = SIMSTATE_BULK_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert_model_calculator_consistency(sevenn_model, sevenn_calculator, sim_state)
