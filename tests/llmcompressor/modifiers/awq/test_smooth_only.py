"""
Tests for AWQ smooth_only functionality
"""

import pytest
import torch
from torch.nn import Linear, Module, Sequential

from llmcompressor.core import State
from llmcompressor.modifiers.awq import AWQModifier


class SimpleModel(Module):
    """Simple model for testing AWQ smooth_only"""

    def __init__(self):
        super().__init__()
        self.layer1 = Linear(128, 128)
        self.layer2 = Linear(128, 128)
        self.layer3 = Linear(128, 128)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


@pytest.mark.unit
def test_awq_smooth_only_parameter():
    """Test that smooth_only parameter is correctly set"""
    # Test default value (False)
    modifier = AWQModifier()
    assert modifier.smooth_only is False

    # Test explicit True
    modifier = AWQModifier(smooth_only=True)
    assert modifier.smooth_only is True

    # Test explicit False
    modifier = AWQModifier(smooth_only=False)
    assert modifier.smooth_only is False


@pytest.mark.unit
def test_awq_smooth_only_no_quantization_init():
    """Test that quantization is not initialized in smooth_only mode"""
    model = SimpleModel()
    state = State(model=model)

    # Create modifier with smooth_only=True but with a quantization scheme
    # In smooth_only mode, the quantization scheme should be ignored
    modifier = AWQModifier(
        smooth_only=True,
        scheme="W4A16",  # This should be ignored in smooth_only mode
        ignore=["layer3"],
    )

    # Initialize the modifier
    modifier.on_initialize(state)

    # In smooth_only mode, modules should NOT have quantization_scheme attached
    # because we skip QuantizationMixin.initialize_quantization
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            # Verify that quantization_scheme was NOT attached
            assert not hasattr(
                module, "quantization_scheme"
            ), f"Module {name} should not have quantization_scheme in smooth_only mode"


@pytest.mark.unit
def test_awq_smooth_only_mappings_resolved():
    """Test that AWQ mappings are still resolved in smooth_only mode"""
    model = SimpleModel()
    state = State(model=model)

    # Create modifier with smooth_only=True and custom mappings
    modifier = AWQModifier(
        smooth_only=True,
        mappings=[
            {"smooth_layer": "layer1", "balance_layers": ["layer2"]},
        ],
    )

    # Initialize the modifier
    modifier.on_initialize(state)

    # Verify that mappings were resolved even in smooth_only mode
    assert len(modifier._resolved_mappings) > 0, (
        "AWQ mappings should be resolved in smooth_only mode "
        "because they are needed for smoothing"
    )


@pytest.mark.unit
def test_awq_smooth_only_with_quantization_config():
    """Test that quantization config is ignored in smooth_only mode"""
    model = SimpleModel()
    state = State(model=model)

    # Create modifier with both smooth_only=True and quantization config
    modifier = AWQModifier(
        smooth_only=True,
        scheme="W4A16_ASYM",
        targets=["Linear"],
        ignore=["layer3"],
    )

    # Initialize the modifier
    modifier.on_initialize(state)

    # Verify that quantization was NOT initialized
    # Check that modules don't have quantization artifacts
    for name, module in model.named_modules():
        if isinstance(module, Linear) and "layer3" not in name:
            # Should NOT have quantization_scheme in smooth_only mode
            assert not hasattr(module, "quantization_scheme"), (
                f"Module {name} should not be quantized in smooth_only mode, "
                "even when scheme is provided"
            )


@pytest.mark.unit
def test_awq_smooth_only_dtype_preservation():
    """Test that model dtype is preserved in smooth_only mode"""
    # Create model in bfloat16
    model = SimpleModel().to(dtype=torch.bfloat16)
    original_dtype = next(model.parameters()).dtype

    state = State(model=model)

    modifier = AWQModifier(
        smooth_only=True,
        ignore=["layer3"],
    )

    # Initialize the modifier
    modifier.on_initialize(state)

    # Verify dtype is preserved
    current_dtype = next(model.parameters()).dtype
    assert current_dtype == original_dtype == torch.bfloat16, (
        f"Model dtype should be preserved as bfloat16, "
        f"but got {current_dtype}"
    )


@pytest.mark.unit
def test_awq_normal_mode_still_works():
    """Test that normal AWQ mode (with quantization) still works"""
    model = SimpleModel()
    state = State(model=model)

    # Create modifier with smooth_only=False (default)
    modifier = AWQModifier(
        smooth_only=False,
        scheme="W4A16",
        targets=["Linear"],
        ignore=["layer3"],
    )

    # Initialize the modifier
    modifier.on_initialize(state)

    # In normal mode, modules SHOULD have quantization_scheme attached
    has_quantization = False
    for name, module in model.named_modules():
        if isinstance(module, Linear) and "layer3" not in name:
            if hasattr(module, "quantization_scheme"):
                has_quantization = True
                break

    assert has_quantization, (
        "In normal mode (smooth_only=False), modules should have "
        "quantization_scheme attached"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
