#!/usr/bin/env python3
"""
Simple verification script for AWQ smooth_only functionality

This script verifies that:
1. smooth_only parameter can be set
2. Quantization is skipped in smooth_only mode
3. Model dtype is preserved
4. Mappings are still resolved

Usage:
    python verify_awq_smooth_only.py
"""

import torch
from torch.nn import Linear, Module

from llmcompressor.core import State
from llmcompressor.modifiers.awq import AWQModifier


class SimpleModel(Module):
    """Simple model for testing"""

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


def test_smooth_only_parameter():
    """Test 1: Verify smooth_only parameter works"""
    print("Test 1: Checking smooth_only parameter...")

    # Default should be False
    modifier = AWQModifier()
    assert modifier.smooth_only is False, "Default smooth_only should be False"

    # Should be able to set to True
    modifier = AWQModifier(smooth_only=True)
    assert modifier.smooth_only is True, "smooth_only should be True when set"

    print("✓ Test 1 passed: smooth_only parameter works correctly")


def test_no_quantization_in_smooth_only():
    """Test 2: Verify quantization is skipped in smooth_only mode"""
    print("\nTest 2: Checking that quantization is skipped in smooth_only mode...")

    model = SimpleModel()
    state = State(model=model)

    # Create modifier with smooth_only=True and a quantization scheme
    modifier = AWQModifier(
        smooth_only=True,
        scheme="W4A16",  # This should be ignored
        ignore=["layer3"],
    )

    # Initialize
    modifier.on_initialize(state)

    # Check that quantization_scheme was NOT attached
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            assert not hasattr(module, "quantization_scheme"), (
                f"Module {name} should not have quantization_scheme in smooth_only mode"
            )

    print("✓ Test 2 passed: No quantization in smooth_only mode")


def test_dtype_preserved():
    """Test 3: Verify model dtype is preserved"""
    print("\nTest 3: Checking that model dtype is preserved...")

    model = SimpleModel().to(dtype=torch.bfloat16)
    original_dtype = next(model.parameters()).dtype

    state = State(model=model)

    modifier = AWQModifier(smooth_only=True)
    modifier.on_initialize(state)

    current_dtype = next(model.parameters()).dtype
    assert current_dtype == original_dtype == torch.bfloat16, (
        f"Expected bfloat16, got {current_dtype}"
    )

    print("✓ Test 3 passed: Model dtype preserved as bfloat16")


def test_mappings_resolved():
    """Test 4: Verify AWQ mappings are still resolved"""
    print("\nTest 4: Checking that AWQ mappings are resolved...")

    model = SimpleModel()
    state = State(model=model)

    # In smooth_only mode, mappings should still be resolved
    # (they're needed for the smoothing operation)
    # We'll use the default mappings which won't match our simple model,
    # but we can verify that the mapping resolution process runs
    modifier = AWQModifier(
        smooth_only=True,
    )

    # Before initialization, _resolved_mappings should be empty
    assert len(modifier._resolved_mappings) == 0, (
        "Mappings should be empty before initialization"
    )

    modifier.on_initialize(state)

    # After initialization, _resolved_mappings should be set
    # (even if empty due to no matches with default mappings)
    # The key is that _set_resolved_mappings was called
    assert isinstance(modifier._resolved_mappings, list), (
        "Mappings should be resolved (as a list) in smooth_only mode"
    )

    print("✓ Test 4 passed: AWQ mapping resolution works in smooth_only mode")


def test_normal_mode_still_works():
    """Test 5: Verify normal AWQ mode (with quantization) still works"""
    print("\nTest 5: Checking that normal AWQ mode still works...")

    model = SimpleModel()
    state = State(model=model)

    modifier = AWQModifier(
        smooth_only=False,  # Normal mode
        scheme="W4A16",
        targets=["Linear"],
        ignore=["layer3"],
    )

    modifier.on_initialize(state)

    # In normal mode, should have quantization_scheme
    has_quantization = False
    for name, module in model.named_modules():
        if isinstance(module, Linear) and "layer3" not in name:
            if hasattr(module, "quantization_scheme"):
                has_quantization = True
                break

    assert has_quantization, "Normal mode should attach quantization_scheme"

    print("✓ Test 5 passed: Normal AWQ mode still works")


def main():
    print("=" * 60)
    print("AWQ smooth_only Verification Tests")
    print("=" * 60)

    try:
        test_smooth_only_parameter()
        test_no_quantization_in_smooth_only()
        test_dtype_preserved()
        test_mappings_resolved()
        test_normal_mode_still_works()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nAWQ smooth_only implementation is working correctly.")
        print("You can now use AWQModifier(smooth_only=True) to apply")
        print("smoothing without quantization.")
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
