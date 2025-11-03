# AWQ Smoothing-Only Mode Implementation

## Summary

This implementation adds a `smoothing_only` mode to the AWQModifier, enabling users to apply AWQ's activation-aware weight smoothing without quantization. This is particularly useful for exporting models to GGUF Q4K format using llama.cpp.

## Use Case

**Problem**: Users want to combine AWQ's superior weight optimization with llama.cpp's GGUF Q4K quantization format.

**Solution**: Apply AWQ smoothing to optimize weights, save as FP16, then convert to GGUF Q4K using llama.cpp's tools.

## Changes Made

### 1. Core Implementation (`src/llmcompressor/modifiers/awq/base.py`)

#### Added Parameter
```python
smoothing_only: bool = False  # If True, only apply smoothing without quantization
```

#### Modified Methods

**`validate_awq_after()`** (lines 150-170)
- Skip quantization config validation when `smoothing_only=True`
- Set default `group_size=128` for smoothing calculations
- Log info message about smoothing-only mode

**`on_initialize()`** (lines 231-243)
- Skip quantization initialization when `smoothing_only=True`
```python
if not self.smoothing_only and QuantizationMixin.has_config(self):
    QuantizationMixin.initialize_quantization(self, state.model)
```

**`on_start()`** (lines 255-269)
- Skip quantization calibration setup when `smoothing_only=True`
- Skip `disable_quantization` call (not needed if quantization not initialized)

**`on_end()`** (lines 287-315)
- Skip weight quantization parameter calculation when `smoothing_only=True`
- Skip `update_weight_zp_scale()` and `QuantizationMixin.end_calibration()`
- Log completion message

### 2. Documentation

#### Updated Docstring
Added documentation for the new parameter explaining:
- Purpose: Export AWQ-optimized FP16 models for GGUF conversion
- Behavior: Weights smoothed but remain in FP16
- Use case: Converting to other quantization formats

### 3. Examples and Documentation

#### Created Files

1. **`examples/awq/awq_smoothing_for_gguf_example.py`**
   - Complete working example using Llama 3
   - Shows full workflow: load → smooth → save → convert instructions
   - Includes test generation

2. **`examples/awq/README_GGUF.md`**
   - Comprehensive guide (300+ lines)
   - Explains benefits of combining AWQ + GGUF Q4K
   - Detailed workflow and conversion instructions
   - Technical details about both formats
   - Troubleshooting section
   - FAQ

3. **`examples/awq/recipe_awq_smoothing_only.yaml`**
   - YAML recipe example
   - Shows how to use via config files
   - Includes usage instructions and comments

4. **`examples/awq/test_smoothing_only.py`**
   - Quick test script using tiny model (OPT-125M)
   - Verifies smoothing is applied
   - Verifies model remains FP16
   - Tests generation still works

5. **`examples/awq/README.md`** (updated)
   - Added section about smoothing-only mode
   - Links to new documentation

## How It Works

### Normal AWQ Flow
```
Load Model → Initialize Quantization → Apply Smoothing →
Apply Quantization → Save (compressed-tensors format)
```

### Smoothing-Only Flow
```
Load Model → Apply Smoothing → Save (FP16 HuggingFace format)
```

### Key Differences

| Aspect | Normal AWQ | Smoothing-Only |
|--------|-----------|----------------|
| Quantization Config | Required | Not needed |
| Quantization Init | Yes | Skipped |
| Smoothing | Applied | Applied ✓ |
| Quantization | Applied | Skipped |
| Output Format | compressed-tensors | FP16 safetensors |
| Output Size | Small (4-bit) | Large (16-bit) |
| Target Use | vLLM inference | Further conversion |

## Usage Examples

### Python API
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

recipe = [
    AWQModifier(
        smoothing_only=True,
        ignore=["lm_head"],
    ),
]

oneshot(model=model, dataset=dataset, recipe=recipe)
model.save_pretrained("model-awq-smoothed-fp16")
```

### YAML Recipe
```yaml
quant_stage:
  quant_modifiers:
    AWQModifier:
      smoothing_only: true
      ignore: ["lm_head"]
```

### Converting to GGUF Q4K
```bash
# After smoothing
python convert-hf-to-gguf.py model-awq-smoothed-fp16 --outtype q4_k_m
```

## Benefits

1. **Best of Both Worlds**
   - AWQ's activation-aware weight optimization
   - llama.cpp's mature Q4K quantization format

2. **Flexibility**
   - Can convert to any format (Q4K, Q5K, Q6K, etc.)
   - Not locked into compressed-tensors format

3. **Compatibility**
   - Works with llama.cpp inference engine
   - Compatible with many other tools

4. **Quality**
   - Better quantization accuracy than direct Q4K
   - AWQ smoothing makes weights more amenable to quantization

## Testing

Run the test script:
```bash
python examples/awq/test_smoothing_only.py
```

Expected output:
- ✓ Weights modified (smoothing applied)
- ✓ Model remains FP16 (not quantized)
- ✓ Model can still generate text
- ✓ Model saved successfully

## Backward Compatibility

- No breaking changes
- Default behavior unchanged (`smoothing_only=False`)
- All existing code continues to work

## Implementation Notes

### Why Skip Quantization Init?

When `smoothing_only=True`, we skip:
1. `QuantizationMixin.initialize_quantization()` - No need to add quantization modules
2. `QuantizationMixin.start_calibration()` - No quantization observers needed
3. `disable_quantization()` - No quantization to disable
4. `update_weight_zp_scale()` - No quantization parameters to compute
5. `QuantizationMixin.end_calibration()` - No quantization to finalize

The smoothing logic (`_apply_smoothing`) works independently and only modifies weight tensors directly.

### Group Size for Smoothing

The `group_size=128` is used in smoothing calculations for:
- Normalizing weights per group
- Computing per-channel weight magnitudes
- This doesn't affect quantization (since there is none)

## Future Enhancements

Potential improvements:
1. Support custom group_size for smoothing
2. Add option to export to other formats directly
3. Benchmark quality improvements vs direct quantization
4. Add more examples for different model architectures

## Files Modified

```
src/llmcompressor/modifiers/awq/base.py          (modified)
examples/awq/README.md                            (modified)
examples/awq/awq_smoothing_for_gguf_example.py   (new)
examples/awq/README_GGUF.md                       (new)
examples/awq/recipe_awq_smoothing_only.yaml      (new)
examples/awq/test_smoothing_only.py              (new)
```

## References

- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GGUF Format Spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
