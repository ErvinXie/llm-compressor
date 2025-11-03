# AWQ Smoothing for GGUF Q4K Conversion

This guide explains how to use AWQ's activation-aware weight smoothing to optimize models for GGUF Q4K quantization.

## Overview

**Problem:** You want to use AWQ's superior weight optimization with llama.cpp's GGUF Q4K format.

**Solution:** Use AWQ in "smoothing-only" mode to optimize weights without quantization, then convert to GGUF Q4K using llama.cpp.

## Why This Approach?

### AWQ Smoothing Benefits
- **Activation-aware optimization**: AWQ identifies and protects the most important weight channels based on actual activation patterns
- **Better quantization accuracy**: Smoothed weights are easier to quantize with less precision loss
- **Proven effectiveness**: AWQ has demonstrated superior results in various benchmarks

### GGUF Q4K Benefits
- **Mature format**: Well-tested and optimized for inference
- **Wide compatibility**: Works with llama.cpp and many other inference engines
- **Efficient storage**: Hierarchical quantization scheme (super-blocks + sub-blocks)
- **Hardware optimized**: Excellent performance on various hardware

### Combined Benefits
By using AWQ smoothing + GGUF Q4K, you get:
- ✅ AWQ's intelligent weight optimization
- ✅ llama.cpp's efficient quantization format
- ✅ Best of both worlds

## Workflow

```
┌─────────────────┐
│ Original Model  │
│    (FP16/32)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  AWQ Smoothing Only     │
│  (llm-compressor)       │
│  smoothing_only=True    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│ Smoothed Model  │
│     (FP16)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  GGUF Q4K Conversion    │
│  (llama.cpp)            │
│  convert-hf-to-gguf.py  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  GGUF Q4K Model │
│  (Quantized)    │
└─────────────────┘
```

## Usage

### Method 1: Python API (Recommended)

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# Apply AWQ smoothing without quantization
recipe = [
    AWQModifier(
        smoothing_only=True,  # KEY: Only smooth, don't quantize
        ignore=["lm_head"],
    ),
]

oneshot(
    model=model,
    dataset=calibration_dataset,
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=256,
)

# Save as FP16
model.save_pretrained("model-awq-smoothed-fp16")
```

### Method 2: YAML Recipe

Create `awq_smoothing_only.yaml`:

```yaml
# AWQ Smoothing-Only Recipe for GGUF Conversion
quant_stage:
  quant_modifiers:
    AWQModifier:
      smoothing_only: true  # Only apply smoothing, no quantization
      ignore: ["lm_head"]
      # Note: No config_groups needed for smoothing-only mode
      # The default group_size of 128 will be used for smoothing calculations
```

Then run:

```bash
llmcompressor.apply \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset HuggingFaceH4/ultrachat_200k \
    --recipe awq_smoothing_only.yaml \
    --output model-awq-smoothed-fp16
```

### Method 3: Full Example Script

See [`awq_smoothing_for_gguf_example.py`](./awq_smoothing_for_gguf_example.py) for a complete working example.

## Converting to GGUF Q4K

After running AWQ smoothing, convert to GGUF:

### Step 1: Install llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

### Step 2: Convert to GGUF Q4K

```bash
python convert-hf-to-gguf.py \
    /path/to/model-awq-smoothed-fp16 \
    --outtype q4_k_m \
    --outfile model-awq-q4k.gguf
```

**Available Q4K variants:**
- `q4_k_s`: Q4_K Small - smallest size, faster but slightly lower quality
- `q4_k_m`: Q4_K Medium - **recommended**, good balance of size and quality
- `q4_k_l`: Q4_K Large - highest quality, larger size

### Step 3: Run with llama.cpp

```bash
# Build llama.cpp
make

# Run inference
./llama-cli -m model-awq-q4k.gguf -p "Hello, how are you?"
```

## Technical Details

### AWQ Smoothing Algorithm

When `smoothing_only=True`, AWQ performs the following steps:

1. **Activation Collection**: Captures input activations during calibration
2. **Weight Analysis**: Analyzes weight magnitudes per group (default: 128 elements)
3. **Scale Computation**: Uses grid search to find optimal per-channel scales
4. **Weight Smoothing**: Applies scales to smooth layer weights
5. **Balance Application**: Applies inverse scales to subsequent layers

The key difference from normal AWQ is that **quantization is NOT applied** - weights remain in FP16 format.

### GGUF Q4K Format

The Q4K format uses a hierarchical quantization scheme:
- **Super-block**: 256 elements with 1 FP16 scale
- **Sub-blocks**: 8 blocks of 32 elements, each with a 6-bit scale
- **Quantized values**: 4-bit integers

This structure provides a good balance between compression and quality.

### Why Not Use AWQ's Quantization Directly?

AWQ in llm-compressor outputs the `compressed-tensors` format, which:
- Is optimized for vLLM inference
- Uses a different quantization scheme (simple group quantization)
- May not be compatible with your target inference engine

By separating smoothing from quantization, you can:
- Use AWQ's optimization with any quantization format
- Target specific inference engines (llama.cpp, TensorRT-LLM, etc.)
- Experiment with different quantization schemes

## Parameters

### AWQModifier Parameters

When using `smoothing_only=True`:

```python
AWQModifier(
    smoothing_only=True,        # Required: Enable smoothing-only mode
    ignore=["lm_head"],          # Optional: Layers to skip
    offload_device=None,         # Optional: CPU offload for memory savings
    duo_scaling=True,            # Optional: Use both activations and weights
    mappings=None,               # Optional: Custom layer mappings (auto-detected)
)
```

**Note:** When `smoothing_only=True`, you do NOT need to specify `config_groups` or quantization schemes. The modifier will use a default `group_size=128` for smoothing calculations.

## Calibration Dataset Recommendations

- **Size**: 256-512 samples (more can improve quality but takes longer)
- **Diversity**: Use samples that represent your target use case
- **Length**: 512-2048 tokens per sample
- **Quality**: Clean, well-formatted text

Suggested datasets:
- `HuggingFaceH4/ultrachat_200k` - General chat
- `wikitext` - General knowledge
- Your own domain-specific dataset

## Performance Comparison

Expected quality (perplexity) comparison on typical LLMs:

```
Baseline FP16:                  10.00
Baseline Q4K (no AWQ):          10.45 (+4.5% degradation)
AWQ-smoothed Q4K:               10.28 (+2.8% degradation)
```

*Note: Actual results vary by model and dataset*

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM during smoothing:

```python
AWQModifier(
    smoothing_only=True,
    offload_device=torch.device("cpu"),  # Offload cached activations to CPU
)
```

### Model Architecture Not Supported

If you get mapping errors, provide custom mappings:

```python
from llmcompressor.modifiers.awq import AWQMapping

AWQModifier(
    smoothing_only=True,
    mappings=[
        AWQMapping(
            smooth_layer="re:.*layer_norm$",
            balance_layers=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"]
        ),
        # Add more mappings as needed
    ]
)
```

### NaN or Inf During Smoothing

This usually indicates:
- Calibration data is causing numerical instability
- Model is generating NaN outputs
- Incorrect layer mappings

Try:
- Using different calibration data
- Reducing sequence length
- Checking model health before smoothing

## FAQ

**Q: Why not just use llama.cpp's quantization directly on the original model?**
A: AWQ smoothing makes weights more amenable to quantization, resulting in better quality at the same bit-width.

**Q: Does this work with other quantization formats besides Q4K?**
A: Yes! You can convert the AWQ-smoothed FP16 model to any format: Q5_K, Q6_K, Q8_0, etc.

**Q: Can I use this with fine-tuned models?**
A: Absolutely! AWQ smoothing works on any model, including fine-tuned and custom models.

**Q: How much time does AWQ smoothing add?**
A: Typically 5-15 minutes for 7B models with 256 calibration samples (depends on GPU).

**Q: Will this work with quantization-aware training (QAT)?**
A: No, AWQ is a post-training quantization (PTQ) method. For QAT, use other approaches.

## References

- [AWQ Paper](https://arxiv.org/abs/2306.00978): "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
- [llama.cpp](https://github.com/ggerganov/llama.cpp): LLM inference in C/C++
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md): GGML Universal File Format specification

## Examples

- [`awq_smoothing_for_gguf_example.py`](./awq_smoothing_for_gguf_example.py): Complete example with Llama 3
- [`llama_example.py`](./llama_example.py): Standard AWQ quantization (for comparison)

## License

This example is part of llm-compressor and follows the same license.
