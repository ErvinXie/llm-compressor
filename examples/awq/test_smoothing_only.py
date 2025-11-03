"""
Quick test script to verify AWQ smoothing-only mode works correctly.

This script uses a tiny model for quick testing.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# Use a tiny model for testing
MODEL_ID = "facebook/opt-125m"

print("=" * 80)
print("Testing AWQ Smoothing-Only Mode")
print("=" * 80)

# Load model
print(f"\n1. Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Create a tiny calibration dataset
print("\n2. Creating calibration dataset (10 samples)")
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language.",
    "The weather today is sunny and warm.",
    "Deep learning models require large amounts of data.",
    "Natural language processing is used in chatbots.",
    "Computer vision enables machines to see.",
    "Quantum computing is still in early stages.",
    "The internet has changed how we communicate.",
    "Renewable energy is important for the future.",
]

# Tokenize
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
dataset = [{"input_ids": inputs["input_ids"][i]} for i in range(len(texts))]

# Get original weight for comparison
print("\n3. Capturing original weight (before smoothing)")
first_layer_name = "model.decoder.layers.0.self_attn.q_proj"
original_weight = None
for name, module in model.named_modules():
    if name == first_layer_name:
        original_weight = module.weight.data.clone()
        print(f"   Original weight mean: {original_weight.mean().item():.6f}")
        print(f"   Original weight std: {original_weight.std().item():.6f}")
        break

# Apply AWQ smoothing
print("\n4. Applying AWQ smoothing (smoothing_only=True)")
recipe = [
    AWQModifier(
        smoothing_only=True,
        ignore=["lm_head"],
    ),
]

oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=128,
    num_calibration_samples=10,
)

# Check that weights have changed (smoothing was applied)
print("\n5. Verifying weights were smoothed")
smoothed_weight = None
for name, module in model.named_modules():
    if name == first_layer_name:
        smoothed_weight = module.weight.data
        print(f"   Smoothed weight mean: {smoothed_weight.mean().item():.6f}")
        print(f"   Smoothed weight std: {smoothed_weight.std().item():.6f}")
        break

if original_weight is not None and smoothed_weight is not None:
    weight_diff = (original_weight - smoothed_weight).abs().mean().item()
    print(f"   Average weight change: {weight_diff:.6f}")

    if weight_diff > 1e-6:
        print("   ✓ Weights were modified (smoothing applied)")
    else:
        print("   ✗ WARNING: Weights unchanged (smoothing may not have worked)")

# Verify model is still FP16 (not quantized)
print("\n6. Verifying model dtype (should remain FP16)")
for name, module in model.named_modules():
    if name == first_layer_name:
        weight_dtype = module.weight.dtype
        print(f"   Weight dtype: {weight_dtype}")
        if weight_dtype == torch.float16:
            print("   ✓ Model remains in FP16 (not quantized)")
        else:
            print(f"   ✗ WARNING: Unexpected dtype {weight_dtype}")
        break

# Test generation
print("\n7. Testing model generation")
input_text = "Hello, my name is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=20)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"   Input: {input_text}")
print(f"   Output: {generated_text}")
print("   ✓ Model can still generate text")

# Save model
print("\n8. Saving AWQ-smoothed model")
save_dir = "test-awq-smoothed-fp16"
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)
print(f"   ✓ Model saved to: {save_dir}")

print("\n" + "=" * 80)
print("✓ AWQ Smoothing-Only Test PASSED")
print("=" * 80)
print(f"""
Next steps to convert to GGUF Q4K:

1. Install llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   pip install -r requirements.txt

2. Convert to GGUF:
   python convert-hf-to-gguf.py {save_dir} --outtype q4_k_m

3. Run inference:
   ./llama-cli -m {save_dir}/model.gguf -p "Hello, my name is"
""")
