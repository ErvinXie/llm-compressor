"""
AWQ Smoothing-Only Example for GGUF Q4K Conversion

This example demonstrates how to use AWQ's activation-aware weight smoothing
without applying quantization. The resulting FP16 model can then be converted
to GGUF Q4K format using llama.cpp's conversion tools.

Workflow:
1. Apply AWQ smoothing to optimize weights (this script)
2. Save model in FP16 format
3. Convert to GGUF Q4K using llama.cpp (separate step)

The benefit of this approach is that it combines:
- AWQ's activation-aware weight optimization
- llama.cpp's mature Q4K quantization format
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# ============================================================================
# STEP 1: Load Model
# ============================================================================
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ============================================================================
# STEP 2: Prepare Calibration Dataset
# ============================================================================
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 256  # 256-512 samples recommended
MAX_SEQUENCE_LENGTH = 512

print(f"Loading calibration dataset: {DATASET_ID}")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


# ============================================================================
# STEP 3: Apply AWQ Smoothing (Without Quantization)
# ============================================================================
print("\n" + "=" * 80)
print("Applying AWQ weight smoothing...")
print("=" * 80)

recipe = [
    AWQModifier(
        smoothing_only=True,  # KEY PARAMETER: Only smooth, don't quantize
        ignore=["lm_head"],
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ============================================================================
# STEP 4: Save AWQ-Smoothed FP16 Model
# ============================================================================
SAVE_DIR = MODEL_ID.split("/")[-1] + "-awq-smoothed-fp16"

print("\n" + "=" * 80)
print(f"Saving AWQ-smoothed FP16 model to: {SAVE_DIR}")
print("=" * 80)

# Save in standard HuggingFace format (FP16, no compression)
model.save_pretrained(SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\n✓ Model saved successfully to {SAVE_DIR}")

# ============================================================================
# STEP 5: Next Steps - Convert to GGUF Q4K
# ============================================================================
print("\n" + "=" * 80)
print("NEXT STEPS: Convert to GGUF Q4K format")
print("=" * 80)
print("""
To convert the AWQ-smoothed model to GGUF Q4K format:

1. Clone llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp

2. Install requirements:
   pip install -r requirements.txt

3. Convert to GGUF Q4K:
   python convert-hf-to-gguf.py \\
       /path/to/{save_dir} \\
       --outtype q4_k_m \\
       --outfile {save_dir}/model.gguf

Alternative quantization types:
   --outtype q4_k_s  # Q4_K Small (smaller size, slightly lower quality)
   --outtype q4_k_m  # Q4_K Medium (recommended, good balance)
   --outtype q4_k_l  # Q4_K Large (larger size, better quality)

The resulting GGUF file will combine:
✓ AWQ's activation-aware weight smoothing
✓ llama.cpp's optimized Q4K quantization format
✓ Compatibility with llama.cpp inference engine
""".format(
    save_dir=SAVE_DIR
))

print("\n" + "=" * 80)
print("OPTIONAL: Test generation with the smoothed model")
print("=" * 80)

from llmcompressor.utils import dispatch_for_generation

dispatch_for_generation(model)
input_ids = tokenizer(
    "Hello, my name is", return_tensors="pt"
).input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=50)
print("\nSample generation:")
print(tokenizer.decode(output[0]))
print("\n" + "=" * 80)
