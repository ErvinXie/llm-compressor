"""
AWQ Smoothing for Qwen3-30B-A3B MoE Model (for GGUF Q4K Conversion)

This script applies AWQ smoothing to the Qwen3-30B-A3B (128 experts, activate 8) model
and saves it in FP16 format for subsequent GGUF Q4K conversion using llama.cpp.

Model characteristics:
- Architecture: Qwen3MoeForCausalLM
- Size: ~30B parameters (189GB in bfloat16)
- Experts: 128 experts, 8 activated per token
- Layers: 48 layers
- Context: 40960 tokens

Expected processing time: 30-60 minutes (depending on GPU)
Expected output size: ~60GB (FP16)

Usage:
    # Basic usage with defaults
    python qwen3_30b_a3b_smoothing_for_gguf.py

    # Custom paths
    python qwen3_30b_a3b_smoothing_for_gguf.py \
        --model_path /path/to/your/model \
        --output_dir /path/to/output

    # Full customization
    python qwen3_30b_a3b_smoothing_for_gguf.py \
        --model_path /mnt/data/models/Qwen3-30B-A3B-250425 \
        --output_dir Qwen3-30B-A3B-awq-smoothed-fp16 \
        --num_calibration_samples 512 \
        --max_seq_length 2048 \
        --offload_to_cpu
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply AWQ smoothing to Qwen3-30B-A3B for GGUF conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/data/models/Qwen3-30B-A3B-250425",
        help="Path to the input model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Qwen3-30B-A3B-awq-smoothed-fp16",
        help="Directory to save the AWQ-smoothed model",
    )

    # Calibration settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset to use",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train_sft",
        help="Dataset split to use for calibration",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=256,
        help="Number of calibration samples (256-512 recommended)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration",
    )

    # Memory optimization
    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        help="Offload cached activations to CPU (use if OOM)",
    )

    # Model loading options
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "auto"],
        help="Data type for loading the model",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ============================================================================
    # Display Configuration
    # ============================================================================
    print("=" * 80)
    print("AWQ Smoothing for Qwen3-30B-A3B MoE Model")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Output path: {args.output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Calibration samples: {args.num_calibration_samples}")
    print(f"Sequence length: {args.max_seq_length}")
    print(f"Offload to CPU: {args.offload_to_cpu}")
    print("=" * 80)

    # ============================================================================
    # STEP 1: Load Model
    # ============================================================================
    print("\n[1/5] Loading model...")
    print("This may take a few minutes for a 189GB model...")

    # Convert dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "auto": "auto",
    }
    torch_dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",           # Automatic device placement
        trust_remote_code=True,
    )

    print(f"✓ Model loaded successfully")
    print(f"  - Model type: {model.config.architectures[0]}")
    print(f"  - Layers: {model.config.num_hidden_layers}")
    print(f"  - Experts: {model.config.num_experts}")
    print(f"  - Active experts per token: {model.config.num_experts_per_tok}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # ============================================================================
    # STEP 2: Prepare Calibration Dataset
    # ============================================================================
    print(f"\n[2/5] Loading calibration dataset...")
    print(f"Dataset: {args.dataset}")

    ds = load_dataset(
        args.dataset,
        split=f"{args.dataset_split}[:{args.num_calibration_samples}]"
    )
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
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    print(f"✓ Dataset loaded: {len(ds)} samples")

    # ============================================================================
    # STEP 3: Apply AWQ Smoothing (Without Quantization)
    # ============================================================================
    print("\n[3/5] Applying AWQ smoothing...")
    print("This will take 30-60 minutes depending on your GPU...")
    print("Progress will be shown below:")

    recipe = [
        AWQModifier(
            smoothing_only=True,  # KEY: Only smooth, don't quantize
            ignore=["lm_head"],
            offload_device=torch.device("cpu") if args.offload_to_cpu else None,
            duo_scaling=True,     # Use both activations and weights
        ),
    ]

    try:
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.num_calibration_samples,
        )
        print("\n✓ AWQ smoothing completed successfully")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n✗ Out of memory error!")
            print("Try adding --offload_to_cpu flag")
            raise
        else:
            raise

    # ============================================================================
    # STEP 4: Save AWQ-Smoothed Model
    # ============================================================================
    print(f"\n[4/5] Saving AWQ-smoothed model...")
    print(f"Output directory: {args.output_dir}")
    print("This may take several minutes for large models...")

    # Convert to float16 before saving (reduces size from bfloat16)
    model = model.to(torch.float16)

    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,  # Use safetensors format
    )
    tokenizer.save_pretrained(args.output_dir)

    print(f"✓ Model saved successfully")

    # ============================================================================
    # STEP 5: Verify and Next Steps
    # ============================================================================
    print("\n[5/5] Verification")

    # Quick sanity check
    print("Running quick generation test...")
    dispatch_for_generation(model)
    input_ids = tokenizer(
        "Hello, what can you do?", return_tensors="pt"
    ).input_ids.to(model.device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50, do_sample=False)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Sample output: {generated_text[:200]}...")
    print("✓ Model can generate text successfully")

    # ============================================================================
    # Summary and Next Steps
    # ============================================================================
    print("\n" + "=" * 80)
    print("✓ AWQ SMOOTHING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"""
Model Information:
  - Input:  {args.model_path}
  - Output: {args.output_dir}
  - Format: FP16 safetensors
  - Size:   ~60GB (estimated)

What was done:
  ✓ Applied activation-aware weight smoothing
  ✓ Protected salient weight channels
  ✓ Optimized weights for better quantization
  ✓ Model remains in FP16 (not quantized)

NEXT STEPS: Convert to GGUF Q4K Format
═══════════════════════════════════════════════════════════════

1. Install llama.cpp (if not already installed):
   ───────────────────────────────────────────────────
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   pip install -r requirements.txt
   make  # Compile for your platform

2. Convert to GGUF Q4K Medium (recommended):
   ───────────────────────────────────────────────────
   python convert-hf-to-gguf.py \\
       {args.output_dir} \\
       --outtype q4_k_m \\
       --outfile {args.output_dir}.q4_k_m.gguf

   Alternative quantization types:
   • q4_k_s  - Q4_K Small (smaller, faster, slightly lower quality)
   • q4_k_m  - Q4_K Medium (recommended, good balance)
   • q4_k_l  - Q4_K Large (higher quality, larger size)
   • q5_k_m  - Q5_K Medium (even higher quality)
   • q6_k    - Q6_K (near FP16 quality)

3. Test inference with llama.cpp:
   ───────────────────────────────────────────────────
   ./llama-cli \\
       -m {args.output_dir}.q4_k_m.gguf \\
       -p "Hello, what can you do?" \\
       -n 100 \\
       --temp 0.7

4. Run as a server (optional):
   ───────────────────────────────────────────────────
   ./llama-server \\
       -m {args.output_dir}.q4_k_m.gguf \\
       --host 0.0.0.0 \\
       --port 8080

Expected Results:
═══════════════════════════════════════════════════════════════
• GGUF file size: ~15-20GB (Q4K_M format)
• Quality: Better than direct Q4K (thanks to AWQ smoothing)
• Speed: Excellent inference speed with llama.cpp
• Compatibility: Works with llama.cpp and compatible tools

Tips:
═══════════════════════════════════════════════════════════════
• Use Q4_K_M for best quality/size balance
• Use Q4_K_S if you need smaller size
• Use Q5_K_M or Q6_K if quality is more important than size
• Test with your specific use cases to find the best quantization

For more information:
═══════════════════════════════════════════════════════════════
• AWQ Paper: https://arxiv.org/abs/2306.00978
• llama.cpp: https://github.com/ggerganov/llama.cpp
• GGUF Docs: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
""")

    print("=" * 80)


if __name__ == "__main__":
    main()
