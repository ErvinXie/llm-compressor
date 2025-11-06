"""
AWQ Smooth-Only Example for Qwen3 30B A3B MoE Model

This script demonstrates how to apply AWQ smoothing without quantization,
keeping the model weights in bf16 precision.

Usage:
    python awq_smooth_only_example.py \
        --model_path /path/to/qwen3-30b-a3b \
        --output_path /path/to/smoothed-model \
        --calibration_data /path/to/calibration.json \
        --num_samples 512
"""

import argparse
import torch
from datasets import load_dataset

from llmcompressor.entrypoints.oneshot import oneshot


def prepare_calibration_dataset(dataset_name="HuggingFaceH4/ultrachat_200k", num_samples=512):
    """
    Load and prepare calibration data for AWQ smoothing

    Args:
        dataset_name: Name of the dataset to use for calibration
        num_samples: Number of calibration samples to use

    Returns:
        Dataset object with a 'text' column
    """
    print(f"Loading calibration dataset: {dataset_name}")

    # Load dataset
    dataset = load_dataset(dataset_name, split="train_sft")
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

    # Map to extract text from messages
    def extract_text(sample):
        # Extract text from the first message
        text = sample["messages"][0]["content"]
        return {"text": text}

    # Map the dataset to have a 'text' column
    calibration_data = dataset.map(extract_text, remove_columns=dataset.column_names)

    print(f"Loaded {len(calibration_data)} calibration samples")
    return calibration_data


def main():
    parser = argparse.ArgumentParser(description="Apply AWQ smoothing without quantization")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Qwen3 30B A3B model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the smoothed model"
    )
    parser.add_argument(
        "--calibration_dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Dataset to use for calibration (default: HuggingFaceH4/ultrachat_200k)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)"
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Path to custom recipe YAML file (optional)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        help="Model precision (default: bfloat16)"
    )
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        default=True,
        help="Offload cached activations to CPU to save GPU memory (default: True)"
    )
    parser.add_argument(
        "--no-offload",
        dest="offload_to_cpu",
        action="store_false",
        help="Disable offloading to CPU (may cause OOM on large models)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AWQ Smooth-Only: Qwen3 30B A3B")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_path}")
    print(f"Calibration samples: {args.num_samples}")
    print(f"Precision: {args.precision}")
    print(f"Offload to CPU: {args.offload_to_cpu}")
    print("=" * 60)

    # Load calibration data
    print("\n[1/4] Loading calibration data...")
    calibration_data = prepare_calibration_dataset(
        args.calibration_dataset,
        args.num_samples
    )

    # Create AWQ smooth-only modifier
    print("\n[2/4] Setting up AWQ smooth-only modifier...")

    if args.recipe:
        # Use custom recipe from YAML file
        print(f"Using custom recipe: {args.recipe}")
        recipe = args.recipe
    else:
        # Create modifier programmatically
        print("Using programmatic configuration")
        from llmcompressor.modifiers.awq import AWQModifier

        # Determine offload device
        offload_device = torch.device("cpu") if args.offload_to_cpu else None
        if args.offload_to_cpu:
            print("  → Offloading cached activations to CPU to save GPU memory")

        # Create AWQ modifier with smooth_only=True
        recipe = AWQModifier(
            smooth_only=True,  # This is the key parameter!
            ignore=["lm_head"],
            duo_scaling=True,
            offload_device=offload_device,  # Offload to CPU to save GPU memory
        )

    # Apply smoothing
    print("\n[3/4] Applying AWQ smoothing (this may take a while)...")
    print("Note: Model weights will be adjusted but remain in bf16 precision")

    try:
        oneshot(
            model=args.model_path,
            dataset=calibration_data,
            recipe=recipe,
            output_dir=args.output_path,
            max_seq_length=2048,
            num_calibration_samples=args.num_samples,
            precision=args.precision,
        )

        print("\n[4/4] Saving smoothed model...")
        print(f"✓ Smoothed model saved to: {args.output_path}")

        # Verify the output
        print("\n" + "=" * 60)
        print("SUCCESS: AWQ smoothing completed!")
        print("=" * 60)
        print("\nNext steps:")
        print(f"1. Load the model from: {args.output_path}")
        print("2. The model weights are adjusted but still in bf16")
        print("3. You can now apply custom quantization or use as-is")
        print("\nModel statistics:")
        import os
        model_size = sum(
            os.path.getsize(os.path.join(args.output_path, f))
            for f in os.listdir(args.output_path)
            if os.path.isfile(os.path.join(args.output_path, f))
        ) / (1024 ** 3)  # Convert to GB
        print(f"  - Total size: {model_size:.2f} GB")
        print(f"  - Precision: bf16 (no quantization applied)")

    except Exception as e:
        print(f"\n✗ Error during smoothing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
