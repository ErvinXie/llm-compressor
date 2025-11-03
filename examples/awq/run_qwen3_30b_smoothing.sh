#!/bin/bash

# Quick Start Script for Qwen3-30B-A3B AWQ Smoothing
# This script applies AWQ smoothing and prepares the model for GGUF conversion

set -e  # Exit on error

# ============================================================================
# Default Configuration (can be overridden via arguments)
# ============================================================================
MODEL_PATH="${1:-/mnt/data/models/Qwen3-30B-A3B-250425}"
OUTPUT_DIR="${2:-Qwen3-30B-A3B-awq-smoothed-fp16}"
CALIBRATION_SAMPLES="${3:-256}"
MAX_SEQ_LENGTH="${4:-2048}"

# Display usage if --help is passed
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [MODEL_PATH] [OUTPUT_DIR] [CALIBRATION_SAMPLES] [MAX_SEQ_LENGTH]"
    echo ""
    echo "Arguments:"
    echo "  MODEL_PATH            Path to input model (default: /mnt/data/models/Qwen3-30B-A3B-250425)"
    echo "  OUTPUT_DIR            Output directory (default: Qwen3-30B-A3B-awq-smoothed-fp16)"
    echo "  CALIBRATION_SAMPLES   Number of calibration samples (default: 256)"
    echo "  MAX_SEQ_LENGTH        Max sequence length (default: 2048)"
    echo ""
    echo "Examples:"
    echo "  # Use defaults"
    echo "  $0"
    echo ""
    echo "  # Custom model path and output"
    echo "  $0 /path/to/model /path/to/output"
    echo ""
    echo "  # Full customization"
    echo "  $0 /path/to/model /path/to/output 512 2048"
    exit 0
fi

echo "=========================================="
echo "AWQ Smoothing for Qwen3-30B-A3B"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Calibration samples: $CALIBRATION_SAMPLES"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "=========================================="
echo ""

# ============================================================================
# Check if model exists
# ============================================================================
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo "✓ Model found at $MODEL_PATH"
echo ""

# ============================================================================
# Check if output directory already exists
# ============================================================================
if [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️  Warning: Output directory $OUTPUT_DIR already exists"
    read -p "Do you want to overwrite it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -rf "$OUTPUT_DIR"
fi

# ============================================================================
# Run AWQ Smoothing
# ============================================================================
echo "Starting AWQ smoothing..."
echo "This will take 30-60 minutes depending on your GPU"
echo ""

python qwen3_30b_a3b_smoothing_for_gguf.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_calibration_samples "$CALIBRATION_SAMPLES" \
    --max_seq_length "$MAX_SEQ_LENGTH"

# ============================================================================
# Check if successful
# ============================================================================
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "✓ AWQ Smoothing Completed Successfully!"
    echo "=========================================="
    echo ""
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Convert to GGUF Q4K:"
    echo "   python convert-hf-to-gguf.py $OUTPUT_DIR --outtype q4_k_m"
    echo ""
    echo "2. Run inference with llama.cpp:"
    echo "   ./llama-cli -m $OUTPUT_DIR.q4_k_m.gguf -p 'Hello!'"
    echo ""
else
    echo ""
    echo "❌ Error: Output directory not created"
    echo "Something went wrong during AWQ smoothing"
    exit 1
fi
