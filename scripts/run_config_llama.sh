#!/bin/bash
set -e

# =============================================================================
# run_config_llama.sh - Run MIMIR analysis pipeline for Llama models
# =============================================================================
#
# Description:
#   This script runs the MIMIR membership inference analysis pipeline for
#   Llama models which require multiple GPUs. The code expects:
#   - device_map="balanced_low_0" spreads the model across GPUs 0 and 1
#   - self.device is auto-detected from model's first layer
#   - Therefore we expose GPUs 0 and 1 where:
#     * GPUs 0-1: Model loading via device_map
#     * GPU 1: Inference operations (typically where first layer resides)
#
# Usage:
#   ./scripts/run_config_llama.sh <config_filename> [shard_id]
#
# Arguments:
#   config_filename - Name of config file in configs/ directory
#   shard_id        - (Optional) Shard ID to pass as SHARD_ID env var to run.py
#
# Example:
#   ./scripts/run_config_llama.sh Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.lite.json
#   ./scripts/run_config_llama.sh Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.lite.json 3
#
# Notes:
#   - Config file must exist in the configs/ directory
#   - Sets CUDA_VISIBLE_DEVICES=0,1 to expose two GPUs
#   - Requires conda environments: 'mimir' and 'analysis'
#   - Blackwell GPU (GPU 2) not used due to PyTorch compatibility (sm_120)
# =============================================================================

if [ -z "$1" ]; then
    echo "Error: Config filename argument required"
    echo "Usage: $0 <config_filename>"
    echo "Example: $0 Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.lite.json"
    exit 1
fi

CONFIG_FILENAME="$1"
CONFIG_FILE="configs/$CONFIG_FILENAME"
CONFIG_BASENAME=$(basename "$CONFIG_FILENAME" .json)

# Export SHARD_ID if provided (run.py reads it from env)
if [ -n "$2" ]; then
    export SHARD_ID="$2"
    echo "Running with SHARD_ID=$SHARD_ID"
fi

OUTPUT_CSV="${CONFIG_BASENAME}.csv"
OUTPUT_CSV_SHARD="${CONFIG_BASENAME}.shard_${SHARD_ID}.csv"
OUTPUT_DIR="csvs"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file does not exist: $CONFIG_FILE"
    exit 1
fi

# If this is a bisection config, BISECTION_QUERIES_PER_TOKEN must be set
if [[ "$CONFIG_FILENAME" == *"bisection"* ]]; then
    if [ -z "$BISECTION_QUERIES_PER_TOKEN" ]; then
        echo "ERROR: BISECTION_QUERIES_PER_TOKEN not set for bisection config: $CONFIG_FILENAME"
        exit 1
    fi
    echo "Running bisection with K=$BISECTION_QUERIES_PER_TOKEN"
    OUTPUT_CSV="${CONFIG_BASENAME}.k${BISECTION_QUERIES_PER_TOKEN}.csv"
    OUTPUT_CSV_SHARD="${CONFIG_BASENAME}.k${BISECTION_QUERIES_PER_TOKEN}.shard_${SHARD_ID}.csv"
fi

# Skip shard 8 for bothbins
if [[ "$CONFIG_FILENAME" == *bothbins* && "$SHARD_ID" == "9" ]]; then
    echo "Skipping shard 9 for bothbins: $CONFIG_FILENAME"
    exit 0
fi

# Early exit if shard output already exists
if [ -n "$SHARD_ID" ] && [ -f "$OUTPUT_DIR/$OUTPUT_CSV_SHARD" ]; then
    echo "Output shard already exists, skipping: $OUTPUT_DIR/$OUTPUT_CSV_SHARD"
    exit 0
fi

# Validate config file
echo "Validating config file..."
python config_validator.py "$CONFIG_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Config validation failed"
    exit 1
fi

echo "Using GPUs 0,1 for Llama model (device_map uses 0-1, inference on cuda:1)"

# Pass BISECTION_QUERIES_PER_TOKEN to both conda run commands if set
if [ -n "$BISECTION_QUERIES_PER_TOKEN" ]; then
    export CUDA_VISIBLE_DEVICES=0,1 && BISECTION_QUERIES_PER_TOKEN=$BISECTION_QUERIES_PER_TOKEN MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir2 python run.py --config "$CONFIG_FILE"
    time BISECTION_QUERIES_PER_TOKEN=$BISECTION_QUERIES_PER_TOKEN conda run --live-stream -n analysis python build_output.py --config "$CONFIG_BASENAME"
else
    export CUDA_VISIBLE_DEVICES=0,1 && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir2 python run.py --config "$CONFIG_FILE"
    time conda run --live-stream -n analysis python build_output.py --config "$CONFIG_BASENAME"
fi

# Check output file exists and is fresh
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "❌ Error: Expected output file not found: $OUTPUT_CSV"
    exit 1
fi

FILE_AGE=$(( $(date +%s) - $(stat -c %Y "$OUTPUT_CSV") ))
if [ "$FILE_AGE" -gt 60 ]; then
    echo "ERROR: Output file is older than 60 seconds (age: ${FILE_AGE}s): $OUTPUT_CSV"
    exit 1
fi

# Rename to shard filename if SHARD_ID set, then move to output dir
mkdir -p "$OUTPUT_DIR"
if [ -n "$SHARD_ID" ]; then
    mv "$OUTPUT_CSV" "$OUTPUT_DIR/$OUTPUT_CSV_SHARD"
    echo "✅ Output: $OUTPUT_DIR/$OUTPUT_CSV_SHARD"
else
    mv "$OUTPUT_CSV" "$OUTPUT_DIR/$OUTPUT_CSV"
    echo "✅ Output: $OUTPUT_DIR/$OUTPUT_CSV"
fi
