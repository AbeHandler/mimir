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
#   ./scripts/run_config_llama.sh <config_filename>
#
# Arguments:
#   config_filename - Name of config file in configs/ directory
#
# Example:
#   ./scripts/run_config_llama.sh Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.lite.json
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
fi

# Validate config file
echo "Validating config file..."
python config_validator.py "$CONFIG_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Config validation failed"
    exit 1
fi

# Extract config name for build_output.py (remove .json suffix)
CONFIG_BASENAME=$(basename "$CONFIG_FILENAME" .json)

echo "Using GPUs 0,1 for Llama model (device_map uses 0-1, inference on cuda:1)"

# Pass BISECTION_QUERIES_PER_TOKEN to both conda run commands if set
if [ -n "$BISECTION_QUERIES_PER_TOKEN" ]; then
    export CUDA_VISIBLE_DEVICES=0,1 && BISECTION_QUERIES_PER_TOKEN=$BISECTION_QUERIES_PER_TOKEN MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config "$CONFIG_FILE"
    time BISECTION_QUERIES_PER_TOKEN=$BISECTION_QUERIES_PER_TOKEN conda run --live-stream -n analysis python build_output.py --config "$CONFIG_BASENAME"
else
    export CUDA_VISIBLE_DEVICES=0,1 && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config "$CONFIG_FILE"
    time conda run --live-stream -n analysis python build_output.py --config "$CONFIG_BASENAME"
fi
