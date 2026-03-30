#!/bin/bash
set -e

# =============================================================================
# run_config.sh - Run MIMIR analysis pipeline with a specified config
# =============================================================================
#
# Description:
#   This script runs the complete MIMIR membership inference analysis pipeline:
#   1. Runs the main MIMIR analysis (run.py) using the specified config file
#   2. Processes and flattens the results into CSV format (build_output.py)
#   3. Extracts n-grams with per-token log probabilities (ngram_probs_analyzer.py)
#   4. Merges n-gram analyses from multiple models (merge_ngram_analyses.py)
#
# Usage:
#   ./scripts/run_config.sh <config_filename> [gpu_number]
#
# Arguments:
#   config_filename - Name of config file in configs/ directory (e.g., minhashblocksample_blocks.zlib.json)
#   gpu_number      - Optional GPU number to use (default: 0)
#
# Example:
#   ./scripts/run_config.sh minhashblocksample_blocks.zlib.json
#   ./scripts/run_config.sh minhashblocksample_blocks.zlib.json 1
#
# Notes:
#   - Config file must exist in the configs/ directory
#   - GPU selection via CUDA_VISIBLE_DEVICES
#   - Requires conda environments: 'mimir' and 'analysis'
# =============================================================================

if [ -z "$1" ]; then
    echo "Error: Config filename argument required"
    echo "Usage: $0 <config_filename> [gpu_number]"
    echo "Example: $0 minhashblocksample_blocks.zlib.json"
    echo "Example: $0 minhashblocksample_blocks.zlib.json 1"
    exit 1
fi

CONFIG_FILENAME="$1"
GPU_NUMBER="${2:-0}"  # Default to GPU 0 if not specified
CONFIG_FILE="configs/$CONFIG_FILENAME"

# Auto-detect conda env: if already running in mimirblackwell, use that
if [[ "$CONDA_DEFAULT_ENV" == "mimirblackwell" ]]; then
    MIMIR_ENV="mimirblackwell"
elif [[ "$(hostname)" == *blackwell* ]]; then
    MIMIR_ENV="mimir"
else
    MIMIR_ENV="mimir2"
fi
echo "Using conda env: $MIMIR_ENV"

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

echo "Using GPU $GPU_NUMBER"

# Pass BISECTION_QUERIES_PER_TOKEN to both conda run commands if set
if [ -n "$BISECTION_QUERIES_PER_TOKEN" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_NUMBER && BISECTION_QUERIES_PER_TOKEN=$BISECTION_QUERIES_PER_TOKEN MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n "$MIMIR_ENV" python run.py --config "$CONFIG_FILE"
    time BISECTION_QUERIES_PER_TOKEN=$BISECTION_QUERIES_PER_TOKEN conda run --live-stream -n analysis python build_output.py --config "$CONFIG_BASENAME"
else
    export CUDA_VISIBLE_DEVICES=$GPU_NUMBER && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n "$MIMIR_ENV" python run.py --config "$CONFIG_FILE"
    time conda run --live-stream -n analysis python build_output.py --config "$CONFIG_BASENAME"
fi
