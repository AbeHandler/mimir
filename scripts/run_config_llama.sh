#!/bin/bash

# Script to run mimir with llama models
# Usage: ./scripts/run_config_llama.sh <config_filename>
# Example: ./scripts/run_config_llama.sh llamatmp6check.Y0.lite.json

# Check if config file is provided
if [ -z "$1" ]; then
    echo "Error: No config filename provided"
    echo "Usage: $0 <config_filename>"
    echo "Example: $0 llamatmp6check.Y0.lite.json"
    exit 1
fi

CONFIG_FILENAME="$1"
CONFIG_FILE="configs/$CONFIG_FILENAME"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract config name for build_output.py (remove .json suffix)
CONFIG_BASENAME=$(basename "$CONFIG_FILENAME" .json)

# Set environment variables and run with GPUs 0 and 1 visible
CUDA_VISIBLE_DEVICES=0,1 MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config "$CONFIG_FILE"

# Process and flatten the results into CSV format
conda run --live-stream -n analysis python build_output.py --config "$CONFIG_BASENAME"