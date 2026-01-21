#!/bin/bash

# Script to run mimir with llama models
# Usage: ./scripts/run_config_llama.sh <config_file>
# Example: ./scripts/run_config_llama.sh configs/llamatmp6check.Y0.lite.json

# Check if config file is provided
if [ -z "$1" ]; then
    echo "Error: No config file provided"
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs/llamatmp6check.Y0.lite.json"
    exit 1
fi

CONFIG_FILE="$1"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Set environment variables
export MIMIR_DATA_SOURCE=mimirdata
export MIMIR_CACHE_PATH=mimrcache

# Run with GPUs 0 and 1 visible
CUDA_VISIBLE_DEVICES=0,1 conda run python run.py --config "$CONFIG_FILE"
