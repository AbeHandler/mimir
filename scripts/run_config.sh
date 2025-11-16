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
#
# Usage:
#   ./scripts/run_config.sh <config_filename>
#
# Arguments:
#   config_filename - Name of config file in configs/ directory (e.g., minhashblocksample_blocks.zlib.json)
#
# Example:
#   ./scripts/run_config.sh minhashblocksample_blocks.zlib.json
#
# Notes:
#   - Config file must exist in the configs/ directory
#   - Uses GPU 0 (CUDA_VISIBLE_DEVICES=0)
#   - Requires conda environments: 'mimir' and 'analysis'
# =============================================================================

if [ -z "$1" ]; then
    echo "Error: Config filename argument required"
    echo "Usage: $0 <config_filename>"
    echo "Example: $0 minhashblocksample_blocks.zlib.json"
    exit 1
fi

CONFIG_FILENAME="$1"
CONFIG_FILE="configs/$CONFIG_FILENAME"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file does not exist: $CONFIG_FILE"
    exit 1
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

export CUDA_VISIBLE_DEVICES=0 && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config "$CONFIG_FILE"

time conda run --live-stream -n analysis python build_output.py --config "$CONFIG_BASENAME"