#!/bin/bash

# Script to run mimir with llama models
# Example: ./scripts/run_config_gpt_oss.sh llamatmp6check.Y0.lite.json

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

# Documenting the env here
# conda create --name unslothmimir --clone unsloth
# mamba activate unslothmimir
# pip install simple-parsing && pip install nltk && pip install matplotlib && pip install openai && pip install ai2-olmo


# Extract config name for build_output.py (remove .json suffix)
CONFIG_BASENAME=$(basename "$CONFIG_FILENAME" .json)

mamba run -n unslothmimir env CUDA_VISIBLE_DEVICES=1,2 MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache python -u run.py --config $CONFIG_FILE
