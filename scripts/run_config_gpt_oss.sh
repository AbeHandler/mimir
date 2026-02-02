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

# Check if output CSV already exists
OUTPUT_CSV="${CONFIG_BASENAME}.csv"
if [ -f "$OUTPUT_CSV" ]; then
    echo "Output CSV already exists: $OUTPUT_CSV"
    echo "Skipping run. Delete the CSV file to re-run."
    exit 0
fi

# Get the model path from config and set as environment variable
MODEL_PATH=$(python scripts/get_model_path.py $CONFIG_FILE)
export MODEL_PATH

echo "Using model path: $MODEL_PATH"

mamba run -n unslothmimir env SHARD_ID=0 CUDA_VISIBLE_DEVICES=0,2 MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache MODEL_PATH="$MODEL_PATH" python -u run.py --config $CONFIG_FILE

conda run --live-stream -n analysis python build_output.py --config "$CONFIG_BASENAME"