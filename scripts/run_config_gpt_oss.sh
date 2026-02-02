#!/bin/bash

# Script to run mimir with llama models
# Example: ./scripts/run_config_gpt_oss.sh llamatmp6check.Y0.lite.json
# Example with custom shard count: ./scripts/run_config_gpt_oss.sh llamatmp6check.Y0.lite.json 10

# Check if config file is provided
if [ -z "$1" ]; then
    echo "Error: No config filename provided"
    echo "Usage: $0 <config_filename> [max_shards]"
    echo "Example: $0 llamatmp6check.Y0.lite.json"
    echo "Example: $0 llamatmp6check.Y0.lite.json 10"
    exit 1
fi

CONFIG_FILENAME="$1"
CONFIG_FILE="configs/$CONFIG_FILENAME"

# Get max shards (default to 5 if not provided)
MAX_SHARDS="${2:-5}"

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

OUTPUT_CSV="${CONFIG_BASENAME}.csv"
OUTPUT_DIR="csvs/gptoss/"

# Get the model path from config and set as environment variable
MODEL_PATH=$(python scripts/get_model_path.py $CONFIG_FILE)
export MODEL_PATH

echo "Using model path: $MODEL_PATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Loop over shards
for ((SHARD_ID=0; SHARD_ID<MAX_SHARDS; SHARD_ID++)); do
    echo "Processing shard $SHARD_ID/$((MAX_SHARDS-1))..."

    SHARD_OUTPUT_CSV="${OUTPUT_CSV%.csv}.shard_$SHARD_ID.csv"

    # Check if shard-specific output CSV already exists
    if [ -f "$OUTPUT_DIR$SHARD_OUTPUT_CSV" ]; then
        echo "  Output CSV already exists: $OUTPUT_DIR$SHARD_OUTPUT_CSV"
        echo "  Skipping shard $SHARD_ID"
        continue
    fi

    OUTPUTFILE="$OUTPUT_CSV"

    mamba run -n unslothmimir env SHARD_ID=$SHARD_ID CUDA_VISIBLE_DEVICES=0,2 MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache MODEL_PATH="$MODEL_PATH" python -u run.py --config $CONFIG_FILE

    conda run --live-stream -n analysis python build_output.py --config "$CONFIG_BASENAME"

    mv "$OUTPUT_CSV" "$OUTPUT_DIR$SHARD_OUTPUT_CSV"

    echo "  Completed shard $SHARD_ID"
done

echo "All shards processed!"
