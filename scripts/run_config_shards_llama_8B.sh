#!/bin/bash
set -e

CONFIGS=(
    "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.bothbins.lite.json"
    "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.excluded.lite.json"
    "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.bothbins.lite.json"
    "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.excluded.lite.json"
)

for CONFIG_FILENAME in "${CONFIGS[@]}"; do
    echo "Processing config: $CONFIG_FILENAME"
    for SHARD_ID in $(seq 1 10); do
        ./scripts/run_config_llama.sh "$CONFIG_FILENAME" "$SHARD_ID"
    done
done
