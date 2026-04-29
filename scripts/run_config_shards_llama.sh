#!/bin/bash
set -e

CONFIGS=(
    "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.bothbins.lite.json"
    "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.excluded.lite.json"
    "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.bothbins.lite.json"
    "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.excluded.lite.json"
    "Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.bothbins.dcpdd.lite.json"
    "Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.dcpdd.lite.json"
    "Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.bothbins.dcpdd.lite.json"
    "Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.excluded.dcpdd.lite.json"
)

for CONFIG_FILENAME in "${CONFIGS[@]}"; do
    echo "Processing config: $CONFIG_FILENAME"
    for SHARD_ID in $(seq 1 9); do
        ./scripts/run_config_llama.sh "$CONFIG_FILENAME" "$SHARD_ID"
    done
done


python scripts/merge_shards.py -d csvs