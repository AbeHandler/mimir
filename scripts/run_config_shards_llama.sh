#!/bin/bash
set -e

CONFIGS=(
    "llama8b.neighbors_card.20240101_20240115.lite.json"
    "llama8b.neighbors_card.20240130_20240130.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.bothbins.bisection.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.bothbins.clipped.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.bothbins.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.excluded.bisection.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.excluded.clipped.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.excluded.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.bothbins.bisection.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.bothbins.clipped.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.bothbins.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.excluded.bisection.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.excluded.clipped.lite.json"
    #"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.excluded.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.bothbins.bisection.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.bothbins.clipped.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.bothbins.dcpdd.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.bisection.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.clipped.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.dcpdd.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.bothbins.bisection.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.bothbins.clipped.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.bothbins.dcpdd.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.excluded.bisection.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.excluded.clipped.lite.json"
    #"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.excluded.dcpdd.lite.json"
    "llama8b.neighbors_card.20240101_20240115.cloze.json"
    "llama8b.neighbors_card.20240130_20240130.cloze.json"
)

for CONFIG_FILENAME in "${CONFIGS[@]}"; do
    echo "Processing config: $CONFIG_FILENAME"

    if [[ "$CONFIG_FILENAME" == *"bisection"* ]]; then
        for K in 10; do
            echo "  Running with K=$K"
            export BISECTION_QUERIES_PER_TOKEN=$K
            for SHARD_ID in $(seq 1 3); do
                ./scripts/run_config_llama.sh "$CONFIG_FILENAME" "$SHARD_ID"
            done
        done
        unset BISECTION_QUERIES_PER_TOKEN
    else
        for SHARD_ID in $(seq 1 9); do
            ./scripts/run_config_llama.sh "$CONFIG_FILENAME" "$SHARD_ID"
        done
    fi
done


python scripts/merge_shards.py -d csvs
