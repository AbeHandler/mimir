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
    "llama8b.bothbins.blocks.cloze.json"
    "llama8b.excluded.noblocks.cloze.json"
    "llama8b.bothbins.noblocks.cloze.json"
    "llama8b.excluded.blocks.cloze.json"
)

CLOZE_SHARD_SIZE=10  # smaller shards for cloze (default in data_utils.py is 100)
DATA_UTILS=mimir/data_utils.py

restore_data_utils() {
    git checkout -- "$DATA_UTILS"
}
trap restore_data_utils EXIT

for CONFIG_FILENAME in "${CONFIGS[@]}"; do
    echo "Processing config: $CONFIG_FILENAME"

    # Always start each config from a clean data_utils.py.
    restore_data_utils

    if [[ "$CONFIG_FILENAME" == *cloze* ]]; then
        sed -i "s/shard_size=100)/shard_size=$CLOZE_SHARD_SIZE)/g" "$DATA_UTILS"
    fi

    # Per-config shard cap, set by total document count.
    # neighbors_card.20240101_20240115.lite: ~80k pair URLs → 8000 shards
    # neighbors_card.20240130_20240130.lite: ~9k pair URLs  → 900 shards
    if [[ "$CONFIG_FILENAME" == "llama8b.neighbors_card.20240101_20240115.lite.json" ]]; then
        MAX_SHARD=50  # this can go up to 8k eventually but lets start w/ 1K
    elif [[ "$CONFIG_FILENAME" == "llama8b.neighbors_card.20240130_20240130.lite.json" ]]; then
        MAX_SHARD=50   # this can go up to 900
    elif [[ "$CONFIG_FILENAME" == *cloze* ]]; then
        MAX_SHARD=2
    else
        MAX_SHARD=100
    fi

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
        for SHARD_ID in $(seq 1 "$MAX_SHARD"); do
            ./scripts/run_config_llama.sh "$CONFIG_FILENAME" "$SHARD_ID"
            python scripts/merge_shards.py -d csvs
        done
    fi
done


python scripts/merge_shards.py -d csvs
