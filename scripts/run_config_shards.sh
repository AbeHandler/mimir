#!/bin/bash
set -e

GPU_NUMBER="${1:-0}"  # Default to GPU 0 if not specified

# real line is here ==> for CONFIG_FILENAME in "excluded-docs.blocks.clipped.json" "excluded-docs.noblocks.clipped.json" "bothbins.blocks.lite.json" "bothbins.noblocks.lite.json" "excluded-docs.blocks.lite.json" "excluded-docs.noblocks.lite.json" "confounddataset.blocks.lite.json" "confounddataset.noblocks.lite.json" "matching_neighbors.blocks.lite.json"; do

# express line below for now 1/7/2025
for CONFIG_FILENAME in "excluded-docs.blocks.dcpdd.json" "excluded-docs.noblocks.dcpdd.json"; do
    echo "Processing config: $CONFIG_FILENAME"

    # Use 20 shards for excluded-docs and bothbins, 60 for matching_neighbors, 36 for others
    # Exception: bisection configs only use 1 shard due to high cost
    if [[ "$CONFIG_FILENAME" == matching_neighbors.* ]]; then
        MAX_SHARD=59
    elif [[ "$CONFIG_FILENAME" == *"bisection"* ]]; then
        MAX_SHARD=0
    elif [[ "$CONFIG_FILENAME" == excluded-docs.* || "$CONFIG_FILENAME" == bothbins.* ]]; then
        MAX_SHARD=19
    else
        MAX_SHARD=36
    fi

    # For bisection configs, loop over K values; otherwise run once
    if [[ "$CONFIG_FILENAME" == *"bisection"* ]]; then
        for K in 5 10 15 50; do
            echo "  Running with K=$K"
            export BISECTION_QUERIES_PER_TOKEN=$K
            for SHARD_ID in $(seq 0 $MAX_SHARD); do
                ./scripts/run_config_shard.sh "$SHARD_ID" "$CONFIG_FILENAME" "$GPU_NUMBER"
            done
        done
        unset BISECTION_QUERIES_PER_TOKEN
    else
        for SHARD_ID in $(seq 0 $MAX_SHARD); do
            ./scripts/run_config_shard.sh "$SHARD_ID" "$CONFIG_FILENAME" "$GPU_NUMBER"
        done
    fi
done


python scripts/merge_shards.py -d csvs/confounddataset