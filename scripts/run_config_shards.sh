#!/bin/bash
set -e

for CONFIG_FILENAME in "bothbins.blocks.lite.json" "bothbins.noblocks.lite.json" "excluded-docs.blocks.lite.json" "excluded-docs.noblocks.lite.json" "confounddataset.blocks.lite.json" "confounddataset.noblocks.lite.json" "matching_neighbors.blocks.lite.json"; do
    echo "Processing config: $CONFIG_FILENAME"

    # Use 20 shards for excluded-docs and bothbins, 60 for matching_neighbors, 36 for others
    if [[ "$CONFIG_FILENAME" == matching_neighbors.* ]]; then
        MAX_SHARD=59
    elif [[ "$CONFIG_FILENAME" == excluded-docs.* || "$CONFIG_FILENAME" == bothbins.* ]]; then
        MAX_SHARD=19
    else
        MAX_SHARD=36
    fi

    for SHARD_ID in $(seq 0 $MAX_SHARD); do
        ./scripts/run_config_shard.sh "$SHARD_ID" "$CONFIG_FILENAME"
    done
done


python scripts/merge_shards.py -d csvs/confounddataset