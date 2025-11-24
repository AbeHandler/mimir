#!/bin/bash
set -e

for CONFIG_FILENAME in "confounddataset.blocks.lite.json" "confounddataset.noblocks.lite.json"; do
    echo "Processing config: $CONFIG_FILENAME"
    for SHARD_ID in $(seq 0 36); do
        ./scripts/run_config_shard.sh "$SHARD_ID" "$CONFIG_FILENAME"
    done
done
