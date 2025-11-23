#!/bin/bash
set -e

for SHARD_ID in $(seq 0 36); do
    ./scripts/run_config_shard.sh "$SHARD_ID"
done
