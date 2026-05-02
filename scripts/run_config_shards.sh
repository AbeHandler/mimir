#!/bin/bash
set -e

# NOTE: For cloze configs, activate the `mimirblackwell` conda env first.
#   conda activate mimirblackwell && ./scripts/run_config_shards.sh 2
# run_config.sh detects $CONDA_DEFAULT_ENV == mimirblackwell and uses that env;
# otherwise on non-blackwell hosts it falls back to mimir2, which errors on cloze.

GPU_NUMBER="${1:-0}"  # Default to GPU 0 if not specified

# real line is here ==> for CONFIG_FILENAME in "excluded-docs.blocks.clipped.json" "excluded-docs.noblocks.clipped.json" "bothbins.blocks.lite.json" "bothbins.noblocks.lite.json" "excluded-docs.blocks.lite.json" "excluded-docs.noblocks.lite.json" "confounddataset.blocks.lite.json" "confounddataset.noblocks.lite.json" "matching_neighbors.blocks.lite.json"; do
# # "excluded-docs.blocks.dcpdd.json" "excluded-docs.noblocks.dcpdd.json" "bothbins.noblocks.dcpdd.json" "bothbins.blocks.dcpdd.json" "excluded-docs.blocks.rlhf.lite.json"; do

# go back to cloze later=> "excluded-docs.noblocks.cloze.json" "excluded-docs.blocks.cloze.json" "bothbins.noblocks.cloze.json" "bothbins.blocks.cloze.json"

for CONFIG_FILENAME in "matching_neighbors.blocks.rlhf.lite.json"; do
    echo "Processing config: $CONFIG_FILENAME"

    # Use 20 shards for excluded-docs and bothbins, 60 for matching_neighbors, 36 for others
    # Exception: bisection and cloze configs only use 1 shard due to high cost
    if [[ "$CONFIG_FILENAME" == matching_neighbors.* ]]; then
        MAX_SHARD=59
    elif [[ "$CONFIG_FILENAME" == *"bisection"* ]]; then
        MAX_SHARD=0
    elif [[ "$CONFIG_FILENAME" == *"cloze"* ]]; then
        MAX_SHARD=0
    elif [[ "$CONFIG_FILENAME" == excluded-docs.* || "$CONFIG_FILENAME" == bothbins.* ]]; then
        MAX_SHARD=5
    elif [[ "$CONFIG_FILENAME" == *"matching_neighbors.blocks.rlhf.lite"* ]]; then
        MAX_SHARD=19
    else
        MAX_SHARD=36
    fi

    # For bisection configs, loop over K values; otherwise run once
    if [[ "$CONFIG_FILENAME" == *"bisection"* ]]; then
        for K in 10; do
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
