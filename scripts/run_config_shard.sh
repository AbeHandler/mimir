#!/bin/bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <SHARD_ID> <CONFIG_FILENAME>"
    echo "Example: ./scripts/run_config_shard.sh 0 confounddataset.blocks.lite.json"
    exit 1
fi

CONFIG_FILENAME="$2"
CONFIG_BASENAME=$(basename "$CONFIG_FILENAME" .json)
OUTPUT_CSV="${CONFIG_BASENAME}.csv"
OUTPUT_CSV_SHARD="${CONFIG_BASENAME}.shard_${1}.csv"
OUTPUT_DIR="csvs/confounddataset"

# Early exit if output shard already exists
if [ -f "$OUTPUT_DIR/$OUTPUT_CSV_SHARD" ]; then
    echo "Output shard already exists, skipping: $OUTPUT_DIR/$OUTPUT_CSV_SHARD"
    exit 0
fi

export SHARD_ID="$1"
echo "Running with SHARD_ID=$SHARD_ID"

./scripts/run_config.sh "$CONFIG_FILENAME"

# Check that output file exists and was created within the last minute
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "ERROR: Output file does not exist: $OUTPUT_CSV"
    exit 1
fi

FILE_AGE=$(( $(date +%s) - $(stat -c %Y "$OUTPUT_CSV") ))
if [ "$FILE_AGE" -gt 60 ]; then
    echo "ERROR: Output file is older than 60 seconds (age: ${FILE_AGE}s): $OUTPUT_CSV"
    exit 1
fi

# Rename to include SHARD_ID and move to output directory
mkdir -p "$OUTPUT_DIR"
mv "$OUTPUT_CSV" "$OUTPUT_DIR/$OUTPUT_CSV_SHARD"

echo "Output: $OUTPUT_DIR/$OUTPUT_CSV_SHARD"
