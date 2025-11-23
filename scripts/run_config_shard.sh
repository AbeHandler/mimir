#!/bin/bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <SHARD_ID> <config_filename>"
    echo "Example: $0 0 confounddataset.blocks.lite.json"
    exit 1
fi

CONFIG_FILENAME="$2"
CONFIG_BASENAME=$(basename "$CONFIG_FILENAME" .json)
OUTPUT_CSV="${CONFIG_BASENAME}.csv"

export SHARD_ID="$1"
echo "Running with SHARD_ID=$SHARD_ID"

./scripts/run_config.sh "$CONFIG_FILENAME"

# Check that output file exists and was created within the last minute
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "ERROR: Output file does not exist: $OUTPUT_CSV"
    exit 1
fi

FILE_AGE=$(( $(date +%s) - $(stat -f %m "$OUTPUT_CSV") ))
if [ "$FILE_AGE" -gt 60 ]; then
    echo "ERROR: Output file is older than 60 seconds (age: ${FILE_AGE}s): $OUTPUT_CSV"
    exit 1
fi

# Rename to include SHARD_ID
OUTPUT_CSV_SHARD="${CONFIG_BASENAME}.shard_${SHARD_ID}.csv"
mv "$OUTPUT_CSV" "$OUTPUT_CSV_SHARD"

echo "Output: $OUTPUT_CSV_SHARD"
