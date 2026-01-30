#!/bin/bash
# Usage: ./scripts/run_single_publishers.sh [gpu_number]
# Example: ./scripts/run_single_publishers.sh 1
#
# To regenerate configs/single_publishers.txt from HuggingFace:
#   python scripts/fetch_scm_publishers.py

set -e

GPU_NUMBER="${1:-0}"  # Default to GPU 0 if not specified

echo "Using GPU: ${GPU_NUMBER}"

# Read publishers from config file and shuffle for random sampling
mapfile -t PUBLISHERS < <(shuf configs/single_publishers.txt)

for PUBLISHER in "${PUBLISHERS[@]}"; do
    echo "Processing publisher: ${PUBLISHER}"

    # Check if output files already exist
    if [[ -f "csvs/${PUBLISHER}_scm.blocks.lite.csv" && -f "csvs/${PUBLISHER}_scm.noblocks.lite.csv" ]]; then
        echo "Skipping ${PUBLISHER}: output files already exist"
        echo "---"
        continue
    fi

    python scripts/R2/scm/create_single_publisher_config.py -publisher ${PUBLISHER}

    ./scripts/run_config.sh ${PUBLISHER}_scm.blocks.lite.json ${GPU_NUMBER}
    ./scripts/run_config.sh ${PUBLISHER}_scm.noblocks.lite.json ${GPU_NUMBER}

    mv *${PUBLISHER}* csvs/

    echo "Completed: ${PUBLISHER}"
    echo "---"
done