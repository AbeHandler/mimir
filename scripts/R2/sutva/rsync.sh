#!/bin/bash

set -e

ME=abe@172.20.98.203
ssh $ME "find /home/abe/mimir/tmp_results/ -name '*loss_results*' | grep 05" > files_to_sync.txt

ssh $ME "find /home/abe/mimir/tmp_results/ -name '*config*' | grep 05" >> files_to_sync.txt

rsync -avP --files-from=files_to_sync.txt $ME:/ ./server_files/

# Process all loss_results.json files to extract n-grams
echo "Processing all loss_results.json files..."
find ./server_files -name 'loss_results.json' | while read -r json_file; do
    #echo "Processing: $json_file"
    python extract_ngrams_from_json.py --json "$json_file" --n 7
done

echo "Done processing all files!"


python scripts/R2/sutva/correlations_7gm_counts_and_loss.py