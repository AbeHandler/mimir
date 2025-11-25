#!/bin/bash
set -e

# =============================================================================
# postprocess_ngrams.sh - Postprocess n-grams from blocks/noblocks model pairs
# =============================================================================
#
# Description:
#   This script postprocesses results from paired blocks/noblocks experiments:
#   1. Extracts n-grams with per-token log probabilities from both models
#   2. Verifies n-grams match between models
#   3. Merges them into a single JSONL file with both models' log probs
#
# Usage:
#   ./scripts/postprocess_ngrams.sh <config1> <config2>
#
# Arguments:
#   config1 - First config file (e.g., confounddatasetxpress.blocks.lite.json)
#   config2 - Second config file (e.g., confounddatasetxpress.noblocks.lite.json)
#
# Example:
#   ./scripts/postprocess_ngrams.sh \
#     confounddatasetxpress.blocks.lite.json \
#     confounddatasetxpress.noblocks.lite.json
#
# Requirements:
#   - Both config files must exist in configs/ directory
#   - Both experiments must have been run already (loss_results.json exists)
#   - Requires conda environment: 'mimir'
#
# Output:
#   - Creates ngrams_analysis.json for each model
#   - Creates merged_ngrams.jsonl in the first config's results directory
# =============================================================================

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Two config filenames required"
    echo "Usage: $0 <config1> <config2>"
    echo "Example: $0 confounddatasetxpress.blocks.lite.json confounddatasetxpress.noblocks.lite.json"
    exit 1
fi

CONFIG1_FILENAME="$1"
CONFIG2_FILENAME="$2"
CONFIG1="configs/$CONFIG1_FILENAME"
CONFIG2="configs/$CONFIG2_FILENAME"

# Check if both config files exist
if [ ! -f "$CONFIG1" ]; then
    echo "Error: Config file does not exist: $CONFIG1"
    exit 1
fi

if [ ! -f "$CONFIG2" ]; then
    echo "Error: Config file does not exist: $CONFIG2"
    exit 1
fi

echo "Processing n-gram pair:"
echo "  Config 1: $CONFIG1"
echo "  Config 2: $CONFIG2"
echo ""

# Extract experiment details from configs to find results directories
parse_config() {
    local config_file=$1
    python -c "
import json
import os
with open('$config_file', 'r') as f:
    config = json.load(f)

# Reconstruct the results path following the pattern in run.py (lines 487-498)
exp_name = config['experiment_name']
base_model_slug = config['base_model'].replace('/', '_')

# Build sf path
sf = os.path.join(exp_name, base_model_slug)

# Check which branch to use
if config.get('specific_source') and not config.get('ourdataset'):
    # This would need sourcename_process logic, but unlikely to be used
    raise ValueError('specific_source not supported in postprocessing script')
elif config.get('ourdataset'):
    sf = os.path.join(sf, config['ourdataset'])
else:
    raise ValueError('Neither specific_source nor ourdataset found in config')

# Get tmp_results path (default is 'tmp_results')
tmp_results = config.get('env_config', {}).get('tmp_results', 'tmp_results')
results_path = os.path.join(tmp_results, sf)

model_name = config['base_model']

print(f\"{results_path}|{model_name}\")
"
}

# Get results paths and model names
MODEL1_INFO=$(parse_config "$CONFIG1")
MODEL2_INFO=$(parse_config "$CONFIG2")

MODEL1_RESULTS_DIR=$(echo "$MODEL1_INFO" | cut -d'|' -f1)
MODEL1_NAME=$(echo "$MODEL1_INFO" | cut -d'|' -f2)

MODEL2_RESULTS_DIR=$(echo "$MODEL2_INFO" | cut -d'|' -f1)
MODEL2_NAME=$(echo "$MODEL2_INFO" | cut -d'|' -f2)

echo "Results directories:"
echo "  Model 1: $MODEL1_RESULTS_DIR"
echo "  Model 2: $MODEL2_RESULTS_DIR"
echo ""

# Check if results exist
if [ ! -f "$MODEL1_RESULTS_DIR/loss_results.json" ]; then
    echo "Error: Model 1 results not found: $MODEL1_RESULTS_DIR/loss_results.json"
    echo "Run the experiment first using: ./scripts/run_config.sh $CONFIG1_FILENAME"
    exit 1
fi

if [ ! -f "$MODEL2_RESULTS_DIR/loss_results.json" ]; then
    echo "Error: Model 2 results not found: $MODEL2_RESULTS_DIR/loss_results.json"
    echo "Run the experiment first using: ./scripts/run_config.sh $CONFIG2_FILENAME"
    exit 1
fi

# Determine model names for output (blocks/noblocks or use model names)
MODEL1_SHORTNAME="model1"
MODEL2_SHORTNAME="model2"

if [[ "$CONFIG1_FILENAME" == *"blocks"* ]] && [[ "$CONFIG2_FILENAME" == *"noblocks"* ]]; then
    MODEL1_SHORTNAME="blocksbin"
    MODEL2_SHORTNAME="noblocksbin"
elif [[ "$CONFIG1_FILENAME" == *"noblocks"* ]] && [[ "$CONFIG2_FILENAME" == *"blocks"* ]]; then
    MODEL1_SHORTNAME="noblocksbin"
    MODEL2_SHORTNAME="blocksbin"
fi

echo "Model short names:"
echo "  Model 1: $MODEL1_SHORTNAME"
echo "  Model 2: $MODEL2_SHORTNAME"
echo ""

# Step 1: Extract n-grams from model 1
echo "Step 1/3: Extracting n-grams from model 1..."
conda run -n mimir python scripts/ngram_probs_analyzer.py \
    "$MODEL1_RESULTS_DIR" \
    --model "$MODEL1_NAME" \
    --n 13

# Step 2: Extract n-grams from model 2
echo ""
echo "Step 2/3: Extracting n-grams from model 2..."
conda run -n mimir python scripts/ngram_probs_analyzer.py \
    "$MODEL2_RESULTS_DIR" \
    --model "$MODEL2_NAME" \
    --n 13

# Step 3: Merge the analyses
echo ""
echo "Step 3/3: Merging n-gram analyses..."
OUTPUT_FILE="$MODEL1_RESULTS_DIR/merged_ngrams.jsonl"

conda run -n mimir python scripts/merge_ngram_analyses.py \
    "$MODEL1_RESULTS_DIR/ngrams_analysis.json" \
    "$MODEL2_RESULTS_DIR/ngrams_analysis.json" \
    --model1-name "$MODEL1_SHORTNAME" \
    --model2-name "$MODEL2_SHORTNAME" \
    --output "$OUTPUT_FILE"

echo ""
echo "============================================"
echo "Postprocessing complete!"
echo "============================================"
echo "Merged output: $OUTPUT_FILE"
echo ""
