#! /usr/bin/env bash
# Usage: bash scripts/R2/reviewer2/generate_delta_0_plots.sh [--max-steps 5000] [--seed 1234]

set -e  # Exit on error

MAX_STEPS="${MAX_STEPS:-5000}"
SEED="${SEED:-1234}"

echo "Using max_steps=$MAX_STEPS, seed=$SEED"

echo "Step 1: Generating ATU data..."
python scripts/R2/reviewer2/check_for_delta_0_appendix.py --max-steps "$MAX_STEPS" --seed "$SEED"

echo ""
echo "Step 2: Generating ATT data..."
python scripts/R2/reviewer2/check_for_delta_1_appendix_seed_names.py --max-steps "$MAX_STEPS" --seed "$SEED"

echo ""
echo "Step 3: Creating ECDF plots for each method..."

# Loop through common MIA methods
for method in loss min_k zlib; do
    echo "  Generating plot for method: $method"
    Rscript scripts/R2/reviewer2/check_for_delta_0_appendix.R $method "$MAX_STEPS" "$SEED"
done

echo ""
echo "Done! Plots saved to figures/delta_0_appendix_ecdf_*.png"
