#! /usr/bin/env bash

set -e  # Exit on error

echo "Step 1: Generating ATU data..."
python scripts/R2/reviewer2/check_for_delta_0_appendix.py

echo ""
echo "Step 2: Generating ATT data..."
python scripts/R2/reviewer2/check_for_delta_1_appendix.py

echo ""
echo "Step 3: Creating ECDF plots for each method..."

# Loop through common MIA methods
for method in loss min_k zlib; do
    echo "  Generating plot for method: $method"
    Rscript scripts/R2/reviewer2/check_for_delta_0_appendix.R $method
done

echo ""
echo "Done! Plots saved to figures/delta_0_appendix_ecdf_*.png"
