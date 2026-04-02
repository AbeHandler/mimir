#! /usr/bin/env bash
# Usage: SEED=1 bash scripts/R2/reviewer2/final_results_per_seed.sh
# Usage: bash scripts/R2/reviewer2/final_results_per_seed.sh -all

set -e  # Exit on error

MAX_STEPS="${MAX_STEPS:-5000}"
SEED="${SEED:-1234}"
STEPS_K="$((MAX_STEPS / 1000))k"

if [ "${1:-}" = "-all" ]; then
    for seed in 1 2 3; do
        SEED="$seed" bash scripts/R2/reviewer2/final_results_per_seed.sh
    done

    head -1 "/tmp/att_appendix_steps${STEPS_K}_seed1.csv" > /tmp/att_appendix_all.csv
    tail -n +2 -q /tmp/att_appendix_steps${STEPS_K}_seed[123].csv >> /tmp/att_appendix_all.csv

    head -1 "/tmp/atu_appendix_steps${STEPS_K}_seed1.csv" > /tmp/atu_appendix_all.csv
    tail -n +2 -q /tmp/atu_appendix_steps${STEPS_K}_seed[123].csv >> /tmp/atu_appendix_all.csv

    echo "Merged all seeds:"
    echo "  /tmp/att_appendix_steps${STEPS_K}_seed1.csv: $(wc -l < /tmp/att_appendix_steps${STEPS_K}_seed1.csv) lines"
    echo "  /tmp/att_appendix_all.csv: $(wc -l < /tmp/att_appendix_all.csv) lines (expect 3n-2)"
    echo "  /tmp/atu_appendix_all.csv: $(wc -l < /tmp/atu_appendix_all.csv) lines (expect 3n-2)"

    for method in loss min_k zlib; do
        echo "  Generating merged plot for method: $method"
        Rscript scripts/R2/reviewer2/check_for_delta_0_appendix.R "$method" "$MAX_STEPS" "all"
    done

    echo "Done! Merged plots saved to figures/delta_0_appendix_ecdf_*_all.png"
    exit 0
fi

echo "Using max_steps=$MAX_STEPS, seed=$SEED"

echo "Step 1: Generating ATU data..."
python scripts/R2/reviewer2/check_for_delta_0_appendix.py --max-steps "$MAX_STEPS" --seed "$SEED"

echo ""
echo "Step 2: Generating ATT data..."
python scripts/R2/reviewer2/check_for_delta_1_appendix_seed_names.py --max-steps "$MAX_STEPS" --seed "$SEED"

echo ""
echo "Step 3: Creating ECDF plots for each method..."

for method in loss min_k zlib; do
    echo "  Generating plot for method: $method"
    Rscript scripts/R2/reviewer2/check_for_delta_0_appendix.R $method "$MAX_STEPS" "$SEED"
done

echo ""
echo "Done! Plots saved to figures/delta_0_appendix_ecdf_*.png"
