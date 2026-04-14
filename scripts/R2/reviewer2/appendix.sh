
#!/bin/bash
# Rsync all_shards CSVs for seeds 1-3 from blackwell.
# Needed by check_for_delta_0_appendix.py and check_for_delta_1_appendix_seed_names.py.
#
# Usage:
#   bash scripts/R2/reviewer2/rsync_appendix_csvs.sh

set -euo pipefail

rm -rf /tmp/*.csv

ssh -p 30003 abe@128.138.93.183 \
    'find mimir/csvs/confounddataset/ -name "pythia-*_seed[123]_*.all_shards.csv.gz"' \
    | while read -r f; do
        rsync -avz -e "ssh -p 30003" "abe@128.138.93.183:${f}" csvs/confounddataset/
    done

echo "Done. Spot check:"
echo "  ls csvs/confounddataset/pythia-45m_lr1e-3_steps5k_seed[123]_*.all_shards.csv.gz"

SEED=1 bash scripts/R2/reviewer2/final_results_per_seed.sh
SEED=2 bash scripts/R2/reviewer2/final_results_per_seed.sh
SEED=3 bash scripts/R2/reviewer2/final_results_per_seed.sh

# Merge per-seed CSVs into combined files
MAX_STEPS="${MAX_STEPS:-5000}"
STEPS_K="$((MAX_STEPS / 1000))k"

head -1 "/tmp/att_appendix_steps${STEPS_K}_seed1.csv" > /tmp/att_appendix_all.csv
tail -n +2 -q /tmp/att_appendix_steps${STEPS_K}_seed[123].csv >> /tmp/att_appendix_all.csv

head -1 "/tmp/atu_appendix_steps${STEPS_K}_seed1.csv" > /tmp/atu_appendix_all.csv
tail -n +2 -q /tmp/atu_appendix_steps${STEPS_K}_seed[123].csv >> /tmp/atu_appendix_all.csv

echo "Merged all seeds:"
echo "  /tmp/att_appendix_steps${STEPS_K}_seed1.csv: $(wc -l < /tmp/att_appendix_steps${STEPS_K}_seed1.csv) lines"
echo "  /tmp/att_appendix_all.csv: $(wc -l < /tmp/att_appendix_all.csv) lines (expect 3n-2)"
echo "  /tmp/atu_appendix_all.csv: $(wc -l < /tmp/atu_appendix_all.csv) lines (expect 3n-2)"

for method in loss; do
    echo "  Generating merged plot for method: $method"
    Rscript scripts/R2/reviewer2/check_for_delta_0_appendix.R "$method" "$MAX_STEPS" "all"
done

echo "Done! Merged plots saved to figures/delta_0_appendix_ecdf_*_all.png"


python scripts/R2/reviewer2/average_delta_jsons.py --delta 0
python scripts/R2/reviewer2/average_delta_jsons.py --delta 1

python scripts/R2/reviewer2/compare_att_vs_atu.py

python scripts/R2/reviewer2/report_auc.py
