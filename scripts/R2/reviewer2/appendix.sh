
#!/bin/bash
# Rsync all_shards CSVs for seeds 1-3 from blackwell.
# Needed by check_for_delta_0_appendix.py and check_for_delta_1_appendix_seed_names.py.
#
# Usage:
#   bash scripts/R2/reviewer2/rsync_appendix_csvs.sh

set -euo pipefail

rm -rf /tmp/att_appendix_all.csv

ssh -p 30003 abe@128.138.93.183 \
    'find mimir/csvs/confounddataset/ -name "pythia-*_seed[123]_*.all_shards.csv.gz"' \
    | while read -r f; do
        rsync -avz -e "ssh -p 30003" "abe@128.138.93.183:${f}" csvs/confounddataset/
    done

echo "Done. Spot check:"
echo "  ls csvs/confounddataset/pythia-45m_lr1e-3_steps5k_seed[123]_*.all_shards.csv.gz"

bash scripts/R2/reviewer2/final_results_per_seed.sh -all