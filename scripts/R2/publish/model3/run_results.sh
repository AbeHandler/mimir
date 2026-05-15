#!/bin/bash
set -e

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MIMIR_DATA_SOURCE="${MIMIR_DATA_SOURCE:-mimirdata}"
export MIMIR_CACHE_PATH="${MIMIR_CACHE_PATH:-mimrcache}"

python ${REPO_ROOT}/scripts/R2/publish/model3/go.py --analyze -repo-root ${REPO_ROOT}

# Sanity check that the loss scores in each CSV match a from-scratch LOSS
# attack on the corresponding base_model (per the config JSONs).
SANITY=${REPO_ROOT}/scripts/R2/sanity_check_mimir_ids.py
DS=abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered
CSV_DIR=${REPO_ROOT}/csvs

python ${SANITY} \
    -base-model abehandler/sutva_click2houston_com_2022-05-01_pair1_treated_run1 \
    -dataset ${DS} \
    -urls-file ${REPO_ROOT}/scripts/R2/publish/model3/sanity_check_urls.sutva.txt \
    -reference-csv ${CSV_DIR}/sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.csv

python ${SANITY} \
    -base-model abehandler/sutva_click2houston_com_2022-05-01_pair2_treated_run3 \
    -dataset ${DS} \
    -urls-file ${REPO_ROOT}/scripts/R2/publish/model3/sanity_check_urls.sutva.txt \
    -reference-csv ${CSV_DIR}/sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.csv

python ${SANITY} \
    -base-model abehandler/sutva_click2houston_com_2022-05-01_pair2_control_run4 \
    -dataset ${DS} \
    -urls-file ${REPO_ROOT}/scripts/R2/publish/model3/sanity_check_urls.sutva.txt \
    -reference-csv ${CSV_DIR}/sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.csv