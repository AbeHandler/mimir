#!/bin/bash
set -e

rsync -av --relative \
    "abe@172.20.98.203:~/mimir/./tmp_results/excluded-docs/abehandlerorg_dobolyilab-blockbench-blocksbin_rlhf_dpo_hh-rlhf_e1_lr5e-5_r8/abehandlerorg/excluded-docs/min_k_results.json" \
    "abe@172.20.98.203:~/mimir/./tmp_results/excluded-docs/abehandlerorg_dobolyilab-blockbench-noblocksbin_rlhf_dpo_hh-rlhf_e1_lr5e-5_r8/abehandlerorg/excluded-docs/min_k_results.json" \
    ~/mimir/
