#!/bin/bash
set -e

rsync -av --relative \
    "abe@172.20.98.203:~/mimir/./tmp_results/excluded-docs/abehandlerorg_dobolyilab-blockbench-blocksbin_rlhf_dpo_hh-rlhf_e1_lr5e-5_r8/abehandlerorg/excluded-docs/min_k_results.json" \
    "abe@172.20.98.203:~/mimir/./tmp_results/excluded-docs/abehandlerorg_dobolyilab-blockbench-noblocksbin_rlhf_dpo_hh-rlhf_e1_lr5e-5_r8/abehandlerorg/excluded-docs/min_k_results.json" \
    "abe@172.20.98.203:~/mimir/./tmp_results/abehandlerorg/cptllama_excluded_20240101_20240115-Y0/abehandlerorg_Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0/abehandlerorg/cptllama_excluded_20240101_20240115/min_k_results.json" \
    "abe@172.20.98.203:~/mimir/./tmp_results/abehandlerorg/cptllama_excluded_20240101_20240115-Y1/abehandlerorg_Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1/abehandlerorg/cptllama_excluded_20240101_20240115/min_k_results.json" \
    "abe@172.20.98.203:~/mimir/./tmp_results/abehandlerorg/cptllama_excluded_20240130_20240130-Y1/abehandlerorg_Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1/abehandlerorg/cptllama_excluded_20240130_20240130/min_k_results.json" \
    "abe@172.20.98.203:~/mimir/./tmp_results/abehandlerorg/cptllama_excluded_20240130_20240130-Y0/abehandlerorg_Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0/abehandlerorg/cptllama_excluded_20240130_20240130/min_k_results.json" \
    ~/mimir/

python scripts/R2/min_k_check_for_response_doc.py \
    -blocks   tmp_results/abehandlerorg/cptllama_excluded_20240101_20240115-Y1/abehandlerorg_Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1/abehandlerorg/cptllama_excluded_20240101_20240115/min_k_results.json \
    -noblocks tmp_results/abehandlerorg/cptllama_excluded_20240101_20240115-Y0/abehandlerorg_Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0/abehandlerorg/cptllama_excluded_20240101_20240115/min_k_results.json \
    -out      results/min_k_check_for_response_doc/cptllama_excluded_20240101_20240115.parquet

python scripts/R2/min_k_check_for_response_doc.py \
    -blocks   tmp_results/abehandlerorg/cptllama_excluded_20240130_20240130-Y1/abehandlerorg_Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1/abehandlerorg/cptllama_excluded_20240130_20240130/min_k_results.json \
    -noblocks tmp_results/abehandlerorg/cptllama_excluded_20240130_20240130-Y0/abehandlerorg_Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0/abehandlerorg/cptllama_excluded_20240130_20240130/min_k_results.json \
    -out      results/min_k_check_for_response_doc/cptllama_excluded_20240130_20240130_70B.parquet
