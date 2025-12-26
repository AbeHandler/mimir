#!/usr/bin/env bash

set -e

mkdir -p tmp/sentences

# Dataset to query against (the control dataset)
INDEX_DATASET='abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4'
N_RESULTS=100

echo "=== Reconstructing sentences from MIMIR results ==="
echo ""

# Pair 1 treated
echo "Processing pair1 treated..."
python reconstructor.py --output-root tmp/sentences --results-json tmp_results/sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/abehandler_sutva_click2houston_com_2022-05-01_pair1_treated_run1/abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/loss_results.json

# Pair 2 treated
echo ""
echo "Processing pair2 treated..."
python reconstructor.py --output-root tmp/sentences --results-json tmp_results/sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/abehandler_sutva_click2houston_com_2022-05-01_pair2_treated_run3/abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/loss_results.json

# Pair 2 control
echo ""
echo "Processing pair2 control..."
python reconstructor.py --output-root tmp/sentences --results-json tmp_results/sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/abehandler_sutva_click2houston_com_2022-05-01_pair2_control_run4/abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/loss_results.json

echo ""
echo "=== Querying sentence-level index for similar sentences ==="
echo ""

# Query pair1 treated sentences against the index
echo "Querying pair1 treated sentences..."
python ~/dolma/scripts/R2/sutva/query_card_sentences.py \
    --index-dataset $INDEX_DATASET \
    --query-file tmp/sentences/tmp_results/sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/abehandler_sutva_click2houston_com_2022-05-01_pair1_treated_run1/abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/sentences.jsonl \
    --sentence-field text \
    --n-results $N_RESULTS \
    --output-file tmp/sentences/tmp_results/sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/abehandler_sutva_click2houston_com_2022-05-01_pair1_treated_run1/abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/sentence_matches.jsonl.gz

# Query pair2 treated sentences against the index
echo ""
echo "Querying pair2 treated sentences..."
python ~/dolma/scripts/R2/sutva/query_card_sentences.py \
    --index-dataset $INDEX_DATASET \
    --query-file tmp/sentences/tmp_results/sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/abehandler_sutva_click2houston_com_2022-05-01_pair2_treated_run3/abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/sentences.jsonl \
    --sentence-field text \
    --n-results $N_RESULTS \
    --output-file tmp/sentences/tmp_results/sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/abehandler_sutva_click2houston_com_2022-05-01_pair2_treated_run3/abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/sentence_matches.jsonl.gz

# Query pair2 control sentences against the index
echo ""
echo "Querying pair2 control sentences..."
python ~/dolma/scripts/R2/sutva/query_card_sentences.py \
    --index-dataset $INDEX_DATASET \
    --query-file tmp/sentences/tmp_results/sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/abehandler_sutva_click2houston_com_2022-05-01_pair2_control_run4/abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/sentences.jsonl \
    --sentence-field text \
    --n-results $N_RESULTS \
    --output-file tmp/sentences/tmp_results/sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/abehandler_sutva_click2houston_com_2022-05-01_pair2_control_run4/abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered/sentence_matches.jsonl.gz

echo ""
echo "✓ Pipeline complete!"