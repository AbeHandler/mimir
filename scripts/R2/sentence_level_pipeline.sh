#!/usr/bin/env bash

set -e

mkdir -p tmp/sentences

# Dataset to query against (the control dataset)
INDEX_DATASET='abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4'
N_RESULTS=100

echo "=== Reconstructing sentences from MIMIR results ==="
echo ""

#                                                    
# 88                      88                         
# ""                      88                         
#                         88                         
# 88 8b,dPPYba,   ,adPPYb,88  ,adPPYba, 8b,     ,d8  
# 88 88P'   `"8a a8"    `Y88 a8P_____88  `Y8, ,8P'   
# 88 88       88 8b       88 8PP"""""""    )888(     
# 88 88       88 "8a,   ,d88 "8b,   ,aa  ,d8" "8b,   
# 88 88       88  `"8bbdP"Y8  `"Ybbd8"' 8P'     `Y8  
#                                                    
#                                                    

# Step 1: Create the sentence-level Annoy index if it doesn't exist
echo "=== Initializing sentence-level Annoy index ==="
echo ""
if [ ! -f "data/interim/sutva/${INDEX_DATASET##*/}/annoy_sentences/index.ann" ]; then
    echo "Creating sentence-level index for ${INDEX_DATASET}..."
    CUDA_VISIBLE_DEVICES=0 python ~/dolma/scripts/R2/sutva/init_card_sentences.py \
        --dataset-name $INDEX_DATASET
    echo ""
else
    echo "✓ Sentence-level index already exists, skipping initialization"
    echo ""
fi

#                                                                                                                       
#                                                                                                                       
#                                                                      ,d                                        ,d     
#                                                                      88                                        88     
# 8b,dPPYba,  ,adPPYba,  ,adPPYba,  ,adPPYba,  8b,dPPYba,  ,adPPYba, MM88MMM 8b,dPPYba, 88       88  ,adPPYba, MM88MMM  
# 88P'   "Y8 a8P_____88 a8"     "" a8"     "8a 88P'   `"8a I8[    ""   88    88P'   "Y8 88       88 a8"     ""   88     
# 88         8PP""""""" 8b         8b       d8 88       88  `"Y8ba,    88    88         88       88 8b           88     
# 88         "8b,   ,aa "8a,   ,aa "8a,   ,a8" 88       88 aa    ]8I   88,   88         "8a,   ,a88 "8a,   ,aa   88,    
# 88          `"Ybbd8"'  `"Ybbd8"'  `"YbbdP"'  88       88 `"YbbdP"'   "Y888 88          `"YbbdP'Y8  `"Ybbd8"'   "Y888  
#                                                                                                                       
#                                                                                                                       

echo "=== Reconstructing sentences from loss results ==="
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

#                                                            
#                                                            
#                                                            
#                                                            
#  ,adPPYb,d8 88       88  ,adPPYba, 8b,dPPYba, 8b       d8  
# a8"    `Y88 88       88 a8P_____88 88P'   "Y8 `8b     d8'  
# 8b       88 88       88 8PP""""""" 88          `8b   d8'   
# "8a    ,d88 "8a,   ,a88 "8b,   ,aa 88           `8b,d8'    
#  `"YbbdP'88  `"YbbdP'Y8  `"Ybbd8"' 88             Y88'     
#          88                                       d8'      
#          88                                      d8'       


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



#TODO for each sentence in doc, what is the ATE?
