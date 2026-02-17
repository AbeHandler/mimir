#!/usr/bin/env bash


for SHARD_ID in $(seq 1 15); do
    ./scripts/run_config_llama.sh Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.bothbins.lite.json "$SHARD_ID"
    ./scripts/run_config_llama.sh Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.bothbins.lite.json "$SHARD_ID"
    ./scripts/run_config_llama.sh Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.lite.json "$SHARD_ID"
    ./scripts/run_config_llama.sh Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.excluded.lite.json "$SHARD_ID"
done


./scripts/run_single_publishers.sh



# these dont need shards
./scripts/run_config_llama.sh Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.bothbins.lite.json
./scripts/run_config_llama.sh Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.bothbins.lite.json
./scripts/run_config_llama.sh Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.lite.json
./scripts/run_config_llama.sh Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.excluded.lite.json

exit 0

./scripts/run_config_shards.sh 0

./scripts/run_config_gpt_oss.sh gptoss.20b.2024-07-30-to-2024-07-30-Y1.excluded.lite.json 97

./scripts/run_config_gpt_oss.sh gptoss.20b.2024-07-30-to-2024-07-30-Y0.excluded.lite.json 97

./scripts/run_config_gpt_oss.sh gptoss.20b.2024-07-30-to-2024-07-30-Y0.bothbins.lite.json 100

./scripts/run_config_gpt_oss.sh gptoss.20b.2024-07-30-to-2024-07-30-Y1.bothbins.lite.json 100

