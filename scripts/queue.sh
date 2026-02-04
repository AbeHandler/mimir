#!/usr/bin/env bash

./scripts/run_config.sh bothbins.blocks.dcpdd.json

./scripts/run_config.sh bothbins.noblocks.dcpdd.json

./scripts/run_config_gpt_oss.sh gptoss.20b.2024-07-30-to-2024-07-30-Y1.excluded.lite.json 97

./scripts/run_config_gpt_oss.sh gptoss.20b.2024-07-30-to-2024-07-30-Y0.excluded.lite.json 97

./scripts/run_config_gpt_oss.sh gptoss.20b.2024-07-30-to-2024-07-30-Y0.bothbins.lite.json 100

./scripts/run_config_gpt_oss.sh gptoss.20b.2024-07-30-to-2024-07-30-Y1.bothbins.lite.json 100

./scripts/run_single_publishers.sh