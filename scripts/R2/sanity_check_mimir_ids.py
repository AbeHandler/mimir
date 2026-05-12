#!/usr/bin/env python3
"""
Standalone script to sanity check mimir ids coming out of run config shards
Especially useful for experiment one. 

To run this do a git pull on the Mimir repo from Duan et al. 

$ cd /tmp/ && git clone git@github.com:iamgroot42/mimir.git && cd mimir && cp ~/mimir/scripts/R2/sanity_check_mimir_ids.py . && export MIMIR_DATA_SOURCE=mimirdata && export MIMIR_CACHE_PATH=mimrcache

- I see minor deviations running this locally on CPU vs. the numbers coming out of the pipeline but these are minor differences.

(mimr) ➜  mimir git:(main) ✗ cat /Users/abha4861/mimir/csvs/confounddataset/bothbins.blocks.lite.shard_17.csv | shuf | grep "loss," | head -1
loss,member,https://www.shutterstock.com/image-photo/pretty-young-girl-holding-daisy-flower-51854116,0.2879748012241104
LOSS score:  0.2863

Usage:
    python test_cloze_vs_loss.py
"""
import argparse
import json

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from mimir.config import ExperimentConfig, EnvironmentConfig
from mimir.models import LanguageModel
from mimir.attacks.loss import LOSSAttack


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check mimir ids/loss")
    parser.add_argument(
        "-base-model",
        default="dobolyilab/blockbench-blocksbin",
        help="HF model id to load",
    )
    parser.add_argument(
        "-dataset",
        default="abehandlerorg/bothbins",
        help="HF dataset id (train split assumed)",
    )
    parser.add_argument(
        "-output",
        default="audit.jsonl",
        help="Output jsonl path",
    )
    return parser.parse_args()


def create_minimal_config(base_model):
    """Create a minimal config for testing."""
    env_config = EnvironmentConfig(
        device="cuda:0",
        cache_dir="/tmp/cache",
    )

    config = ExperimentConfig(
        experiment_name="test_cloze_vs_loss",
        base_model=base_model,
        dataset_member="dummy",  # Not used in this test
        dataset_nonmember="dummy",  # Not used in this test
        env_config=env_config,
        random_seed=42,
    )

    return config


def main():
    args = parse_args()

    print("=" * 60)
    print("Sanity testing MIMIR output")
    print("=" * 60)

    config = create_minimal_config(args.base_model)
    print(f"\nDevice: {config.env_config.device}")
    print(f"Model: {config.base_model}")

    print(f"\nLoading model...")
    model = LanguageModel(config)
    model.load()

    loss_attack = LOSSAttack(config, model)

    from datasets import load_dataset

    ds = load_dataset(args.dataset)["train"]

    urls = ['https://www.bovnews.com/2022/06/09/updating-the-investment-thesis-box-inc-box-and-capital-one-financial-corporation-cof/']
    urls.append('https://www.news5cleveland.com/news/national/gas-prices-are-falling-at-a-historic-rate-heres-why-experts-say-it-will-continue')

    ds = ds.filter(lambda x: x["url"] in urls)

    df = ds.to_pandas()
    test_examples = list(zip(df["url"].to_list(), df["text"].to_list()))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    with open(args.output, "w") as fout:
        for i, (doc_id, text) in enumerate(test_examples, 1):
            print(f"\nExample {i}: {text[:50]}...")

            probs = model.get_probabilities(text)

            loss_score = loss_attack._attack(text, probs=probs)

            print(f"  id:          {doc_id}")
            print(f"  LOSS score:  {loss_score:.4f}")

            fout.write(json.dumps({"id": doc_id, "loss": float(loss_score)}) + "\n")

    print(f"\nWrote {len(test_examples)} records to {args.output}")
    print("exit")

if __name__ == "__main__":
    main()
