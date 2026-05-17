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
from pathlib import Path

import pandas as pd
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
    parser.add_argument(
        "-reference-csv",
        default="/Users/abha4861/mimir/csvs/confounddataset/bothbins.blocks.lite.all_shards.csv.gz",
        help="Reference csv(.gz) with columns method,membership,doc_id,score to diff against",
    )
    parser.add_argument(
        "-urls-file",
        default=str(Path(__file__).resolve().parent / "publish" / "model3"
                    / "sanity_check_urls.bothbins.txt"),
        help="Text file with one URL per line; only docs whose url is in this list are scored.",
    )
    parser.add_argument(
        "-tolerance",
        type=float,
        default=1e-3,
        help="Absolute+relative tolerance for comparing recomputed loss vs reference CSV.",
    )
    return parser.parse_args()


def load_reference_member_loss(reference_csv: str) -> dict:
    """Read `loss,member,<doc_id>,<score>` rows from the reference CSV.

    Returns {doc_id: score}. Supports .csv and .csv.gz; assumes the columns
    `method, membership, doc_id, score` (the schema used by `run.py`).
    """
    df = pd.read_csv(reference_csv)
    df = df[(df["method"] == "loss") & (df["membership"] == "member")]
    return dict(zip(df["doc_id"], df["score"]))


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

    with open(args.urls_file) as f:
        urls = [line.strip() for line in f if line.strip()]
    assert urls, f"no urls loaded from {args.urls_file}"
    print(f"Loaded {len(urls)} url(s) from {args.urls_file}")

    url_set = set(urls)
    ds = ds.filter(lambda x: x["url"] in url_set)

    df = ds.to_pandas()
    test_examples = list(zip(df["url"].to_list(), df["text"].to_list()))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    ref_loss = load_reference_member_loss(args.reference_csv)
    print(f"Loaded {len(ref_loss)} reference (loss,member) rows from {args.reference_csv}")

    mismatches = []
    with open(args.output, "a") as fout:
        for i, (doc_id, text) in enumerate(test_examples, 1):
            print(f"\nExample {i}: {text[:50]}...")

            probs = model.get_probabilities(text)

            loss_score = loss_attack._attack(text, probs=probs)

            ref_score = ref_loss.get(doc_id)
            if ref_score is None:
                status = "MISSING_REF"
            elif np.isclose(loss_score, ref_score,
                            rtol=args.tolerance, atol=args.tolerance):
                status = "OK"
            else:
                status = "MISMATCH"
                mismatches.append((doc_id, loss_score, ref_score))

            print(f"  id:          {doc_id}")
            print(f"  LOSS score:  {loss_score:.6f}")
            print(f"  ref score:   "
                  f"{'n/a' if ref_score is None else f'{ref_score:.6f}'}")
            print(f"  status:      {status} (tol={args.tolerance:g})")

            fout.write(json.dumps({
                "id": doc_id,
                "loss": float(loss_score),
                "ref_loss": None if ref_score is None else float(ref_score),
                "tolerance": args.tolerance,
                "status": status,
                "dataset": args.dataset,
                "base_model": args.base_model,
                "attack": "loss",
                "reference_csv": args.reference_csv,
            }) + "\n")

    print(f"\nWrote {len(test_examples)} records to {args.output}")
    if mismatches:
        raise AssertionError(
            f"{len(mismatches)} doc(s) disagreed with reference beyond "
            f"tol={args.tolerance:g}: " +
            ", ".join(f"{d}: got {g:.6f}, ref {r:.6f}" for d, g, r in mismatches)
        )
    print("exit")

if __name__ == "__main__":
    main()
