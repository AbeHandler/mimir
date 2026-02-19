"""
Sanity check: correlate GPT-OSS cloze scores with mimir membership scores.

Merges jan2022 rollups (mean_neg_logprob) with mimir merged CSV (score)
and prints the Pearson correlation between the two.

Usage:
    python scripts/R2/inthewild/sanity_check_cloze.py
    python scripts/R2/inthewild/sanity_check_cloze.py \
        --rollups data/interim/R2/scm/inthewild/rollups/jan2022_urls_rollups.jsonl.gz \
        --mimir results/process_in_the_wild_jan_22/gptoss_mimir_merged.csv

Spot checks:
    head -2 results/process_in_the_wild_jan_22/gptoss_mimir_merged.csv
    zcat ~/dolma/data/interim/R2/scm/inthewild/rollups/jan2022_urls_rollups.jsonl.gz | head -2
"""
import argparse

import pandas as pd

ROLLUPS_PATH = "~/dolma/data/interim/R2/scm/inthewild/rollups/jan2022_urls_rollups.jsonl.gz"
MIMIR_PATH = "results/process_in_the_wild_jan_22/gptoss_mimir_merged.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlate GPT-OSS cloze scores with mimir membership scores"
    )
    parser.add_argument(
        "--rollups",
        default=ROLLUPS_PATH,
        help=f"Path to consolidated rollups jsonl.gz (default: {ROLLUPS_PATH})",
    )
    parser.add_argument(
        "--mimir",
        default=MIMIR_PATH,
        help=f"Path to mimir merged CSV (default: {MIMIR_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rollups = pd.read_json(args.rollups, lines=True, compression="gzip")
    print(f"Rollups: {len(rollups)} rows")

    mimir = pd.read_csv(args.mimir).rename(columns={"doc_id": "url"})
    print(f"Mimir:   {len(mimir)} rows")

    merged = rollups.merge(mimir, on="url")
    print(f"Merged:  {len(merged)} rows")

    if merged.empty:
        raise ValueError("Merge produced 0 rows — check that URL formats match")

    corr = merged[["score", "mean_neg_logprob"]].corr()
    print("\nCorrelation (score vs mean_neg_logprob):")
    print(corr)


if __name__ == "__main__":
    main()
