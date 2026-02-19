"""
Join mimir results CSV shards with the analyze_neighbor_log jsonl on url.

Usage:
    python scripts/R2/inthewild/process_in_the_wild_jan_22.py
    python scripts/R2/inthewild/process_in_the_wild_jan_22.py --csv-dir csvs/gptoss --pattern "*in-the-wild*.csv"
"""
import argparse
from pathlib import Path

import pandas as pd


JSONL_PATH = "~/dolma/logs/scripts/R2/extract/inthewild/analyze_neighbor_log.jsonl"
OUTPUT_PATH = "results/process_in_the_wild_jan_22/gptoss_mimir_merged.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir",
        default="csvs/gptoss",
        help="Directory containing shard CSVs (default: csvs/gptoss)",
    )
    parser.add_argument(
        "--pattern",
        default="*in-the-wild*.csv",
        help="Glob pattern for shard files (default: *in-the-wild*.csv)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help=f"Output path for merged CSV (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def load_shards(csv_dir: str, pattern: str) -> pd.DataFrame:
    paths = sorted(Path(csv_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {csv_dir}")
    print(f"Loading {len(paths)} shard(s) from {csv_dir}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load_neighbor_log(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def main():
    args = parse_args()

    results_df = load_shards(args.csv_dir, args.pattern)
    neighbor_df = load_neighbor_log(JSONL_PATH)

    # filter to member rows scored by loss, drop dupes
    results_df = results_df[
        (results_df["membership"] == "member") & (results_df["method"] == "loss")
    ].drop_duplicates(subset=["doc_id"])

    # doc_id in results maps to query_url in the jsonl
    merged = results_df.merge(neighbor_df, left_on="doc_id", right_on="query_url", how="left")

    print(f"Rows: {len(merged)}, matched: {merged['query_url'].notna().sum()}")

    lt_cols = [c for c in merged.columns if c.startswith("lt_")]
    for col in lt_cols:
        r = merged["score"].corr(merged[col])
        print(f"{col}: r={r:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Saved merged data -> {out}")


if __name__ == "__main__":
    main()
