"""
Join mimir results CSV shards with the analyze_neighbor_log jsonl on url.

Usage:
    python tmp.py --csv-dir csvs/gptoss
    python tmp.py --csv-dir csvs/gptoss --pattern "*in-the-wild*.csv"
"""
import argparse
from pathlib import Path

import pandas as pd


JSONL_PATH = "~/dolma/logs/scripts/R2/extract/inthewild/analyze_neighbor_log.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", required=True, help="Directory containing shard CSVs")
    parser.add_argument(
        "--pattern",
        default="*in-the-wild*.csv",
        help="Glob pattern for shard files (default: *in-the-wild*.csv)",
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

    lt_cols = [c for c in merged.columns if c.startswith("lt_")][0:8]
    for col in lt_cols:
        r = merged["score"].corr(merged[col])
        print(col, r)

if __name__ == "__main__":
    main()
