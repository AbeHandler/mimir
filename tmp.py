"""
Join mimir results CSV with the analyze_neighbor_log jsonl on url.

Usage:
    python tmp.py --csv path/to/results.csv
"""
import argparse
import pandas as pd


JSONL_PATH = "~/dolma/logs/scripts/R2/extract/inthewild/analyze_neighbor_log.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to mimir results CSV")
    return parser.parse_args()


def load_neighbor_log(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def main():
    args = parse_args()

    results_df = pd.read_csv(args.csv)
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


if __name__ == "__main__":
    main()
