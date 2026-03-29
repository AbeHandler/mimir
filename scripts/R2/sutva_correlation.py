"""Correlate neighbor count differences (delta_run) with score differences (delta_y) per method.

Usage:
    python scripts/R2/tmp.py --enriched-jsonl /tmp/sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered_in_sutva_click2houston_com_2022-05-01_pair1_control_run2_top100.enriched.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enriched-jsonl",
        type=Path,
        required=True,
        help="Path to enriched JSONL file with neighbor similarity and training membership",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.5,
        help="Similarity threshold for counting close neighbors (default: 0.5)",
    )
    parser.add_argument(
        "--control-file",
        type=Path,
        default=Path("~/mimir/csvs/sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.csv"),
        help="Control scores CSV",
    )
    parser.add_argument(
        "--treated-run1-file",
        type=Path,
        default=Path("~/mimir/csvs/sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.csv"),
        help="Treated run1 scores CSV",
    )
    parser.add_argument(
        "--treated-run2-file",
        type=Path,
        default=Path("~/mimir/csvs/sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.csv"),
        help="Treated run2 (run3) scores CSV",
    )
    return parser.parse_args()


def load_pair(treated_file, control_file, treated_col="blocks", control_col="noblocks"):
    df1 = pd.read_csv(treated_file).rename(columns={"score": treated_col})
    df2 = pd.read_csv(control_file).rename(columns={"score": control_col})
    pair = df1.merge(df2, on=["doc_id", "method", "membership"])
    pair["delta"] = pair[treated_col] - pair[control_col]
    pair = pair[pair["membership"] == "member"].copy()
    return pair


def build_both(control_file, treated_run1_file, treated_run2_file):
    pair1 = load_pair(treated_run1_file, control_file).rename(columns={"blocks": "run1"})
    pair1 = pair1[pair1["membership"] == "member"].copy().drop(columns=["noblocks", "delta"])

    pair2 = load_pair(treated_run2_file, control_file).rename(columns={"blocks": "run2"})
    pair2 = pair2[pair2["membership"] == "member"].copy().drop(columns=["noblocks", "delta"])

    both = pair1.merge(pair2, on=["doc_id", "method", "membership"]).drop(columns=["membership"])
    both["delta_y"] = both["run1"] - both["run2"]
    return both


def build_neighbor_counts(enriched_jsonl, epsilon):
    L = []
    with open(enriched_jsonl) as f:
        for line in f:
            r = json.loads(line)
            close = [n for n in r["neighbors"] if n["similarity"] >= epsilon]
            in_run1 = sum(1 for n in close if n["in_training"]["sutva_click2houston_com_2022-05-01_pair1_treated_run1"])
            in_run3 = sum(1 for n in close if n["in_training"]["sutva_click2houston_com_2022-05-01_pair2_treated_run3"])
            L.append({"in_run1": in_run1, "in_run3": in_run3, "doc_id": r["query_url"]})

    df = pd.DataFrame(L)
    df["delta_run"] = df["in_run1"] - df["in_run3"]
    return df


def main():
    args = parse_args()

    both = build_both(
        args.control_file.expanduser(),
        args.treated_run1_file.expanduser(),
        args.treated_run2_file.expanduser(),
    )
    neighbor_counts = build_neighbor_counts(args.enriched_jsonl, args.epsilon)
    both2 = neighbor_counts.merge(both, on="doc_id")

    for method in both2["method"].unique():
        print(method)
        print(both2[both2["method"] == method][["delta_run", "delta_y"]].corr(method="kendall"))


if __name__ == "__main__":
    main()
