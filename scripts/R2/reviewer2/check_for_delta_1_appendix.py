"""
Check contamination by comparing loss scores between contaminated and uncontaminated models.

Usage:
  python check_contamination.py --contaminated-file <file> --uncontaminated-file <file>
"""
import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare loss scores between contaminated and uncontaminated models"
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--contaminated-file",
        type=str,
        default=None,
    )
    # this is the rate limiter => pythia-45m_lr1e-3_steps5k_seed1234_on_contaminated.json
    parser.add_argument(
        "--uncontaminated-file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV file for merged results with delta scores",
    )
    args = parser.parse_args()
    steps_k = f"{args.max_steps // 1000}k"
    seed_suffix = "" if args.seed == 1234 else f"_seed{args.seed}"
    if args.contaminated_file is None:
        args.contaminated_file = f"csvs/confounddataset/pythia-45m_lr1e-3_steps{steps_k}_seed{args.seed}_interleave0.02_contaminated{seed_suffix}.all_shards.csv.gz"
    if args.uncontaminated_file is None:
        args.uncontaminated_file = f"csvs/confounddataset/pythia-45m_lr1e-3_steps{steps_k}_seed{args.seed}_on_contaminated{seed_suffix}.all_shards.csv.gz"
    return args


def main() -> None:
    args = parse_args()

    contaminated_path = Path(args.contaminated_file)
    uncontaminated_path = Path(args.uncontaminated_file)

    if not contaminated_path.exists():
        raise FileNotFoundError(f"Contaminated file not found: {contaminated_path}")
    if not uncontaminated_path.exists():
        raise FileNotFoundError(f"Uncontaminated file not found: {uncontaminated_path}")

    print(f"Reading contaminated model results: {contaminated_path}")
    contaminated = pd.read_csv(contaminated_path)
    contaminated = contaminated[contaminated["membership"] == "member"].copy().drop(columns="membership")
    contaminated = contaminated.rename(columns={"score": "contaminated"})
    print(len(contaminated))

    print(f"Reading uncontaminated model results: {uncontaminated_path}")
    uncontaminated = pd.read_csv(uncontaminated_path)
    uncontaminated = uncontaminated[uncontaminated["membership"] == "member"].copy().drop(columns="membership")
    uncontaminated = uncontaminated.rename(columns={"score": "uncontaminated"})

    print(f"Merging on doc_id and method...")
    both = uncontaminated.merge(contaminated, on=["doc_id", "method"])

    print(f"Calculating delta (uncontaminated - contaminated)...")
    both["delta"] = both["uncontaminated"] - both["contaminated"]

    print("Counts")
    print(both["method"].value_counts())

    # Write ATT data for R plotting
    both.to_csv("/tmp/att_appendix.csv", index=False)
    print(f"\nWrote ATT data to: /tmp/att_appendix.csv")

    print(f"\nResults by method:")
    print(f"  Total rows: {len(both):,}")

    mean_by_method = both[["delta", "method"]].groupby(["method"]).mean()
    print(f"\nMean delta (uncontaminated - contaminated) by method:")
    print(mean_by_method)

    if args.output:
        output_path = Path(args.output)
        both.to_csv(output_path, index=False, compression="gzip" if output_path.suffix == ".gz" else None)
        print(f"\nWrote merged results to: {output_path}")


if __name__ == "__main__":
    main()