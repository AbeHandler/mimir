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
    parser.add_argument(
        "--contaminated-file",
        type=str,
        required=True,
        help="Path to contaminated model results CSV (e.g., *_interleave*_contaminated.all_shards.csv.gz)",
    )
    parser.add_argument(
        "--uncontaminated-file",
        type=str,
        required=True,
        help="Path to uncontaminated model results CSV (e.g., *_on_contaminated.all_shards.csv.gz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV file for merged results with delta scores",
    )
    return parser.parse_args()


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
    contaminated = contaminated[contaminated["method"] == "loss"].copy().drop(columns="method")
    contaminated = contaminated[contaminated["membership"] == "member"].copy().drop(columns="membership")
    contaminated = contaminated.rename(columns={"score": "contaminated"})

    print(f"Reading uncontaminated model results: {uncontaminated_path}")
    uncontaminated = pd.read_csv(uncontaminated_path)
    uncontaminated = uncontaminated[uncontaminated["method"] == "loss"].copy().drop(columns="method")
    uncontaminated = uncontaminated[uncontaminated["membership"] == "member"].copy().drop(columns="membership")
    uncontaminated = uncontaminated.rename(columns={"score": "uncontaminated"})

    print(f"Merging on doc_id...")
    both = uncontaminated.merge(contaminated, on="doc_id")

    print(f"Calculating delta (uncontaminated - contaminated)...")
    both["delta"] = both["uncontaminated"] - both["contaminated"]

    mean_delta = both["delta"].mean()
    print(f"\nResults:")
    print(f"  Contaminated docs: {len(both):,}")
    print(f"  Mean delta (uncontaminated - contaminated): {mean_delta:.6f}")
    print(f"  Mean contaminated score: {both['contaminated'].mean():.6f}")
    print(f"  Mean uncontaminated score: {both['uncontaminated'].mean():.6f}")

    if args.output:
        output_path = Path(args.output)
        both.to_csv(output_path, index=False, compression="gzip" if output_path.suffix == ".gz" else None)
        print(f"\nWrote merged results to: {output_path}")


if __name__ == "__main__":
    main()