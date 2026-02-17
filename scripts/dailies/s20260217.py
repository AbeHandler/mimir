"""
Daily script: 2026-02-17
"""

import pandas as pd


"""
Merge sharded CSVs into a single file per logical dataset.

Shard naming pattern:
    {base}.shard_{N}.csv  ->  {base}.csv

Usage:
    python tmp.py --input-dir csvs/
    python tmp.py --input-dir csvs/ --output-dir merged/
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

SHARD_PATTERN = re.compile(r"^(.+)\.shard_\d+\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge sharded CSVs.")
    parser.add_argument("--input-dir", type=Path, default="csvs",
                        help="Directory containing sharded CSVs.")
    parser.add_argument("--output-dir", type=Path, default="merged",
                        help="Output directory (default: same as input-dir).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be merged without writing.")
    return parser.parse_args()


def group_shards(input_dir: Path) -> dict[str, list[Path]]:
    """Group shard files by their logical base name."""
    groups = defaultdict(list)
    for f in sorted(input_dir.glob("*.csv")):
        m = SHARD_PATTERN.match(f.name)
        if m:
            base = m.group(1)
            groups[base].append(f)
    return groups


def merge_group(base: str, shards: list[Path], output_dir: Path, dry_run: bool):
    # Sort shards by shard number to preserve order
    shards = sorted(shards, key=lambda p: int(re.search(r"shard_(\d+)", p.name).group(1)))
    out_path = output_dir / f"{base}.csv"

    print(f"\n{base}")
    for s in shards:
        print(f"  + {s.name}")
    print(f"  -> {out_path.name}")

    if dry_run:
        return

    dfs = [pd.read_csv(s) for s in shards]

    # Sanity check: columns match across shards
    cols = [set(df.columns) for df in dfs]
    if len(set(frozenset(c) for c in cols)) > 1:
        raise ValueError(f"Column mismatch across shards for {base}: {cols}")

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(out_path, index=False)
    print(f"  Wrote {len(merged):,} rows to {out_path}")


def main():
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = group_shards(args.input_dir)
    if not groups:
        print("No sharded CSVs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(groups)} logical file(s) to merge.")
    for base, shards in sorted(groups.items()):
        merge_group(base, shards, output_dir, dry_run=args.dry_run)


def process():

	for run in ["excluded", "bothbins"]:
		
		Y1 = pd.read_csv(f"merged/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.{run}.lite.csv").rename(columns={"score": "blocks"}).drop(columns=["membership"])
		Y0 = pd.read_csv(f"merged/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.{run}.lite.csv").rename(columns={"score": "noblocks"}).drop(columns=["membership"])

		B = Y1.merge(Y0, on=["method", "doc_id"])
		B["delta"] = B["blocks"] - B["noblocks"]

		print(run)
		print(B[["delta", "method"]].groupby("method").mean())

if __name__ == "__main__":
    main()
    process()


