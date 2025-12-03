#!/usr/bin/env python3
"""
Merge sharded CSV files by their common prefix.

This script reads all CSV files in a directory that contain 'shard_' in their name,
groups them by their common prefix, and merges each group into a single gzipped CSV file
named {prefix}.all_shards.csv.gz
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd


def find_sharded_files(directory):
    """Find all CSV files with 'shard_' in their name."""
    csv_dir = Path(directory)
    if not csv_dir.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    sharded_files = [f for f in csv_dir.glob("*.csv") if "shard_" in f.name]
    return sharded_files


def extract_prefix(filename):
    """Extract the prefix before '.shard_X.csv'"""
    # Remove the file extension
    name = filename.stem  # Gets filename without .csv

    # Find the position of '.shard_' and extract everything before it
    shard_pos = name.find(".shard_")
    if shard_pos != -1:
        return name[:shard_pos]
    return name


def group_by_prefix(files):
    """Group files by their common prefix."""
    prefix_groups = defaultdict(list)

    for file_path in files:
        prefix = extract_prefix(file_path)
        prefix_groups[prefix].append(file_path)

    # Sort files within each group by shard number
    for prefix in prefix_groups:
        prefix_groups[prefix].sort(key=lambda x: int(x.stem.split("shard_")[1]))

    return prefix_groups


def merge_shards(file_list, output_path):
    """Merge all CSV files in the list into a single gzipped CSV."""
    print(f"  Merging {len(file_list)} shards...")

    dfs = []
    for file_path in file_list:
        print(f"    Reading {file_path.name}")
        df = pd.read_csv(file_path)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_path, index=False, compression='gzip')
    print(f"  Saved {len(merged_df)} rows to {output_path.name}")
    return len(merged_df)


def merge_sharded_csvs(directory, output_dir=None, verbose=True):
    """
    Merge sharded CSV files by their common prefix.

    Args:
        directory: Directory containing sharded CSV files
        output_dir: Output directory (defaults to same as input directory)
        verbose: Whether to print progress messages

    Returns:
        dict: Dictionary mapping prefix to output file path
    """
    # Find all sharded files
    if verbose:
        print(f"Scanning directory: {directory}")
    sharded_files = find_sharded_files(directory)
    if verbose:
        print(f"Found {len(sharded_files)} sharded CSV files")

    if not sharded_files:
        if verbose:
            print("No sharded files found!")
        return {}

    # Group by prefix
    prefix_groups = group_by_prefix(sharded_files)
    if verbose:
        print(f"\nFound {len(prefix_groups)} unique prefixes:")
        for prefix, files in prefix_groups.items():
            print(f"  {prefix}: {len(files)} shards")

    # Determine output directory
    output_path = Path(output_dir) if output_dir else Path(directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # Merge each group
    if verbose:
        print("\nMerging shards:")

    output_files = {}
    for prefix, files in prefix_groups.items():
        output_file = output_path / f"{prefix}.all_shards.csv.gz"
        if verbose:
            print(f"\n{prefix}:")
        merge_shards(files, output_file)
        output_files[prefix] = output_file

    if verbose:
        print("\nDone!")

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded CSV files by common prefix"
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Directory containing sharded CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to same as input directory)"
    )

    args = parser.parse_args()
    merge_sharded_csvs(args.directory, args.output_dir, verbose=True)


if __name__ == "__main__":
    main()
