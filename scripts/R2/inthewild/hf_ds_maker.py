#!/usr/bin/env python3
"""
Subset a HuggingFace dataset to rows whose URL matches URLs in a JSONL log file.

Caches the result to disk. Uses a .done marker to detect stale caches.
If the marker is older than --max-age-days, wipes and rebuilds.

Usage:
    python scripts/R2/inthewild/hf_ds_maker.py
    python scripts/R2/inthewild/hf_ds_maker.py \
        --jsonl ~/dolma/logs/scripts/R2/extract/inthewild/analyze_neighbor_log.jsonl \
        --dataset abehandlerorg/ccnews-jan2022 \
        --max-age-days 7
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_JSONL = os.path.expanduser(
    "~/dolma/logs/scripts/R2/extract/inthewild/analyze_neighbor_log.jsonl"
)
DEFAULT_DATASET = "abehandlerorg/ccnews-jan2022"
DEFAULT_HF_URL_FIELD = "url"
DEFAULT_JSONL_URL_FIELD = "query_url"
DEFAULT_MAX_AGE_DAYS = 7


def load_query_urls(jsonl_path: str, url_field: str) -> set:
    """Read query URLs from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file.
        url_field: Key name for the URL in each JSON record.

    Returns:
        Set of URL strings.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    urls = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            url = record.get(url_field)
            if url:
                urls.add(url)

    print(f"Loaded {len(urls)} unique URLs from {path}")
    return urls


def cache_dir_for(dataset: str, jsonl_path: str) -> Path:
    """Derive the cache directory path from dataset and log file names.

    Pattern: /tmp/<dataset_org>/<dataset_name>/<log_stem>

    Args:
        dataset: HuggingFace dataset id, e.g. "org/name".
        jsonl_path: Path to the source JSONL file.

    Returns:
        Path to the cache directory.
    """
    parts = dataset.replace("/", os.sep)
    log_stem = Path(jsonl_path).stem
    return Path("/tmp") / parts / log_stem


def is_cache_fresh(cache_dir: Path, max_age_days: int) -> bool:
    """Return True if the .done marker exists and is younger than max_age_days.

    Args:
        cache_dir: Directory containing the .done marker.
        max_age_days: Maximum acceptable age in days.

    Returns:
        True if cache is fresh, False otherwise.
    """
    marker = cache_dir / ".done"
    if not marker.exists():
        return False

    mtime = datetime.fromtimestamp(marker.stat().st_mtime, tz=timezone.utc)
    age_days = (datetime.now(tz=timezone.utc) - mtime).days
    return age_days < max_age_days


def wipe_cache(cache_dir: Path) -> None:
    """Remove all files in the cache directory.

    Args:
        cache_dir: Directory to wipe.
    """
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Wiped stale cache at {cache_dir}")


def load_hf_dataset_as_df(dataset: str, hf_url_field: str) -> pd.DataFrame:
    """Download a HuggingFace dataset and return it as a DataFrame.

    Args:
        dataset: HuggingFace dataset id.
        hf_url_field: Name of the URL column in the dataset.

    Returns:
        DataFrame of all records.
    """
    print(f"Loading HuggingFace dataset: {dataset} ...")
    ds = load_dataset(dataset, split="train")

    records = []
    for record in tqdm(ds, desc="Streaming records"):
        records.append(record)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} rows; columns: {list(df.columns)}")

    if hf_url_field not in df.columns:
        raise ValueError(
            f"URL field '{hf_url_field}' not found in dataset columns: {list(df.columns)}"
        )
    return df


def build_subset(
    jsonl_path: str,
    dataset: str,
    hf_url_field: str,
    jsonl_url_field: str,
    max_age_days: int,
) -> pd.DataFrame:
    """Return the subset of the HF dataset whose URLs appear in the JSONL file.

    Caches to disk under /tmp/<dataset>/<log_stem>/subset.parquet.
    Rebuilds if the .done marker is missing or older than max_age_days.

    Args:
        jsonl_path: Path to the JSONL log file.
        dataset: HuggingFace dataset id.
        hf_url_field: URL column name in the HF dataset.
        jsonl_url_field: URL key in each JSONL record.
        max_age_days: How many days before the cache is considered stale.

    Returns:
        DataFrame of matched rows.
    """
    cache_dir = cache_dir_for(dataset, jsonl_path)
    parquet_path = cache_dir / "subset.parquet"
    marker = cache_dir / ".done"

    if is_cache_fresh(cache_dir, max_age_days):
        print(f"Cache is fresh (< {max_age_days} days). Loading from {parquet_path}")
        return pd.read_parquet(parquet_path)

    if cache_dir.exists():
        wipe_cache(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    query_urls = load_query_urls(jsonl_path, jsonl_url_field)
    df_hf = load_hf_dataset_as_df(dataset, hf_url_field)

    print(f"Filtering {len(df_hf)} rows to {len(query_urls)} query URLs ...")
    df_subset = df_hf[df_hf[hf_url_field].isin(query_urls)].copy()
    print(f"Matched {len(df_subset)} rows")

    df_subset.to_parquet(parquet_path, index=False)
    marker.touch()
    print(f"Saved subset to {parquet_path}")
    print(f"Wrote .done marker at {marker}")

    # Spot-check output
    print("\n--- Spot check ---")
    print(f"  parquet path : {parquet_path}")
    print(f"  rows         : {len(df_subset)}")
    print(f"  columns      : {list(df_subset.columns)}")
    if len(df_subset) > 0:
        print(f"  first URL    : {df_subset[hf_url_field].iloc[0]}")
    print("------------------\n")

    return df_subset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Subset a HuggingFace dataset by URLs found in a JSONL log file."
    )
    parser.add_argument(
        "--jsonl",
        default=DEFAULT_JSONL,
        help=f"Path to the JSONL log file (default: {DEFAULT_JSONL})",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset id (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--hf-url-field",
        default=DEFAULT_HF_URL_FIELD,
        help=f"URL column name in the HF dataset (default: {DEFAULT_HF_URL_FIELD})",
    )
    parser.add_argument(
        "--jsonl-url-field",
        default=DEFAULT_JSONL_URL_FIELD,
        help=f"URL key in each JSONL record (default: {DEFAULT_JSONL_URL_FIELD})",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=DEFAULT_MAX_AGE_DAYS,
        help=f"Days before cache is considered stale (default: {DEFAULT_MAX_AGE_DAYS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_subset(
        jsonl_path=args.jsonl,
        dataset=args.dataset,
        hf_url_field=args.hf_url_field,
        jsonl_url_field=args.jsonl_url_field,
        max_age_days=args.max_age_days,
    )
    print(f"Done. DataFrame shape: {df.shape}")


if __name__ == "__main__":
    main()
