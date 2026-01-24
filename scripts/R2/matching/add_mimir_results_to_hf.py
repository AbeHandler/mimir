#!/usr/bin/env python3
"""
Analyze matching neighbors dataset to find unique methods per URL.

This script:
1. Reads the local CSV file (matching_neighbors.blocks.lite.all_shards.csv.gz)
2. Loads the HuggingFace dataset (abehandlerorg/matching_neighbors)
3. Matches URLs from HF dataset with scores for each method in the CSV
4. Filters to only 'member' membership
5. Matches doc_id from CSV with url from HF dataset
6. Finds unique methods per document ID
"""

import gzip
import os
import pandas as pd
from datasets import load_dataset
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def load_csv_data(csv_path, score_col_name='neighbor_score'):
    """Load and filter the CSV data.

    Args:
        csv_path: Path to the gzipped CSV file
        score_col_name: Name to use for the score column in the output
    """
    print(f"Loading CSV from {csv_path}...")

    # Read compressed CSV
    with gzip.open(csv_path, 'rt') as f:
        df = pd.read_csv(f)

    print(f"Loaded {len(df)} total rows")

    # Filter to only members
    df_members = df[df['membership'] == 'member'].copy()
    print(f"Filtered to {len(df_members)} member rows")

    # Rename score column to the desired name
    df_members = df_members.rename(columns={'score': score_col_name})

    return df_members


def load_hf_dataset():
    """Load the HuggingFace dataset."""
    print("Loading HuggingFace dataset...")
    dataset = load_dataset('abehandlerorg/matching_neighbors', split='train')

    # Convert streaming dataset to pandas with progress bar
    print("Converting to pandas DataFrame...")
    records = []
    for record in tqdm(dataset, desc="Loading records"):
        records.append(record)

    df_hf = pd.DataFrame(records)
    print(f"Loaded {len(df_hf)} rows from HuggingFace")

    return df_hf


def match_and_analyze(df_matching, df_excluded, df_hf):
    """Match CSV and HF data to create blocked_doc + neighbor + method rows.

    Args:
        df_matching: matching_neighbors data (doc_id matches url in HF)
        df_excluded: excluded-docs data (doc_id matches query_id in HF)
        df_hf: HuggingFace dataset (has query_id=blocked doc, url=neighbor)
    """
    print("\nJoining HF dataset with both score datasets...")

    # First, merge HF with excluded-docs to get blocked_doc scores
    # HF query_id matches excluded-docs doc_id
    hf_with_blocked = pd.merge(
        df_hf,
        df_excluded[['doc_id', 'method', 'blocked_score']],
        left_on='query_id',
        right_on='doc_id',
        how='inner',
        suffixes=('', '_blocked')
    )

    print(f"Matched {len(hf_with_blocked)} HF rows with blocked doc scores")

    # Now merge with matching_neighbors to get neighbor scores
    # HF url matches matching_neighbors doc_id
    merged = pd.merge(
        hf_with_blocked,
        df_matching[['doc_id', 'method', 'neighbor_score']],
        left_on=['url', 'method'],
        right_on=['doc_id', 'method'],
        how='outer',
        suffixes=('', '_neighbor')
    )

    print(f"Total rows after merging with neighbor scores: {len(merged)}")

    # Clean up - drop duplicate doc_id columns
    if 'doc_id_blocked' in merged.columns:
        merged = merged.drop(columns=['doc_id_blocked'])
    if 'doc_id_neighbor' in merged.columns:
        merged = merged.drop(columns=['doc_id_neighbor'])
    if 'doc_id' in merged.columns:
        merged = merged.drop(columns=['doc_id'])

    # Create detailed view
    detailed_results = []
    for _, row in merged.iterrows():
        detailed_results.append({
            'neighbor_url': row['url'],
            'blocked_doc_id': row['query_id'],
            'method': row['method'],
            'blocked_doc_score': row.get('blocked_score'),
            'neighbor_score': row.get('neighbor_score'),
            'neighbor_rank': row.get('neighbor_rank'),
            'distance': row.get('distance'),
            'crawl_date': row.get('crawl_date')
        })

    df_detailed = pd.DataFrame(detailed_results)

    # Group by URL and find unique methods
    print("\nFinding unique methods per URL...")
    url_methods = df_detailed.groupby('neighbor_url').agg({
        'method': lambda x: sorted(list(set(x))),
        'neighbor_score': lambda x: [s for s in x if pd.notna(s)],
        'blocked_doc_score': lambda x: [s for s in x if pd.notna(s)]
    }).reset_index()

    url_methods['num_methods'] = url_methods['method'].apply(len)

    return url_methods, df_detailed


def print_summary(url_methods, df_detailed):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nTotal unique URLs: {len(url_methods)}")
    print(f"\nDistribution of methods per URL:")
    print(url_methods['num_methods'].value_counts().sort_index())

    print(f"\nURLs with multiple methods: {len(url_methods[url_methods['num_methods'] > 1])}")

    print("\nTop 10 URLs by number of methods:")
    top_urls = url_methods.nlargest(10, 'num_methods')[['neighbor_url', 'method', 'num_methods']]
    for _, row in top_urls.iterrows():
        print(f"\n  {row['neighbor_url']}")
        print(f"    Methods ({row['num_methods']}): {', '.join(row['method'])}")

    print("\n" + "="*60)
    print("METHOD DISTRIBUTION")
    print("="*60)
    print(df_detailed['method'].value_counts())


def main():
    """Main execution function."""
    # File paths
    matching_csv_path = 'csvs/confounddataset/matching_neighbors.blocks.lite.all_shards.csv.gz'
    excluded_csv_path = 'csvs/confounddataset/excluded-docs.blocks.lite.all_shards.csv.gz'

    # Load data
    df_matching = load_csv_data(matching_csv_path, score_col_name='neighbor_score')
    df_excluded = load_csv_data(excluded_csv_path, score_col_name='blocked_score')
    df_hf = load_hf_dataset()

    # Match and analyze
    url_methods, df_detailed = match_and_analyze(df_matching, df_excluded, df_hf)

    # Print summary
    print_summary(url_methods, df_detailed)

    tmp_dir = Path(os.environ.get('TMPDIR', os.environ.get('TMP', '/tmp')))
    detailed_output_path = tmp_dir / 'matching_neighbors_detailed.csv'
    print(f"Saving detailed results to {detailed_output_path}...")
    df_detailed.to_csv(detailed_output_path, index=False)

    print("\nDone!")


if __name__ == '__main__':
    main()
