#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from scipy import stats
import numpy as np
import os

# update the all_shards docs to ensure you get the latest versions
# os.system("python scripts/merge_shards.py -d csvs/confounddataset")


def load_MIA_scores(template: str, excluded_urls: set = None, bothbins_urls: set = None) -> pd.DataFrame:
    """
    Load MIA scores for blocks and noblocks treatments, merge, and calculate delta.
    Automatically filters by URL set based on template file name pattern.

    Args:
        template: Path template with {} placeholder for 'blocks'/'noblocks'
                 Example: 'csvs/confounddataset/excluded-docs.{}.lite.all_shards.csv.gz'
        excluded_urls: Set of excluded URLs (optional, auto-detected from template)
        bothbins_urls: Set of bothbins URLs (optional, auto-detected from template)

    Returns:
        Merged dataframe with 'blocks', 'noblocks', and 'delta' columns, filtered by appropriate URL set
    """
    noblocks_path = template.format('noblocks')
    blocks_path = template.format('blocks')

    noblocks = pd.read_csv(noblocks_path).rename(columns={"score": "noblocks"})
    blocks = pd.read_csv(blocks_path).rename(columns={"score": "blocks"})

    merged = noblocks.merge(blocks, on=['doc_id', 'method', 'membership'])
    merged = merged[merged["membership"] == "member"].copy()
    merged = merged.drop_duplicates()
    merged["delta"] = merged["blocks"] - merged["noblocks"]

    # Auto-detect URL filtering based on template name
    if 'bothbins' in template:
        if bothbins_urls is None:
            bothbins_urls = load_url_set("/Users/abha4861/dolma/data/interim/R2/cleaning/verified_bothbins_urls.txt")
        merged = merged[merged['doc_id'].isin(bothbins_urls)].copy()
    elif 'excluded-docs' in template:
        if excluded_urls is None:
            excluded_urls = load_url_set("/Users/abha4861/dolma/data/interim/R2/cleaning/verified_excluded_urls.txt")
        merged = merged[merged['doc_id'].isin(excluded_urls)].copy()

    return merged


def load_url_set(filepath: str) -> set:
    """
    Load URLs from a text file into a set.

    Args:
        filepath: Path to text file with one URL per line

    Returns:
        Set of URLs with whitespace stripped
    """
    with open(filepath) as f:
        return set(line.strip() for line in f)


def print_mean_delta_by_method(df: pd.DataFrame) -> None:
    """
    Print mean delta grouped by method.

    Args:
        df: DataFrame with 'method' and 'delta' columns
    """
    print(df[["method", "delta"]].groupby("method").mean().reset_index())


def compare_two_distributions(group1: np.ndarray, group2: np.ndarray) -> dict:
    """
    Compare two distributions using statistical tests (pure function).

    Args:
        group1: Array of values for first group
        group2: Array of values for second group

    Returns:
        Dictionary with statistical comparison results
    """
    # Mann-Whitney U test (non-parametric test for difference in distributions)
    mw_result = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    # Independent samples t-test
    t_result = stats.ttest_ind(group1, group2, alternative='two-sided')

    # Calculate statistics
    return {
        'group1_mean': np.mean(group1),
        'group2_mean': np.mean(group2),
        'group1_median': np.median(group1),
        'group2_median': np.median(group2),
        'n_group1': len(group1),
        'n_group2': len(group2),
        't_statistic': t_result.statistic,
        't_pvalue': t_result.pvalue,
        'mw_statistic': mw_result.statistic,
        'mw_pvalue': mw_result.pvalue,
        't_significant': t_result.pvalue < 0.05,
        'mw_significant': mw_result.pvalue < 0.05
    }


def compare_att_vs_atu(att_df: pd.DataFrame, atu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare ATT vs ATU distributions using statistical tests.

    Args:
        att_df: DataFrame with ATT data (must have 'method' and 'delta' columns)
        atu_df: DataFrame with ATU data (must have 'method' and 'delta' columns)

    Returns:
        DataFrame with comparison statistics for each method
    """
    # Get unique methods
    unique_methods = sorted(set(att_df['method'].unique()) & set(atu_df['method'].unique()))

    results = []
    for method in unique_methods:
        att_deltas = att_df[att_df['method'] == method]['delta'].values
        atu_deltas = atu_df[atu_df['method'] == method]['delta'].values

        # Use pure function for statistical comparison
        stats_result = compare_two_distributions(att_deltas, atu_deltas)

        # Add method name and rename keys to match expected output
        results.append({
            'method': method,
            'att_mean': stats_result['group1_mean'],
            'atu_mean': stats_result['group2_mean'],
            'att_median': stats_result['group1_median'],
            'atu_median': stats_result['group2_median'],
            'n_att': stats_result['n_group1'],
            'n_atu': stats_result['n_group2'],
            't_statistic': stats_result['t_statistic'],
            't_pvalue': stats_result['t_pvalue'],
            'mw_statistic': stats_result['mw_statistic'],
            'mw_pvalue': stats_result['mw_pvalue'],
            't_significant': stats_result['t_significant'],
            'mw_significant': stats_result['mw_significant']
        })

    return pd.DataFrame(results)


#
#
#              ,d      ,d
#              88      88
# ,adPPYYba, MM88MMM MM88MMM
# ""     `Y8   88      88
# ,adPPPPP88   88      88
# 88,    ,88   88,     88,
# `"8bbdP"Y8   "Y888   "Y888
#
#


ATT = load_MIA_scores('csvs/confounddataset/excluded-docs.{}.lite.all_shards.csv.gz')
ATT = ATT.sample(n=50_000, random_state=42)

ATT.to_csv("/tmp/att.csv", index=False)

print_mean_delta_by_method(ATT)



#                                                             
#          88                                 88          88  
#          88                                 88          88  
#          88                                 88          88  
#  ,adPPYb,88  ,adPPYba, 8b,dPPYba,   ,adPPYb,88  ,adPPYb,88  
# a8"    `Y88 a8"     "" 88P'    "8a a8"    `Y88 a8"    `Y88  
# 8b       88 8b         88       d8 8b       88 8b       88  
# "8a,   ,d88 "8a,   ,aa 88b,   ,a8" "8a,   ,d88 "8a,   ,d88  
#  `"8bbdP"Y8  `"Ybbd8"' 88`YbbdP"'   `"8bbdP"Y8  `"8bbdP"Y8  
#                        88
#                        88

ATT = load_MIA_scores('csvs/confounddataset/excluded-docs.{}.dcpdd.all_shards.csv.gz')
ATT = ATT.sample(n=50_000, random_state=42)
dcp = ATT[ATT["method"] != "loss"].copy()
print_mean_delta_by_method(dcp)


#                                 
#                                 
#              ,d                 
#              88                 
# ,adPPYYba, MM88MMM 88       88  
# ""     `Y8   88    88       88  
# ,adPPPPP88   88    88       88  
# 88,    ,88   88,   "8a,   ,a88  
# `"8bbdP"Y8   "Y888  `"YbbdP'Y8  
#                                 
#                               

print("[*] ATU")

ATU = load_MIA_scores('csvs/confounddataset/bothbins.{}.lite.all_shards.csv.gz')
ATU = ATU.sample(n=50_000, random_state=42)

ATU.to_csv("/tmp/atu.csv", index=False)

print_mean_delta_by_method(ATU)

ATU = load_MIA_scores('csvs/confounddataset/bothbins.{}.dcpdd.all_shards.csv.gz')
ATU = ATU.sample(n=50_000, random_state=42)
dcp = ATU[ATU["method"] != "loss"].copy()

print_mean_delta_by_method(dcp)


#
#
#              ,d      ,d                                            ,d
#              88      88                                            88
# ,adPPYYba, MM88MMM MM88MMM    8b       d8 ,adPPYba,      ,adPPYYba, MM88MMM 88       88
# ""     `Y8   88      88       `8b     d8' I8[    ""      ""     `Y8   88    88       88
# ,adPPPPP88   88      88        `8b   d8'   `"Y8ba,       ,adPPPPP88   88    88       88
# 88,    ,88   88,     88,        `8b,d8'   aa    ]8I      88,    ,88   88,   "8a,   ,a88
# `"8bbdP"Y8   "Y888   "Y888        "8"     `"YbbdP"'      `"8bbdP"Y8   "Y888  `"YbbdP'Y8
#
#

# print("\n" + "="*80)
# print("ATT vs ATU Comparison: Non-parametric tests by method")
# print("="*80 + "\n")

results_df = compare_att_vs_atu(ATT, ATU)
print(results_df.to_string(index=False))
#print("\n" + "="*80)
#print(f"T-test significant differences (p < 0.05): {results_df['t_significant'].sum()} out of {len(results_df)} methods")
#print(f"Mann-Whitney significant differences (p < 0.05): {results_df['mw_significant'].sum()} out of {len(results_df)} methods")
#print("="*80)



#                                   
#            88 88            ad88  
#            88 88           d8"    
#            88 88           88     
# 8b,dPPYba, 88 88,dPPYba, MM88MMM  
# 88P'   "Y8 88 88P'    "8a  88     
# 88         88 88       88  88     
# 88         88 88       88  88     
# 88         88 88       88  88     
#                                   
#                                   
print("[*] RLHF")
both = load_MIA_scores('csvs/confounddataset/excluded-docs.{}.rlhf.lite.all_shards.csv.gz')

print_mean_delta_by_method(both)
