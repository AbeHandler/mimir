#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from scipy import stats
import numpy as np
import json
import pandas as pd
import os
from sigfig import round as sf_round

# update the all_shards docs to ensure you get the latest versions
# os.system("python scripts/merge_shards.py -d csvs/confounddataset")
# update the all_shards docs to ensure you get the latest versions
# os.system("python scripts/merge_shards.py -d csvs")

def _load_shards(pattern_template, skip_shards, score_col):
    """Load and concat shards, filtering to members only."""
    dfs = []
    for n in range(1, 17):
        if n not in skip_shards:
            dfs.append(pd.read_csv(pattern_template.format(n)))
    df = pd.concat(dfs)
    df = df[df["membership"] == "member"].copy()
    df = df.drop(columns=["membership"])
    return df.rename(columns={"score": score_col})




def compute_delta_8B(fileclass: str) -> pd.DataFrame:
    y0_path = f'csvs/Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.{fileclass}.lite.all_shards.csv.gz'
    y0 = pd.read_csv(y0_path)
    y0 = y0[y0["membership"] == "member"].copy().rename(columns={"score": "noblocks"}).drop(columns=["membership"])

    y1_path = f'csvs/Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.{fileclass}.lite.all_shards.csv.gz'
    y1 = pd.read_csv(y1_path)
    y1 = y1[y1["membership"] == "member"].copy().rename(columns={"score": "blocks"}).drop(columns=["membership"])

    merged = y0.merge(y1, on=["doc_id", "method"])
    merged["delta"] = merged["blocks"] - merged["noblocks"]
    merged["template"] = f"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-X.{fileclass}.lite.all_shards.csv.gz"
    return merged





def _load_70b_condition(dataset, skip_shards):
    """Load Y0/Y1 shards for one dataset condition, merge, and compute delta."""
    base = "csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30"
    y0 = _load_shards(f"{base}-Y0.{dataset}.lite.shard_{{}}.csv", skip_shards, "noblocks")
    y1 = _load_shards(f"{base}-Y1.{dataset}.lite.shard_{{}}.csv", skip_shards, "blocks")
    merged = y1.merge(y0, on=["doc_id", "method"])
    merged["delta"] = merged["blocks"] - merged["noblocks"]
    merged["method"] = "70b-" + merged["method"]
    return merged


def load_llama_70b():
    conditions = [
        ("bothbins", {8, 9, 12}),
        ("excluded", {8, 9, 12, 13, 15, 16}),
    ]
    for dataset, skip_shards in conditions:
        merged = _load_70b_condition(dataset, skip_shards)
        print(f'[*] {dataset}')
        print(merged[["method", "delta"]].groupby("method").mean())



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
        Merged dataframe with 'blocks', 'noblocks', 'delta', and 'template' columns.
        Template column contains filename with {} replaced by X (e.g., 'excluded-docs.X.lite.all_shards.csv.gz')
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

    # Add template column: extract filename and replace {} with X
    template_name = template.split("/")[-1].replace("{}", "X")
    merged["template"] = template_name

    if len(merged) > 50000: # may happen when stuff is running
        merged = merged.sample(n=50_000, random_state=42)
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


def process_scores(df: pd.DataFrame) -> list:
    """
    Generate JSONL lines for mean delta grouped by method.
    Automatically detects ATT/ATU label from template column.
    Adds +RLHF suffix to method if template contains .rlhf.

    Args:
        df: DataFrame with 'method', 'delta', and 'template' columns

    Returns:
        List of JSON strings (one per method)
    """
    # Get template to determine label
    template = df['template'].iloc[0]
    if 'bothbins' in template:
        label = 'ATU'
    elif 'excluded' in template:
        label = 'ATT'
    else:
        label = 'unknown'

    result = df[["method", "delta"]].groupby("method").agg(
        delta=("delta", "mean"),
        count=("delta", "count")
    ).reset_index()
    result["delta"] = result["delta"].apply(lambda x: sf_round(x, sigfigs=2))

    # Add +RLHF suffix if template contains .rlhf.
    if '.rlhf.' in template:
        result['method'] = result['method'] + '+RLHF'

    lines = []
    for _, row in result.iterrows():
        record = row.to_dict()
        record['label'] = label
        record['template'] = template
        if "Llama-3.1-8B-Instruct" in template:
            record["test_case"] = "CPT-8B"
        else:
            record["test_case"] = template_to_case[template.split("X.").pop()]

        lines.append(pd.Series(record).to_json())

    return lines


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

if __name__ == "__main__":

    template_to_case = {
        'rlhf.lite.all_shards.csv.gz': 'PT+rlhf',
        'rlhf.dcpdd.lite.all_shards.csv.gz': 'PT+rlhf',
        'rlhf.clipped.all_shards.csv.gz': 'PT+rlhf',
        'clipped.all_shards.csv.gz': 'PT',
        'lite.all_shards.csv.gz': 'PT',
        'dcpdd.all_shards.csv.gz': 'PT',
        "cloze.all_shards.csv.gz": "PT",
        "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-X.bothbins.lite.all_shards.csv.gz": "CPT-8B",
        "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-X.excluded.lite.all_shards.csv.gz": "CPT-8B",
        'bisection.k10.all_shards.csv.gz': "PT",
        'rlhf.bisection.k10.all_shards.csv.gz': "PT+rlhf"
    }

    template_patterns = [
        'csvs/confounddataset/{}.{}.rlhf.lite.all_shards.csv.gz',
        'csvs/confounddataset/{}.{}.clipped.all_shards.csv.gz', # this gets clipped and skipped
        'csvs/confounddataset/{}.{}.lite.all_shards.csv.gz',
        'csvs/confounddataset/{}.{}.cloze.all_shards.csv.gz',
        'csvs/confounddataset/{}.{}.dcpdd.all_shards.csv.gz',
        'csvs/confounddataset/{}.{}.rlhf.dcpdd.lite.all_shards.csv.gz',
        'csvs/confounddataset/{}.{}.rlhf.clipped.all_shards.csv.gz',
        'csvs/confounddataset/{}.{}.bisection.k10.all_shards.csv.gz',
        'csvs/confounddataset/{}.{}.rlhf.bisection.k10.all_shards.csv.gz'
    ]
    templates = []
    for base in ['bothbins', 'excluded-docs']:
        for template in template_patterns:
            templates.append(template.format(base, '{}'))

    all_results = []

    for template in templates:
        scores = load_MIA_scores(template)
        if ".dcpdd." in template or ".clipped." in template or ".skipped." in template:
            scores = scores[scores["method"] != "loss"].copy()
        all_results.extend(process_scores(scores))

    # 8B test case
    for fileclass in ["bothbins", "excluded"]:
        df = compute_delta_8B(fileclass)
        all_results.extend(process_scores(df))


    # Write all results to JSONL file
    output_file = "results/ate_vs_atu_for_R2/mean_delta_by_method.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        all_results.sort(key=lambda x: json.loads(x)["test_case"])
        for line in all_results:
            f.write(line + '\n')

    print(f"Results written to {output_file}")
