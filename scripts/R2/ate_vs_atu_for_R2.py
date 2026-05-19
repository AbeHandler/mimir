#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from scipy import stats
import numpy as np
import json
import pandas as pd
import argparse
import os
import glob
import re
import sys
from pathlib import Path
from sigfig import round as sf_round

# update the all_shards docs to ensure you get the latest versions
# os.system("python scripts/merge_shards.py -d csvs/confounddataset")
# update the all_shards docs to ensure you get the latest versions
# os.system("python scripts/merge_shards.py -d csvs")


def push():
    # https://dash.cloudflare.com/1d736c1e8da83d40f1eda75419d90b86/r2/default/buckets/misqsi
    # sync needed files to cloudflare
    path_to = ["/Users/abha4861/mimir/csvs", "/Users/abha4861/mimir/csvs/confounddataset/"]
    for path in path_to:
        prefix = path.lstrip("/").rstrip("/")
        end_point = 'https://1d736c1e8da83d40f1eda75419d90b86.r2.cloudflarestorage.com'
        sync_csvs = f'''aws s3 sync {path} s3://misqsi/{prefix} --endpoint-url {end_point} --profile r2 --size-only --exclude "*" --include "*all_shards.csv.gz"'''
        os.system(sync_csvs)

    path = "/Users/abha4861/dolma/data/interim/R2/cleaning/"
    end_point = 'https://1d736c1e8da83d40f1eda75419d90b86.r2.cloudflarestorage.com'
    prefix = path.lstrip("/").rstrip("/")
    sync_csvs = f'''aws s3 sync {path} s3://misqsi/{prefix} --endpoint-url {end_point} --profile r2 --size-only --exclude "*" --include "*verified_*"'''
    os.system(sync_csvs)


def pull(dir: str = None):
    # pull from cloudflare the files that push() uploads
    # local destination is relative to CWD (or `dir` if provided)
    base = os.path.abspath(dir) if dir else os.getcwd()
    end_point = 'https://1d736c1e8da83d40f1eda75419d90b86.r2.cloudflarestorage.com'

    pulls = [
        ("Users/abha4861/mimir/csvs", "csvs", "*all_shards.csv.gz"),
        ("Users/abha4861/mimir/csvs/confounddataset", "csvs/confounddataset", "*all_shards.csv.gz"),
        ("Users/abha4861/dolma/data/interim/R2/cleaning", "data/interim/R2/cleaning", "*verified_*"),
    ]
    for remote_prefix, local_sub, include in pulls:
        local = os.path.join(base, local_sub)
        os.makedirs(local, exist_ok=True)
        cmd = f'''aws s3 sync s3://misqsi/{remote_prefix} {local} --endpoint-url {end_point} --profile r2 --size-only --exclude "*" --include "{include}"'''
        os.system(cmd)

def consolidate_llama_files(
    base: str = "csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30",
    datasets: tuple = ("bothbins", "excluded"),
    arms: tuple = ("Y0", "Y1"),
    suffix: str = "lite",
) -> None:
    """Consolidate per-shard llama CSVs into one all_shards.csv.gz per (arm, dataset).

    Globs for `{base}-{arm}.{dataset}.{suffix}.shard_*.csv` and writes
    `{base}-{arm}.{dataset}.{suffix}.all_shards.csv.gz`. Clobbers existing output.
    """
    import gzip
    for arm in arms:
        for dataset in datasets:
            stem = f"{base}-{arm}.{dataset}.{suffix}"
            out_path = f"{stem}.all_shards.csv.gz"
            shard_paths = sorted(
                glob.glob(f"{stem}.shard_*.csv"),
                key=lambda p: int(re.search(r"shard_(\d+)\.csv$", p).group(1)),
            )
            if not shard_paths:
                raise FileNotFoundError(f"no shards found for {stem}")
            df = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
            with gzip.GzipFile(out_path, mode="wb", mtime=0) as gz:
                df.to_csv(gz, index=False)
            print(out_path)




def load_8b_cloze(fileclass: str) -> pd.DataFrame:
    """Load llama8b cloze blocks/noblocks, merge, compute delta."""
    noblocks_path = f"csvs/llama8b.{fileclass}.noblocks.cloze.all_shards.csv.gz"
    blocks_path = f"csvs/llama8b.{fileclass}.blocks.cloze.all_shards.csv.gz"

    noblocks = pd.read_csv(noblocks_path).rename(columns={"score": "noblocks"})
    blocks = pd.read_csv(blocks_path).rename(columns={"score": "blocks"})

    merged = noblocks.merge(blocks, on=["doc_id", "method", "membership"])
    merged = merged[merged["membership"] == "member"].copy()
    merged = merged.drop_duplicates()
    merged["delta"] = merged["blocks"] - merged["noblocks"]

    if fileclass not in ("bothbins", "excluded"):
        raise ValueError(f"unknown fileclass {fileclass}")

    merged["template"] = f"llama8b.{fileclass}.X.cloze.all_shards.csv.gz"
    return merged


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



def compute_delta_8B_clipped(fileclass: str) -> pd.DataFrame:
    y0_path = f'csvs/Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.{fileclass}.clipped.lite.all_shards.csv.gz'
    y0 = pd.read_csv(y0_path)
    y0 = y0[y0["membership"] == "member"].copy().rename(columns={"score": "noblocks"}).drop(columns=["membership"])

    y1_path = f'csvs/Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.{fileclass}.clipped.lite.all_shards.csv.gz'
    y1 = pd.read_csv(y1_path)
    y1 = y1[y1["membership"] == "member"].copy().rename(columns={"score": "blocks"}).drop(columns=["membership"])

    merged = y0.merge(y1, on=["doc_id", "method"])
    merged["delta"] = merged["blocks"] - merged["noblocks"]
    merged["template"] = f"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-X.{fileclass}.clipped.lite.all_shards.csv.gz"
    return merged


def compute_delta_70B_clipped(fileclass: str) -> pd.DataFrame:
    y0_path = f'csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.{fileclass}.clipped.lite.all_shards.csv.gz'
    y0 = pd.read_csv(y0_path)
    y0 = y0[y0["membership"] == "member"].copy().rename(columns={"score": "noblocks"}).drop(columns=["membership"])

    y1_path = f'csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.{fileclass}.clipped.lite.all_shards.csv.gz'
    y1 = pd.read_csv(y1_path)
    y1 = y1[y1["membership"] == "member"].copy().rename(columns={"score": "blocks"}).drop(columns=["membership"])

    merged = y0.merge(y1, on=["doc_id", "method"])
    merged["delta"] = merged["blocks"] - merged["noblocks"]
    merged["template"] = f"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-X.{fileclass}.clipped.lite.all_shards.csv.gz"
    return merged


def compute_delta_70B_dcpdd(fileclass: str) -> pd.DataFrame:
    '''
    csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.bothbins.dcpdd.lite.all_shards.csv.gz
    csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.excluded.dcpdd.lite.all_shards.csv.gz
    csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.bothbins.dcpdd.lite.all_shards.csv.gz
    csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.excluded.dcpdd.lite.all_shards.csv.gz
    '''
    y0_path = f'csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.{fileclass}.dcpdd.lite.all_shards.csv.gz'
    y0 = pd.read_csv(y0_path)
    y0 = y0[y0["membership"] == "member"].copy().rename(columns={"score": "noblocks"}).drop(columns=["membership"])

    y1_path = f'csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.{fileclass}.dcpdd.lite.all_shards.csv.gz'
    y1 = pd.read_csv(y1_path)
    y1 = y1[y1["membership"] == "member"].copy().rename(columns={"score": "blocks"}).drop(columns=["membership"])

    merged = y0.merge(y1, on=["doc_id", "method"])
    merged["delta"] = merged["blocks"] - merged["noblocks"]
    merged["template"] = f"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-X.{fileclass}.lite.all_shards.csv.gz"
    merged = merged[merged["method"] == 'dc_pdd'].copy()
    return merged


def compute_delta_8B_bisection(fileclass: str) -> pd.DataFrame:
    y0_path = f'csvs/Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y0.{fileclass}.bisection.lite.k10.all_shards.csv.gz'
    y0 = pd.read_csv(y0_path)
    y0 = y0[y0["membership"] == "member"].copy().rename(columns={"score": "noblocks"}).drop(columns=["membership"])

    y1_path = f'csvs/Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.{fileclass}.bisection.lite.k10.all_shards.csv.gz'
    y1 = pd.read_csv(y1_path)
    y1 = y1[y1["membership"] == "member"].copy().rename(columns={"score": "blocks"}).drop(columns=["membership"])

    merged = y0.merge(y1, on=["doc_id", "method"])
    merged["delta"] = merged["blocks"] - merged["noblocks"]
    merged["template"] = f"Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-X.{fileclass}.bisection.lite.k10.all_shards.csv.gz"
    return merged


def compute_delta_70B_bisection(fileclass: str) -> pd.DataFrame:
    y0_path = f'csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y0.{fileclass}.bisection.lite.k10.all_shards.csv.gz'
    y0 = pd.read_csv(y0_path)
    y0 = y0[y0["membership"] == "member"].copy().rename(columns={"score": "noblocks"}).drop(columns=["membership"])

    y1_path = f'csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-Y1.{fileclass}.bisection.lite.k10.all_shards.csv.gz'
    y1 = pd.read_csv(y1_path)
    y1 = y1[y1["membership"] == "member"].copy().rename(columns={"score": "blocks"}).drop(columns=["membership"])

    merged = y0.merge(y1, on=["doc_id", "method"])
    merged["delta"] = merged["blocks"] - merged["noblocks"]
    merged["template"] = f"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-X.{fileclass}.bisection.lite.k10.all_shards.csv.gz"
    return merged


def load_llama_70b(all_results):
    """Read pre-consolidated 70B all_shards.csv.gz files and print per-method delta means."""
    base = "csvs/Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30"
    for dataset in ("bothbins", "excluded"):
        y0 = pd.read_csv(f"{base}-Y0.{dataset}.lite.all_shards.csv.gz")
        y0 = y0[y0["membership"] == "member"].copy().rename(columns={"score": "noblocks"}).drop(columns=["membership"])
        y1 = pd.read_csv(f"{base}-Y1.{dataset}.lite.all_shards.csv.gz")
        y1 = y1[y1["membership"] == "member"].copy().rename(columns={"score": "blocks"}).drop(columns=["membership"])
        merged = y1.merge(y0, on=["doc_id", "method"])
        merged["delta"] = merged["blocks"] - merged["noblocks"]
        merged["method"] = "70b-" + merged["method"]
        merged["template"] = f"Llama-3.3-70B-Instruct-bnb-4bit_cptllama-2024-01-30-to-2024-01-30-X.{dataset}.lite.all_shards.csv.gz"
        all_results.extend(process_scores(merged))


def load_MIA_scores(
    template: str,
    excluded_urls: set = None,
    bothbins_urls: set = None,
    path_to_clean: Path = Path("/Users/abha4861/dolma/data/interim/R2/cleaning"),
) -> pd.DataFrame:
    """
    Load MIA scores for blocks and noblocks treatments, merge, and calculate delta.
    Automatically filters by URL set based on template file name pattern.

    Args:
        template: Path template with {} placeholder for 'blocks'/'noblocks'
                 Example: 'csvs/confounddataset/excluded-docs.{}.lite.all_shards.csv.gz'
        excluded_urls: Set of excluded URLs (optional, auto-detected from template)
        bothbins_urls: Set of bothbins URLs (optional, auto-detected from template)
        path_to_clean: Directory containing verified_bothbins_urls.txt and
                       verified_excluded_urls.txt. Defaults to the dolma checkout.

    Returns:
        Merged dataframe with 'blocks', 'noblocks', 'delta', and 'template' columns.
        Template column contains filename with {} replaced by X (e.g., 'excluded-docs.X.lite.all_shards.csv.gz')
    """
    path_to_clean = Path(path_to_clean)
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
            bothbins_urls = load_url_set(path_to_clean / "verified_bothbins_urls.txt")
        merged = merged[merged['doc_id'].isin(bothbins_urls)].copy()
    elif 'excluded-docs' in template:
        if excluded_urls is None:
            excluded_urls = load_url_set(path_to_clean / "verified_excluded_urls.txt")
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
        elif "Llama-3.3-70B-Instruct" in template:
            record["test_case"] = "CPT-70B"
        else:
            record["test_case"] = template_to_case[template.split("X.").pop()]

        lines.append(pd.Series(record).to_json())

    return lines


def pretty_print_results(all_results: list) -> None:
    """Print a (test_case, method) -> (att, atu, n_att, n_atu) table from JSONL records.

    Rows colored green if att > atu, red otherwise.
    """
    GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
    rows = [json.loads(line) for line in all_results]
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index=["test_case", "method"],
        columns="label",
        values=["delta", "count"],
        aggfunc="first",
    )
    pivot.columns = [f"{v}_{l.lower()}" for v, l in pivot.columns]
    pivot = pivot.rename(columns={
        "delta_att": "att", "delta_atu": "atu",
        "count_att": "n_att", "count_atu": "n_atu",
    })
    pivot = pivot[[c for c in ("att", "atu", "n_att", "n_atu") if c in pivot.columns]].reset_index()

    header = f"{'test_case':<20} {'method':<25} {'att':>12} {'atu':>12} {'n_att':>8} {'n_atu':>8}"
    print(header)
    print("-" * len(header))
    for _, r in pivot.iterrows():
        att, atu = r.get("att"), r.get("atu")
        color = GREEN if pd.notna(att) and pd.notna(atu) and att > atu else RED
        att_s = f"{att:.4g}" if pd.notna(att) else "-"
        atu_s = f"{atu:.4g}" if pd.notna(atu) else "-"
        n_att_s = f"{int(r['n_att'])}" if pd.notna(r.get("n_att")) else "-"
        n_atu_s = f"{int(r['n_atu'])}" if pd.notna(r.get("n_atu")) else "-"
        print(f"{color}{r['test_case']:<20} {r['method']:<25} {att_s:>12} {atu_s:>12} {n_att_s:>8} {n_atu_s:>8}{RESET}")


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

def parse_args():
    p = argparse.ArgumentParser(description="ATE vs ATU analysis for R2.")
    p.add_argument(
        "-mode",
        choices=["push", "pull", "run"],
        default="run",
        help="push: upload local files to R2; pull: download from R2; run: analysis (default).",
    )
    p.add_argument(
        "-data-dir",
        default=None,
        help="Local destination dir for -mode pull (relative to CWD). Defaults to CWD.",
    )
    p.add_argument(
        "-path-to-clean",
        default="/Users/abha4861/dolma/data/interim/R2/cleaning",
        help="Directory containing verified_{bothbins,excluded}_urls.txt (used by -mode run).",
    )
    return p.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if args.mode == "push":
        consolidate_llama_files() # helpful for the push step. only need to push conslidated
        push()
        sys.exit(0)

    if args.mode == "pull":
        pull(dir=args.data_dir)
        sys.exit(0)

    

    for base in ['bothbins', 'excluded']:
        r = compute_delta_70B_dcpdd(base)
        print(base, "70b dcpdd", r["delta"].mean())

    template_to_case = {
        'rlhf.lite.all_shards.csv.gz': 'PT+rlhf',
        'rlhf.dcpdd.lite.all_shards.csv.gz': 'PT+rlhf',
        'rlhf.clipped.all_shards.csv.gz': 'PT+rlhf',
        'clipped.all_shards.csv.gz': 'PT',
        'lite.all_shards.csv.gz': 'PT',
        'dcpdd.all_shards.csv.gz': 'PT',
        "cloze.all_shards.csv.gz": "PT",
        'rlhf.cloze.all_shards.csv.gz': 'PT+rlhf',
        'bisection.k10.all_shards.csv.gz': "PT",
        'rlhf.bisection.k10.all_shards.csv.gz': "PT+rlhf"
    }

    template_patterns = [
        'csvs/confounddataset/{}.{}.rlhf.lite.all_shards.csv.gz',
        'csvs/confounddataset/{}.{}.clipped.all_shards.csv.gz', # this gets clipped and skipped
        'csvs/confounddataset/{}.{}.lite.all_shards.csv.gz',
        'csvs/confounddataset/{}.{}.cloze.all_shards.csv.gz',
        'csvs/confounddataset/{}.{}.rlhf.cloze.all_shards.csv.gz',
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
    load_llama_70b(all_results)

    for template in templates:
        scores = load_MIA_scores(template, path_to_clean=Path(args.path_to_clean))
        if ".dcpdd." in template or ".clipped." in template or ".skipped." in template:
            scores = scores[scores["method"] != "loss"].copy()
        all_results.extend(process_scores(scores))

    # 8B test case
    for fileclass in ["bothbins", "excluded"]:
        df = compute_delta_8B(fileclass)
        all_results.extend(process_scores(df))

    # 8B clipped test case
    for fileclass in ["bothbins", "excluded"]:
        df = compute_delta_8B_clipped(fileclass)
        df = df[df["method"] != "loss"].copy()
        all_results.extend(process_scores(df))

    # 70B clipped test case
    for fileclass in ["bothbins", "excluded"]:
        df = compute_delta_70B_clipped(fileclass)
        df = df[df["method"] != "loss"].copy()
        all_results.extend(process_scores(df))

    # 8B bisection test case
    for fileclass in ["bothbins", "excluded"]:
        df = compute_delta_8B_bisection(fileclass)
        all_results.extend(process_scores(df))

    # 70B bisection test case
    for fileclass in ["bothbins", "excluded"]:
        df = compute_delta_70B_bisection(fileclass)
        all_results.extend(process_scores(df))

    # llama8b cloze test case
    for fileclass in ["bothbins", "excluded"]:
        df = load_8b_cloze(fileclass)
        all_results.extend(process_scores(df))


    # Write all results to JSONL file
    output_file = "results/ate_vs_atu_for_R2/mean_delta_by_method.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        all_results.sort(key=lambda x: json.loads(x)["test_case"])
        for line in all_results:
            f.write(line + '\n')

    print(f"Results written to {output_file}")

    pretty_print_results(all_results)

    # === Significance testing: ATT vs ATU per test case ===
    sig_pairs = []  # list of (test_case_label, att_df, atu_df)

    # Confound dataset templates
    for pattern in template_patterns:
        att = load_MIA_scores(pattern.format('excluded-docs', '{}'), path_to_clean=Path(args.path_to_clean))
        atu = load_MIA_scores(pattern.format('bothbins', '{}'), path_to_clean=Path(args.path_to_clean))
        if ".dcpdd." in pattern or ".clipped." in pattern or ".skipped." in pattern:
            att = att[att["method"] != "loss"].copy()
            atu = atu[atu["method"] != "loss"].copy()
        sig_pairs.append((pattern, att, atu))

    # Paired loaders (mirror the blocks above)
    for label, loader, drop_loss in [
        ("8B",            compute_delta_8B,            False),
        ("8B_clipped",    compute_delta_8B_clipped,    True),
        ("70B_clipped",   compute_delta_70B_clipped,   True),
        ("8B_bisection",  compute_delta_8B_bisection,  False),
        ("70B_bisection", compute_delta_70B_bisection, False),
        ("8B_cloze",      load_8b_cloze,               False),
    ]:
        att = loader("excluded")
        atu = loader("bothbins")
        if drop_loss:
            att = att[att["method"] != "loss"].copy()
            atu = atu[atu["method"] != "loss"].copy()
        sig_pairs.append((label, att, atu))

    sig_lines = []
    for label, att_df, atu_df in sig_pairs:
        cmp_df = compare_att_vs_atu(att_df, atu_df)
        cmp_df["test_case"] = label
        for _, row in cmp_df.iterrows():
            sig_lines.append(row.to_json())

    sig_file = "results/ate_vs_atu_for_R2/att_vs_atu_significance.jsonl"
    with open(sig_file, 'w') as f:
        for line in sig_lines:
            f.write(line + '\n')
    print(f"Significance results written to {sig_file}")

    # === Studentized permutation test (randomization inference) ===
    from scripts.R2.randomization_inference import permutation_test_studentized

    perm_lines = []
    for label, att_df, atu_df in sig_pairs:
        shared_methods = sorted(set(att_df['method'].unique()) & set(atu_df['method'].unique()))
        for method in shared_methods:
            att_deltas = att_df[att_df['method'] == method]['delta'].values
            atu_deltas = atu_df[atu_df['method'] == method]['delta'].values
            S_obs, p_value, _ = permutation_test_studentized(
                att_deltas, atu_deltas, n_permutations=10000
            )
            perm_lines.append(json.dumps({
                "test_case": label,
                "method": method,
                "n_att": int(len(att_deltas)),
                "n_atu": int(len(atu_deltas)),
                "studentized_stat": float(S_obs),
                "perm_pvalue": float(p_value),
                "perm_significant": bool(p_value < 0.05),
            }))

    perm_file = "results/ate_vs_atu_for_R2/att_vs_atu_permutation.jsonl"
    with open(perm_file, 'w') as f:
        for line in perm_lines:
            f.write(line + '\n')
    print(f"Permutation results written to {perm_file}")

    # === Dump ATT / ATU per-doc deltas for density.R ===
    att_dump = load_MIA_scores('csvs/confounddataset/excluded-docs.{}.lite.all_shards.csv.gz', path_to_clean=Path(args.path_to_clean))
    att_dump.to_csv("/tmp/att.csv", index=False)
    print("Wrote /tmp/att.csv")

    atu_dump = load_MIA_scores('csvs/confounddataset/bothbins.{}.lite.all_shards.csv.gz', path_to_clean=Path(args.path_to_clean))
    atu_dump.to_csv("/tmp/atu.csv", index=False)
    print("Wrote /tmp/atu.csv")

    import subprocess
    density_script = os.path.join(os.path.dirname(__file__), "..", "..", "density.R")
    density_script = os.path.abspath(density_script)
    subprocess.run(["Rscript", density_script], check=True)
