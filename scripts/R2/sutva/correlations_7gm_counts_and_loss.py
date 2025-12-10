#!/usr/bin/env python3
"""Analyze correlation between ngram counts and loss results."""

from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, pearsonr


def main():
    # Load ngram counts. See dolma/7gm for how to get this file 
    ngram_counts = pd.read_csv(
        "data/exports/sutva_click2houston_com_2022-05-01_ngram_matrix.csv"
    ).rename(columns={"text": "ngram"})

    base_dir = Path("server_files")
    results = []

    for path in base_dir.rglob("*loss_results*csv"):
        if "take" in path.as_posix():
            continue

        print(path)
        ngrams = pd.read_csv(path)

        # Extract corpus count name from path
        corpus_count_name = (
            path.as_posix()
            .replace("server_files/home/abe/mimir/tmp_results/", "")
            .split("/")[0]
            .replace("_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered", "")
        )
        print(corpus_count_name)

        print(f"Before merge: {len(ngrams)}")
        ngrams = ngrams.merge(ngram_counts, on="ngram")
        print(f"After merge: {len(ngrams)}")

        tau_result = kendalltau(ngrams[corpus_count_name], ngrams['loss'])
        pearson_result = pearsonr(ngrams[corpus_count_name], ngrams['loss'])

        print(f"Kendall tau: {tau_result}")
        print(f"Pearson r: {pearson_result}")
        print()

        # Store results
        results.append({
            'corpus_count_name': corpus_count_name,
            'tau': tau_result.statistic,
            'tau_pvalue': tau_result.pvalue,
            'rho': pearson_result.statistic,
            'rho_pvalue': pearson_result.pvalue
        })

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)

    # Ensure results directory exists
    output_path = Path("results")
    output_path.mkdir(exist_ok=True)

    # Write to CSV
    output_file = output_path / "7gmcountxloss.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"\nSummary:\n{results_df}")

def replace_name(run):
    if run == "run1":
        return "$\\mathcal{M}^{\\bm{D}}_{Y^1}$"
    if run == "run3":
        return "$\\mathcal{M}^{\\bm{D}'}_{Y^1}$"
    if run == "run4":
        return "$\\mathcal{M}^{\\neg \\bm{D}}_{Y^0}$"

def print_as_booktabs():
    import pandas as pd
    df = pd.read_csv("results/7gmcountxloss.csv")
    
    assert (df["tau_pvalue"] < 10**-3).all()
    assert (df["rho_pvalue"] < 10**-3).all()
    
    # Extract run number from corpus_count_name
    df["run"] = df["corpus_count_name"].str.extract(r"(run\d+)")

    df["run"] = df["run"].apply(replace_name)
    
    # Format for booktabs
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Correlation between 7-gram counts and loss}")
    print(r"\label{tab:7gmcountxloss}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Model & $\tau$ & $\rho$ \\")
    print(r"\midrule")
    for _, row in df.iterrows():
        print(f"{row['run']} & {row['tau']:.3f} & {row['rho']:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    #main()
    print_as_booktabs()