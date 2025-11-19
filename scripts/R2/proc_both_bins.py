import os
import pandas as pd
from scipy import stats
from datasets import load_dataset
from scipy.stats import wilcoxon

def load_both_bins_with_minhash_count(
    local_csv_path: str = "bothbins.noblocks.lite.csv",
    hf_dataset_name: str = "abehandlerorg/bothbins",
    output_path: str = None,
) -> pd.DataFrame:
    """
    Add minhash_count from HuggingFace dataset to local CSV file.

    Args:
        local_csv_path: Path to the local CSV file with doc_id field
        hf_dataset_name: Name of the HuggingFace dataset to load
        output_path: Path to save the updated CSV (if None, overwrites local_csv_path)

    Returns:
        DataFrame with minhash_count added
    """
    # Load local CSV
    print(f"Loading local CSV from {local_csv_path}...")
    local_df = pd.read_csv(local_csv_path)
    print(f"Local CSV shape: {local_df.shape}")
    print(f"Columns: {list(local_df.columns)}")

    # Load HuggingFace dataset
    print(f"\nLoading HuggingFace dataset: {hf_dataset_name}...")
    hf_dataset = load_dataset(hf_dataset_name, split="train")

    # Convert to pandas DataFrame and select relevant columns
    hf_df = hf_dataset.to_pandas()[["url", "minhash_count"]]
    print(f"HF dataset shape: {hf_df.shape}")
    print(f"Columns: {list(hf_df.columns)}")

    # Merge on doc_id (local) = url (HF)
    print(f"\nMerging on doc_id = url...")
    merged_df = local_df.merge(
        hf_df,
        left_on="doc_id",
        right_on="url",
        how="left"
    ).drop(columns=["url"]).rename(columns={"minhash_count":"size"})

    return merged_df

noblocks = load_both_bins_with_minhash_count().rename(columns={"score": "noblocks_score"})
blocks = pd.read_csv("bothbins.blocks.lite.csv").rename(columns={"score": "blocks_score"})

D = noblocks.merge(blocks, on=["doc_id", "method", "membership"])

D = D[~D["size"].isna()].copy()

D["size"] = D["size"].astype(int)

D = D[D["membership"] == "member"].copy()

D = D.sample(n=2500, random_state=42)

D["delta"] = D["blocks_score"] - D["noblocks_score"]

D["size_bin"] = pd.cut(D["size"], bins=range(0, 50, 5), right=True)


for method in D["method"].unique():
    method_data = D[D["method"] == method]

    df = method_data.groupby("size_bin", observed=True).agg(
            delta_mean=("delta", "mean"),
            count=("delta", "count")
        ).reset_index()
    df['method'] = method

    stat, p = wilcoxon(method_data["delta"].to_list(), alternative="greater")
    print(f"{method_data['delta'].mean():.3g}", method, p/7)

    df.to_csv(f"data/interim/bothbins/{method}.csv", index=False)

os.system("Rscript scripts/R2/plot_all_methods_line_chart_both_bins.R")