"""
Process confound dataset by combining sharded CSVs and merging with HuggingFace dataset.

This script:
1. Loads all CSV shards from csvs/confounddataset/
2. Concatenates them into one DataFrame
3. Merges with abehandlerorg/confounddataset on id (doc_id -> url)
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path


def load_local_csvs(csv_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSV files from the directory."""
    csv_files = sorted(csv_dir.glob("*shard*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    print(f"Found {len(csv_files)} CSV files")

    dfs = []
    for csv_file in csv_files:
        print(f"  Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined DataFrame: {len(combined_df)} rows")

    # Drop membership column and deduplicate
    if "membership" in combined_df.columns:
        combined_df = combined_df.drop(columns=["membership"])
        print("Dropped 'membership' column")
    combined_df = combined_df.drop_duplicates()
    print(f"After deduplication: {len(combined_df)} rows")

    return combined_df


def load_hf_dataset() -> pd.DataFrame:
    """Load the HuggingFace dataset and convert to DataFrame."""
    print("Loading HuggingFace dataset: abehandlerorg/confounddataset...")
    ds = load_dataset("abehandlerorg/confounddataset", split="train")
    hf_df = ds.to_pandas()
    print(f"HuggingFace dataset: {len(hf_df)} rows")

    return hf_df


def merge_datasets(local_df: pd.DataFrame, hf_df: pd.DataFrame) -> pd.DataFrame:
    """Merge local CSV data with HuggingFace dataset on id (doc_id -> url)."""
    print("Merging datasets on doc_id -> url...")

    merged_df = local_df.merge(
        hf_df,
        left_on="doc_id",
        right_on="url",
        how="left"
    )

    # Report merge statistics
    matched = merged_df["url"].notna().sum()
    unmatched = merged_df["url"].isna().sum()
    print(f"Merged DataFrame: {len(merged_df)} rows")
    print(f"  Matched: {matched} ({matched/len(merged_df)*100:.1f}%)")
    print(f"  Unmatched: {unmatched} ({unmatched/len(merged_df)*100:.1f}%)")

    return merged_df


def main(csv_dir):
    local_df = load_local_csvs(csv_dir)
    hf_df = load_hf_dataset()
    merged_df = merge_datasets(local_df, hf_df)
    return merged_df


def analysis():

    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as smf
    from scipy import stats
    from tqdm import tqdm as tqdm

    # Read the CSV
    df = pd.read_csv("csvs/confounddataset/full.csv", dtype={
        'in_blocksbin': 'int8',
        'in_noblocksbin': 'int8',
        'group': 'category',
        'method': 'category',
        'source_domain': 'category',
        'tag': 'category'
    })

    df = df[df['method'] == 'loss'].copy()

    #                                                                                                                
    # 88          88                                                        88          88                       88  
    # 88          ""                                                        88          ""                       88  
    # 88                                                                    88                                   88  
    # 88,dPPYba,  88  ,adPPYba, 8b,dPPYba, ,adPPYYba, 8b,dPPYba,  ,adPPYba, 88,dPPYba,  88  ,adPPYba, ,adPPYYba, 88  
    # 88P'    "8a 88 a8P_____88 88P'   "Y8 ""     `Y8 88P'   "Y8 a8"     "" 88P'    "8a 88 a8"     "" ""     `Y8 88  
    # 88       88 88 8PP""""""" 88         ,adPPPPP88 88         8b         88       88 88 8b         ,adPPPPP88 88  
    # 88       88 88 "8b,   ,aa 88         88,    ,88 88         "8a,   ,aa 88       88 88 "8a,   ,aa 88,    ,88 88  
    # 88       88 88  `"Ybbd8"' 88         `"8bbdP"Y8 88          `"Ybbd8"' 88       88 88  `"Ybbd8"' `"8bbdP"Y8 88  
    #                                                                                                                
    #                                                                                                                

    model = smf.mixedlm("score ~ 1", df, groups=df["source_domain"])
    result = model.fit()

    print(result.summary())

    #                                                                                          
    #                                                                                          
    #                                                       ,d                          ,d     
    #                                                       88                          88     
    # 8b,dPPYba,   ,adPPYba, 8b,dPPYba, 88,dPYba,,adPYba, MM88MMM ,adPPYba, ,adPPYba, MM88MMM  
    # 88P'    "8a a8P_____88 88P'   "Y8 88P'   "88"    "8a  88   a8P_____88 I8[    ""   88     
    # 88       d8 8PP""""""" 88         88      88      88  88   8PP"""""""  `"Y8ba,    88     
    # 88b,   ,a8" "8b,   ,aa 88         88      88      88  88,  "8b,   ,aa aa    ]8I   88,    
    # 88`YbbdP"'   `"Ybbd8"' 88         88      88      88  "Y888 `"Ybbd8"' `"YbbdP"'   "Y888  
    # 88                                                                                       
    # 88                                                                                       


    # Permutation test focusing on variance across domain means
    import numpy as np

    def test_statistic(df):
        domain_means = df.groupby('source_domain')['score'].mean()
        return domain_means.var()  # or .std()

    obs_stat = test_statistic(df)

    # Permutation
    n_perm = 10000
    perm_stats = []
    for _ in tqdm(range(n_perm)):
        df_perm = df.copy()
        df_perm['score'] = np.random.permutation(df_perm['score'])
        perm_stats.append(test_statistic(df_perm))
    p_value = (np.array(perm_stats) >= obs_stat).mean()
    print(p_value)



if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    csv_dir = project_root / "csvs" / "confounddataset"
    merged_df = main(csv_dir)
    print("\nFinal DataFrame columns:")
    print(merged_df.columns.tolist())
    print(f"\nShape: {merged_df.shape}")
    outf = csv_dir / "full.csv"
    merged_df.to_csv(outf)
