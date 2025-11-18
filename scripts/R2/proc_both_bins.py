import pandas as pd
from scipy import stats

noblocks = pd.read_csv("bothbins.noblocks.lite.csv").rename(columns={"score": "noblocks_score"})
blocks = pd.read_csv("bothbins.blocks.lite.csv").rename(columns={"score": "blocks_score"})

D = noblocks.merge(blocks, on=["doc_id", "method", "membership"])
D = D[D["membership"] == "member"].copy()

D["delta"] = D["blocks_score"] - D["noblocks_score"]

for method in D["method"].unique():
    method_data = D[D["method"] == method]["delta"]
    
    # One-sample t-test against 0
    t_stat, p_value = stats.ttest_1samp(method_data, 0)
    
    print(f"\nMethod: {method}")
    print(f"Mean delta: {method_data.mean():.4f}")
    print(f"Sd : {method_data.std():.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

    cohen_d = method_data.mean() / method_data.std()
    print(f"Cohen's d: {cohen_d:.4f}")

print(D[["method", "delta"]].groupby("method").mean())
