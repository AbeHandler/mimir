import pandas as pd
from scipy.stats import wilcoxon

contaminated = pd.read_csv('csvs/confounddataset/pythia-45m_lr1e-3_steps5k_seed1234_interleave0.02_uncontaminated_insample_regular_training_data.all_shards.csv.gz').rename(columns={"score": "contaminated"})
uncontaminated = pd.read_csv('csvs/confounddataset/pythia-45m_lr1e-3_steps5k_seed1234_uncontaminated_insample_regular_training_data.all_shards.csv.gz').rename(columns={"score": "uncontaminated"})

contaminated = contaminated[contaminated["membership"] == "member"].copy()
uncontaminated = uncontaminated[uncontaminated["membership"] == "member"].copy()

both = contaminated.merge(uncontaminated, on =["method", "doc_id"])


both["delta"] = both["uncontaminated"] - both["contaminated"]  # uncontaminated - contaminated (to match ATT)

# Write ATU data for R plotting
both.to_csv("/tmp/atu_appendix.csv", index=False)

print("Mean delta by method:")
print(both[["delta", "method"]].groupby(["method"]).mean())
print("\n" + "="*80)
print("Wilcoxon signed-rank test (H0: delta = 0, H1: delta > 0)")
print("="*80)

for method in both["method"].unique():
    method_data = both[both["method"] == method]["delta"]
    # One-sided test: alternative='greater' tests if delta > 0
    stat, p_value = wilcoxon(method_data)
    n = len(method_data)
    mean_delta = method_data.mean()
    median_delta = method_data.median()

    print(f"\nMethod: {method}")
    print(f"  n = {n}")
    print(f"  Mean delta = {mean_delta:.6f}")
    print(f"  Median delta = {median_delta:.6f}")
    print(f"  Test statistic = {stat:.2f}")
    print(f"  p-value = {p_value:.6e}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"  Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")