import argparse
import json

import pandas as pd
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


args = parse_args()
steps_k = f"{args.max_steps // 1000}k"
seed_suffix = "" if args.seed == 1234 else f"_seed{args.seed}"

contaminated = pd.read_csv(f'csvs/confounddataset/pythia-45m_lr1e-3_steps{steps_k}_seed{args.seed}_interleave0.02_uncontaminated_insample_regular_training_data{seed_suffix}.all_shards.csv.gz').rename(columns={"score": "contaminated"})
uncontaminated = pd.read_csv(f'csvs/confounddataset/pythia-45m_lr1e-3_steps{steps_k}_seed{args.seed}_uncontaminated_insample_regular_training_data{seed_suffix}.all_shards.csv.gz').rename(columns={"score": "uncontaminated"})

contaminated = contaminated[contaminated["membership"] == "member"].copy()
uncontaminated = uncontaminated[uncontaminated["membership"] == "member"].copy()

both = contaminated.merge(uncontaminated, on =["method", "doc_id"])


both["delta"] = both["uncontaminated"] - both["contaminated"]  # uncontaminated - contaminated (to match ATT)

# Write ATU data for R plotting
both.to_csv(f"/tmp/atu_appendix_steps{steps_k}_seed{args.seed}.csv", index=False)

results = {}
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

    contam_scores = both[both["method"] == method]["contaminated"]
    uncontam_scores = both[both["method"] == method]["uncontaminated"]

    results[method] = {
        "n": n,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "var_contaminated": contam_scores.var(),
        "var_uncontaminated": uncontam_scores.var(),
        "test_statistic": stat,
        "p_value": p_value,
    }

    print(f"\nMethod: {method}")
    print(f"  n = {n}")
    print(f"  Mean delta = {mean_delta:.6f}")
    print(f"  Median delta = {median_delta:.6f}")
    print(f"  Test statistic = {stat:.2f}")
    print(f"  p-value = {p_value:.6e}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"  Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")

output_path = f"/tmp/delta_0_steps{steps_k}_seed{args.seed}.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")