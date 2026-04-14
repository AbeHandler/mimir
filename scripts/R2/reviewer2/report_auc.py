"""ROC AUC for the uncontaminated model: in-sample training docs vs out-of-sample test docs.

Usage:
    python scripts/R2/reviewer2/report_auc.py --seed 1
    python scripts/R2/reviewer2/report_auc.py --seeds 1 2 3
"""
import argparse
import json

import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--steps-k", type=str, default="5k")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_results = {}
    for seed in args.seeds:
        # uncontaminated model: member=training docs, nonmember=test instances
        path = (
            f"csvs/confounddataset/pythia-45m_lr1e-3_steps{args.steps_k}_seed{seed}"
            f"_uncontaminated_vs_pythia-45m_lr1e-3_steps{args.steps_k}_seed{seed}"
            f"_interleave0.02_test_instances_seed{seed}.all_shards.csv.gz"
        )
        df = pd.read_csv(path)
        df["label"] = df["membership"].map({"member": 1, "nonmember": 0})

        seed_results = {}
        print(f"seed={seed}  ({path})")
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method]
            auc = roc_auc_score(sub["label"], -sub["score"])
            seed_results[method] = {"auc": auc, "n": len(sub)}
            print(f"  {method}: AUC={auc:.4f} (n={len(sub)})")
        print()

        all_results[f"seed{seed}"] = seed_results

    # Average across seeds
    methods = all_results[f"seed{args.seeds[0]}"].keys()
    print("=" * 40)
    print("Average across seeds:")
    averaged = {}
    for method in methods:
        aucs = [all_results[f"seed{s}"][method]["auc"] for s in args.seeds]
        mean_auc = sum(aucs) / len(aucs)
        var_auc = sum((a - mean_auc) ** 2 for a in aucs) / (len(aucs) - 1)
        averaged[method] = {"mean_auc": mean_auc, "var_auc": var_auc, "n_seeds": len(aucs)}
        print(f"  {method}: AUC={mean_auc:.4f} (var={var_auc:.6f})")

    all_results["average"] = averaged

    output = f"/tmp/auc_steps{args.steps_k}.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
