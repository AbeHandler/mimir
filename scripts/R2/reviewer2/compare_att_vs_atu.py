"""Compare ATT (delta=1) vs ATU (delta=0) per seed per method.

Tests whether contaminated docs have larger effects than uncontaminated docs.
Mann-Whitney U, one-sided: H1: ATT > ATU.

Usage:
    python scripts/R2/reviewer2/compare_att_vs_atu.py --seed 1
    python scripts/R2/reviewer2/compare_att_vs_atu.py --seed 1 --seed 2 --seed 3
"""
import argparse
import json

import pandas as pd
from scipy.stats import mannwhitneyu


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--steps-k", type=str, default="5k")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_results = {}
    for seed in args.seeds:
        att = pd.read_csv(f"/tmp/att_appendix_steps{args.steps_k}_seed{seed}.csv")
        atu = pd.read_csv(f"/tmp/atu_appendix_steps{args.steps_k}_seed{seed}.csv")

        seed_results = {}
        for method in att["method"].unique():
            att_delta = att[att["method"] == method]["delta"]
            atu_delta = atu[atu["method"] == method]["delta"]

            stat, p_value = mannwhitneyu(
                att_delta, atu_delta, alternative="greater"
            )

            seed_results[method] = {
                "n_att": len(att_delta),
                "n_atu": len(atu_delta),
                "mean_att": att_delta.mean(),
                "mean_atu": atu_delta.mean(),
                "var_atu": atu_delta.var(),
                "U_statistic": stat,
                "p_value": p_value,
            }

            print(f"seed={seed} method={method}:")
            print(f"  ATT mean={att_delta.mean():.6f} (n={len(att_delta)})")
            print(f"  ATU mean={atu_delta.mean():.6f} var={atu_delta.var():.6f} (n={len(atu_delta)})")
            print(f"  U={stat:.0f}  p={p_value:.6e}")
            print(f"  ATT > ATU at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
            print()

        all_results[f"seed{seed}"] = seed_results

    # Average ATU variance across seeds
    print("=" * 60)
    print("Average ATU variance across seeds:")
    methods = all_results[f"seed{args.seeds[0]}"].keys()
    for method in methods:
        vars = [all_results[f"seed{s}"][method]["var_atu"] for s in args.seeds]
        print(f"  {method}: mean_var_atu={sum(vars)/len(vars):.6f}")

    output = f"/tmp/att_vs_atu_steps{args.steps_k}.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
