"""Average delta JSON results across seeds.

Usage:
    python scripts/R2/reviewer2/average_delta_jsons.py --delta 0
    python scripts/R2/reviewer2/average_delta_jsons.py --delta 1
"""
import argparse
import json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--delta", type=int, required=True, choices=[0, 1])
    p.add_argument("--steps-k", type=str, default="5k")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results = []
    for seed in args.seeds:
        path = f"/tmp/delta_{args.delta}_steps{args.steps_k}_seed{seed}.json"
        with open(path) as f:
            results.append(json.load(f))
        print(f"Loaded {path}")

    methods = results[0].keys()
    n_seeds = len(results)
    averaged = {}
    for method in methods:
        fields = list(results[0][method].keys())
        averaged[method] = {}
        for field in fields:
            values = [r[method][field] for r in results]
            mean = sum(values) / n_seeds
            var = sum((v - mean) ** 2 for v in values) / (n_seeds - 1)
            averaged[method][field] = mean
            averaged[method][f"{field}_var"] = var
        averaged[method]["n_seeds"] = n_seeds

    output = f"/tmp/delta_{args.delta}_steps{args.steps_k}_all.json"
    with open(output, "w") as f:
        json.dump(averaged, f, indent=2)
    print(f"\nAveraged {len(results)} seeds -> {output}")
    print(json.dumps(averaged, indent=2))


if __name__ == "__main__":
    main()
