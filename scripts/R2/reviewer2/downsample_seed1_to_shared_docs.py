"""Downsample the seed 1 Appendix csvs to the docs that all three seeds scored.

Seed 1 scored a large sample of contaminated docs (about 28.4K); seeds 2 and 3 were
replications on a smaller sample (1,010 docs, the same docs in both). Fig 1A and Table 1A
are cleaner if every seed uses the same documents, so this keeps only the doc_ids that
appear in the contaminated files of all three seeds. The cut is deterministic (no random
sampling). The full seed 1 files are left untouched; the downsampled copies go to -out.

Usage:
    python scripts/R2/reviewer2/downsample_seed1_to_shared_docs.py
    python scripts/R2/reviewer2/downsample_seed1_to_shared_docs.py -out /tmp/csvs_1k

Spot check:
    python -c "import pandas as pd; d=pd.read_csv('csvs/confounddataset_1k/pythia-45m_lr1e-3_steps5k_seed1_interleave0.02_contaminated_seed1.all_shards.csv.gz'); print(d.groupby(['method','membership']).size())"
"""
import argparse
from pathlib import Path

import pandas as pd

SEEDS = [1, 2, 3]
# the two seed 1 files that are much larger than the other seeds'
DOWNSAMPLE_TAGS = ["interleave0.02_contaminated", "on_contaminated_oos"]
MIN_SHARED_DOCS = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-csvs", type=Path, default=Path("csvs/confounddataset"))
    p.add_argument("-out", type=Path, default=Path("csvs/confounddataset_1k"))
    return p.parse_args()


def csv_name(seed: int, tag: str) -> str:
    return f"pythia-45m_lr1e-3_steps5k_seed{seed}_{tag}_seed{seed}.all_shards.csv.gz"


def shared_doc_ids(csvs: Path) -> set:
    ids = []
    for seed in SEEDS:
        df = pd.read_csv(csvs / csv_name(seed, "interleave0.02_contaminated"))
        ids.append(set(df.loc[df["membership"] == "member", "doc_id"]))
    shared = set.intersection(*ids)
    if len(shared) < MIN_SHARED_DOCS:
        raise ValueError(f"only {len(shared)} doc_ids shared across seeds {SEEDS}")
    return shared


def main() -> None:
    args = parse_args()
    shared = shared_doc_ids(args.csvs)
    print(f"doc_ids shared across seeds {SEEDS}: {len(shared):,}")

    args.out.mkdir(parents=True, exist_ok=True)
    for tag in DOWNSAMPLE_TAGS:
        df = pd.read_csv(args.csvs / csv_name(1, tag))
        kept = df[df["doc_id"].isin(shared)]
        out_path = args.out / csv_name(1, tag)
        kept.to_csv(out_path, index=False)
        print(f"{out_path}: {len(df):,} -> {len(kept):,} rows")


if __name__ == "__main__":
    main()
