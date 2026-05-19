#!/usr/bin/env python3
"""

The point of this script is reproducibility.
A fresh `iamgroot42/mimir` clone should re-derive the score column of the published CSVs.

Reference CSVs are pulled on demand from the R2 bucket the paper uses
(s3://misqsi/...), so there is no dependency on a local `csvs/` tree.
CSVs are cached at `-cache-dir` (default /tmp/6_1_sanity_csvs) and
reused across reruns.

# Mimir does not appear active but this script was checked on
# May 19th, 2026 when the head MIMIR commit is 1b6fd649eeeecc887275a2336c7da808ee58757d

# To run
$ git clone https://github.com/iamgroot42/mimir # fresh mimir clone
$ cd mimir && mkdir -p scripts/R2/publish/ # setup dirs
# copy from local modification of mimir
$ cp ~/mimir/scripts/R2/publish/6.1_sanity.py scripts/R2/publish/

# AWS profile "r2" must be configured (see ~/.aws/credentials) so the
# script can pull reference CSVs from cloudflare R2.
# setting CUDA_VISIBLE here is needed for llama to run


$ python scripts/R2/publish/6.1_sanity.py -cache-dir /tmp/61/
$ CUDA_VISIBLE_DEVICES=0,1 MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache python scripts/R2/publish/6.1_sanity.py -cache-dir /tmp/61/

"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


R2_ENDPOINT = "https://1d736c1e8da83d40f1eda75419d90b86.r2.cloudflarestorage.com"
R2_BUCKET = "misqsi"
R2_BASE_PREFIX = "Users/abha4861/mimir/csvs"
R2_AWS_PROFILE = os.environ.get("R2_AWS_PROFILE", "r2")


def r2_fetch(remote_key: str, dest: Path) -> Path:
    """Download a single object from R2 into `dest` using aws s3 cp.

    Mirrors the pull pattern in scripts/R2/ate_vs_atu_for_R2.py:pull
    (same endpoint, bucket, profile). Skips the download if `dest`
    already exists so reruns are fast.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "cp",
        f"s3://{R2_BUCKET}/{remote_key}", str(dest),
        "--endpoint-url", R2_ENDPOINT,
        "--profile", R2_AWS_PROFILE,
    ]
    print(f"[6.1_sanity] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return dest


def _compat_patch_llama_tokenizer():
    """Make a fresh `iamgroot42/mimir` clone work with current transformers.

    This just patches the llama branch of the mimir models file to use the tokenizer
    included w/ our models on HF
    """
    models_py = Path(__file__).resolve().parents[3] / "mimir" / "models.py"
    if not models_py.exists():
        return
    src = models_py.read_text()
    needle = "transformers.LlamaTokenizer.from_pretrained"
    if needle not in src:
        return
    patched = src.replace(needle, "transformers.AutoTokenizer.from_pretrained")
    models_py.write_text(patched)
    print(f"[6.1_sanity] patched {models_py}: LlamaTokenizer -> AutoTokenizer")


_compat_patch_llama_tokenizer()

import numpy as np
import pandas as pd

from mimir.config import EnvironmentConfig, ExperimentConfig
from mimir.models import LanguageModel
from mimir.attacks.loss import LOSSAttack
from mimir.attacks.zlib import ZLIBAttack
from mimir.attacks.min_k import MinKProbAttack


METHODS = ("loss", "zlib", "min_k")


def parse_args():
    p = argparse.ArgumentParser(description="Section 6.1 mimir-output sanity checks.")
    p.add_argument(
        "-mode",
        choices=["pull", "run"],
        default="run",
        help="pull: just download reference CSVs into -cache-dir and exit. "
             "run: pull (idempotent) then score. Default: run.",
    )
    p.add_argument(
        "-cache-dir",
        default=str(Path(tempfile.gettempdir()) / "6_1_sanity_csvs"),
        help="Local directory to hold reference CSVs pulled from R2.",
    )
    p.add_argument(
        "-tolerance",
        type=float,
        default=1e-3,
        help="Absolute+relative tolerance for score comparison.",
    )
    p.add_argument(
        "-n-per-cell",
        type=int,
        default=3,
        help="Number of member rows to spot-check per method.",
    )
    p.add_argument(
        "-seed",
        type=int,
        default=0,
        help="RNG seed for sampling rows from the reference CSV.",
    )
    p.add_argument(
        "-output",
        default="audit_6_1.jsonl",
        help="Append per-row audit records to this jsonl file.",
    )
    p.add_argument(
        "-device",
        default="cuda:0",
        help="Device passed to mimir EnvironmentConfig.",
    )
    return p.parse_args()


def build_combos(cache_dir: Path):
    """PT (blockbench) combos + one CPT-8B Y1 combo.

    Each combo maps a base_model + HF dataset to a remote R2 key for the
    per-treatment all_shards.csv.gz produced by run.py and a local
    cache path inside `cache_dir`. The CSV is pulled on demand so the
    script does not depend on a local `csvs/` tree. The 8B test case in
    ate_vs_atu_for_R2.py is a paired Y1 - Y0 delta; we sanity-check the
    Y1 ("blocks") side here, which is enough to verify the CSV.
    """
    entries = [
        (
            "PT bothbins blocks",
            "dobolyilab/blockbench-blocksbin",
            "abehandlerorg/bothbins",
            "confounddataset/bothbins.blocks.lite.all_shards.csv.gz",
        ),
        (
            "PT bothbins noblocks",
            "dobolyilab/blockbench-noblocksbin",
            "abehandlerorg/bothbins",
            "confounddataset/bothbins.noblocks.lite.all_shards.csv.gz",
        ),
        (
            "PT excluded-docs blocks",
            "dobolyilab/blockbench-blocksbin",
            "abehandlerorg/excluded-docs",
            "confounddataset/excluded-docs.blocks.lite.all_shards.csv.gz",
        ),
        (
            "CPT-8B bothbins Y1",
            "abehandlerorg/Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1",
            "abehandlerorg/cptllama_bothbins_20240101_20240115",
            "Llama-3.1-8B-Instruct-bnb-4bit_cptllama-2024-01-01-to-2024-01-15-Y1.bothbins.lite.all_shards.csv.gz",
        ),
    ]
    out = []
    for label, base_model, dataset, rel_key in entries:
        remote_key = f"{R2_BASE_PREFIX}/{rel_key}"
        local_csv = cache_dir / rel_key
        out.append((label, base_model, dataset, remote_key, local_csv))
    return out


def load_model(base_model: str, device: str):
    env_config = EnvironmentConfig(device=device, cache_dir="/tmp/cache")
    config = ExperimentConfig(
        experiment_name="6_1_sanity",
        base_model=base_model,
        dataset_member="dummy",
        dataset_nonmember="dummy",
        env_config=env_config,
        random_seed=42,
    )
    model = LanguageModel(config)
    model.load()
    return config, model


def load_split_text(dataset: str, train: bool) -> dict:
    """Return {url: text} for the given HF split."""
    from datasets import load_dataset

    split = "train" if train else "test"
    ds = load_dataset(dataset)[split]
    df = ds.to_pandas()
    if "url" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"dataset {dataset} split {split} missing url/text columns: "
            f"got {list(df.columns)}"
        )
    return dict(zip(df["url"].to_list(), df["text"].to_list()))


def sample_reference_rows(reference_csv: Path, n_per_cell: int, rng: random.Random) -> pd.DataFrame:
    """Pick up to n_per_cell member rows per method."""
    df = pd.read_csv(reference_csv)
    df = df[df["membership"] == "member"]
    pieces = []
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        idx = list(sub.index)
        rng.shuffle(idx)
        pieces.append(sub.loc[idx[:n_per_cell]])
    if not pieces:
        raise ValueError(
            f"{reference_csv}: no member rows matched any of {METHODS}"
        )
    return pd.concat(pieces, ignore_index=True)


def score_row(method: str, attacks: dict, text: str, model) -> float:
    """Recompute one score for (method, text) using cached probs."""
    probs = model.get_probabilities(text)
    if method not in attacks:
        raise ValueError(f"unknown method {method}")
    return float(attacks[method]._attack(text, probs=probs))


def pull_all(combos) -> None:
    """Download every combo's reference CSV from R2 into its cache path.

    Idempotent: existing files are kept (r2_fetch short-circuits on
    existence). Run this once standalone via `-mode pull`, or let
    `-mode run` invoke it before scoring.
    """
    for label, _, _, remote_key, local_csv in combos:
        print(f"[6.1_sanity] pulling {label}")
        r2_fetch(remote_key, Path(local_csv))


def check_combo(label, base_model, dataset, remote_key, reference_csv, args, fout) -> list:
    """Score the sampled rows and return list of mismatch tuples."""
    print("=" * 70)
    print(f"[6.1 sanity] {label}")
    print(f"  base-model:    {base_model}")
    print(f"  dataset:       {dataset}")
    print(f"  remote-csv:    s3://{R2_BUCKET}/{remote_key}")
    print(f"  reference-csv: {reference_csv}")
    print("=" * 70)
    if not Path(reference_csv).exists():
        raise FileNotFoundError(
            f"reference csv missing for combo {label!r}: {reference_csv}. "
            "Run with -mode pull first (or re-run with -mode run)."
        )

    rng = random.Random(args.seed)
    rows = sample_reference_rows(Path(reference_csv), args.n_per_cell, rng)
    print(
        f"sampled {len(rows)} member row(s) across "
        f"{rows.groupby(['method']).size().to_dict()}"
    )

    config, model = load_model(base_model, args.device)
    attacks = {
        "loss": LOSSAttack(config, model),
        "zlib": ZLIBAttack(config, model),
        "min_k": MinKProbAttack(config, model),
    }

    member_text = load_split_text(dataset, train=True)

    mismatches = []
    for _, row in rows.iterrows():
        method = row["method"]
        doc_id = row["doc_id"]
        ref = float(row["score"])

        text = member_text.get(doc_id)
        if text is None:
            status = "MISSING_TEXT"
            fresh = None
        else:
            fresh = score_row(method, attacks, text, model)
            if np.isclose(fresh, ref, rtol=args.tolerance, atol=args.tolerance):
                status = "OK"
            else:
                status = "MISMATCH"
                mismatches.append((label, method, doc_id, fresh, ref))

        fresh_s = "n/a" if fresh is None else f"{fresh:.6f}"
        print(f"  [{method:>5}] {doc_id[:60]:60s}")
        print(f"    fresh: {fresh_s:>12}    ref: {ref:.6f}    status: {status}")

        fout.write(json.dumps({
            "combo": label,
            "base_model": base_model,
            "dataset": dataset,
            "reference_csv": str(reference_csv),
            "method": method,
            "doc_id": doc_id,
            "fresh_score": fresh,
            "ref_score": ref,
            "tolerance": args.tolerance,
            "status": status,
        }) + "\n")
    return mismatches


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[6.1_sanity] reference CSV cache: {cache_dir}")

    combos = build_combos(cache_dir)

    if args.mode == "pull":
        pull_all(combos)
        print(f"[6.1_sanity] pulled {len(combos)} reference CSV(s) into {cache_dir}")
        return

    pull_all(combos)

    all_mismatches = []
    with open(args.output, "a") as fout:
        for combo in combos:
            all_mismatches.extend(check_combo(*combo, args=args, fout=fout))

    print()
    if all_mismatches:
        details = "\n  ".join(
            f"{lab} | {m} {d}: fresh={g:.6f} ref={r:.6f}"
            for lab, m, d, g, r in all_mismatches
        )
        raise SystemExit(
            f"{len(all_mismatches)} row(s) disagreed beyond tol={args.tolerance:g}:\n  {details}"
        )
    print(f"All sanity checks passed across {len(combos)} combo(s).")


if __name__ == "__main__":
    main()
