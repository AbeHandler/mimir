#!/usr/bin/env python3
"""
Run the LOSS attack on a HuggingFace dataset (or local .jsonl) and write per-example scores to JSONL.

-dataset may be either an HF hub id or a path to a local .jsonl file (detected by suffix + existence).

Output path is derived from the dataset (+ model + split) unless -output is passed explicitly.
A .done marker is written once all indices are accounted for in the output JSONL.

Conda env: gptoss

Usage (blackwell):
    conda activate gptoss
    MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache CUDA_VISIBLE_DEVICES=0 \
        python -m scripts.R2.get_loss_for_hf_dataset \
        -dataset abehandlerorg/localnewsinthewild \
        -model openai/gpt-oss-20b \
        -device cuda:0
"""
import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, load_dataset

from mimir.config import EnvironmentConfig, ExperimentConfig
from mimir.models import LanguageModel
from mimir.attacks.loss import LOSSAttack


DEFAULT_DATASET = "abehandlerorg/localnewsinthewild"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_COL = "text"
DEFAULT_OUTPUT_DIR = "results/get_loss_for_hf_dataset"
DEFAULT_CACHE_DIR = "/mnt/storage/abe/tmp"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-dataset", default=DEFAULT_DATASET)
    p.add_argument("-split", default=DEFAULT_SPLIT)
    p.add_argument("-text-col", default=DEFAULT_TEXT_COL)
    p.add_argument("-model", default=DEFAULT_MODEL)
    p.add_argument("-device", default="cuda")
    p.add_argument("-cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("-output", default=None, help="Override auto-derived output jsonl path")
    p.add_argument("-output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("-limit", type=int, default=None)
    p.add_argument("-seed", type=int, default=42)
    return p.parse_args()


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def derive_output_path(args):
    if args.output:
        return Path(args.output)
    ds_path = Path(args.dataset)
    if ds_path.suffix == ".jsonl" and ds_path.exists():
        ds_slug = _slug(ds_path.stem)
    else:
        ds_slug = _slug(args.dataset)
    model_slug = _slug(args.model)
    name = f"{ds_slug}.{args.split}.{model_slug}.jsonl"
    return Path(args.output_dir) / name


def load_input_dataset(dataset, split):
    """Load an HF hub dataset or a local .jsonl file into an in-memory Dataset."""
    path = Path(dataset)
    if path.suffix == ".jsonl" and path.exists():
        records = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return Dataset.from_list(records)
    return load_dataset(dataset, split=split)


def build_config(args):
    env_config = EnvironmentConfig(device=args.device, cache_dir=args.cache_dir)
    return ExperimentConfig(
        experiment_name="get_loss_for_hf_dataset",
        base_model=args.model,
        dataset_member="dummy",
        dataset_nonmember="dummy",
        env_config=env_config,
        random_seed=args.seed,
    )


def already_done(output_path):
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["index"])
            except Exception:
                continue
    return done


def main():
    args = parse_args()
    random.seed(args.seed)

    output_path = derive_output_path(args)
    done_marker = output_path.with_suffix(output_path.suffix + ".done")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if done_marker.exists():
        print(f"[skip] done marker exists: {done_marker}")
        print("  Delete the marker (and optionally the jsonl) to rerun.")
        return

    config = build_config(args)
    model = LanguageModel(config)
    model.load()
    loss_attack = LOSSAttack(config, model)

    ds = load_input_dataset(args.dataset, args.split)
    all_indices = list(range(len(ds)))
    indices = list(all_indices)
    random.shuffle(indices)
    if args.limit is not None:
        indices = indices[: args.limit]

    done = already_done(output_path)
    script_path = str(Path(__file__).resolve())
    print(f"[output] {output_path}")
    print(f"[dataset] size={len(ds)} already_done={len(done)}")

    with output_path.open("a") as fout:
        for i in indices:
            if i in done:
                continue
            row = ds[i]
            text = row.get(args.text_col)
            if not text:
                raise ValueError(f"empty/missing {args.text_col!r} at index {i} in {args.dataset}")

            probs = model.get_probabilities(text)
            loss_score = loss_attack._attack(text, probs=probs)

            rec = {
                "index": i,
                "url": row.get("url"),
                "loss": float(loss_score),
                "dataset": args.dataset,
                "split": args.split,
                "model": args.model,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "written_by": script_path,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            done.add(i)
            print(f"[{i}] loss={loss_score:.4f}")

    # Completeness check: only write .done if every input index (respecting -limit) is present.
    expected = set(all_indices if args.limit is None else all_indices[: args.limit])
    final_done = already_done(output_path)
    missing = expected - final_done
    if missing:
        print(f"\n[incomplete] {len(missing)}/{len(expected)} indices missing from {output_path}")
        print("  Rerun to fill gaps, or wipe the jsonl for a clean restart.")
        print(f"  Example missing indices: {sorted(list(missing))[:10]}")
    else:
        done_marker.write_text(datetime.now(timezone.utc).isoformat() + "\n")
        print(f"\n[done] wrote marker: {done_marker}")

    # Spot checks
    print("\nSpot check commands:")
    print(f"  wc -l {output_path}")
    print(f"  head -1 {output_path} | python -m json.tool")
    print(f"  shuf {output_path} | head -3")


if __name__ == "__main__":
    main()
