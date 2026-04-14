#!/usr/bin/env python3
"""
Run the LOSS attack on a HuggingFace dataset and write per-example scores to JSONL.

Conda env: gptoss

Usage:
    conda activate gptoss
    MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache \
        CUDA_VISIBLE_DEVICES=2 python -m scripts.R2.get_loss_for_hf_dataset \
        -dataset abehandlerorg/localnewsinthewild \
        -model openai/gpt-oss-20b \
        -output results/get_loss_for_hf_dataset/localnewsinthewild.jsonl

Defaults match the above, so this also works:
    MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache \
        CUDA_VISIBLE_DEVICES=2 python -m scripts.R2.get_loss_for_hf_dataset
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from mimir.config import EnvironmentConfig, ExperimentConfig
from mimir.models import LanguageModel
from mimir.attacks.loss import LOSSAttack


DEFAULT_DATASET = "abehandlerorg/localnewsinthewild"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_COL = "text"
DEFAULT_OUTPUT = "results/get_loss_for_hf_dataset/localnewsinthewild.jsonl"
DEFAULT_CACHE_DIR = "/mnt/storage/abe/tmp"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-dataset", default=DEFAULT_DATASET)
    p.add_argument("-split", default=DEFAULT_SPLIT)
    p.add_argument("-text-col", default=DEFAULT_TEXT_COL)
    p.add_argument("-model", default=DEFAULT_MODEL)
    p.add_argument("-device", default="cuda")
    p.add_argument("-cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("-output", default=DEFAULT_OUTPUT)
    p.add_argument("-limit", type=int, default=None)
    p.add_argument("-seed", type=int, default=42)
    return p.parse_args()


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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    model = LanguageModel(config)
    model.load()
    loss_attack = LOSSAttack(config, model)

    ds = load_dataset(args.dataset, split=args.split)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    if args.limit is not None:
        indices = indices[: args.limit]

    done = already_done(output_path)
    script_path = str(Path(__file__).resolve())

    with output_path.open("a") as fout:
        for i in indices:
            if i in done:
                continue
            row = ds[i]
            text = row.get(args.text_col)
            if not text:
                continue

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
            print(f"[{i}] loss={loss_score:.4f}")

    # Spot checks
    print("\nSpot check commands:")
    print(f"  wc -l {output_path}")
    print(f"  head -1 {output_path} | python -m json.tool")
    print(f"  shuf {output_path} | head -3")


if __name__ == "__main__":
    main()
