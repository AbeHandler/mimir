"""
Orchestrator for model3 dc_pdd SUTVA analysis on David's server.

Steps:
  run     - compute + analyze
  compute - Run run_config.sh for each dc_pdd config, move CSV to csvs/R2/model3/
  analyze - Compare treated vs control MIA scores (paired by doc_id)
  flush   - Delete outputs for a clean rerun

Usage:
  python scripts/R2/publish/model3/go.py --run
  python scripts/R2/publish/model3/go.py --compute
  python scripts/R2/publish/model3/go.py --analyze
  python scripts/R2/publish/model3/go.py --flush
  python scripts/R2/publish/model3/go.py --dry-run
"""

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NAME = "model3_dc_pdd"

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = REPO_ROOT / "csvs" / "R2" / "model3"
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_config.sh"

CONFIGS = [
    "sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.dc_pdd.json",
    "sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.dc_pdd.json",
    "sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.dc_pdd.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description=NAME)
    parser.add_argument("--run", action="store_true", help="compute + analyze")
    parser.add_argument("--compute", action="store_true", help="Run mimir for each config")
    parser.add_argument("--analyze", action="store_true", help="Compare treated vs control")
    parser.add_argument("--flush", action="store_true", help="Delete outputs for clean rerun")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Report status of all configs")
    return parser.parse_args()


def _cached_csv(cfg_name):
    return CACHE_DIR / cfg_name.replace(".json", ".csv")


def _uncollected_csv(cfg_name):
    return REPO_ROOT / cfg_name.replace(".json", ".csv")


def _mark_todo_and_done():
    done, todo = [], []
    for cfg in CONFIGS:
        if _cached_csv(cfg).exists():
            done.append(cfg)
        else:
            todo.append(cfg)
    return done, todo


def dry_run():
    """Print status of all configs. Returns (done, todo) for reuse."""
    done, todo = _mark_todo_and_done()
    if not todo:
        print(f"✅ compute step done ({len(done)}/{len(CONFIGS)})")
    else:
        print(f"Total: {len(CONFIGS)}  Done: {len(done)}  To run: {len(todo)}")
        for cfg in todo:
            print(f"  to run     {Path(cfg).stem[:70]}")
    return done, todo


def collect_to_cache(cfg):
    """Move output CSV from repo root into CACHE_DIR."""
    src = _uncollected_csv(cfg)
    if not src.exists():
        raise FileNotFoundError(f"Expected output not found: {src}")
    dest = _cached_csv(cfg)
    src.rename(dest)
    print(f"  -> {dest.relative_to(REPO_ROOT)}")


def compute(gpu: int):
    """Run each config then immediately move CSV to csvs/R2/model3/."""
    _, todo = dry_run()

    if not todo:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    random.shuffle(todo)

    for cfg in todo:
        cmd = [str(RUN_SCRIPT), cfg, str(gpu)]
        print(f"\nRUN: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            raise RuntimeError(f"run_config.sh failed for {cfg}")

        collect_to_cache(cfg)


def run(gpu: int):
    """compute + analyze."""
    compute(gpu)
    analyze()


# Pairs: (treated_cfg, control_cfg)
PAIRS = [
    (CONFIGS[0], CONFIGS[1]),  # pair1_treated_run1 vs pair2_control_run4
    (CONFIGS[2], CONFIGS[1]),  # pair2_treated_run3 vs pair2_control_run4
]


def _load_scores(cfg_name):
    """Load CSV, return DataFrame with method/membership/doc_id/score."""
    df = pd.read_csv(_cached_csv(cfg_name))
    return df[df["method"] == "dc_pdd"].copy()


def _compare_treated_vs_control(treated_df, control_df, method):
    """Compare scores for same doc_ids. Returns dict with stats."""
    t = treated_df[(treated_df["method"] == method) & (treated_df["membership"] == "member")].rename(columns={"score": "blocked"})
    c = control_df[(control_df["method"] == method) & (control_df["membership"] == "member")].rename(columns={"score": "unblocked"})
    merged = t.merge(c, on=["doc_id"])
    merged["delta"] = merged["blocked"] - merged["unblocked"]
    merged["pct_change"] = merged["delta"] / merged["unblocked"].abs()

    mean_blocked = merged["blocked"].mean()
    mean_unblocked = merged["unblocked"].mean()
    mean_delta = merged["delta"].mean()
    mean_pct = merged["pct_change"].mean()

    return {
        "method": method,
        "n": len(merged),
        "mean_blocked": mean_blocked,
        "mean_unblocked": mean_unblocked,
        "mean_delta": mean_delta,
        "mean_pct_change": mean_pct,
    }


def _validate_analyze():
    """Check all CSVs exist before analyzing."""
    missing = [cfg for pair in PAIRS for cfg in pair if not _cached_csv(cfg).exists()]
    if missing:
        names = "\n  ".join(Path(c).stem for c in missing)
        raise FileNotFoundError(f"Missing CSVs (run --compute first):\n  {names}")


def analyze():
    """Compare treated vs control MIA scores, paired by doc_id."""

    _validate_analyze()

    for treated_cfg, control_cfg in PAIRS:
        treated = _load_scores(treated_cfg)
        control = _load_scores(control_cfg)

        treated_label = Path(treated_cfg).stem.split("_pair")[0].split("sutva_")[-1]
        print(f"\n{'=' * 70}")
        print(f"Treated: {Path(treated_cfg).stem[:70]}")
        print(f"Control: {Path(control_cfg).stem[:70]}")
        print(f"{'=' * 70}")

        r = _compare_treated_vs_control(treated, control, "dc_pdd")
        print(f"  blocked={r['mean_blocked']:.4f}  "
              f"unblocked={r['mean_unblocked']:.4f}  "
              f"delta={r['mean_delta']:+.4f}  "
              f"pct={r['mean_pct_change']:+.1%}  n={r['n']:,}")


def flush():
    """Delete CACHE_DIR for a clean rerun."""
    if not CACHE_DIR.exists():
        print("Nothing to flush.")
        return
    print(f"rm -rf {CACHE_DIR}")
    shutil.rmtree(CACHE_DIR)
    print("Flushed.")


if __name__ == "__main__":
    args = parse_args()

    if args.dry_run:
        dry_run()
    elif args.run:
        run(args.gpu)
    elif args.compute:
        compute(args.gpu)
    elif args.analyze:
        analyze()
    elif args.flush:
        flush()
    else:
        raise ValueError("Provide --run, --compute, --analyze, --flush, or --dry-run")
