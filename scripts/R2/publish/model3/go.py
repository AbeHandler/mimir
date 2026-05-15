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

NAME = "model3"

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = REPO_ROOT / "csvs"
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_config.sh"

METHOD = 'loss'

CONFIGS = [
    "sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.json",
    "sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.json",
    "sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description=NAME)
    parser.add_argument("--run", action="store_true", help="compute + analyze")
    parser.add_argument("--compute", action="store_true", help="Run mimir for each config")
    parser.add_argument("--analyze", action="store_true", help="Compare treated vs control")
    parser.add_argument("--flush", action="store_true", help="Delete outputs for clean rerun")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Report status of all configs")
    parser.add_argument("-repo-root", dest="repo_root", type=Path,
                        default=REPO_ROOT,
                        help=f"Path to the mimir repo root (default: {REPO_ROOT})")
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
    out = df[df["method"] == METHOD].copy()
    assert len(out) > 0
    return out


def _compare_treated_vs_control(treated_df, control_df, method):
    """Compare scores for same doc_ids. Returns dict with stats."""
    t = treated_df[(treated_df["method"] == method) & (treated_df["membership"] == "member")].rename(columns={"score": "blocked"})
    c = control_df[(control_df["method"] == method) & (control_df["membership"] == "member")].rename(columns={"score": "unblocked"})
    merged = t.merge(c, on=["doc_id"])
    merged["delta"] = merged["blocked"] - merged["unblocked"]

    return {
        "method": method,
        "n": len(merged),
        "mean_blocked": merged["blocked"].mean(),
        "mean_unblocked": merged["unblocked"].mean(),
        "mean_delta": merged["delta"].mean(),
    }


def _validate_analyze():
    """Check all CSVs exist before analyzing."""
    missing = [cfg for pair in PAIRS for cfg in pair if not _cached_csv(cfg).exists()]
    if missing:
        names = "\n  ".join(Path(c).stem for c in missing)
        raise FileNotFoundError(f"Missing CSVs (run --compute first):\n  {names}")


def sync():
    """Sync the CSVs (which hold the `loss` method scores) and their configs to R2.

    Uses `aws s3 sync --size-only` so reruns are no-ops once files are uploaded.
    """
    end_point = "https://1d736c1e8da83d40f1eda75419d90b86.r2.cloudflarestorage.com"

    extra = [
        "sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered_take2.json",
    ]
    all_cfgs = CONFIGS + extra

    csv_includes = " ".join(f'--include "{_cached_csv(cfg).name}"' for cfg in all_cfgs)
    csv_path = str(CACHE_DIR) + "/"
    csv_prefix = csv_path.lstrip("/").rstrip("/")
    subprocess.run(
        f'aws s3 sync {csv_path} s3://misqsi/{csv_prefix} '
        f'--endpoint-url {end_point} --profile r2 --size-only '
        f'--exclude "*" {csv_includes}',
        shell=True, check=True,
    )

    cfg_includes = " ".join(f'--include "{cfg}"' for cfg in all_cfgs)
    cfg_path = str(REPO_ROOT / "configs") + "/"
    cfg_prefix = cfg_path.lstrip("/").rstrip("/")
    subprocess.run(
        f'aws s3 sync {cfg_path} s3://misqsi/{cfg_prefix} '
        f'--endpoint-url {end_point} --profile r2 --size-only '
        f'--exclude "*" {cfg_includes}',
        shell=True, check=True,
    )


def analyze():
    """Compare treated vs control MIA scores, paired by doc_id."""

    _validate_analyze()
    sync()

    results = []
    for treated_cfg, control_cfg in PAIRS:
        treated = _load_scores(treated_cfg)
        control = _load_scores(control_cfg)

        r = _compare_treated_vs_control(treated, control, "loss")
        results.append(r)

    # pct change in delta across pairs
    d1 = results[0]["mean_delta"]
    d2 = results[1]["mean_delta"]
    pct_change = (d2 - d1) / abs(d1)
    print(f"pct_change={pct_change:+.1%}")


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

    REPO_ROOT = args.repo_root.resolve()
    CACHE_DIR = REPO_ROOT / "csvs"
    RUN_SCRIPT = REPO_ROOT / "scripts" / "run_config.sh"

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
