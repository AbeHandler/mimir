"""
Verify the model3 SUTVA result by re-running --analyze on R2-hosted CSVs.

- Pulls supporting CSVs from remote into a fresh /tmp dir
- Runs the number from Section 6.3

"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


R2_ENDPOINT = "https://1d736c1e8da83d40f1eda75419d90b86.r2.cloudflarestorage.com"
R2_PREFIX = "s3://misqsi/Users/abha4861/mimir/csvs/model3"

METHOD = "loss"

CONFIGS = [
    "sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.json",
    "sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.json",
    "sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.json",
]

PAIRS = [
    (CONFIGS[0], CONFIGS[1]),
    (CONFIGS[2], CONFIGS[1]),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Verify model3 by re-analyzing R2 CSVs.")
    parser.add_argument("-tmp-dir", dest="tmp_dir", type=Path, default=None,
                        help="Optional explicit tmp dir (default: a fresh mkdtemp).")
    return parser.parse_args()


def _cached_csv(tmp_dir: Path, cfg: str) -> Path:
    return tmp_dir / cfg.replace(".json", ".csv")


def pull_from_r2(tmp_dir: Path):
    """Pull each needed CSV from R2 into tmp_dir; skip files already on disk."""
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for cfg in CONFIGS:
        dest = _cached_csv(tmp_dir, cfg)
        if dest.exists():
            print(f"  ⚡ cached  {dest.name}")
            continue
        csv_name = cfg.replace(".json", ".csv")
        cmd = (
            f'aws s3 cp {R2_PREFIX}/{csv_name} {dest} '
            f'--endpoint-url {R2_ENDPOINT} --profile r2'
        )
        subprocess.run(cmd, shell=True, check=True)
        assert dest.exists(), f"pull failed: {dest} not on disk"


def _load_scores(tmp_dir: Path, cfg: str) -> pd.DataFrame:
    df = pd.read_csv(_cached_csv(tmp_dir, cfg))
    out = df[df["method"] == METHOD].copy()
    assert len(out) > 0, f"no method={METHOD!r} rows in {_cached_csv(tmp_dir, cfg)}"
    return out


def _compare_treated_vs_control(treated_df: pd.DataFrame,
                                control_df: pd.DataFrame,
                                method: str) -> dict:
    t = treated_df.rename(columns={"score": "blocked"})
    c = control_df.rename(columns={"score": "unblocked"})
    merged = t.merge(c, on=["doc_id"])
    merged["delta"] = merged["blocked"] - merged["unblocked"]
    return {
        "method": method,
        "n": len(merged),
        "mean_blocked": merged["blocked"].mean(),
        "mean_unblocked": merged["unblocked"].mean(),
        "mean_delta": merged["delta"].mean(),
    }


def analyze(tmp_dir: Path):
    results = []
    for treated_cfg, control_cfg in PAIRS:
        treated = _load_scores(tmp_dir, treated_cfg)
        control = _load_scores(tmp_dir, control_cfg)
        r = _compare_treated_vs_control(treated, control, METHOD)
        results.append(r)

    d1 = results[0]["mean_delta"]
    d2 = results[1]["mean_delta"]
    pct_change = (d2 - d1) / abs(d1)
    print(f"pct_change={pct_change:+.1%}")


def main():
    args = parse_args()
    if args.tmp_dir is not None:
        tmp_dir = args.tmp_dir.resolve()
        owns_tmp = False
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="model3_verification_"))
        owns_tmp = True
    print(f"Using tmp dir: {tmp_dir}")

    try:
        pull_from_r2(tmp_dir)
        analyze(tmp_dir)
    finally:
        if owns_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
