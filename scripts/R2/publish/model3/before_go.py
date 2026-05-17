"""Filter the 3 model3 CSVs to only method=='loss' & membership=='member'.

Reads from <REPO_ROOT>/csvs/<cfg_stem>.csv and writes the filtered copy to
<REPO_ROOT>/csvs/model3/<cfg_stem>.csv (creating the dir if missing).

Usage:
    python scripts/R2/publish/model3/before_go.py
    python scripts/R2/publish/model3/before_go.py -repo-root /home/abe/mimir
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd


METHOD = "loss"
MEMBERSHIP = "member"

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[4]

CONFIGS = [
    "sutva_click2houston_com_2022-05-01_pair1_treated_run1_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.json",
    "sutva_click2houston_com_2022-05-01_pair2_control_run4_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.json",
    "sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered.json",
    "sutva_click2houston_com_2022-05-01_pair2_treated_run3_sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered_take2.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Filter model3 CSVs to loss/member rows.")
    parser.add_argument("-repo-root", dest="repo_root", type=Path,
                        default=REPO_ROOT_DEFAULT,
                        help=f"Path to the mimir repo root (default: {REPO_ROOT_DEFAULT})")
    return parser.parse_args()


def sync_to_r2(dest_dir: Path, repo_root: Path):
    """Sync the filtered model3 CSVs (and their configs) to R2.

    Uses `aws s3 sync --size-only` so reruns are no-ops once uploaded.
    Mirrors the style of scripts/R2/ate_vs_atu_for_R2.py :: sync().
    """
    end_point = "https://1d736c1e8da83d40f1eda75419d90b86.r2.cloudflarestorage.com"

    csv_includes = " ".join(
        f'--include "{cfg.replace(".json", ".csv")}"' for cfg in CONFIGS
    )
    csv_path = str(dest_dir) + "/"
    csv_prefix = csv_path.lstrip("/").rstrip("/")
    subprocess.run(
        f'aws s3 sync {csv_path} s3://misqsi/{csv_prefix} '
        f'--endpoint-url {end_point} --profile r2 --size-only '
        f'--exclude "*" {csv_includes}',
        shell=True, check=True,
    )

    cfg_includes = " ".join(f'--include "{cfg}"' for cfg in CONFIGS)
    cfg_path = str(repo_root / "configs") + "/"
    cfg_prefix = cfg_path.lstrip("/").rstrip("/")
    subprocess.run(
        f'aws s3 sync {cfg_path} s3://misqsi/{cfg_prefix} '
        f'--endpoint-url {end_point} --profile r2 --size-only '
        f'--exclude "*" {cfg_includes}',
        shell=True, check=True,
    )


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    src_dir = repo_root / "csvs"
    dest_dir = src_dir / "model3"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for cfg in CONFIGS:
        stem = cfg.replace(".json", "")
        src = src_dir / f"{stem}.csv"
        dest = dest_dir / f"{stem}.csv"

        assert src.exists(), f"missing input: {src}"
        df = pd.read_csv(src)
        out = df[(df["method"] == METHOD) & (df["membership"] == MEMBERSHIP)].copy()
        assert len(out) > 0, f"no rows with method={METHOD!r}, membership={MEMBERSHIP!r} in {src}"
        out.to_csv(dest, index=False)
        print(f"{src.name}: {len(df)} -> {len(out)} rows  ->  {dest}")

    sync_to_r2(dest_dir, repo_root)


if __name__ == "__main__":
    main()
