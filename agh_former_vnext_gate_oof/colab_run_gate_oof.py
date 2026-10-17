#!/usr/bin/env python3
"""Drive-backed Colab presets for the AGH-vNext Gate-OOF experiment."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path("/content/comparative-study")
DATA_ROOT = Path("/content/drive/MyDrive/orthodontic/data/dataset")
SOURCE_ROOT = Path("/content/drive/MyDrive/orthodontic/agh_vnext_runs/publication_cv_seed42")
PREPROCESSING_ROOT = Path(
    "/content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/publication_cv_stage1_v4_seed42"
)
RUN_ROOT = Path("/content/drive/MyDrive/orthodontic/agh_vnext_gate_oof_runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("gate_oof_preflight", "gate_oof_fold1", "gate_oof_cv"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--source-run-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--preprocessing-root", type=Path, default=PREPROCESSING_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    for label, path in (
        ("dataset", args.data_root),
        ("source run", args.source_run_root),
        ("preprocessing", args.preprocessing_root),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    command = [
        sys.executable, "-u",
        str(CODE_ROOT / "agh_former_vnext_gate_oof/run_gate_oof.py"),
        "--preset", args.preset,
        "--data-root", str(args.data_root),
        "--source-run-root", str(args.source_run_root),
        "--preprocessing-root", str(args.preprocessing_root),
        "--output-dir", str(args.run_root / f"publication_gate_oof_seed{args.seed}"),
        "--seed", str(args.seed),
        "--device", "auto",
    ]
    print("Working directory:", CODE_ROOT, flush=True)
    print("Running:", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=str(CODE_ROOT), check=True, env=environment)


if __name__ == "__main__":
    main()
