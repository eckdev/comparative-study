import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root():
    here = Path(__file__).resolve().parent
    if (here.parent / "shared_splits").exists():
        return here.parent
    return Path("/content/drive/MyDrive/comparative-study")


def main():
    parser = argparse.ArgumentParser(description="Colab presets for AGH-Former orthodontic landmark localization.")
    parser.add_argument("--preset", choices=["smoke", "a100", "a100_16k"], default="smoke")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = repo_root()
    work_dir = root / "agh_former_orthodontic_comparison"
    data_root = root / "data" / "dataset"
    splits_json = root / "shared_splits" / "orthodontic_180_60_60_seed42.json"
    transform_dir = root / "palnet_orthodontic_comparison" / "transforms" / "orthodontic_procrustes_rigid_20260627_143801"

    common = [
        sys.executable,
        "-u",
        str(work_dir / "run_orthodontic_aghformer.py"),
        "--data-root",
        str(data_root),
        "--splits-json",
        str(splits_json),
        "--transformation-dir",
        str(transform_dir),
        "--device",
        args.device,
    ]

    if args.preset == "smoke":
        cmd = common + [
            "--output-dir",
            str(work_dir / "runs" / "aghformer_smoke_colab"),
            "--surface-points",
            "512",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--batch-size",
            "2",
            "--width",
            "64",
            "--blocks",
            "1",
            "--heads",
            "4",
            "--topk",
            "10",
            "--max-samples",
            "24",
        ]
    elif args.preset == "a100":
        cmd = common + [
            "--output-dir",
            str(work_dir / "runs" / "aghformer_p12000_w192_b4_e220"),
            "--surface-points",
            "12000",
            "--eval-surface-points",
            "12000",
            "--epochs",
            "220",
            "--patience",
            "35",
            "--batch-size",
            "2",
            "--lr",
            "0.0008",
            "--weight-decay",
            "0.0001",
            "--width",
            "192",
            "--blocks",
            "4",
            "--heads",
            "6",
            "--mlp-ratio",
            "2.0",
            "--heatmap-sigma-start",
            "5.0",
            "--heatmap-sigma-end",
            "2.5",
            "--coord-weight",
            "0.45",
            "--structure-weight",
            "0.08",
            "--symmetry-weight",
            "0.02",
            "--clinical-weight",
            "0.05",
            "--uncertainty-weight",
            "0.02",
            "--rotation-aug-deg",
            "4.0",
            "--point-jitter-std",
            "0.003",
            "--feature-dropout",
            "0.05",
            "--topk",
            "30",
            "--temperature",
            "1.0",
        ]
    else:
        cmd = common + [
            "--output-dir",
            str(work_dir / "runs" / "aghformer_p16000_w192_b4_e240"),
            "--surface-points",
            "16000",
            "--eval-surface-points",
            "16000",
            "--epochs",
            "240",
            "--patience",
            "40",
            "--batch-size",
            "1",
            "--lr",
            "0.0006",
            "--weight-decay",
            "0.0001",
            "--width",
            "192",
            "--blocks",
            "4",
            "--heads",
            "6",
            "--mlp-ratio",
            "2.0",
            "--heatmap-sigma-start",
            "5.0",
            "--heatmap-sigma-end",
            "2.5",
            "--coord-weight",
            "0.45",
            "--structure-weight",
            "0.08",
            "--symmetry-weight",
            "0.02",
            "--clinical-weight",
            "0.05",
            "--uncertainty-weight",
            "0.02",
            "--rotation-aug-deg",
            "4.0",
            "--point-jitter-std",
            "0.003",
            "--feature-dropout",
            "0.05",
            "--topk",
            "30",
        ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True, env=env)


if __name__ == "__main__":
    main()
