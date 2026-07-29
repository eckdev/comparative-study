import argparse
import os
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path("/content/comparative-study")
DRIVE_ROOT = Path("/content/drive/MyDrive/orthodontic")
AGH_RUN = DRIVE_ROOT / "diffusion_runs" / "aghformer_v6_stage2_raw_fine_refiner_p12000"
if not AGH_RUN.exists():
    AGH_RUN = CODE_ROOT / "agh_former_orthodontic_comparison" / "aghformer_v6_stage2_raw_fine_refiner_p12000"

DATA_ROOT = DRIVE_ROOT / "data" / "dataset"
RUN_ROOT = DRIVE_ROOT / "hardnet_runs"


def main():
    parser = argparse.ArgumentParser(description="Colab runner for AGH-HardNet.")
    parser.add_argument("--preset", choices=["smoke", "oracle", "full"], default="smoke")
    parser.add_argument("--agh-run", type=Path, default=AGH_RUN)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()

    work_dir = CODE_ROOT / "agh_hardnet_refiner"
    train_csv = args.agh_run / "stage1_predictions_train.csv"
    val_csv = args.agh_run / "refined_predictions_val.csv"
    test_csv = args.agh_run / "refined_predictions_test.csv"
    cache_dir = args.agh_run / "stage1_point_cache"
    output_dir = args.run_root / f"agh_hardnet_{args.preset}"

    cmd = [
        sys.executable,
        "-u",
        str(work_dir / "run_agh_hardnet_refiner.py"),
        "--train-pred-csv",
        str(train_csv),
        "--val-pred-csv",
        str(val_csv),
        "--test-pred-csv",
        str(test_csv),
        "--point-cache-dir",
        str(cache_dir),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(output_dir),
        "--device",
        "auto",
        "--mixed-precision",
    ]
    if args.preset == "oracle":
        cmd += ["--oracle-only", "--oracle-patch-points", "2048", "--max-surface-points", "12000"]
    elif args.preset == "smoke":
        cmd += [
            "--epochs",
            "2",
            "--patience",
            "2",
            "--batch-size",
            "4",
            "--patch-points",
            "256",
            "--oracle-patch-points",
            "512",
            "--max-surface-points",
            "2048",
            "--hidden-dim",
            "64",
            "--max-items",
            "24",
            "--no-tqdm",
        ]
    else:
        cmd += [
            "--epochs",
            "120",
            "--patience",
            "25",
            "--batch-size",
            "12",
            "--patch-points",
            "2048",
            "--oracle-patch-points",
            "4096",
            "--max-surface-points",
            "12000",
            "--hidden-dim",
            "192",
            "--radius-mm",
            "45",
            "--trichion-radius-mm",
            "55",
            "--sigma-mm",
            "2.5",
            "--topk",
            "20",
        ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True, env=env)


if __name__ == "__main__":
    main()
