import argparse
from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path("/content/comparative-study")
DATA_ROOT = Path("/content/drive/MyDrive/orthodontic/data/dataset")
RUN_ROOT = Path("/content/drive/MyDrive/orthodontic/atlas_spnet_runs")


def main():
    parser = argparse.ArgumentParser(description="Colab presets for Atlas-SPNet.")
    parser.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiment", default="E7_full_atlas_spnet")
    args = parser.parse_args()

    work_dir = CODE_ROOT / "atlas_spnet_orthodontic_comparison"
    common = [
        sys.executable,
        "-u",
        str(work_dir / "run_atlas_spnet.py"),
        "--data-root",
        str(DATA_ROOT),
        "--patient-key",
        "gender_subject",
        "--device",
        args.device,
        "--experiment",
        args.experiment,
    ]
    if args.preset == "smoke":
        cmd = common + [
            "--output-dir",
            str(RUN_ROOT / "smoke"),
            "--folds",
            "2",
            "--surface-points",
            "1024",
            "--patch-points",
            "128",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--batch-size",
            "2",
            "--width",
            "64",
            "--heads",
            "4",
            "--max-samples",
            "24",
            "--num-workers",
            "0",
        ]
    else:
        cmd = common + [
            "--output-dir",
            str(RUN_ROOT / "full_atlas_spnet"),
            "--folds",
            "5",
            "--surface-points",
            "12000",
            "--patch-points",
            "512",
            "--epochs",
            "200",
            "--patience",
            "35",
            "--batch-size",
            "2",
            "--width",
            "192",
            "--heads",
            "4",
            "--num-workers",
            "2",
            "--mixed-precision",
        ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True)


if __name__ == "__main__":
    main()
