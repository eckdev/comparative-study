import argparse
from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path("/content/comparative-study")
DATA_ROOT = Path("/content/drive/MyDrive/orthodontic/data/dataset")
SPLITS_JSON = CODE_ROOT / "shared_splits/orthodontic_180_60_60_seed42.json"
TRANSFORM_DIR = Path("/content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801")
RUN_ROOT = Path("/content/drive/MyDrive/orthodontic/diffusion_runs")
BASE_RUN_DIR = RUN_ROOT / "aghformer_v6_stage2_raw_fine_refiner_p12000"
BASE_PREDICTION_CANDIDATES = [
    RUN_ROOT / "aghformer_v12_stage3_core20_refiner_v6",
    RUN_ROOT / "aghformer_v11_stage3_mid_refiner_v6",
    BASE_RUN_DIR,
]


def choose_base_prediction_dir():
    for candidate in BASE_PREDICTION_CANDIDATES:
        if (candidate / "base_stage2_predictions_train.csv").exists() or (candidate / "refined_predictions_train.csv").exists():
            return candidate
    return BASE_RUN_DIR


def main():
    parser = argparse.ArgumentParser(description="Colab presets for core20 2.5D local heatmap refiner.")
    parser.add_argument("--preset", choices=["smoke", "a100"], default="smoke")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    work_dir = CODE_ROOT / "core20_2p5d_refinement"
    base_prediction_dir = choose_base_prediction_dir()
    common = [
        sys.executable,
        "-u",
        str(work_dir / "run_core20_2p5d_refiner.py"),
        "--data-root",
        str(DATA_ROOT),
        "--splits-json",
        str(SPLITS_JSON),
        "--base-run-dir",
        str(BASE_RUN_DIR),
        "--base-prediction-dir",
        str(base_prediction_dir),
        "--device",
        args.device,
    ]
    if TRANSFORM_DIR.exists():
        common.extend(["--transformation-dir", str(TRANSFORM_DIR)])

    if args.preset == "smoke":
        cmd = common + [
            "--output-dir",
            str(RUN_ROOT / "core20_2p5d_smoke"),
            "--grid-size",
            "48",
            "--patch-radius-mm",
            "8",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--batch-size",
            "64",
            "--width",
            "32",
            "--max-samples",
            "24",
            "--num-workers",
            "0",
        ]
    else:
        cmd = common + [
            "--output-dir",
            str(RUN_ROOT / "core20_2p5d_p96_r8_e160"),
            "--grid-size",
            "96",
            "--patch-radius-mm",
            "8",
            "--epochs",
            "160",
            "--patience",
            "30",
            "--batch-size",
            "128",
            "--width",
            "64",
            "--lr",
            "0.0008",
            "--clinical-weight",
            "0.35",
            "--coord-weight",
            "1.0",
            "--heatmap-weight",
            "1.0",
            "--improvement-weight",
            "0.35",
            "--delta-reg-weight",
            "0.01",
            "--focus-min-mm",
            "1.5",
            "--focus-max-mm",
            "3.2",
            "--focus-weight",
            "2.0",
            "--num-workers",
            "2",
        ]

    print("Base run:", BASE_RUN_DIR, flush=True)
    print("Base prediction dir:", base_prediction_dir, flush=True)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True)


if __name__ == "__main__":
    main()
