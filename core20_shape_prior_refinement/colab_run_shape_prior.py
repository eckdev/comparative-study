import argparse
from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path("/content/comparative-study")
RUN_ROOT = Path("/content/drive/MyDrive/orthodontic/diffusion_runs")
PREDICTION_CANDIDATES = [
    RUN_ROOT / "aghformer_v12_stage3_core20_refiner_v6",
    RUN_ROOT / "aghformer_v11_stage3_mid_refiner_v6",
    RUN_ROOT / "aghformer_v6_stage2_raw_fine_refiner_p12000",
]


def choose_prediction_dir():
    for candidate in PREDICTION_CANDIDATES:
        if (candidate / "base_stage2_predictions_train.csv").exists() or (candidate / "refined_predictions_train.csv").exists():
            return candidate
    return PREDICTION_CANDIDATES[-1]


def main():
    parser = argparse.ArgumentParser(description="Colab runner for shape-prior residual refiner.")
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-landmarks", default="all")
    parser.add_argument("--gate-landmarks", default="all")
    parser.add_argument("--selection-metric", choices=["all", "core20", "target"], default="core20")
    parser.add_argument("--min-val-improvement-mm", default="0.0")
    args = parser.parse_args()

    work_dir = CODE_ROOT / "core20_shape_prior_refinement"
    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else choose_prediction_dir()
    output_dir = Path(args.output_dir) if args.output_dir else RUN_ROOT / "shape_prior_residual_refiner"

    cmd = [
        sys.executable,
        "-u",
        str(work_dir / "run_shape_prior_refiner.py"),
        "--prediction-dir",
        str(prediction_dir),
        "--output-dir",
        str(output_dir),
        "--target-landmarks",
        args.target_landmarks,
        "--gate-landmarks",
        args.gate_landmarks,
        "--selection-metric",
        args.selection_metric,
        "--min-val-improvement-mm",
        args.min_val_improvement_mm,
    ]
    print("Prediction dir:", prediction_dir, flush=True)
    print("Output dir:", output_dir, flush=True)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True)


if __name__ == "__main__":
    main()
