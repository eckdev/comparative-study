import argparse
from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path("/content/comparative-study")
DRIVE_ROOT = Path("/content/drive/MyDrive/orthodontic")
RUN_ROOT = Path("/content/drive/MyDrive/orthodontic/diffusion_runs")
DATA_ROOT = DRIVE_ROOT / "data" / "dataset"
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
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--splits-json", default=str(CODE_ROOT / "shared_splits" / "orthodontic_180_60_60_seed42.json"))
    parser.add_argument(
        "--transformation-dir",
        default=str(DRIVE_ROOT / "transforms" / "orthodontic_procrustes_rigid_20260627_143801"),
    )
    parser.add_argument("--target-landmarks", default="all")
    parser.add_argument("--gate-landmarks", default="all")
    parser.add_argument("--selection-metric", choices=["all", "core20", "target"], default="core20")
    parser.add_argument("--final-policy", choices=["shape_prior", "gated"], default="shape_prior")
    parser.add_argument("--min-val-improvement-mm", default="0.0")
    parser.add_argument("--bootstrap-iters", default="2000")
    parser.add_argument("--skip-refined-train-export", action="store_true")
    args = parser.parse_args()

    work_dir = CODE_ROOT / "core20_shape_prior_refinement"
    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else choose_prediction_dir()
    output_dir = Path(args.output_dir) if args.output_dir else RUN_ROOT / "shape_prior_residual_refiner"
    if not (prediction_dir / "refined_predictions_train.csv").exists() and not args.skip_refined_train_export:
        export_script = CODE_ROOT / "agh_former_orthodontic_comparison" / "export_stage2_train_predictions.py"
        export_cmd = [
            sys.executable,
            "-u",
            str(export_script),
            "--stage2-run-dir",
            str(prediction_dir),
            "--data-root",
            str(args.data_root),
            "--splits-json",
            str(args.splits_json),
            "--device",
            "auto",
            "--eval-batch-size",
            "256",
        ]
        transform_dir = Path(args.transformation_dir)
        if transform_dir.exists():
            export_cmd += ["--transformation-dir", str(transform_dir)]
        print("refined_predictions_train.csv not found; exporting it first.", flush=True)
        print("Running:", " ".join(export_cmd), flush=True)
        subprocess.run(export_cmd, cwd=str(CODE_ROOT / "agh_former_orthodontic_comparison"), check=True)

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
        "--final-policy",
        args.final_policy,
        "--min-val-improvement-mm",
        args.min_val_improvement_mm,
        "--bootstrap-iters",
        args.bootstrap_iters,
    ]
    print("Prediction dir:", prediction_dir, flush=True)
    print("Output dir:", output_dir, flush=True)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True)


if __name__ == "__main__":
    main()
