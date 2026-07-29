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


PRESETS = {
    "global_full": {
        "feature_mode": "full",
        "calibration_mode": "global",
        "output_name": "shape_prior_residual_refiner",
    },
    "per_landmark_flat": {
        "feature_mode": "flat",
        "calibration_mode": "per_landmark",
        "output_name": "shape_prior_local_per_landmark_flat",
    },
    "per_landmark_flat_meta": {
        "feature_mode": "flat_meta",
        "calibration_mode": "per_landmark",
        "output_name": "shape_prior_local_per_landmark_flat_meta",
    },
    "per_landmark_full": {
        "feature_mode": "full",
        "calibration_mode": "per_landmark",
        "output_name": "shape_prior_local_per_landmark",
    },
}


def choose_prediction_dir():
    for candidate in PREDICTION_CANDIDATES:
        if (candidate / "base_stage2_predictions_train.csv").exists() or (candidate / "refined_predictions_train.csv").exists():
            return candidate
    return PREDICTION_CANDIDATES[-1]


def main():
    parser = argparse.ArgumentParser(description="Colab runner for shape-prior residual refiner.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
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
    parser.add_argument("--feature-mode", choices=["flat", "flat_meta", "full"], default="full")
    parser.add_argument("--calibration-mode", choices=["global", "per_landmark"], default="global")
    parser.add_argument("--l2-grid", default="0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000")
    parser.add_argument("--shrinkage-grid", default="0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.75,0.9,1.0")
    parser.add_argument("--selection-metric", choices=["all", "core20", "target"], default="core20")
    parser.add_argument("--final-policy", choices=["shape_prior", "gated"], default="shape_prior")
    parser.add_argument("--min-val-improvement-mm", default="0.0")
    parser.add_argument("--bootstrap-iters", default="2000")
    parser.add_argument("--skip-refined-train-export", action="store_true")
    args = parser.parse_args()

    work_dir = CODE_ROOT / "core20_shape_prior_refinement"
    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else choose_prediction_dir()
    if args.preset:
        preset = PRESETS[args.preset]
        args.feature_mode = preset["feature_mode"]
        args.calibration_mode = preset["calibration_mode"]
        output_dir = Path(args.output_dir) if args.output_dir else RUN_ROOT / preset["output_name"]
    else:
        output_dir = Path(args.output_dir) if args.output_dir else RUN_ROOT / "shape_prior_residual_refiner"
        output_name = output_dir.name.lower()
        if args.feature_mode == "full" and "flat_meta" in output_name:
            args.feature_mode = "flat_meta"
        elif args.feature_mode == "full" and "flat" in output_name:
            args.feature_mode = "flat"
        if args.calibration_mode == "global" and "per_landmark" in output_name:
            args.calibration_mode = "per_landmark"

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
        "--feature-mode",
        args.feature_mode,
        "--calibration-mode",
        args.calibration_mode,
        "--l2-grid",
        args.l2_grid,
        "--shrinkage-grid",
        args.shrinkage_grid,
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
    print("Feature mode:", args.feature_mode, flush=True)
    print("Calibration mode:", args.calibration_mode, flush=True)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True)


if __name__ == "__main__":
    main()
