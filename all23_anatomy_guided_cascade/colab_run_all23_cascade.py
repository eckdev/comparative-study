import argparse
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path("/content/comparative-study")
DRIVE_ROOT = Path("/content/drive/MyDrive/orthodontic")
RUN_ROOT = DRIVE_ROOT / "diffusion_runs"
DATA_ROOT = DRIVE_ROOT / "data/dataset"
TRANSFORM_DIR = DRIVE_ROOT / "transforms/orthodontic_procrustes_rigid_20260627_143801"
SPLITS_JSON = CODE_ROOT / "shared_splits/orthodontic_180_60_60_seed42.json"
DEFAULT_BASE = RUN_ROOT / "aghformer_v6_stage2_raw_fine_refiner_p12000"
DEFAULT_INITIAL = RUN_ROOT / "shape_prior_stacker"


PRESETS = {
    "smoke": {
        "output_name": "all23_cascade_smoke",
        "candidate_points": 512,
        "width": 64,
        "epochs": 2,
        "patience": 2,
        "batch_size": 2,
        "eval_batch_size": 2,
        "max_samples": 12,
        "bootstrap_iters": 100,
    },
    "a100": {
        "output_name": "all23_anatomy_guided_cascade_v1",
        "candidate_points": 4096,
        "width": 192,
        "epochs": 160,
        "patience": 30,
        "batch_size": 4,
        "eval_batch_size": 4,
        "max_samples": None,
        "bootstrap_iters": 2000,
    },
}


def ensure_train_predictions(base_dir, data_root, splits_json, transform_dir, device):
    train_csv = base_dir / "refined_predictions_train.csv"
    if train_csv.exists():
        print("Stage2 train predictions found:", train_csv, flush=True)
        return
    exporter = CODE_ROOT / "agh_former_orthodontic_comparison/export_stage2_train_predictions.py"
    command = [
        sys.executable,
        "-u",
        str(exporter),
        "--stage2-run-dir",
        str(base_dir),
        "--data-root",
        str(data_root),
        "--splits-json",
        str(splits_json),
        "--device",
        device,
    ]
    if transform_dir.exists():
        command.extend(["--transformation-dir", str(transform_dir)])
    print("refined_predictions_train.csv is missing; exporting it once.", flush=True)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(exporter.parent), check=True)


def main():
    parser = argparse.ArgumentParser(description="Colab runner for the all-23 anatomy-guided cascade.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="a100")
    parser.add_argument("--base-prediction-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--initial-prediction-dir", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--point-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--splits-json", type=Path, default=SPLITS_JSON)
    parser.add_argument("--transformation-dir", type=Path, default=TRANSFORM_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    work_dir = CODE_ROOT / "all23_anatomy_guided_cascade"
    point_cache_dir = args.point_cache_dir or args.base_prediction_dir / "stage1_point_cache"
    output_dir = args.output_dir or RUN_ROOT / preset["output_name"]

    if not args.base_prediction_dir.exists():
        raise FileNotFoundError(f"Base AGH run was not found: {args.base_prediction_dir}")
    if not args.initial_prediction_dir.exists():
        raise FileNotFoundError(
            f"Initial all-23 prediction run was not found: {args.initial_prediction_dir}. "
            "Run the shape-prior stacker first or pass --initial-prediction-dir."
        )
    if not point_cache_dir.exists():
        raise FileNotFoundError(f"Stage1 point cache was not found: {point_cache_dir}")

    ensure_train_predictions(
        args.base_prediction_dir,
        args.data_root,
        args.splits_json,
        args.transformation_dir,
        args.device,
    )

    command = [
        sys.executable,
        "-u",
        str(work_dir / "run_all23_cascade.py"),
        "--base-prediction-dir",
        str(args.base_prediction_dir),
        "--initial-prediction-dir",
        str(args.initial_prediction_dir),
        "--point-cache-dir",
        str(point_cache_dir),
        "--output-dir",
        str(output_dir),
        "--candidate-points",
        str(preset["candidate_points"]),
        "--width",
        str(preset["width"]),
        "--epochs",
        str(preset["epochs"]),
        "--patience",
        str(preset["patience"]),
        "--batch-size",
        str(preset["batch_size"]),
        "--eval-batch-size",
        str(preset["eval_batch_size"]),
        "--bootstrap-iters",
        str(preset["bootstrap_iters"]),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ]
    if preset["max_samples"] is not None:
        command.extend(["--max-samples", str(preset["max_samples"])])
    if not args.no_mixed_precision:
        command.append("--mixed-precision")

    print("Preset:", args.preset, flush=True)
    print("Base prediction dir:", args.base_prediction_dir, flush=True)
    print("Initial all-23 prediction dir:", args.initial_prediction_dir, flush=True)
    print("Point cache dir:", point_cache_dir, flush=True)
    print("Output dir:", output_dir, flush=True)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(work_dir), check=True)


if __name__ == "__main__":
    main()
