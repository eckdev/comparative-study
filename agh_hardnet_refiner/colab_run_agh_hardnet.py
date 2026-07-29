import argparse
import os
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path("/content/comparative-study")
DRIVE_ROOT = Path("/content/drive/MyDrive/orthodontic")
DEFAULT_AGH_RUN = DRIVE_ROOT / "diffusion_runs" / "aghformer_v6_stage2_raw_fine_refiner_p12000"

DATA_ROOT = DRIVE_ROOT / "data" / "dataset"
RUN_ROOT = DRIVE_ROOT / "hardnet_runs"


def has_required_predictions(path):
    path = Path(path)
    return (path / "refined_predictions_val.csv").exists() and (path / "refined_predictions_test.csv").exists()


def resolve_agh_run(requested):
    requested = Path(requested)
    if has_required_predictions(requested):
        return requested

    search_roots = [
        DRIVE_ROOT / "diffusion_runs",
        DRIVE_ROOT,
        CODE_ROOT / "agh_former_orthodontic_comparison",
    ]
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(path.parent for path in root.rglob("refined_predictions_val.csv"))

    valid = []
    for candidate in candidates:
        if has_required_predictions(candidate):
            score = 0
            name = candidate.name.lower()
            if "aghformer_v6" in name:
                score += 4
            if "stage2" in name:
                score += 2
            if "raw_fine" in name:
                score += 1
            valid.append((score, candidate))
    if valid:
        valid.sort(key=lambda item: (-item[0], str(item[1])))
        return valid[0][1]

    raise FileNotFoundError(
        "Could not find AGH run directory with refined_predictions_val.csv and "
        "refined_predictions_test.csv. Pass it explicitly, for example:\n"
        "python -u colab_run_agh_hardnet.py --preset oracle "
        "--agh-run /content/drive/MyDrive/orthodontic/diffusion_runs/<aghformer_run_folder>"
    )


def main():
    parser = argparse.ArgumentParser(description="Colab runner for AGH-HardNet.")
    parser.add_argument("--preset", choices=["smoke", "oracle", "full", "trichion", "gonion"], default="smoke")
    parser.add_argument("--agh-run", type=Path, default=DEFAULT_AGH_RUN)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--splits-json", type=Path, default=CODE_ROOT / "shared_splits" / "orthodontic_180_60_60_seed42.json")
    parser.add_argument(
        "--transformation-dir",
        type=Path,
        default=DRIVE_ROOT / "transforms" / "orthodontic_procrustes_rigid_20260627_143801",
    )
    parser.add_argument("--skip-refined-train-export", action="store_true")
    args = parser.parse_args()

    work_dir = CODE_ROOT / "agh_hardnet_refiner"
    args.agh_run = resolve_agh_run(args.agh_run)
    print("AGH_RUN:", args.agh_run, flush=True)
    refined_train_csv = args.agh_run / "refined_predictions_train.csv"
    if args.preset != "oracle" and not refined_train_csv.exists() and not args.skip_refined_train_export:
        export_script = CODE_ROOT / "agh_former_orthodontic_comparison" / "export_stage2_train_predictions.py"
        export_cmd = [
            sys.executable,
            "-u",
            str(export_script),
            "--stage2-run-dir",
            str(args.agh_run),
            "--data-root",
            str(args.data_root),
            "--splits-json",
            str(args.splits_json),
            "--device",
            "auto",
            "--eval-batch-size",
            "256",
        ]
        if args.transformation_dir.exists():
            export_cmd += ["--transformation-dir", str(args.transformation_dir)]
        print("refined_predictions_train.csv not found; exporting it first.", flush=True)
        print("Running:", " ".join(export_cmd), flush=True)
        subprocess.run(export_cmd, cwd=str(CODE_ROOT / "agh_former_orthodontic_comparison"), check=True)

    train_csv = refined_train_csv if refined_train_csv.exists() else args.agh_run / "stage1_predictions_train.csv"
    print("TRAIN_PRED_CSV:", train_csv, flush=True)
    val_csv = args.agh_run / "refined_predictions_val.csv"
    test_csv = args.agh_run / "refined_predictions_test.csv"
    cache_dir = args.agh_run / "stage1_point_cache"
    if args.preset == "trichion":
        output_dir = args.run_root / "agh_hardnet_trichion_candidate"
    elif args.preset == "gonion":
        output_dir = args.run_root / "agh_hardnet_gonion_candidate"
    else:
        output_dir = args.run_root / f"agh_hardnet_{args.preset}"

    if not train_csv.exists() and args.preset != "oracle":
        raise FileNotFoundError(f"Missing train prediction file: {train_csv}")
    if not cache_dir.exists():
        print(f"Warning: point cache not found, falling back to --data-root meshes: {cache_dir}", flush=True)

    cmd = [
        sys.executable,
        "-u",
        str(work_dir / "run_agh_hardnet_refiner.py"),
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
        cmd += ["--train-pred-csv", str(train_csv)]
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
        cmd += ["--train-pred-csv", str(train_csv)]
        landmark_set = "hard3"
        if args.preset == "trichion":
            landmark_set = "trichion"
        elif args.preset == "gonion":
            landmark_set = "gonion"
        cmd += [
            "--landmark-set",
            landmark_set,
            "--epochs",
            "160",
            "--patience",
            "35",
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
            "40",
            "--trichion-radius-mm",
            "45",
            "--sigma-mm",
            "1.75",
            "--topk",
            "12",
            "--temperature",
            "0.45",
            "--final-mode",
            "pred",
            "--residual-limit-mm",
            "0.5",
            "--center-prior-weight",
            "3.0",
            "--candidate-blend",
            "0.35",
            "--coord-weight",
            "0.25",
            "--weighted-coord-weight",
            "0.7",
            "--heatmap-weight",
            "0.8",
            "--hard-ce-weight",
            "0.8",
            "--clinical-weight",
            "0.25",
            "--entropy-weight",
            "0.002",
            "--residual-reg-weight",
            "0.02",
        ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True, env=env)


if __name__ == "__main__":
    main()
