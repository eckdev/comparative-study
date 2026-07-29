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
    parser.add_argument("--preset", choices=["smoke", "oracle", "full"], default="smoke")
    parser.add_argument("--agh-run", type=Path, default=DEFAULT_AGH_RUN)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()

    work_dir = CODE_ROOT / "agh_hardnet_refiner"
    args.agh_run = resolve_agh_run(args.agh_run)
    print("AGH_RUN:", args.agh_run, flush=True)
    train_csv = args.agh_run / "stage1_predictions_train.csv"
    val_csv = args.agh_run / "refined_predictions_val.csv"
    test_csv = args.agh_run / "refined_predictions_test.csv"
    cache_dir = args.agh_run / "stage1_point_cache"
    output_dir = args.run_root / f"agh_hardnet_{args.preset}"

    if not train_csv.exists() and args.preset != "oracle":
        raise FileNotFoundError(f"Missing train prediction file: {train_csv}")
    if not cache_dir.exists():
        print(f"Warning: point cache not found, falling back to --data-root meshes: {cache_dir}", flush=True)

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
