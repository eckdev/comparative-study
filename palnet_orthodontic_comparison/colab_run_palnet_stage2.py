import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def build_command(args):
    repo_root = Path(args.repo_root)
    palnet_root = repo_root / "palnet_orthodontic_comparison"
    upstream_dir = palnet_root / "upstream"

    data_root = Path(args.data_root)
    splits_json = Path(args.splits_json)
    run_root = Path(args.run_root)
    stage1_model = Path(args.stage1_model)
    output_dir = run_root / args.output_name

    cmd = [
        sys.executable,
        "-u",
        str(upstream_dir / "run_orthodontic.py"),
        "--data-root",
        str(data_root),
        "--splits-json",
        str(splits_json),
        "--output-dir",
        str(output_dir),
        "--stage1-model-path",
        str(stage1_model),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--patch-size",
        str(args.stage1_patch_size),
        "--refiner-patch-size",
        str(args.refiner_patch_size),
        "--surface-points",
        str(args.surface_points),
        "--lr",
        str(args.lr),
        "--snap-k",
        str(args.snap_k),
        "--model",
        "PALNET",
        "--loss",
        "combined",
        "--template-mode",
        args.template_mode,
        "--train-refiner",
        "--refine-center",
        "stage1",
        "--residual-target",
        "--landmark-weighting",
        "val_error",
        "--center-jitter-mm",
        str(args.center_jitter_mm),
        "--point-noise-mm",
        str(args.point_noise_mm),
        "--point-dropout",
        str(args.point_dropout),
        "--refiner-snap-k-candidates",
        args.refiner_snap_k_candidates,
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]

    if args.transformation_dir:
        cmd.extend(["--transformation-dir", str(Path(args.transformation_dir))])

    if args.max_train_samples:
        cmd.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_val_samples:
        cmd.extend(["--max-val-samples", str(args.max_val_samples)])
    if args.max_test_samples:
        cmd.extend(["--max-test-samples", str(args.max_test_samples)])

    return cmd, upstream_dir, output_dir


def assert_path(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} bulunamadi: {path}")


def main():
    parser = argparse.ArgumentParser(description="Colab runner for PAL-Net Stage 2 residual refiner.")
    parser.add_argument("--repo-root", default="/content/comparative-study")
    parser.add_argument("--data-root", default="/content/drive/MyDrive/orthodontic/data/dataset")
    parser.add_argument(
        "--splits-json",
        default="/content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json",
    )
    parser.add_argument(
        "--transformation-dir",
        default="/content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801",
    )
    parser.add_argument("--run-root", default="/content/drive/MyDrive/orthodontic/palnet_runs")
    parser.add_argument(
        "--stage1-model",
        default="/content/drive/MyDrive/orthodontic/palnet_runs/palnet_procrustes_p1000_surface100k_e200/best_model.pth",
    )
    parser.add_argument("--output-name", default="palnet_stage2_refiner_p500_jitter1_seed42")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--stage1-patch-size", type=int, default=1000)
    parser.add_argument("--refiner-patch-size", type=int, default=500)
    parser.add_argument("--surface-points", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--snap-k", type=int, default=1)
    parser.add_argument("--template-mode", choices=["global", "class", "gender", "class_gender"], default="class_gender")
    parser.add_argument("--center-jitter-mm", type=float, default=1.0)
    parser.add_argument("--point-noise-mm", type=float, default=0.1)
    parser.add_argument("--point-dropout", type=float, default=0.05)
    parser.add_argument("--refiner-snap-k-candidates", default="1,3,5")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    assert_path(args.repo_root, "Repo root")
    assert_path(args.data_root, "Dataset")
    assert_path(args.splits_json, "Shared split")
    assert_path(args.stage1_model, "Stage 1 best_model.pth")
    if args.transformation_dir:
        assert_path(args.transformation_dir, "Transformation directory")

    cmd, upstream_dir, output_dir = build_command(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_MININTERVAL"] = "1"

    print("PAL-Net Stage 2 residual refiner basliyor.", flush=True)
    print("Output:", output_dir, flush=True)
    print("Command:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(upstream_dir), check=True, env=env)


if __name__ == "__main__":
    main()
