import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def assert_path(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} bulunamadi: {path}")


def build_command(args):
    repo_root = Path(args.repo_root)
    pointnet_dir = repo_root / "pointnet2_orthodontic_comparison"
    script_path = pointnet_dir / "run_orthodontic_pointnet2.py"
    output_dir = Path(args.run_root) / args.output_name

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--data-root",
        str(Path(args.data_root)),
        "--splits-json",
        str(Path(args.splits_json)),
        "--output-dir",
        str(output_dir),
        "--surface-points",
        str(args.surface_points),
        "--eval-surface-points",
        str(args.eval_surface_points),
        "--sa1-points",
        str(args.sa1_points),
        "--sa2-points",
        str(args.sa2_points),
        "--sa3-points",
        str(args.sa3_points),
        "--nsample",
        str(args.nsample),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--optimizer",
        args.optimizer,
        "--scheduler",
        args.scheduler,
        "--target-mode",
        args.target_mode,
        "--heatmap-sigma",
        str(args.heatmap_sigma),
        "--coord-weight",
        str(args.coord_weight),
        "--coord-temperature",
        str(args.coord_temperature),
        "--postprocess",
        args.postprocess,
        "--topk",
        str(args.topk),
        "--temperature",
        str(args.temperature),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]

    if args.use_normals:
        cmd.append("--use-normals")
    else:
        cmd.append("--no-use-normals")
    if args.transformation_dir:
        cmd.extend(["--transformation-dir", str(Path(args.transformation_dir))])
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])

    return cmd, pointnet_dir, output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Colab runner for PointNet++ with shared 180/60/60 split and extended orthodontic metrics."
    )
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
    parser.add_argument("--run-root", default="/content/drive/MyDrive/orthodontic/pointnet2_runs")
    parser.add_argument("--output-name", default="pointnet2_shared_metrics_p4096_e200_topk20")
    parser.add_argument("--surface-points", type=int, default=4096)
    parser.add_argument("--eval-surface-points", type=int, default=4096)
    parser.add_argument("--sa1-points", type=int, default=1024)
    parser.add_argument("--sa2-points", type=int, default=256)
    parser.add_argument("--sa3-points", type=int, default=64)
    parser.add_argument("--nsample", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="cosine")
    parser.add_argument("--target-mode", choices=["gaussian", "mask"], default="gaussian")
    parser.add_argument("--heatmap-sigma", type=float, default=3.5)
    parser.add_argument("--use-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coord-weight", type=float, default=0.25)
    parser.add_argument("--coord-temperature", type=float, default=1.0)
    parser.add_argument("--postprocess", choices=["softmax", "topk_softmax", "argmax"], default="topk_softmax")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    assert_path(args.repo_root, "Repo root")
    assert_path(args.data_root, "Dataset")
    assert_path(args.splits_json, "Shared split JSON")
    if args.transformation_dir:
        assert_path(args.transformation_dir, "Transformation directory")

    cmd, pointnet_dir, output_dir = build_command(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_MININTERVAL"] = "1"

    print("PointNet++ ortak split + yeni metrik kosusu basliyor.", flush=True)
    print("Output:", output_dir, flush=True)
    print("Command:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(pointnet_dir), check=True, env=env)


if __name__ == "__main__":
    main()
