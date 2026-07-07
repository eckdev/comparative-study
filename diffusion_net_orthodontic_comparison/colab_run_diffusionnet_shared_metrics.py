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
    diffusion_dir = repo_root / "diffusion_net_orthodontic_comparison"
    script_path = diffusion_dir / "run_orthodontic_diffusion.py"
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
        "--k-eig",
        str(args.k_eig),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--width",
        str(args.width),
        "--blocks",
        str(args.blocks),
        "--mlp-hidden-dims",
        args.mlp_hidden_dims,
        "--loss-mode",
        args.loss_mode,
        "--mask-radius",
        str(args.mask_radius),
        "--input-features",
        args.input_features,
        "--postprocess",
        args.postprocess,
        "--refine-topk",
        str(args.refine_topk),
        "--refine-temperature",
        str(args.refine_temperature),
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]

    if args.transformation_dir:
        cmd.extend(["--transformation-dir", str(Path(args.transformation_dir))])
    if args.use_mesh_vertices:
        cmd.append("--use-mesh-vertices")
    if args.evaluate_only:
        cmd.append("--evaluate-only")
        if args.model_path:
            cmd.extend(["--model-path", str(Path(args.model_path))])

    return cmd, diffusion_dir, output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Colab runner for DiffusionNet with shared 180/60/60 split and extended orthodontic metrics."
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
    parser.add_argument("--run-root", default="/content/drive/MyDrive/orthodontic/diffusion_runs")
    parser.add_argument("--output-name", default="diffusionnet_shared_metrics_p12000_k96_w192_b8_e220_topk30")
    parser.add_argument("--surface-points", type=int, default=12000)
    parser.add_argument("--k-eig", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--mlp-hidden-dims", default="384")
    parser.add_argument("--loss-mode", choices=["ce", "mask_bce"], default="mask_bce")
    parser.add_argument("--mask-radius", type=float, default=3.5)
    parser.add_argument("--input-features", choices=["xyz", "hks", "xyz_hks"], default="xyz")
    parser.add_argument(
        "--postprocess",
        choices=["argmax", "topk_softmax", "softmax", "sigmoid_weighted"],
        default="topk_softmax",
    )
    parser.add_argument("--refine-topk", type=int, default=30)
    parser.add_argument("--refine-temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-mesh-vertices", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    assert_path(args.repo_root, "Repo root")
    assert_path(args.data_root, "Dataset")
    assert_path(args.splits_json, "Shared split JSON")
    if args.transformation_dir:
        assert_path(args.transformation_dir, "Transformation directory")
    if args.evaluate_only and args.model_path:
        assert_path(args.model_path, "Model checkpoint")

    cmd, diffusion_dir, output_dir = build_command(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_MININTERVAL"] = "1"

    print("DiffusionNet ortak split + yeni metrik kosusu basliyor.", flush=True)
    print("Output:", output_dir, flush=True)
    print("Command:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(diffusion_dir), check=True, env=env)


if __name__ == "__main__":
    main()
