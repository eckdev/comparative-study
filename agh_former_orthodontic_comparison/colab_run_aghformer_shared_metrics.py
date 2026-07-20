import argparse
import os
import subprocess
import sys
from pathlib import Path


DATA_ROOT = Path("/content/drive/MyDrive/orthodontic/data/dataset")
SPLITS_JSON = Path("/content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json")
TRANSFORM_DIR = Path("/content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801")
RUN_ROOT = Path("/content/drive/MyDrive/orthodontic/diffusion_runs")


def repo_root(explicit=None):
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    return here.parent


def has_paired_dataset(path):
    path = Path(path)
    return any(path.glob("Class*/men/*.ply")) or any(path.glob("Class*/women/*.ply"))


def main():
    parser = argparse.ArgumentParser(description="Colab presets for AGH-Former orthodontic landmark localization.")
    parser.add_argument("--preset", choices=["smoke", "a100", "a100_16k", "stage2", "stage2_smoke"], default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--splits-json", default=str(SPLITS_JSON))
    parser.add_argument("--transformation-dir", default=str(TRANSFORM_DIR))
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--stage1-run-dir", default=None)
    args = parser.parse_args()

    root = repo_root(args.repo_root)
    work_dir = root / "agh_former_orthodontic_comparison"
    data_root = Path(args.data_root)
    splits_json = Path(args.splits_json)
    transform_dir = Path(args.transformation_dir)
    run_root = Path(args.run_root)
    use_transforms = transform_dir.exists()

    print(f"CODE_ROOT = {root}", flush=True)
    print(f"DATA_ROOT = {data_root}", flush=True)
    print(f"SPLITS_JSON = {splits_json}", flush=True)
    print(f"TRANSFORM_DIR = {transform_dir}", flush=True)
    print(f"RUN_ROOT = {run_root}", flush=True)
    print(f"USE_TRANSFORMS = {use_transforms}", flush=True)
    if not has_paired_dataset(data_root):
        raise SystemExit(
            "Dataset bulunamadi. Beklenen yapi: "
            f"{data_root}/Class1/men/*.ply ve "
            f"{data_root}/Class1/Class1-Landmark/men/Class1_M1.txt"
        )
    if not splits_json.exists():
        raise SystemExit(f"Split dosyasi bulunamadi: {splits_json}")

    common = [
        sys.executable,
        "-u",
        str(work_dir / "run_orthodontic_aghformer.py"),
        "--data-root",
        str(data_root),
        "--splits-json",
        str(splits_json),
        "--device",
        args.device,
    ]
    if use_transforms:
        common.extend(["--transformation-dir", str(transform_dir)])

    if args.preset == "smoke":
        cmd = common + [
            "--output-dir",
            str(run_root / "aghformer_v2_template_smoke_colab"),
            "--surface-points",
            "512",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--batch-size",
            "2",
            "--width",
            "64",
            "--blocks",
            "1",
            "--heads",
            "4",
            "--topk",
            "10",
            "--template-mode",
            "class_gender",
            "--prediction-mode",
            "direct",
            "--selection-metric",
            "raw",
            "--coord-weight",
            "1.0",
            "--heatmap-positive-weight",
            "20",
            "--heatmap-ce-weight",
            "0.05",
            "--max-samples",
            "24",
        ]
    elif args.preset == "a100":
        cmd = common + [
            "--output-dir",
            str(run_root / "aghformer_v2_template_p12000_w192_b4_e220"),
            "--surface-points",
            "12000",
            "--eval-surface-points",
            "12000",
            "--epochs",
            "220",
            "--patience",
            "35",
            "--batch-size",
            "2",
            "--lr",
            "0.0008",
            "--weight-decay",
            "0.0001",
            "--width",
            "192",
            "--blocks",
            "4",
            "--heads",
            "6",
            "--mlp-ratio",
            "2.0",
            "--heatmap-sigma-start",
            "5.0",
            "--heatmap-sigma-end",
            "2.5",
            "--heatmap-loss",
            "weighted_mse",
            "--heatmap-positive-weight",
            "20",
            "--heatmap-ce-weight",
            "0.05",
            "--template-mode",
            "class_gender",
            "--prediction-mode",
            "direct",
            "--selection-metric",
            "raw",
            "--residual-scale",
            "0.18",
            "--coord-weight",
            "1.0",
            "--structure-weight",
            "0.08",
            "--symmetry-weight",
            "0.02",
            "--clinical-weight",
            "0.05",
            "--uncertainty-weight",
            "0.02",
            "--rotation-aug-deg",
            "2.0",
            "--point-jitter-std",
            "0.001",
            "--feature-dropout",
            "0.05",
            "--topk",
            "30",
            "--temperature",
            "1.0",
        ]
    elif args.preset == "a100_16k":
        cmd = common + [
            "--output-dir",
            str(run_root / "aghformer_v2_template_p16000_w192_b4_e240"),
            "--surface-points",
            "16000",
            "--eval-surface-points",
            "16000",
            "--epochs",
            "240",
            "--patience",
            "40",
            "--batch-size",
            "1",
            "--lr",
            "0.0006",
            "--weight-decay",
            "0.0001",
            "--width",
            "192",
            "--blocks",
            "4",
            "--heads",
            "6",
            "--mlp-ratio",
            "2.0",
            "--heatmap-sigma-start",
            "5.0",
            "--heatmap-sigma-end",
            "2.5",
            "--heatmap-loss",
            "weighted_mse",
            "--heatmap-positive-weight",
            "20",
            "--heatmap-ce-weight",
            "0.05",
            "--template-mode",
            "class_gender",
            "--prediction-mode",
            "direct",
            "--selection-metric",
            "raw",
            "--residual-scale",
            "0.18",
            "--coord-weight",
            "1.0",
            "--structure-weight",
            "0.08",
            "--symmetry-weight",
            "0.02",
            "--clinical-weight",
            "0.05",
            "--uncertainty-weight",
            "0.02",
            "--rotation-aug-deg",
            "2.0",
            "--point-jitter-std",
            "0.001",
            "--feature-dropout",
            "0.05",
            "--topk",
            "30",
        ]
    elif args.preset == "stage2_smoke":
        stage1_run_dir = Path(args.stage1_run_dir) if args.stage1_run_dir else run_root / "aghformer_v2_template_smoke_colab"
        cmd = [
            sys.executable,
            "-u",
            str(work_dir / "run_aghformer_stage2_refiner.py"),
            "--data-root",
            str(data_root),
            "--splits-json",
            str(splits_json),
            "--stage1-run-dir",
            str(stage1_run_dir),
            "--output-dir",
            str(run_root / "aghformer_v4_stage2_heatmap_smoke_colab"),
            "--surface-points",
            "512",
            "--patch-points",
            "128",
            "--patch-radius-mm",
            "15",
            "--patch-heatmap-sigma-mm",
            "3.0",
            "--epochs",
            "2",
            "--patience",
            "2",
            "--batch-size",
            "64",
            "--refiner-width",
            "64",
            "--final-mode",
            "center_delta",
            "--heatmap-refine-weight",
            "0.2",
            "--patch-heatmap-weight",
            "0.25",
            "--patch-heatmap-positive-weight",
            "20",
            "--patch-heatmap-ce-weight",
            "0.05",
            "--projection-mode",
            "topk_distance",
            "--projection-topk",
            "5",
            "--max-samples",
            "24",
            "--device",
            args.device,
        ]
        if use_transforms:
            cmd.extend(["--transformation-dir", str(transform_dir)])
    else:
        stage1_run_dir = Path(args.stage1_run_dir) if args.stage1_run_dir else run_root / "aghformer_v2_template_p12000_w192_b4_e220"
        cmd = [
            sys.executable,
            "-u",
            str(work_dir / "run_aghformer_stage2_refiner.py"),
            "--data-root",
            str(data_root),
            "--splits-json",
            str(splits_json),
            "--stage1-run-dir",
            str(stage1_run_dir),
            "--output-dir",
            str(run_root / "aghformer_v4_stage2_heatmap_refiner_p12000"),
            "--surface-points",
            "12000",
            "--patch-points",
            "1024",
            "--patch-radius-mm",
            "18",
            "--patch-heatmap-sigma-mm",
            "3.0",
            "--stage1-center",
            "snapped",
            "--epochs",
            "160",
            "--patience",
            "30",
            "--batch-size",
            "256",
            "--lr",
            "0.001",
            "--weight-decay",
            "0.0001",
            "--refiner-width",
            "192",
            "--landmark-embedding-dim",
            "48",
            "--residual-limit-mm",
            "12",
            "--final-mode",
            "center_delta",
            "--heatmap-refine-weight",
            "0.2",
            "--heatmap-temperature",
            "0.8",
            "--patch-heatmap-weight",
            "0.25",
            "--patch-heatmap-positive-weight",
            "20",
            "--patch-heatmap-ce-weight",
            "0.05",
            "--eval-coordinate-mode",
            "raw_final",
            "--eval-topk",
            "30",
            "--projection-mode",
            "topk_distance",
            "--projection-topk",
            "5",
            "--center-jitter-mm",
            "1.5",
            "--point-noise-mm",
            "0.1",
            "--point-dropout",
            "0.05",
            "--clinical-weight",
            "0.08",
            "--delta-reg-weight",
            "0.002",
            "--uncertainty-weight",
            "0.01",
            "--device",
            args.device,
        ]
        if use_transforms:
            cmd.extend(["--transformation-dir", str(transform_dir)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True, env=env)


if __name__ == "__main__":
    main()
