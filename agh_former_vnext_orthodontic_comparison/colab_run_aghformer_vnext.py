#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path("/content/comparative-study")
DATA_ROOT = Path("/content/drive/MyDrive/orthodontic/data/dataset")
RUN_ROOT = Path("/content/drive/MyDrive/orthodontic/agh_vnext_runs")
SPLITS_JSON = CODE_ROOT / "shared_splits/orthodontic_180_60_60_seed42.json"
PREPROCESSING_ROOT = Path(
    "/content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/"
    "publication_cv_stage1_v4_seed42"
)


def command_for(preset, seed, fold_indices):
    script = CODE_ROOT / "agh_former_vnext_orthodontic_comparison/run_aghformer_vnext.py"
    common = [
        sys.executable,
        "-u",
        str(script),
        "--data-root",
        str(DATA_ROOT),
        "--device",
        "auto",
        "--mixed-precision",
        "--amp-dtype",
        "bfloat16",
        "--no-tqdm",
        "--seed",
        str(seed),
    ]
    if preset == "smoke":
        return common + [
            "--output-dir",
            str(RUN_ROOT / f"smoke_seed{seed}"),
            "--protocol",
            "fixed",
            "--splits-json",
            str(SPLITS_JSON),
            "--coarse-source",
            "train_template",
            "--train-center-mode",
            "template",
            "--max-samples",
            "24",
            "--folds",
            "2",
            "--roi-points",
            "128",
            "--icp-points",
            "512",
            "--icp-iterations",
            "3",
            "--atlas-size",
            "2",
            "--atlas-iterations",
            "1",
            "--registration-candidates",
            "1",
            "--registration-restarts",
            "1",
            "--roi-radius-scale",
            "1.5",
            "--width",
            "32",
            "--global-blocks",
            "1",
            "--token-blocks",
            "1",
            "--token-surface-points",
            "512",
            "--epochs",
            "2",
            "--min-epochs",
            "1",
            "--patience",
            "1",
            "--gate-stage-epochs",
            "1",
            "--gate-stage-min-epochs",
            "1",
            "--gate-stage-patience",
            "1",
            "--bootstrap-iters",
            "50",
            "--skip-oracle-gate",
            "--max-stage2-val-ale",
            "200",
            "--no-shape-prior",
            "--no-tta",
            "--no-tta-validation",
        ]
    output_name = (
        f"publication_cv_preflight_seed{seed}"
        if preset == "cv_preflight"
        else f"publication_cv_seed{seed}"
    )
    command = common + [
        "--output-dir",
        str(RUN_ROOT / output_name),
        "--protocol",
        "cv",
        "--folds",
        "5",
        "--cv-repeats",
        "1",
        "--coarse-source",
        "stage1_oof",
        "--train-center-mode",
        "external",
        "--alignment",
        "mesh_icp",
        "--registration-roi-expansion",
        "0.75",
        "--roi-points",
        "1024",
        "--roi-radius-scale",
        "1.5",
        "--width",
        "128",
        "--global-blocks",
        "4",
        "--token-blocks",
        "2",
        "--token-surface-points",
        "4096",
        "--epochs",
        "220",
        "--min-epochs",
        "80",
        "--patience",
        "35",
        "--lr",
        "0.0003",
        "--stage1-oof-folds",
        "5",
        "--stage1-oof-fixed-epochs",
        "120",
        "--stage1-width",
        "96",
        "--stage1-blocks",
        "3",
        "--stage1-epochs",
        "100",
        "--stage1-min-epochs",
        "60",
        "--stage1-patience",
        "20",
    ]
    if PREPROCESSING_ROOT.exists():
        command.extend(["--preprocessing-root", str(PREPROCESSING_ROOT)])
    if fold_indices:
        command.extend(["--fold-indices", fold_indices])
    if preset == "cv_preflight":
        command.append("--preflight-only")
    elif preset == "dev_fold1":
        command.extend(["--fold-indices", "1", "--validation-only"])
    return command


def main():
    parser = argparse.ArgumentParser(description="Colab runner for AGH-Former vNext")
    parser.add_argument(
        "--preset",
        choices=("smoke", "cv_preflight", "dev_fold1", "cv"),
        default="smoke",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-indices", default=None)
    args = parser.parse_args()
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    command = command_for(args.preset, args.seed, args.fold_indices)
    print("Working directory:", CODE_ROOT, flush=True)
    print("Running:", " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=str(CODE_ROOT), check=True, env=environment)


if __name__ == "__main__":
    main()
