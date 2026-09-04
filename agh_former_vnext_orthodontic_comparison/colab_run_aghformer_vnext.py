#!/usr/bin/env python3
import argparse
import json
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


def require_hard3_development_gate(output_dir):
    decision_path = Path(output_dir) / "fold_1/hard3_stage3_decision.json"
    if not decision_path.exists():
        raise RuntimeError(
            "Fold 1 Hard3 gate is missing. Run --preset hard3_fold1 before --preset cv: "
            f"{decision_path}"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("run_full_cv", False):
        failed = [
            name for name, passed in decision.get("checks", {}).items() if not passed
        ]
        raise RuntimeError(
            "Fold 1 Hard3 gate failed; full CV is intentionally blocked. "
            f"Failed checks: {failed or ['run_full_cv']}"
        )
    return decision


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
            "--hard3-structured-folds",
            "2",
            "--hard3-structured-epochs",
            "2",
            "--hard3-structured-min-epochs",
            "1",
            "--hard3-structured-patience",
            "1",
            "--hard3-structured-batch-size",
            "4",
            "--hard3-structured-width",
            "16",
            "--hard3-structured-pair-topk",
            "4",
            "--hard3-refiner-mode",
            "dual_view",
            "--hard3-dual-view-folds",
            "2",
            "--hard3-dual-view-epochs",
            "2",
            "--hard3-dual-view-min-epochs",
            "1",
            "--hard3-dual-view-patience",
            "1",
            "--hard3-dual-view-batch-size",
            "4",
            "--hard3-dual-view-image-size",
            "32",
            "--hard3-dual-view-width",
            "8",
            "--hard3-dual-view-final-members",
            "1",
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
        "--hard3-refiner-mode",
        "dual_view",
    ]
    if PREPROCESSING_ROOT.exists():
        command.extend(["--preprocessing-root", str(PREPROCESSING_ROOT)])
    if fold_indices:
        command.extend(["--fold-indices", fold_indices])
    if preset == "cv_preflight":
        command.append("--preflight-only")
    elif preset in ("dev_fold1", "hard3_fold1"):
        command.extend(["--fold-indices", "1", "--validation-only"])
    return command


def main():
    parser = argparse.ArgumentParser(description="Colab runner for AGH-Former vNext")
    parser.add_argument(
        "--preset",
        choices=("smoke", "cv_preflight", "dev_fold1", "hard3_fold1", "cv"),
        default="smoke",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-indices", default=None)
    args = parser.parse_args()
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    command = command_for(args.preset, args.seed, args.fold_indices)
    if args.preset == "cv":
        require_hard3_development_gate(
            RUN_ROOT / f"publication_cv_seed{args.seed}"
        )
    print("Working directory:", CODE_ROOT, flush=True)
    print("Running:", " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=str(CODE_ROOT), check=True, env=environment)


if __name__ == "__main__":
    main()
