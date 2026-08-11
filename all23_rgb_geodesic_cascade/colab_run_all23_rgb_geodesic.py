#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path("/content/comparative-study")
WORK_DIR = CODE_ROOT / "all23_rgb_geodesic_cascade"
DATA_ROOT = Path("/content/drive/MyDrive/orthodontic/data/dataset")
SPLITS_JSON = CODE_ROOT / "shared_splits/orthodontic_180_60_60_seed42.json"
TRANSFORM_DIR = Path("/content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801")
RUN_ROOT = Path("/content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs")
BASE_RUN = Path("/content/drive/MyDrive/orthodontic/diffusion_runs/aghformer_v6_stage2_raw_fine_refiner_p12000")
INITIAL_RUN = Path("/content/drive/MyDrive/orthodontic/diffusion_runs/shape_prior_stacker")


def common(output_name):
    return [
        sys.executable,
        "-u",
        str(WORK_DIR / "run_all23_rgb_geodesic.py"),
        "--data-root", str(DATA_ROOT),
        "--output-dir", str(RUN_ROOT / output_name),
        "--device", "auto",
        "--mixed-precision",
        "--no-tqdm",
    ]


def fixed_external(output_name, experiment):
    return common(output_name) + [
        "--protocol", "fixed",
        "--splits-json", str(SPLITS_JSON),
        "--coarse-source", "external",
        "--base-prediction-dir", str(BASE_RUN),
        "--initial-prediction-dir", str(INITIAL_RUN),
        "--legacy-transformation-dir", str(TRANSFORM_DIR),
        "--alignment", "mesh_icp",
        "--train-center-mode", "synthetic",
        "--experiment", experiment,
    ]


def command_for(preset, seed):
    if preset == "e0":
        return [
            sys.executable, "-u", str(WORK_DIR / "evaluate_baseline.py"),
            "--prediction-dir", str(INITIAL_RUN),
            "--output-dir", str(RUN_ROOT / "ablation_e0_stacker"),
            "--prefix", "stacked",
            "--seed", str(seed),
        ]
    if preset == "smoke":
        return common("smoke") + [
            "--protocol", "fixed",
            "--splits-json", str(SPLITS_JSON),
            "--coarse-source", "stage1_oof",
            "--train-center-mode", "external",
            "--alignment", "mesh_icp",
            "--experiment", "E9",
            "--max-samples", "24",
            "--icp-points", "128",
            "--icp-iterations", "2",
            "--atlas-size", "2",
            "--atlas-iterations", "1",
            "--registration-candidates", "1",
            "--registration-restarts", "1",
            "--roi-points", "32",
            "--roi-mode", "hybrid",
            "--roi-multi-seeds", "2",
            "--width", "32",
            "--global-blocks", "1",
            "--epochs", "2",
            "--min-epochs", "1",
            "--patience", "2",
            "--scheduler-start-epoch", "1",
            "--lr-warmup-epochs", "0",
            "--stage1-oof-folds", "2",
            "--stage1-oof-mode", "fixed_epoch",
            "--stage1-oof-fixed-epochs", "2",
            "--stage1-width", "32",
            "--stage1-blocks", "1",
            "--stage1-topk", "10",
            "--stage1-epochs", "2",
            "--stage1-min-epochs", "1",
            "--stage1-patience", "1",
            "--stage1-scheduler-start-epoch", "1",
            "--stage1-lr-warmup-epochs", "0",
            "--max-stage1-val-ale", "100",
            "--max-stage1-oof-ale", "100",
            "--max-stage1-oof-p95", "100",
            "--max-stage1-oof-val-gap", "100",
            "--max-stage2-val-ale", "100",
            "--hard-landmark-weight", "4.0",
            "--hard-rank-weight", "1.0",
            "--gate-weight", "0.5",
            "--checkpoint-metric", "balanced",
            "--refinement-calibration", "group_scale",
            "--bootstrap-iters", "50",
            "--skip-oracle-gate",
            "--seed", str(seed),
        ]
    if preset == "a100":
        return fixed_external(f"full_fixed_v4_seed{seed}", "E8") + [
            "--amp-dtype", "bfloat16",
            "--roi-points", "512",
            "--width", "128",
            "--global-blocks", "4",
            "--epochs", "200",
            "--patience", "35",
            "--seed", str(seed),
        ]
    if preset in ("cv", "cv_repeated", "cv_preflight", "e9_dev", "e9_cv"):
        repeats = "3" if preset == "cv_repeated" else "1"
        is_e9 = preset in ("e9_dev", "e9_cv")
        output_preset = "cv" if preset == "cv_preflight" else preset
        output_name = (
            f"publication_e9_cv_seed{seed}"
            if is_e9
            else f"publication_{output_preset}_stage1_v4_seed{seed}"
        )
        command = common(output_name) + [
            "--protocol", "cv",
            "--folds", "5",
            "--cv-repeats", repeats,
            "--coarse-source", "stage1_oof",
            "--train-center-mode", "external",
            "--center-jitter-mm", "1.0",
            "--alignment", "mesh_icp",
            "--atlas-size", "8",
            "--atlas-iterations", "2",
            "--registration-candidates", "3",
            "--registration-restarts", "2",
            "--registration-trim-quantile", "0.8",
            "--registration-roi-expansion", "0.75",
            "--experiment", "E9" if is_e9 else "E8",
            "--roi-points", "1024",
            "--roi-radius-scale", "1.5",
            "--roi-mode", "hybrid",
            "--roi-euclidean-scale", "1.25",
            "--roi-multi-seeds", "3",
            "--width", "128",
            "--global-blocks", "4",
            "--epochs", "220",
            "--min-epochs", "80",
            "--patience", "35",
            "--scheduler-start-epoch", "60",
            "--lr-warmup-epochs", "5",
            "--lr", "0.0003",
            "--stage1-oof-folds", "5",
            "--stage1-oof-mode", "fixed_epoch",
            "--stage1-oof-fixed-epochs", "120",
            "--stage1-oof-template-alpha", "auto",
            "--stage1-width", "96",
            "--stage1-blocks", "3",
            "--stage1-epochs", "100",
            "--stage1-min-epochs", "60",
            "--stage1-patience", "20",
            "--stage1-scheduler-start-epoch", "40",
            "--stage1-lr-warmup-epochs", "5",
            "--stage1-lr", "0.0003",
            "--max-stage1-val-ale", "6.0",
            "--max-stage1-oof-ale", "6.0",
            "--max-stage1-oof-p95", "12.0",
            "--max-stage1-oof-val-gap", "2.0",
            "--max-stage2-val-ale", "5.0",
            "--max-val-oracle-p95", "2.0",
            "--max-val-oracle-max", "15.0",
            "--max-val-sample-oracle-ale", "2.0",
            "--max-nonfinite-fraction", "0.01",
            "--max-amp-overflow-fraction", "0.10",
            "--amp-dtype", "bfloat16",
            "--seed", str(seed),
        ]
        if preset == "cv_preflight":
            command.append("--preflight-only")
        if is_e9:
            preprocessing_root = RUN_ROOT / f"publication_cv_stage1_v4_seed{seed}"
            command.extend(
                [
                    "--preprocessing-root", str(preprocessing_root),
                    "--hard-landmark-weight", "4.0",
                    "--hard-rank-weight", "1.0",
                    "--gate-weight", "0.5",
                    "--checkpoint-metric", "balanced",
                    "--checkpoint-hard3-weight", "0.35",
                    "--refinement-calibration", "group_scale",
                ]
            )
            if preset == "e9_dev":
                command.extend(["--fold-indices", "1", "--validation-only"])
        return command
    if preset.startswith("e") and preset[1:].isdigit() and 1 <= int(preset[1:]) <= 9:
        experiment = preset.upper()
        command = fixed_external(f"ablation_{preset}_seed{seed}", experiment) + [
            "--roi-points", "512",
            "--width", "128",
            "--global-blocks", "4",
            "--epochs", "200",
            "--patience", "35",
            "--seed", str(seed),
        ]
        if preset == "e9":
            command.extend(
                [
                    "--hard-landmark-weight", "4.0",
                    "--hard-rank-weight", "1.0",
                    "--gate-weight", "0.5",
                    "--checkpoint-metric", "balanced",
                    "--refinement-calibration", "group_scale",
                ]
            )
        return command
    raise ValueError(
        "Preset must be smoke, a100, cv_preflight, cv, cv_repeated, "
        "e9_dev, e9_cv, or e0..e9"
    )


def validate_paths(preset, seed):
    if preset == "e0":
        required = [INITIAL_RUN]
    else:
        required = [DATA_ROOT]
    if preset not in (
        "cv", "cv_repeated", "cv_preflight", "e9_dev", "e9_cv", "smoke", "e0"
    ):
        required.extend([SPLITS_JSON, TRANSFORM_DIR, BASE_RUN, INITIAL_RUN])
    elif preset == "smoke":
        required.append(SPLITS_JSON)
    if preset in ("e9_dev", "e9_cv"):
        required.append(RUN_ROOT / f"publication_cv_stage1_v4_seed{seed}")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Colab paths:\n- " + "\n- ".join(missing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    validate_paths(args.preset, args.seed)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    command = command_for(args.preset.lower(), args.seed)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    print("Working directory:", WORK_DIR, flush=True)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(CODE_ROOT), check=True, env=environment)


if __name__ == "__main__":
    main()
