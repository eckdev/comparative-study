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
            "--coarse-source", "train_template",
            "--alignment", "mesh_icp",
            "--experiment", "E1",
            "--max-samples", "12",
            "--icp-points", "512",
            "--icp-iterations", "3",
            "--roi-points", "32",
            "--width", "32",
            "--global-blocks", "1",
            "--epochs", "2",
            "--patience", "2",
            "--bootstrap-iters", "50",
            "--skip-oracle-gate",
            "--seed", str(seed),
        ]
    if preset == "a100":
        return fixed_external(f"full_fixed_seed{seed}", "FULL") + [
            "--amp-dtype", "bfloat16",
            "--roi-points", "512",
            "--width", "128",
            "--global-blocks", "4",
            "--epochs", "200",
            "--patience", "35",
            "--seed", str(seed),
        ]
    if preset in ("cv", "cv_repeated", "cv_preflight"):
        repeats = "3" if preset == "cv_repeated" else "1"
        output_preset = "cv" if preset == "cv_preflight" else preset
        command = common(f"publication_{output_preset}_v2_seed{seed}") + [
            "--protocol", "cv",
            "--folds", "5",
            "--cv-repeats", repeats,
            "--coarse-source", "train_template",
            "--train-center-mode", "template",
            "--center-jitter-mm", "1.0",
            "--alignment", "mesh_icp",
            "--experiment", "FULL",
            "--roi-points", "1024",
            "--roi-radius-scale", "1.5",
            "--width", "128",
            "--global-blocks", "4",
            "--epochs", "200",
            "--patience", "35",
            "--lr", "0.0003",
            "--max-nonfinite-fraction", "0.01",
            "--max-amp-overflow-fraction", "0.10",
            "--amp-dtype", "bfloat16",
            "--seed", str(seed),
        ]
        if preset == "cv_preflight":
            command.append("--preflight-only")
        return command
    if preset.startswith("e") and preset[1:].isdigit() and 1 <= int(preset[1:]) <= 7:
        experiment = preset.upper()
        return fixed_external(f"ablation_{preset}_seed{seed}", experiment) + [
            "--roi-points", "512",
            "--width", "128",
            "--global-blocks", "4",
            "--epochs", "200",
            "--patience", "35",
            "--seed", str(seed),
        ]
    raise ValueError("Preset must be smoke, a100, cv_preflight, cv, cv_repeated, or e0..e7")


def validate_paths(preset):
    if preset == "e0":
        required = [INITIAL_RUN]
    else:
        required = [DATA_ROOT]
    if preset not in ("cv", "cv_repeated", "cv_preflight", "smoke", "e0"):
        required.extend([SPLITS_JSON, TRANSFORM_DIR, BASE_RUN, INITIAL_RUN])
    elif preset == "smoke":
        required.append(SPLITS_JSON)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Colab paths:\n- " + "\n- ".join(missing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    validate_paths(args.preset)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    command = command_for(args.preset.lower(), args.seed)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    print("Working directory:", WORK_DIR, flush=True)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(CODE_ROOT), check=True, env=environment)


if __name__ == "__main__":
    main()
