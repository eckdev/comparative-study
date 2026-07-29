import argparse
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path("/content/comparative-study")
DRIVE_ROOT = Path("/content/drive/MyDrive/orthodontic")
RUN_ROOT = DRIVE_ROOT / "diffusion_runs"
DEFAULT_BASE = RUN_ROOT / "aghformer_v6_stage2_raw_fine_refiner_p12000"
DEFAULT_CANDIDATES = [
    RUN_ROOT / "shape_prior_local_v11",
    RUN_ROOT / "shape_prior_local_flat_meta",
    RUN_ROOT / "shape_prior_residual_refiner",
    RUN_ROOT / "shape_prior_local_per_landmark_flat",
    RUN_ROOT / "shape_prior_local_per_landmark",
    RUN_ROOT / "shape_prior_local_per_landmark_flat_meta",
]


def existing_candidate_dirs(paths):
    return [path for path in paths if (path / "predictions_val.csv").exists() and (path / "predictions_test.csv").exists()]


def main():
    parser = argparse.ArgumentParser(description="Colab runner for shape-prior residual stacker.")
    parser.add_argument("--base-prediction-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--candidate-dirs", nargs="*", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT / "shape_prior_stacker")
    parser.add_argument("--l2-grid", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--shrinkage-grid", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--coef-clip", default="1.0")
    parser.add_argument("--bootstrap-iters", default="2000")
    args = parser.parse_args()

    work_dir = CODE_ROOT / "core20_shape_prior_refinement"
    candidates = args.candidate_dirs if args.candidate_dirs is not None else existing_candidate_dirs(DEFAULT_CANDIDATES)
    if not candidates:
        raise FileNotFoundError(
            "No candidate shape-prior prediction directories found. Run colab_run_shape_prior.py variants first, "
            "or pass --candidate-dirs explicitly."
        )
    cmd = [
        sys.executable,
        "-u",
        str(work_dir / "run_shape_prior_stacker.py"),
        "--base-prediction-dir",
        str(args.base_prediction_dir),
        "--candidate-dirs",
        *[str(path) for path in candidates],
        "--output-dir",
        str(args.output_dir),
        "--l2-grid",
        args.l2_grid,
        "--shrinkage-grid",
        args.shrinkage_grid,
        "--coef-clip",
        args.coef_clip,
        "--bootstrap-iters",
        args.bootstrap_iters,
    ]
    print("Base prediction dir:", args.base_prediction_dir, flush=True)
    print("Candidate dirs:", flush=True)
    for candidate in candidates:
        print(" -", candidate, flush=True)
    print("Output dir:", args.output_dir, flush=True)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True)


if __name__ == "__main__":
    main()
