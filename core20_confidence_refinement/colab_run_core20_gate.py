import argparse
from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path("/content/comparative-study")


def main():
    parser = argparse.ArgumentParser(description="Colab runner for core20 confidence refinement.")
    parser.add_argument("--stage3-run-dir", default="/content/drive/MyDrive/orthodontic/diffusion_runs/aghformer_v11_stage3_mid_refiner_v6")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/orthodontic/diffusion_runs/core20_gate_v11")
    parser.add_argument("--target-landmarks", default="2,10,11,12,13,16,19,20")
    args = parser.parse_args()
    work_dir = CODE_ROOT / "core20_confidence_refinement"
    cmd = [
        sys.executable,
        "-u",
        str(work_dir / "run_core20_confidence_refinement.py"),
        "--stage3-run-dir",
        str(Path(args.stage3_run_dir)),
        "--output-dir",
        str(Path(args.output_dir)),
        "--target-landmarks",
        args.target_landmarks,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(work_dir), check=True)


if __name__ == "__main__":
    main()
