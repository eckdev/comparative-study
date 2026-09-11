#!/usr/bin/env python3
"""Colab entry point for the Curve-H3 development and annotated runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agh_former_vnext_orthodontic_comparison.colab_run_aghformer_vnext import (
    CODE_ROOT,
    DATA_ROOT,
    RUN_ROOT,
    command_for,
)


DEFAULT_MANIFEST = Path(
    "/content/drive/MyDrive/orthodontic/annotations/hard3_curves_v1.json"
)


def _replace_argument(command, name, value):
    command = list(command)
    index = command.index(name)
    command[index + 1] = str(value)
    return command


def build_command(args):
    base_preset = "smoke" if args.preset == "pseudo_smoke" else "hard3_fold1"
    command = command_for(base_preset, args.seed, None)
    default_output = (
        RUN_ROOT / f"curve_h3_smoke_seed{args.seed}"
        if args.preset == "pseudo_smoke"
        else RUN_ROOT / f"publication_cv_seed{args.seed}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    command = _replace_argument(command, "--output-dir", output_dir)
    curve_args = [
        "--hard3-refiner-mode",
        "curve_supervised",
        "--hard3-curve-folds",
        str(args.curve_folds),
        "--hard3-curve-epochs",
        str(args.curve_epochs),
        "--hard3-curve-min-epochs",
        str(args.curve_min_epochs),
        "--hard3-curve-patience",
        str(args.curve_patience),
        "--hard3-curve-batch-size",
        str(args.curve_batch_size),
        "--hard3-curve-image-size",
        str(args.curve_image_size),
        "--hard3-curve-width",
        str(args.curve_width),
        "--hard3-curve-blocks",
        str(args.curve_blocks),
        "--hard3-curve-radius-scale",
        str(args.curve_radius_scale),
        "--hard3-curve-neighbors",
        str(args.curve_neighbors),
    ]
    if args.preset == "pseudo_smoke":
        curve_args.extend(
            [
                "--hard3-curve-min-annotated-samples",
                "0",
                "--hard3-curve-allow-pseudo",
                "--hard3-curve-folds",
                "2",
                "--hard3-curve-epochs",
                "2",
                "--hard3-curve-min-epochs",
                "1",
                "--hard3-curve-patience",
                "1",
                "--hard3-curve-image-size",
                "32",
                "--hard3-curve-width",
                "24",
                "--hard3-curve-blocks",
                "1",
                "--bootstrap-iters",
                "50",
            ]
        )
    elif args.preset == "pseudo_fold1":
        curve_args.extend(
            [
                "--hard3-curve-min-annotated-samples",
                "0",
                "--hard3-curve-allow-pseudo",
            ]
        )
    else:
        manifest = Path(args.annotation_manifest)
        if not manifest.exists():
            raise FileNotFoundError(
                "Curve annotation manifest not found: "
                f"{manifest}. Generate the template with prepare_annotations.py."
            )
        curve_args.extend(
            [
                "--hard3-curve-annotation-manifest",
                str(manifest),
                "--hard3-curve-min-annotated-samples",
                str(args.minimum_annotated_samples),
                "--hard3-curve-allow-pseudo",
            ]
        )
    return command + curve_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("pseudo_smoke", "pseudo_fold1", "annotated_fold1"),
        default="pseudo_smoke",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir")
    parser.add_argument("--annotation-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--minimum-annotated-samples", type=int, default=60)
    parser.add_argument("--curve-folds", type=int, default=5)
    parser.add_argument("--curve-epochs", type=int, default=100)
    parser.add_argument("--curve-min-epochs", type=int, default=30)
    parser.add_argument("--curve-patience", type=int, default=15)
    parser.add_argument("--curve-batch-size", type=int, default=8)
    parser.add_argument("--curve-image-size", type=int, default=64)
    parser.add_argument("--curve-width", type=int, default=48)
    parser.add_argument("--curve-blocks", type=int, default=2)
    parser.add_argument("--curve-radius-scale", type=float, default=1.25)
    parser.add_argument("--curve-neighbors", type=int, default=12)
    args = parser.parse_args()
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")
    command = build_command(args)
    print("Working directory:", CODE_ROOT, flush=True)
    print("Running:", " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=str(CODE_ROOT), check=True, env=environment)


if __name__ == "__main__":
    main()
