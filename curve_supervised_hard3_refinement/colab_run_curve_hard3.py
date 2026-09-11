#!/usr/bin/env python3
"""Colab entry point for the Curve-H3 development and annotated runs."""

from __future__ import annotations

import argparse
import json
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
    "/content/drive/MyDrive/orthodontic/annotations/hard3_curves_pilot_v2.json"
)


def _fully_annotated_count(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"hairline", "jaw_left", "jaw_right"}
    return sum(
        required.issubset(row.get("curves", {}))
        and all(len(row["curves"][name]) >= 2 for name in required)
        for row in payload.get("samples", {}).values()
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
        "--hard3-curve-pretrain-epochs",
        str(args.curve_pretrain_epochs),
        "--hard3-curve-pretrain-lr",
        str(args.curve_pretrain_lr),
        "--hard3-curve-real-sample-fraction",
        str(args.real_sample_fraction),
        "--hard3-curve-checkpoint-weight",
        str(args.curve_checkpoint_weight),
        "--hard3-curve-max-annotation-surface-distance-mm",
        str(args.maximum_annotation_surface_distance_mm),
        "--hard3-curve-max-landmark-curve-distance-mm",
        str(args.maximum_landmark_curve_distance_mm),
    ]
    if args.preset == "pseudo_smoke":
        curve_args.extend(
            [
                "--hard3-curve-run-mode",
                "pseudo",
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
                "--hard3-curve-pretrain-epochs",
                "0",
                "--bootstrap-iters",
                "50",
            ]
        )
    elif args.preset == "pseudo_fold1":
        curve_args.extend(
            [
                "--hard3-curve-run-mode",
                "pseudo",
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
        run_mode = "pilot" if args.preset == "annotated_pilot" else "publication"
        minimum = (
            args.pilot_annotated_samples
            if run_mode == "pilot"
            else args.minimum_annotated_samples
        )
        annotated = _fully_annotated_count(manifest)
        if annotated < minimum:
            raise RuntimeError(
                f"{args.preset} requires {minimum} fully annotated samples in "
                f"{manifest}; found {annotated}. Fill hairline, jaw_left and "
                "jaw_right for every selected sample before launching training."
            )
        print(
            f"Curve annotation preflight: {annotated} fully annotated samples",
            flush=True,
        )
        curve_args.extend(
            [
                "--hard3-curve-run-mode",
                run_mode,
                "--hard3-curve-annotation-manifest",
                str(manifest),
                "--hard3-curve-min-annotated-samples",
                str(minimum),
                "--hard3-curve-publication-min-annotated-samples",
                str(args.publication_annotated_samples),
                "--hard3-curve-allow-pseudo",
            ]
        )
    return command + curve_args


def annotation_qa_command(args):
    if args.preset not in ("annotated_pilot", "annotated_fold1"):
        return None
    minimum = (
        args.pilot_annotated_samples
        if args.preset == "annotated_pilot"
        else args.minimum_annotated_samples
    )
    manifest = Path(args.annotation_manifest)
    return [
        sys.executable,
        "-u",
        str(CODE_ROOT / "curve_supervised_hard3_refinement/validate_annotations.py"),
        "--data-root",
        str(DATA_ROOT),
        "--manifest",
        str(manifest),
        "--minimum-annotated-samples",
        str(minimum),
        "--maximum-surface-distance-mm",
        str(args.maximum_annotation_surface_distance_mm),
        "--maximum-landmark-distance-mm",
        str(args.maximum_landmark_curve_distance_mm),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=(
            "pseudo_smoke",
            "pseudo_fold1",
            "annotated_pilot",
            "annotated_fold1",
        ),
        default="pseudo_smoke",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir")
    parser.add_argument("--annotation-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--minimum-annotated-samples", type=int, default=60)
    parser.add_argument("--pilot-annotated-samples", type=int, default=24)
    parser.add_argument("--publication-annotated-samples", type=int, default=60)
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
    parser.add_argument("--curve-pretrain-epochs", type=int, default=20)
    parser.add_argument("--curve-pretrain-lr", type=float, default=5e-4)
    parser.add_argument("--real-sample-fraction", type=float, default=0.5)
    parser.add_argument("--curve-checkpoint-weight", type=float, default=0.05)
    parser.add_argument(
        "--maximum-annotation-surface-distance-mm", type=float, default=5.0
    )
    parser.add_argument("--maximum-landmark-curve-distance-mm", type=float, default=5.0)
    args = parser.parse_args()
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")
    command = build_command(args)
    qa_command = annotation_qa_command(args)
    if qa_command is not None:
        print("Validating curve annotations before training...", flush=True)
        subprocess.run(qa_command, cwd=str(CODE_ROOT), check=True)
    print("Working directory:", CODE_ROOT, flush=True)
    print("Running:", " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=str(CODE_ROOT), check=True, env=environment)


if __name__ == "__main__":
    main()
