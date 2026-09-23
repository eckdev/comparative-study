#!/usr/bin/env python3
"""Run leakage-free DiffusionNet folds on Colab/Google Drive."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO = Path("/content/comparative-study")
DEFAULT_DATA = Path("/content/drive/MyDrive/orthodontic/data/dataset")
DEFAULT_PREPROCESSING = Path(
    "/content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/"
    "publication_cv_stage1_v4_seed42"
)
DEFAULT_RUN_ROOT = Path("/content/drive/MyDrive/orthodontic/diffusion_runs")


def require(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def selected_folds(value, total):
    if not value:
        return list(range(1, total + 1))
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    invalid = [fold for fold in result if fold < 1 or fold > total]
    if invalid:
        raise ValueError(f"Invalid fold indices: {invalid}")
    return result


def validate_preprocessing(fold_spec, preprocessing_fold):
    transform = require(
        preprocessing_fold / "alignment" / "mesh_only_transforms.npz",
        "Fold transform archive",
    )
    alignment = require(
        preprocessing_fold / "alignment" / "alignment_report.json",
        "Fold alignment report",
    )
    split_report = require(
        preprocessing_fold / "split_and_leakage_report.json",
        "Fold preprocessing split report",
    )
    report = json.loads(split_report.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        if list(report["splits"][split]) != list(fold_spec[split]):
            raise ValueError(
                f"Preprocessing split mismatch for fold {fold_spec['fold']} / {split}"
            )
    alignment_payload = json.loads(alignment.read_text(encoding="utf-8"))
    if alignment_payload.get("uses_expert_landmarks") is not False:
        raise ValueError(f"Fold {fold_spec['fold']} alignment is not label-free")
    if alignment_payload.get("scale") is not False:
        raise ValueError(f"Fold {fold_spec['fold']} alignment changes physical scale")
    fitted_ids = set(alignment_payload.get("atlas_sample_ids", []))
    fitted_ids.update(alignment_payload.get("train_template_sample_ids", []))
    if alignment_payload.get("train_medoid_sample_id"):
        fitted_ids.add(alignment_payload["train_medoid_sample_id"])
    if fitted_ids and not fitted_ids <= set(fold_spec["train"]):
        raise ValueError(f"Fold {fold_spec['fold']} alignment was not fit on train only")
    return transform, alignment


def preset_settings(preset):
    if preset == "smoke":
        return {
            "surface_points": 512,
            "k_eig": 16,
            "epochs": 2,
            "min_epochs": 1,
            "patience": 2,
            "width": 32,
            "blocks": 2,
            "mlp_hidden_dims": "64",
            "max_samples": 24,
        }
    return {
        "surface_points": 12000,
        "k_eig": 96,
        "epochs": 220,
        "min_epochs": 80,
        "patience": 35,
        "width": 192,
        "blocks": 8,
        "mlp_hidden_dims": "384",
        "max_samples": None,
    }


def fold_command(args, fold_spec, split_path, transform_path, alignment_path, output_dir):
    settings = preset_settings(args.preset)
    command = [
        sys.executable,
        "-u",
        str(Path(args.repo_root) / "diffusion_net_orthodontic_comparison" / "run_orthodontic_diffusion.py"),
        "--data-root", str(args.data_root),
        "--splits-json", str(split_path),
        "--transformation-npz", str(transform_path),
        "--alignment-report", str(alignment_path),
        "--require-label-free-alignment",
        "--output-dir", str(output_dir),
        "--fold-number", str(fold_spec["fold"]),
        "--surface-points", str(settings["surface_points"]),
        "--k-eig", str(settings["k_eig"]),
        "--epochs", str(settings["epochs"]),
        "--min-epochs", str(settings["min_epochs"]),
        "--patience", str(settings["patience"]),
        "--width", str(settings["width"]),
        "--blocks", str(settings["blocks"]),
        "--mlp-hidden-dims", settings["mlp_hidden_dims"],
        "--loss-mode", "mask_bce",
        "--mask-radius", "3.5",
        "--input-features", "xyz",
        "--postprocess", "topk_softmax",
        "--refine-topk", "30",
        "--refine-temperature", "1.0",
        "--checkpoint-metric", "val_ale",
        "--lr", "0.001",
        "--device", args.device,
        "--seed", str(args.seed),
        "--no-tqdm",
    ]
    if args.resume and not args.force:
        command.append("--resume")
    if settings["max_samples"] is not None:
        command.extend(["--max-samples", str(settings["max_samples"])])
    if args.preset == "preflight":
        command.append("--preflight-only")
    return command


def aggregate(repo_root, output_root, folds, seed, allow_partial=False):
    command = [
        sys.executable,
        "-u",
        str(Path(repo_root) / "shared_metrics" / "aggregate_cv_results.py"),
        "--run-dir", str(output_root),
        "--model", "DiffusionNet",
        "--folds", str(folds),
        "--seed", str(seed),
    ]
    if allow_partial:
        command.append("--allow-partial")
    subprocess.run(command, cwd=str(repo_root), check=True)


def main():
    parser = argparse.ArgumentParser(description="DiffusionNet shared leakage-free CV runner.")
    parser.add_argument("--preset", choices=["preflight", "smoke", "cv"], default="preflight")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_REPO / "shared_splits/orthodontic_5fold_192_48_60_seed42.json",
    )
    parser.add_argument("--preprocessing-root", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--fold-indices", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    require(args.repo_root, "Repository")
    require(args.data_root, "Dataset")
    require(args.manifest, "CV manifest")
    require(args.preprocessing_root, "Shared preprocessing root")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol_path = require(
        args.repo_root
        / "diffusion_net_orthodontic_comparison"
        / "publication_cv_protocol.json",
        "Frozen DiffusionNet CV protocol",
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["manifest_sha256"] != manifest.get("manifest_sha256"):
        raise ValueError("Frozen DiffusionNet protocol and CV manifest hashes do not match")
    folds = manifest["folds"]
    requested = selected_folds(args.fold_indices, len(folds))
    if args.preset == "smoke" and args.fold_indices is None:
        requested = [1]

    default_names = {
        "preflight": f"diffusionnet_cv_preflight_seed{args.seed}",
        "smoke": f"diffusionnet_cv_smoke_seed{args.seed}",
        "cv": f"diffusionnet_publication_cv_seed{args.seed}",
    }
    output_root = args.run_root / (args.output_name or default_names[args.preset])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cv_manifest_snapshot.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_root / "publication_cv_protocol_snapshot.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    for fold in requested:
        fold_spec = folds[fold - 1]
        preprocessing_fold = args.preprocessing_root / f"fold_{fold}"
        transform, alignment = validate_preprocessing(fold_spec, preprocessing_fold)
        output_dir = output_root / f"fold_{fold}"
        output_dir.mkdir(parents=True, exist_ok=True)
        complete = output_dir / "metrics.json"
        predictions = output_dir / "predictions_test.csv"
        if args.preset == "preflight":
            complete = output_dir / "preflight_report.json"
            predictions = complete
        if complete.exists() and predictions.exists() and not args.force:
            print(f"Fold {fold} already complete; skipping: {output_dir}", flush=True)
            continue
        split_path = output_dir / "fold_split.json"
        split_path.write_text(
            json.dumps(
                {
                    "repeat": fold_spec.get("repeat", 1),
                    "fold": fold,
                    "train": fold_spec["train"],
                    "val": fold_spec["val"],
                    "test": fold_spec["test"],
                    "manifest_sha256": manifest.get("manifest_sha256"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        command = fold_command(
            args, fold_spec, split_path, transform, alignment, output_dir
        )
        print(f"\nDiffusionNet fold {fold}/{len(folds)} preset={args.preset}", flush=True)
        print("Running:", " ".join(shlex.quote(part) for part in command), flush=True)
        subprocess.run(command, cwd=str(args.repo_root), check=True, env=environment)

    if args.preset == "cv":
        completed = sum(
            (output_root / f"fold_{fold}" / "predictions_test.csv").exists()
            for fold in range(1, len(folds) + 1)
        )
        if completed:
            aggregate(
                args.repo_root,
                output_root,
                len(folds),
                args.seed,
                allow_partial=completed < len(folds),
            )
        print(f"Completed folds: {completed}/{len(folds)}", flush=True)
    print(f"Output root: {output_root}", flush=True)


if __name__ == "__main__":
    main()
