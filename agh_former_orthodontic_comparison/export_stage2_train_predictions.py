import argparse
import csv
import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def coerce_config(config):
    defaults = {
        "data_root": "../data/dataset",
        "splits_json": None,
        "transformation_dir": None,
        "stage1_run_dir": "",
        "stage1_model_path": None,
        "stage1_batch_size": 2,
        "output_dir": "",
        "surface_points": 12000,
        "heatmap_sigma_start": 5.0,
        "template_mode": "class_gender",
        "stage1_center": "snapped",
        "stage1_postprocess": "topk_softmax",
        "stage1_prediction_mode": "direct",
        "stage1_temperature": 1.0,
        "stage1_topk": 30,
        "stage1_width": 192,
        "stage1_blocks": 4,
        "stage1_heads": 6,
        "stage1_mlp_ratio": 2.0,
        "stage1_dropout": 0.1,
        "stage1_residual_scale": 0.18,
        "symmetry_pairs": "13-16,14-15,17-18,19-20,21-22",
        "midline_indices": "0,1,2,3,4,5,6,7,8,9,10,11,12",
        "patch_points": 1024,
        "patch_radius_mm": 12.0,
        "patch_heatmap_sigma_mm": 2.0,
        "center_jitter_mm": 0.0,
        "point_noise_mm": 0.0,
        "point_dropout": 0.0,
        "refiner_width": 256,
        "landmark_embedding_dim": 64,
        "refiner_dropout": 0.1,
        "residual_limit_mm": 12.0,
        "final_mode": "center_delta",
        "heatmap_refine_weight": 0.05,
        "heatmap_temperature": 0.8,
        "patch_heatmap_weight": 0.25,
        "patch_heatmap_positive_weight": 20.0,
        "patch_heatmap_ce_weight": 0.05,
        "eval_coordinate_mode": "raw_final",
        "eval_topk": 30,
        "projection_mode": "none",
        "projection_topk": 5,
        "selection_metric": "raw",
        "landmark_weighting": "train_error",
        "landmark_weight_min": 0.75,
        "landmark_weight_max": 2.5,
        "epochs": 160,
        "patience": 30,
        "batch_size": 192,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "clinical_weight": 0.08,
        "clinical_threshold_mm": 2.0,
        "delta_reg_weight": 0.002,
        "uncertainty_weight": 0.01,
        "grad_clip": 1.0,
        "test_size": 0.2,
        "val_size": 0.2,
        "seed": 42,
        "device": "auto",
        "max_samples": None,
        "num_workers": 0,
        "no_tqdm": False,
    }
    merged = defaults.copy()
    merged.update(config)
    return Namespace(**merged)


def read_stage1_predictions(path, dataset, indices, stage1_center):
    sample_to_idx = {dataset.samples[i].sample_id: i for i in indices}
    stage1 = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if sample_id not in sample_to_idx:
                continue
            sample_idx = sample_to_idx[sample_id]
            entry = stage1.setdefault(
                sample_idx,
                {
                    "raw": np.zeros((23, 3), dtype=np.float32),
                    "snapped": np.zeros((23, 3), dtype=np.float32),
                    "center": np.zeros((23, 3), dtype=np.float32),
                    "expert": np.zeros((23, 3), dtype=np.float32),
                    "raw_errors": np.zeros(23, dtype=np.float32),
                    "snapped_errors": np.zeros(23, dtype=np.float32),
                },
            )
            lm_idx = int(row["landmark"])
            for prefix in ("stage1_raw", "stage1_snapped", "expert"):
                values = np.asarray([float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")], dtype=np.float32)
                if prefix == "stage1_raw":
                    entry["raw"][lm_idx] = values
                elif prefix == "stage1_snapped":
                    entry["snapped"][lm_idx] = values
                else:
                    entry["expert"][lm_idx] = values
            entry["raw_errors"][lm_idx] = float(row.get("raw_localization_error", 0.0))
            entry["snapped_errors"][lm_idx] = float(row.get("snapped_localization_error", 0.0))
    missing = [dataset.samples[i].sample_id for i in indices if i not in stage1]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing Stage1 predictions for {len(missing)} train samples. First missing: {preview}")
    for entry in stage1.values():
        entry["center"] = entry[stage1_center].copy()
    return stage1


def resolve_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(description="Export AGH-Former Stage2 train refined predictions from an existing run.")
    parser.add_argument("--stage2-run-dir", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits-json", default=None)
    parser.add_argument("--transformation-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    from run_aghformer_stage2_refiner import (
        LocalPatchResidualRefiner,
        Stage2PatchDataset,
        build_stage1_dataset,
        evaluate_refiner,
        save_refiner_outputs,
    )

    stage2_run_dir = Path(args.stage2_run_dir)
    output_csv = stage2_run_dir / "refined_predictions_train.csv"
    if output_csv.exists() and not args.overwrite:
        print(f"Already exists: {output_csv}", flush=True)
        return

    config_path = stage2_run_dir / "config_stage2.json"
    checkpoint_path = stage2_run_dir / "best_refiner.pth"
    stage1_csv = stage2_run_dir / "stage1_predictions_train.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_stage2.json: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_refiner.pth: {checkpoint_path}")
    if not stage1_csv.exists():
        raise FileNotFoundError(f"Missing stage1_predictions_train.csv: {stage1_csv}")

    config = load_json(config_path)
    stage1_config_path = Path(config.get("stage1_run_dir", "")) / "config.json"
    stage1_config = load_json(stage1_config_path) if stage1_config_path.exists() else {}
    run_args = coerce_config(config)
    run_args.output_dir = str(stage2_run_dir)
    if args.data_root is not None:
        run_args.data_root = args.data_root
    if args.splits_json is not None:
        run_args.splits_json = args.splits_json
    if args.transformation_dir is not None:
        run_args.transformation_dir = args.transformation_dir
    run_args.device = args.device
    run_args.num_workers = args.num_workers
    run_args.no_tqdm = args.no_tqdm
    run_args.max_samples = None

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = resolve_device(args.device)
    dataset, train_idx, _, _, _, _ = build_stage1_dataset(run_args, stage1_config)
    stage1_train = read_stage1_predictions(stage1_csv, dataset, train_idx, run_args.stage1_center)
    train_ds = Stage2PatchDataset(
        dataset,
        train_idx,
        stage1_train,
        patch_points=run_args.patch_points,
        patch_radius_mm=run_args.patch_radius_mm,
        center_jitter_mm=0.0,
        point_noise_mm=0.0,
        point_dropout=0.0,
        heatmap_sigma_mm=run_args.patch_heatmap_sigma_mm,
        seed=run_args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    refiner = LocalPatchResidualRefiner(
        input_dim=7,
        width=run_args.refiner_width,
        landmark_dim=run_args.landmark_embedding_dim,
        dropout=run_args.refiner_dropout,
        residual_limit_mm=run_args.residual_limit_mm,
    ).to(device)
    refiner.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Device: {device}", flush=True)
    print(f"Exporting Stage2 train predictions: n={len(train_idx)} -> {output_csv}", flush=True)
    rows, raw_errors, snapped_errors, stage1_errors = evaluate_refiner(refiner, train_loader, dataset, train_idx, device, run_args)
    metrics = save_refiner_outputs(
        stage2_run_dir,
        run_args,
        dataset,
        train_idx,
        rows,
        raw_errors,
        snapped_errors,
        stage1_errors,
        suffix="train",
    )
    print(f"Train Stage1 center ALE: {metrics['stage1_center_baseline']['ale']:.4f}", flush=True)
    print(f"Train Stage2 raw ALE: {metrics['stage2_raw']['ale']:.4f}", flush=True)
    print(f"Train Stage2 snapped ALE: {metrics['stage2_snapped']['ale']:.4f}", flush=True)
    print(f"Saved: {output_csv}", flush=True)


if __name__ == "__main__":
    main()
