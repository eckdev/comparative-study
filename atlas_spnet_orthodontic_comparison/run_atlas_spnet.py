import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.alignment import build_fold_alignment
from src.data import AtlasDataset, discover_samples, fit_normalizer, make_patient_folds, split_leakage_report
from src.metrics import (
    CORE20,
    HARD,
    bootstrap_ci,
    group_rows,
    landmark_rows,
    save_json,
    summarize_errors,
    write_outliers,
    write_predictions,
    write_rows,
)
from src.model import AtlasSPNet
from src.train import evaluate, train_one_epoch
from src.utils import count_parameters, cpu_count_for_torch, ensure_dir, resolve_device, set_seed, write_json


def apply_experiment_defaults(args):
    if args.experiment == "E0_baseline_current_metrics":
        args.use_normals = False
        args.use_curvature = False
        args.use_refinement = False
        args.use_shape_prior = False
        args.structure_weight = 0.0
        args.symmetry_weight = 0.0
        args.confidence_weight = 0.0
        args.clinical_weight = 0.0
    elif args.experiment == "E1_leakage_free_preprocessing":
        args.use_refinement = False
        args.use_shape_prior = False
        args.structure_weight = 0.0
        args.symmetry_weight = 0.0
    elif args.experiment == "E2_normals_curvature_density_features":
        args.use_normals = True
        args.use_curvature = True
        args.use_refinement = False
        args.use_shape_prior = False
    elif args.experiment == "E3_smooth_l1_confidence_loss":
        args.confidence_weight = max(args.confidence_weight, 0.05)
        args.use_refinement = False
        args.use_shape_prior = False
    elif args.experiment == "E4_multi_scale_encoder":
        args.patch_points = max(args.patch_points, 256)
        args.use_refinement = False
        args.use_shape_prior = False
    elif args.experiment == "E5_anatomical_constraints":
        args.structure_weight = max(args.structure_weight, 0.05)
        args.symmetry_weight = max(args.symmetry_weight, 0.03)
        args.use_refinement = False
        args.use_shape_prior = True
    elif args.experiment == "E6_local_refinement_head":
        args.use_refinement = True
        args.use_shape_prior = False
    elif args.experiment == "E7_full_atlas_spnet":
        args.use_refinement = True
        args.use_shape_prior = True
    elif args.experiment == "E8_seed_ensemble":
        args.use_refinement = True
        args.use_shape_prior = True
    return args


def write_csv(path, rows):
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fold_result_row(fold_idx, metrics, training_time, param_count):
    overall = metrics["test"]["overall"]
    core = metrics["test"]["core20"]
    hard = metrics["test"]["hard_landmarks"]
    return {
        "fold": fold_idx,
        "ale": overall["ale"],
        "median": overall["median"],
        "std": overall["std"],
        "p75": overall["p75"],
        "p90": overall["p90"],
        "p95": overall["p95"],
        "p99": overall["p99"],
        "sdr_at_2mm": overall["sdr_at_2mm"],
        "sdr_at_3mm": overall["sdr_at_3mm"],
        "sdr_at_4mm": overall["sdr_at_4mm"],
        "core20_ale": core["ale"],
        "hard_ale": hard["ale"],
        "training_time_sec": training_time,
        "parameter_count": param_count,
    }


def save_eval_outputs(output_dir, split_name, samples, sample_indices, pred, expert, errors, confidence):
    write_predictions(output_dir / f"predictions_{split_name}.csv", samples, sample_indices, pred, expert, errors, confidence)
    write_rows(output_dir / f"landmark_metrics_{split_name}.csv", landmark_rows(errors))
    write_rows(output_dir / f"group_metrics_{split_name}.csv", group_rows(samples, sample_indices, errors))
    write_outliers(output_dir / f"outlier_samples_{split_name}.csv", samples, sample_indices, errors)
    return {
        "overall": summarize_errors(errors),
        "core20": summarize_errors(errors[:, CORE20]),
        "hard_landmarks": summarize_errors(errors[:, HARD]),
        "bootstrap_ale": bootstrap_ci(errors),
    }


def train_fold(args, fold_idx, samples, fold, device):
    fold_dir = ensure_dir(Path(args.output_dir) / f"fold_{fold_idx:02d}")
    split_report = split_leakage_report(samples, fold, args.patient_key)
    write_json(fold_dir / "split_report.json", split_report)
    for pair, checks in split_report["checks"].items():
        if checks["sample_id_overlap"] or checks["patient_key_overlap"]:
            raise RuntimeError(f"Split leakage detected for {pair}: {checks}")

    transforms, _ = build_fold_alignment(samples, fold, fold_dir, reflection=False)
    normalizer = fit_normalizer(samples, fold["train"], transforms, args.surface_points, args.seed + fold_idx)
    write_json(fold_dir / "normalization.json", normalizer)

    cache_dir = fold_dir / "point_cache"
    train_ds = AtlasDataset(
        samples,
        fold["train"],
        transforms,
        normalizer,
        cache_dir=cache_dir,
        num_points=args.surface_points,
        local_geometry_k=args.local_geometry_k,
        use_normals=args.use_normals,
        use_curvature=args.use_curvature,
        seed=args.seed + fold_idx * 1000,
    )
    val_ds = AtlasDataset(samples, fold["val"], transforms, normalizer, cache_dir=cache_dir, num_points=args.surface_points, local_geometry_k=args.local_geometry_k, use_normals=args.use_normals, use_curvature=args.use_curvature, seed=args.seed + fold_idx * 1000)
    test_ds = AtlasDataset(samples, fold["test"], transforms, normalizer, cache_dir=cache_dir, num_points=args.surface_points, local_geometry_k=args.local_geometry_k, use_normals=args.use_normals, use_curvature=args.use_curvature, seed=args.seed + fold_idx * 1000)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    input_dim = 3 + (3 if args.use_normals else 0) + (2 if args.use_curvature else 0)
    model = AtlasSPNet(
        input_dim=input_dim,
        width=args.width,
        heads=args.heads,
        graph_blocks=args.graph_blocks,
        patch_points=args.patch_points,
        dropout=args.dropout,
        use_refinement=args.use_refinement,
        use_shape_prior=args.use_shape_prior,
    ).to(device)
    param_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.02)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.mixed_precision and device.type == "cuda"))

    history = []
    best_val = float("inf")
    no_improve = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_parts = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        val_loss, val_parts, val_pred, val_expert, val_errors, val_conf, val_sample_indices = evaluate(model, val_loader, device, args)
        val_ale = float(val_errors.mean())
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ale": val_ale,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_parts": train_parts,
            "val_parts": val_parts,
        }
        history.append(row)
        print(f"Fold {fold_idx} Epoch {epoch:04d}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f} val_ALE={val_ale:.4f}", flush=True)
        if val_ale < best_val:
            best_val = val_ale
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_ale": val_ale, "args": vars(args)}, fold_dir / "best_model.pth")
            save_eval_outputs(fold_dir, "val", samples, val_sample_indices, val_pred, val_expert, val_errors, val_conf)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping fold {fold_idx} at epoch {epoch}", flush=True)
                break

    training_time = time.time() - start
    write_json(fold_dir / "history.json", history)
    checkpoint = torch.load(fold_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    val_loss, val_parts, val_pred, val_expert, val_errors, val_conf, val_sample_indices = evaluate(model, val_loader, device, args)
    test_loss, test_parts, test_pred, test_expert, test_errors, test_conf, test_sample_indices = evaluate(model, test_loader, device, args)
    val_metrics = save_eval_outputs(fold_dir, "val", samples, val_sample_indices, val_pred, val_expert, val_errors, val_conf)
    test_metrics = save_eval_outputs(fold_dir, "test", samples, test_sample_indices, test_pred, test_expert, test_errors, test_conf)
    metrics = {
        "fold": fold_idx,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_ale": float(checkpoint["val_ale"]),
        "parameter_count": param_count,
        "training_time_sec": training_time,
        "val": val_metrics,
        "test": test_metrics,
    }
    write_json(fold_dir / "metrics.json", metrics)
    return metrics, test_errors, test_sample_indices


def main():
    parser = argparse.ArgumentParser(description="Atlas-SPNet leakage-aware orthodontic 3D landmark training.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patient-key", choices=["gender_subject", "sample_id"], default="gender_subject")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--surface-points", type=int, default=12000)
    parser.add_argument("--patch-points", type=int, default=512)
    parser.add_argument("--local-geometry-k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--graph-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--coarse-weight", type=float, default=0.35)
    parser.add_argument("--confidence-weight", type=float, default=0.05)
    parser.add_argument("--structure-weight", type=float, default=0.05)
    parser.add_argument("--symmetry-weight", type=float, default=0.03)
    parser.add_argument("--clinical-weight", type=float, default=0.05)
    parser.add_argument("--clinical-threshold-mm", type=float, default=2.0)
    parser.add_argument("--clinical-margin-mm", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--use-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-curvature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-refinement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-shape-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--experiment",
        choices=[
            "E0_baseline_current_metrics",
            "E1_leakage_free_preprocessing",
            "E2_normals_curvature_density_features",
            "E3_smooth_l1_confidence_loss",
            "E4_multi_scale_encoder",
            "E5_anatomical_constraints",
            "E6_local_refinement_head",
            "E7_full_atlas_spnet",
            "E8_seed_ensemble",
        ],
        default="E7_full_atlas_spnet",
    )
    args = apply_experiment_defaults(parser.parse_args())

    set_seed(args.seed)
    torch.set_num_threads(cpu_count_for_torch())
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    samples, missing = discover_samples(args.data_root)
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    if not samples:
        raise RuntimeError(f"No paired samples found under {args.data_root}")
    folds = make_patient_folds(samples, args.folds, args.patient_key, args.seed, args.val_fraction)
    write_json(
        output_dir / "experiment_config.json",
        {
            **vars(args),
            "n_samples": len(samples),
            "missing_landmarks": missing,
            "folds": folds,
        },
    )
    fold_rows = []
    all_errors = []
    all_sample_indices = []
    for fold_idx, fold in enumerate(folds):
        print(f"\n=== Fold {fold_idx + 1}/{len(folds)} ===", flush=True)
        metrics, test_errors, test_sample_indices = train_fold(args, fold_idx, samples, fold, device)
        fold_rows.append(fold_result_row(fold_idx, metrics, metrics["training_time_sec"], metrics["parameter_count"]))
        all_errors.append(test_errors)
        all_sample_indices.extend(test_sample_indices)
    all_errors = np.concatenate(all_errors, axis=0)
    write_csv(output_dir / "summary_fold_metrics.csv", fold_rows)
    write_rows(output_dir / "summary_landmark_metrics.csv", landmark_rows(all_errors))
    write_rows(output_dir / "summary_group_metrics.csv", group_rows(samples, all_sample_indices, all_errors))
    write_outliers(output_dir / "outlier_samples.csv", samples, all_sample_indices, all_errors)
    summary = {
        "overall": summarize_errors(all_errors),
        "core20": summarize_errors(all_errors[:, CORE20]),
        "hard_landmarks": summarize_errors(all_errors[:, HARD]),
        "fold_mean_ale": float(np.mean([row["ale"] for row in fold_rows])),
        "fold_std_ale": float(np.std([row["ale"] for row in fold_rows])),
        "bootstrap_ale": bootstrap_ci(all_errors),
        "folds": fold_rows,
    }
    save_json(output_dir / "summary_metrics.json", summary)
    save_json(output_dir / "bootstrap_ci.json", {"overall": summary["bootstrap_ale"], "core20": bootstrap_ci(all_errors[:, CORE20]), "hard_landmarks": bootstrap_ci(all_errors[:, HARD])})
    print("\nAtlas-SPNet summary", flush=True)
    print(f"Overall ALE: {summary['overall']['ale']:.4f}", flush=True)
    print(f"Median: {summary['overall']['median']:.4f}", flush=True)
    print(f"SDR@2mm: {summary['overall']['sdr_at_2mm']:.4f}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
