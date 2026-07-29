import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import HARD_LANDMARKS, oracle_rows, read_prediction_csv
from src.metrics import combined_metrics, ensure_dir, landmark_metrics, oracle_summary, write_csv, write_json
from src.train import build_loader, combine_predictions, evaluate_hard, resolve_device, train_hardnet


def prediction_rows(samples, pred, confidence):
    rows = []
    expert = np.stack([sample.expert for sample in samples], axis=0)
    base = np.stack([sample.base for sample in samples], axis=0)
    base_errors = np.linalg.norm(base - expert, axis=-1)
    final_errors = np.linalg.norm(pred - expert, axis=-1)
    for sample_idx, sample in enumerate(samples):
        for lm in range(23):
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "class": sample.class_name,
                    "gender": sample.gender,
                    "subject_id": sample.subject_id,
                    "landmark": lm,
                    "is_hard_refined": lm in HARD_LANDMARKS,
                    "expert_x": float(expert[sample_idx, lm, 0]),
                    "expert_y": float(expert[sample_idx, lm, 1]),
                    "expert_z": float(expert[sample_idx, lm, 2]),
                    "base_x": float(base[sample_idx, lm, 0]),
                    "base_y": float(base[sample_idx, lm, 1]),
                    "base_z": float(base[sample_idx, lm, 2]),
                    "final_x": float(pred[sample_idx, lm, 0]),
                    "final_y": float(pred[sample_idx, lm, 1]),
                    "final_z": float(pred[sample_idx, lm, 2]),
                    "base_error": float(base_errors[sample_idx, lm]),
                    "final_error": float(final_errors[sample_idx, lm]),
                    "confidence": float(confidence[sample_idx, lm]),
                }
            )
    return rows


def split_metrics(samples):
    expert = np.stack([sample.expert for sample in samples], axis=0)
    base = np.stack([sample.base for sample in samples], axis=0)
    errors = np.linalg.norm(base - expert, axis=-1)
    return combined_metrics(errors)


def run_oracle(args, split_name, samples, output_dir):
    rows = oracle_rows(
        samples,
        data_root=args.data_root,
        point_cache_dir=args.point_cache_dir,
        radius_values=args.oracle_radii,
        patch_points=args.oracle_patch_points,
        max_surface_points=args.max_surface_points,
        seed=args.seed,
    )
    write_csv(output_dir / f"oracle_{split_name}.csv", rows)
    return oracle_summary(rows)


def evaluate_split(model, samples, args, split_name, output_dir):
    loader = build_loader(samples, args, split_name, shuffle=False)
    _, parts, hard_rows = evaluate_hard(model, loader, resolve_device(args.device), args)
    pred, expert, confidence, errors, metrics = combine_predictions(samples, hard_rows, mode=args.final_mode)
    write_csv(output_dir / f"predictions_{split_name}.csv", prediction_rows(samples, pred, confidence))
    write_csv(output_dir / f"landmark_metrics_{split_name}.csv", landmark_metrics(errors))
    return metrics, parts


def main():
    parser = argparse.ArgumentParser(description="AGH-HardNet specialist refiner for LM0/LM21/LM22.")
    parser.add_argument("--train-pred-csv", type=str, default=None)
    parser.add_argument("--val-pred-csv", type=str, required=True)
    parser.add_argument("--test-pred-csv", type=str, required=True)
    parser.add_argument("--base-prefix", type=str, default="auto")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--point-cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--oracle-radii", type=float, nargs="+", default=[30.0, 40.0, 50.0, 60.0])
    parser.add_argument("--oracle-patch-points", type=int, default=4096)
    parser.add_argument("--radius-mm", type=float, default=45.0)
    parser.add_argument("--trichion-radius-mm", type=float, default=55.0)
    parser.add_argument("--patch-points", type=int, default=2048)
    parser.add_argument("--max-surface-points", type=int, default=12000)
    parser.add_argument("--sigma-mm", type=float, default=2.5)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--landmark-embedding-dim", type=int, default=32)
    parser.add_argument("--residual-limit-mm", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coord-beta-mm", type=float, default=1.0)
    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--heatmap-weight", type=float, default=0.35)
    parser.add_argument("--weighted-coord-weight", type=float, default=0.2)
    parser.add_argument("--clinical-weight", type=float, default=0.15)
    parser.add_argument("--clinical-threshold-mm", type=float, default=2.0)
    parser.add_argument("--clinical-margin-mm", type=float, default=0.5)
    parser.add_argument("--nll-weight", type=float, default=0.01)
    parser.add_argument("--residual-reg-weight", type=float, default=0.002)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--final-mode", choices=["pred", "weighted"], default="pred")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    val_samples, val_prefix = read_prediction_csv(args.val_pred_csv, args.base_prefix)
    test_samples, test_prefix = read_prediction_csv(args.test_pred_csv, args.base_prefix)
    train_samples = None
    train_prefix = None
    if args.train_pred_csv:
        train_samples, train_prefix = read_prediction_csv(args.train_pred_csv, args.base_prefix)

    write_json(
        output_dir / "config_hardnet.json",
        {
            **vars(args),
            "detected_prefixes": {"train": train_prefix, "val": val_prefix, "test": test_prefix},
            "n_samples": {
                "train": len(train_samples) if train_samples else 0,
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "hard_landmarks": list(HARD_LANDMARKS),
        },
    )

    print("Running candidate coverage oracle...", flush=True)
    oracle = {
        "val": run_oracle(args, "val", val_samples, output_dir),
        "test": run_oracle(args, "test", test_samples, output_dir),
    }
    write_json(output_dir / "oracle_report.json", oracle)
    print("Base validation metrics:", json.dumps(split_metrics(val_samples)["hard_landmarks"], indent=2), flush=True)
    print("Base test metrics:", json.dumps(split_metrics(test_samples)["hard_landmarks"], indent=2), flush=True)
    if args.oracle_only:
        print(f"Oracle-only results saved to: {output_dir}", flush=True)
        return
    if not train_samples:
        raise ValueError("--train-pred-csv is required unless --oracle-only is set")

    model, history, training_time, best_epoch, best_val = train_hardnet(train_samples, val_samples, args, output_dir)
    write_json(output_dir / "history.json", history)
    val_metrics, val_parts = evaluate_split(model, val_samples, args, "val", output_dir)
    test_metrics, test_parts = evaluate_split(model, test_samples, args, "test", output_dir)
    metrics = {
        "model": "AGH-HardNet specialist refiner",
        "best_epoch": best_epoch,
        "best_val_hard_ale": best_val,
        "training_time_sec": training_time,
        "base": {"val": split_metrics(val_samples), "test": split_metrics(test_samples)},
        "hardnet": {"val": val_metrics, "test": test_metrics},
        "loss_parts": {"val": val_parts, "test": test_parts},
    }
    write_json(output_dir / "metrics_hardnet.json", metrics)
    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"Base test ALE: {metrics['base']['test']['overall']['ale']:.4f}", flush=True)
    print(f"HardNet test ALE: {test_metrics['overall']['ale']:.4f}", flush=True)
    print(f"Base hard3 ALE: {metrics['base']['test']['hard_landmarks']['ale']:.4f}", flush=True)
    print(f"HardNet hard3 ALE: {test_metrics['hard_landmarks']['ale']:.4f}", flush=True)
    print(f"HardNet median: {test_metrics['overall']['median']:.4f}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
