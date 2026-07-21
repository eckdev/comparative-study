import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


HARD_LANDMARKS = {0, 21, 22}
CORE20 = [idx for idx in range(23) if idx not in HARD_LANDMARKS]


def parse_ints(value):
    value = str(value).strip()
    if value.lower() == "core20":
        return CORE20
    if value.lower() == "all":
        return list(range(23))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError("No prediction CSV found. Tried: " + ", ".join(str(path) for path in paths))


def infer_prediction_prefix(row, requested_prefix):
    if f"{requested_prefix}_x" in row:
        return requested_prefix
    for prefix in ("final", "stage3", "stage2_raw", "stage2_snapped", "base"):
        if f"{prefix}_x" in row:
            return prefix
    raise KeyError(f"Could not infer coordinate prefix from columns: {sorted(row.keys())}")


def load_prediction_split(prediction_dir, split, source_prefix="final"):
    prediction_dir = Path(prediction_dir)
    path = first_existing(
        [
            prediction_dir / f"base_stage2_predictions_{split}.csv",
            prediction_dir / f"refined_predictions_{split}.csv",
            prediction_dir / f"stage3_predictions_{split}.csv",
        ]
    )
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    prefix = infer_prediction_prefix(rows[0], source_prefix)
    sample_ids = []
    base = {}
    expert = {}
    metadata = {}
    for row in rows:
        sample_id = row["sample_id"]
        lm_idx = int(row["landmark"])
        if sample_id not in base:
            sample_ids.append(sample_id)
            base[sample_id] = np.zeros((23, 3), dtype=np.float64)
            expert[sample_id] = np.zeros((23, 3), dtype=np.float64)
            metadata[sample_id] = {
                "sample_id": sample_id,
                "class": row.get("class", ""),
                "gender": row.get("gender", ""),
                "subject_id": row.get("subject_id", ""),
            }
        base[sample_id][lm_idx] = [float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")]
        expert[sample_id][lm_idx] = [float(row[f"expert_{axis}"]) for axis in ("x", "y", "z")]
    return {
        "path": str(path),
        "prefix": prefix,
        "sample_ids": sample_ids,
        "base": np.stack([base[sample_id] for sample_id in sample_ids]),
        "expert": np.stack([expert[sample_id] for sample_id in sample_ids]),
        "metadata": metadata,
    }


def metadata_features(sample_ids, metadata):
    rows = []
    for sample_id in sample_ids:
        item = metadata[sample_id]
        class_name = item.get("class", "")
        gender = item.get("gender", "")
        rows.append(
            [
                float(class_name == "Class1"),
                float(class_name == "Class2"),
                float(class_name == "Class3"),
                float(gender == "women"),
                float(gender == "men"),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def pairwise_distance_features(points, pairs):
    feats = []
    for left, right in pairs:
        feats.append(np.linalg.norm(points[:, left] - points[:, right], axis=1))
    return np.stack(feats, axis=1)


def build_features(split, normalizer=None):
    points = split["base"]
    flat = points.reshape(len(points), -1)
    if normalizer is None:
        mean = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-6
        normalizer = {"mean": mean, "std": std}
    flat_norm = (flat - normalizer["mean"]) / normalizer["std"]
    pairs = [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
        (17, 18),
        (19, 20),
        (21, 22),
    ]
    distances = pairwise_distance_features(points, pairs)
    if "distance_mean" not in normalizer:
        normalizer["distance_mean"] = distances.mean(axis=0)
        normalizer["distance_std"] = distances.std(axis=0) + 1e-6
    distance_norm = (distances - normalizer["distance_mean"]) / normalizer["distance_std"]
    meta = metadata_features(split["sample_ids"], split["metadata"])
    ones = np.ones((len(points), 1), dtype=np.float64)
    return np.concatenate([ones, flat_norm, distance_norm, meta], axis=1), normalizer


def ridge_fit(features, targets, l2):
    reg = float(l2) * np.eye(features.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(features.T @ features + reg, features.T @ targets)


def summarize(errors):
    arr = np.asarray(errors, dtype=np.float64)
    flat = arr.reshape(-1)
    out = {
        "ale": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "max": float(flat.max()),
    }
    for threshold in (2.0, 2.5, 3.0):
        key = ("%g" % threshold).replace(".", "_")
        out[f"pck_at_{key}mm"] = float((flat <= threshold).mean())
    if arr.ndim == 2:
        out["per_landmark_ale"] = arr.mean(axis=0).astype(float).tolist()
        out["per_landmark_median"] = np.median(arr, axis=0).astype(float).tolist()
    return out


def validation_gate(base_errors, pred_errors, target_landmarks, min_improvement):
    enabled = []
    for lm_idx in target_landmarks:
        if float(pred_errors[:, lm_idx].mean()) + float(min_improvement) < float(base_errors[:, lm_idx].mean()):
            enabled.append(int(lm_idx))
    return enabled


def apply_gate(base, pred, enabled):
    final = base.copy()
    for lm_idx in enabled:
        final[:, lm_idx] = pred[:, lm_idx]
    return final


def write_prediction_csv(path, split, pred, final, enabled):
    enabled_set = set(enabled)
    base = split["base"]
    expert = split["expert"]
    base_errors = np.linalg.norm(base - expert, axis=-1)
    pred_errors = np.linalg.norm(pred - expert, axis=-1)
    final_errors = np.linalg.norm(final - expert, axis=-1)
    rows = []
    for sample_pos, sample_id in enumerate(split["sample_ids"]):
        meta = split["metadata"][sample_id]
        for lm_idx in range(23):
            rows.append(
                {
                    "sample_id": sample_id,
                    "class": meta.get("class", ""),
                    "gender": meta.get("gender", ""),
                    "subject_id": meta.get("subject_id", ""),
                    "landmark": lm_idx,
                    "enabled": lm_idx in enabled_set,
                    "base_error": float(base_errors[sample_pos, lm_idx]),
                    "shape_prior_error": float(pred_errors[sample_pos, lm_idx]),
                    "final_error": float(final_errors[sample_pos, lm_idx]),
                    "expert_x": float(expert[sample_pos, lm_idx, 0]),
                    "expert_y": float(expert[sample_pos, lm_idx, 1]),
                    "expert_z": float(expert[sample_pos, lm_idx, 2]),
                    "base_x": float(base[sample_pos, lm_idx, 0]),
                    "base_y": float(base[sample_pos, lm_idx, 1]),
                    "base_z": float(base[sample_pos, lm_idx, 2]),
                    "shape_prior_x": float(pred[sample_pos, lm_idx, 0]),
                    "shape_prior_y": float(pred[sample_pos, lm_idx, 1]),
                    "shape_prior_z": float(pred[sample_pos, lm_idx, 2]),
                    "final_x": float(final[sample_pos, lm_idx, 0]),
                    "final_y": float(final[sample_pos, lm_idx, 1]),
                    "final_z": float(final[sample_pos, lm_idx, 2]),
                }
            )
    write_rows(path, rows)


def write_landmark_metrics(path, base_errors, pred_errors, final_errors, enabled):
    rows = []
    enabled_set = set(enabled)
    for lm_idx in range(23):
        base = base_errors[:, lm_idx]
        pred = pred_errors[:, lm_idx]
        final = final_errors[:, lm_idx]
        rows.append(
            {
                "landmark": lm_idx,
                "enabled": lm_idx in enabled_set,
                "base_ale": float(base.mean()),
                "shape_prior_ale": float(pred.mean()),
                "final_ale": float(final.mean()),
                "final_delta": float(final.mean() - base.mean()),
                "base_median": float(np.median(base)),
                "final_median": float(np.median(final)),
                "base_pck_at_2mm": float((base <= 2.0).mean()),
                "final_pck_at_2mm": float((final <= 2.0).mean()),
                "base_pck_at_2_5mm": float((base <= 2.5).mean()),
                "final_pck_at_2_5mm": float((final <= 2.5).mean()),
                "base_pck_at_3mm": float((base <= 3.0).mean()),
                "final_pck_at_3mm": float((final <= 3.0).mean()),
                "improved_count": int((final < base).sum()),
                "worsened_count": int((final > base).sum()),
                "n": int(len(base)),
            }
        )
    write_rows(path, rows)


def class_gender_metrics(split, errors):
    grouped = defaultdict(list)
    for sample_pos, sample_id in enumerate(split["sample_ids"]):
        meta = split["metadata"][sample_id]
        grouped[("class", meta.get("class", ""))].extend(errors[sample_pos].tolist())
        grouped[("gender", meta.get("gender", ""))].extend(errors[sample_pos].tolist())
    rows = []
    for (group_type, group), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "group_type": group_type,
                "group": group,
                "ale": float(arr.mean()),
                "median": float(np.median(arr)),
                "pck_at_2mm": float((arr <= 2.0).mean()),
                "pck_at_2_5mm": float((arr <= 2.5).mean()),
                "pck_at_3mm": float((arr <= 3.0).mean()),
                "n": int(len(arr)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Shape-prior residual calibration for AGH-Former/PAL style predictions.")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-prefix", default="final")
    parser.add_argument("--target-landmarks", default="all")
    parser.add_argument("--gate-landmarks", default="all")
    parser.add_argument("--l2-grid", default="0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000")
    parser.add_argument("--shrinkage-grid", default="0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--selection-metric", choices=["all", "core20", "target"], default="core20")
    parser.add_argument("--min-val-improvement-mm", type=float, default=0.0)
    parser.add_argument("--max-residual-mm", type=float, default=8.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_prediction_split(args.prediction_dir, "train", args.source_prefix)
    val = load_prediction_split(args.prediction_dir, "val", args.source_prefix)
    test = load_prediction_split(args.prediction_dir, "test", args.source_prefix)
    print(f"Prediction dir: {args.prediction_dir}", flush=True)
    print(f"Sources train/val/test: {train['path']} | {val['path']} | {test['path']}", flush=True)
    print(f"Coordinate prefix: {train['prefix']}", flush=True)

    target_landmarks = parse_ints(args.target_landmarks)
    gate_landmarks = parse_ints(args.gate_landmarks)
    metric_landmarks = {
        "all": list(range(23)),
        "core20": CORE20,
        "target": target_landmarks,
    }[args.selection_metric]
    l2_values = [float(value) for value in args.l2_grid.split(",") if value.strip()]
    shrinkage_values = [float(value) for value in args.shrinkage_grid.split(",") if value.strip()]

    train_features, normalizer = build_features(train)
    val_features, _ = build_features(val, normalizer)
    test_features, _ = build_features(test, normalizer)
    train_residual = train["expert"] - train["base"]
    target_mask = np.zeros((23, 1), dtype=np.float64)
    target_mask[target_landmarks] = 1.0
    train_targets = (train_residual * target_mask[None]).reshape(len(train["base"]), -1)

    best = None
    sweep_rows = []
    for l2 in l2_values:
        weights = ridge_fit(train_features, train_targets, l2)
        for shrinkage in shrinkage_values:
            val_residual = (val_features @ weights).reshape(len(val["base"]), 23, 3)
            residual_norm = np.linalg.norm(val_residual, axis=-1, keepdims=True)
            scale = np.minimum(1.0, args.max_residual_mm / np.maximum(residual_norm, 1e-8))
            val_pred = val["base"] + shrinkage * val_residual * scale
            val_errors = np.linalg.norm(val_pred - val["expert"], axis=-1)
            score = float(val_errors[:, metric_landmarks].mean())
            row = {"l2": l2, "shrinkage": shrinkage, "validation_score": score}
            sweep_rows.append(row)
            if best is None or score < best["score"]:
                best = {"score": score, "l2": l2, "shrinkage": shrinkage, "weights": weights}

    weights = best["weights"]

    def predict(split, features):
        residual = (features @ weights).reshape(len(split["base"]), 23, 3)
        residual = residual * target_mask[None]
        residual_norm = np.linalg.norm(residual, axis=-1, keepdims=True)
        scale = np.minimum(1.0, args.max_residual_mm / np.maximum(residual_norm, 1e-8))
        return split["base"] + best["shrinkage"] * residual * scale

    val_pred = predict(val, val_features)
    test_pred = predict(test, test_features)
    val_base_errors = np.linalg.norm(val["base"] - val["expert"], axis=-1)
    test_base_errors = np.linalg.norm(test["base"] - test["expert"], axis=-1)
    val_pred_errors = np.linalg.norm(val_pred - val["expert"], axis=-1)
    test_pred_errors = np.linalg.norm(test_pred - test["expert"], axis=-1)
    enabled = validation_gate(val_base_errors, val_pred_errors, gate_landmarks, args.min_val_improvement_mm)
    val_final = apply_gate(val["base"], val_pred, enabled)
    test_final = apply_gate(test["base"], test_pred, enabled)
    val_final_errors = np.linalg.norm(val_final - val["expert"], axis=-1)
    test_final_errors = np.linalg.norm(test_final - test["expert"], axis=-1)

    write_prediction_csv(output_dir / "predictions_val.csv", val, val_pred, val_final, enabled)
    write_prediction_csv(output_dir / "predictions_test.csv", test, test_pred, test_final, enabled)
    write_landmark_metrics(output_dir / "landmark_metrics_test.csv", test_base_errors, test_pred_errors, test_final_errors, enabled)
    write_rows(output_dir / "delta_analysis_shape_prior.csv", read_rows(output_dir / "landmark_metrics_test.csv"))
    write_rows(output_dir / "group_metrics_test.csv", class_gender_metrics(test, test_final_errors))
    write_rows(output_dir / "sweep_validation.csv", sweep_rows)

    metrics = {
        "model": "Shape-prior residual refiner",
        "prediction_dir": str(args.prediction_dir),
        "source_prefix": train["prefix"],
        "target_landmarks": target_landmarks,
        "gate_landmarks": gate_landmarks,
        "selection_metric": args.selection_metric,
        "best_l2": float(best["l2"]),
        "best_shrinkage": float(best["shrinkage"]),
        "enabled_landmarks": enabled,
        "base_validation": summarize(val_base_errors),
        "shape_prior_validation": summarize(val_pred_errors),
        "gated_validation": summarize(val_final_errors),
        "base_test": summarize(test_base_errors),
        "shape_prior_test": summarize(test_pred_errors),
        "gated_test": summarize(test_final_errors),
        "base_core20_test": summarize(test_base_errors[:, CORE20]),
        "shape_prior_core20_test": summarize(test_pred_errors[:, CORE20]),
        "gated_core20_test": summarize(test_final_errors[:, CORE20]),
    }
    (output_dir / "metrics_shape_prior.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config = {key: value for key, value in vars(args).items()}
    config.update({"best_l2": best["l2"], "best_shrinkage": best["shrinkage"]})
    (output_dir / "config_shape_prior.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"Base ALE: {metrics['base_test']['ale']:.4f}", flush=True)
    print(f"Shape-prior all-target ALE: {metrics['shape_prior_test']['ale']:.4f}", flush=True)
    print(f"Shape-prior gated ALE: {metrics['gated_test']['ale']:.4f}", flush=True)
    print(f"Shape-prior gated median: {metrics['gated_test']['median']:.4f}", flush=True)
    print(f"Core20 base/gated ALE: {metrics['base_core20_test']['ale']:.4f} -> {metrics['gated_core20_test']['ale']:.4f}", flush=True)
    print(f"Best l2={best['l2']} shrinkage={best['shrinkage']}", flush=True)
    print(f"Enabled landmarks: {enabled}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
