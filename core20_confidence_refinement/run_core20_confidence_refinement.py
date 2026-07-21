import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


CORE20 = [idx for idx in range(23) if idx not in {0, 21, 22}]


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


def coord(row, prefix):
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")], dtype=np.float64)


def summarize(errors):
    arr = np.asarray(errors, dtype=np.float64)
    flat = arr.reshape(-1)
    out = {
        "ale": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "max": float(flat.max()),
        "per_landmark_ale": arr.mean(axis=0).astype(float).tolist(),
        "per_landmark_median": np.median(arr, axis=0).astype(float).tolist(),
        "per_sample_ale": arr.mean(axis=1).astype(float).tolist(),
    }
    for threshold in (2.0, 2.5, 3.0):
        key = ("%g" % threshold).replace(".", "_")
        out[f"pck_at_{key}mm"] = float((flat <= threshold).mean())
    return out


def summarize_any(errors):
    arr = np.asarray(errors, dtype=np.float64)
    flat = arr.reshape(-1)
    out = {
        "ale": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "max": float(flat.max()),
    }
    if arr.ndim == 2:
        out["per_landmark_ale"] = arr.mean(axis=0).astype(float).tolist()
        out["per_landmark_median"] = np.median(arr, axis=0).astype(float).tolist()
        out["per_sample_ale"] = arr.mean(axis=1).astype(float).tolist()
    for threshold in (2.0, 2.5, 3.0):
        key = ("%g" % threshold).replace(".", "_")
        out[f"pck_at_{key}mm"] = float((flat <= threshold).mean())
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def standardize_fit(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_apply(x, mean, std):
    return (x - mean[None, :]) / std[None, :]


def train_logistic_regression(x, y, sample_weight, lr=0.05, epochs=2500, l2=0.01):
    n, d = x.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    sw = sample_weight.astype(np.float64)
    sw = sw / max(float(sw.mean()), 1e-8)
    for _ in range(int(epochs)):
        p = sigmoid(x @ w + b)
        diff = (p - y) * sw
        grad_w = (x.T @ diff) / max(n, 1) + float(l2) * w
        grad_b = float(diff.mean())
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
    return w, b


def confusion_metrics(prob, y, threshold):
    pred = prob >= float(threshold)
    yb = y >= 0.5
    tp = int(np.logical_and(pred, yb).sum())
    fp = int(np.logical_and(pred, ~yb).sum())
    tn = int(np.logical_and(~pred, ~yb).sum())
    fn = int(np.logical_and(~pred, yb).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall}


def parse_rows(rows, target_landmarks):
    target_set = set(target_landmarks)
    sample_ids = sorted({row["sample_id"] for row in rows})
    sample_to_pos = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    base = np.zeros((len(sample_ids), 23, 3), dtype=np.float64)
    stage3 = np.zeros_like(base)
    expert = np.zeros_like(base)
    base_errors = np.zeros((len(sample_ids), 23), dtype=np.float64)
    stage3_errors = np.zeros_like(base_errors)
    by_sample = {}
    feature_rows = []
    meta = {}
    for row in rows:
        sample_id = row["sample_id"]
        lm_idx = int(row["landmark"])
        pos = sample_to_pos[sample_id]
        base[pos, lm_idx] = coord(row, "base")
        stage3[pos, lm_idx] = coord(row, "stage3")
        expert[pos, lm_idx] = coord(row, "expert")
        base_errors[pos, lm_idx] = float(row["base_error"])
        stage3_errors[pos, lm_idx] = float(row["stage3_error"])
        meta[sample_id] = {
            "sample_id": sample_id,
            "class": row.get("class", ""),
            "gender": row.get("gender", ""),
            "subject_id": row.get("subject_id", ""),
        }
        by_sample.setdefault(sample_id, {})[lm_idx] = row

    for row in rows:
        sample_id = row["sample_id"]
        lm_idx = int(row["landmark"])
        if lm_idx not in target_set:
            continue
        pos = sample_to_pos[sample_id]
        base_pt = base[pos, lm_idx]
        stage3_pt = stage3[pos, lm_idx]
        delta = stage3_pt - base_pt
        delta_norm = float(np.linalg.norm(delta))
        neighbor_dist_features = []
        for nb in CORE20:
            if nb == lm_idx:
                continue
            neighbor_dist_features.append(float(np.linalg.norm(base[pos, lm_idx] - base[pos, nb])))
        neighbor_dist_features = np.asarray(neighbor_dist_features, dtype=np.float64)
        symmetry_partner = {1: 2, 2: 1, 3: 4, 4: 3, 7: 8, 8: 7, 10: 11, 11: 10, 12: 13, 13: 12, 14: 15, 15: 14, 16: 17, 17: 16, 19: 20, 20: 19}.get(lm_idx)
        if symmetry_partner is not None:
            sym_base = float(np.linalg.norm(base[pos, lm_idx] - base[pos, symmetry_partner]))
            sym_stage3 = float(np.linalg.norm(stage3[pos, lm_idx] - base[pos, symmetry_partner]))
            sym_change = abs(sym_stage3 - sym_base)
        else:
            sym_base = 0.0
            sym_change = 0.0
        base_neighbor_mean = float(neighbor_dist_features.mean()) if len(neighbor_dist_features) else 0.0
        base_neighbor_std = float(neighbor_dist_features.std()) if len(neighbor_dist_features) else 0.0
        feature_rows.append(
            {
                "sample_id": sample_id,
                "landmark": lm_idx,
                "class": row.get("class", ""),
                "gender": row.get("gender", ""),
                "delta_norm": delta_norm,
                "delta_x": float(delta[0]),
                "delta_y": float(delta[1]),
                "delta_z": float(delta[2]),
                "base_neighbor_mean": base_neighbor_mean,
                "base_neighbor_std": base_neighbor_std,
                "sym_base": sym_base,
                "sym_change": sym_change,
                "stage3_enabled_original": 1.0 if str(row.get("enabled", "")).lower() == "true" else 0.0,
                "base_error": base_errors[pos, lm_idx],
                "stage3_error": stage3_errors[pos, lm_idx],
                "improvement": base_errors[pos, lm_idx] - stage3_errors[pos, lm_idx],
            }
        )
    return sample_ids, meta, base, stage3, expert, base_errors, stage3_errors, feature_rows


def featurize(feature_rows, landmark_subset):
    landmark_subset = list(landmark_subset)
    lm_to_pos = {lm: i for i, lm in enumerate(landmark_subset)}
    classes = sorted({row["class"] for row in feature_rows})
    genders = sorted({row["gender"] for row in feature_rows})
    class_to_pos = {value: i for i, value in enumerate(classes)}
    gender_to_pos = {value: i for i, value in enumerate(genders)}
    names = [
        "delta_norm",
        "abs_delta_x",
        "abs_delta_y",
        "abs_delta_z",
        "base_neighbor_mean",
        "base_neighbor_std",
        "sym_base",
        "sym_change",
        "stage3_enabled_original",
    ]
    names += [f"lm_{lm}" for lm in landmark_subset]
    names += [f"class_{value}" for value in classes]
    names += [f"gender_{value}" for value in genders]
    x = []
    y = []
    sample_weight = []
    for row in feature_rows:
        delta_x = float(row["delta_x"])
        delta_y = float(row["delta_y"])
        delta_z = float(row["delta_z"])
        values = [
            float(row["delta_norm"]),
            abs(delta_x),
            abs(delta_y),
            abs(delta_z),
            float(row["base_neighbor_mean"]),
            float(row["base_neighbor_std"]),
            float(row["sym_base"]),
            float(row["sym_change"]),
            float(row["stage3_enabled_original"]),
        ]
        lm_onehot = [0.0] * len(landmark_subset)
        lm_onehot[lm_to_pos[int(row["landmark"])]] = 1.0
        cls_onehot = [0.0] * len(classes)
        cls_onehot[class_to_pos[row["class"]]] = 1.0
        gender_onehot = [0.0] * len(genders)
        gender_onehot[gender_to_pos[row["gender"]]] = 1.0
        values.extend(lm_onehot)
        values.extend(cls_onehot)
        values.extend(gender_onehot)
        x.append(values)
        improvement = float(row["improvement"])
        y.append(1.0 if improvement > 0.0 else 0.0)
        sample_weight.append(1.0 + min(abs(improvement), 2.0))
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(sample_weight, dtype=np.float64), names, {
        "landmarks": landmark_subset,
        "classes": classes,
        "genders": genders,
    }


def featurize_with_schema(feature_rows, schema):
    landmark_subset = schema["landmarks"]
    classes = schema["classes"]
    genders = schema["genders"]
    lm_to_pos = {lm: i for i, lm in enumerate(landmark_subset)}
    class_to_pos = {value: i for i, value in enumerate(classes)}
    gender_to_pos = {value: i for i, value in enumerate(genders)}
    x = []
    for row in feature_rows:
        delta_x = float(row["delta_x"])
        delta_y = float(row["delta_y"])
        delta_z = float(row["delta_z"])
        values = [
            float(row["delta_norm"]),
            abs(delta_x),
            abs(delta_y),
            abs(delta_z),
            float(row["base_neighbor_mean"]),
            float(row["base_neighbor_std"]),
            float(row["sym_base"]),
            float(row["sym_change"]),
            float(row["stage3_enabled_original"]),
        ]
        lm_onehot = [0.0] * len(landmark_subset)
        if int(row["landmark"]) in lm_to_pos:
            lm_onehot[lm_to_pos[int(row["landmark"])]] = 1.0
        cls_onehot = [0.0] * len(classes)
        if row["class"] in class_to_pos:
            cls_onehot[class_to_pos[row["class"]]] = 1.0
        gender_onehot = [0.0] * len(genders)
        if row["gender"] in gender_to_pos:
            gender_onehot[gender_to_pos[row["gender"]]] = 1.0
        values.extend(lm_onehot)
        values.extend(cls_onehot)
        values.extend(gender_onehot)
        x.append(values)
    return np.asarray(x, dtype=np.float64)


def choose_thresholds(feature_rows, probabilities, args):
    thresholds = {}
    rows_by_lm = {}
    probs_by_lm = {}
    for row, prob in zip(feature_rows, probabilities):
        lm = int(row["landmark"])
        rows_by_lm.setdefault(lm, []).append(row)
        probs_by_lm.setdefault(lm, []).append(float(prob))
    for lm, rows in rows_by_lm.items():
        probs = np.asarray(probs_by_lm[lm], dtype=np.float64)
        best = None
        for threshold in np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps):
            total = 0.0
            n_use = 0
            for row, prob in zip(rows, probs):
                use = prob >= threshold
                total += float(row["stage3_error"] if use else row["base_error"])
                n_use += int(use)
            mean = total / max(len(rows), 1)
            base_mean = float(np.mean([float(row["base_error"]) for row in rows]))
            payload = {"threshold": float(threshold), "val_ale": float(mean), "base_val_ale": base_mean, "n_use": int(n_use)}
            if best is None or payload["val_ale"] < best["val_ale"]:
                best = payload
        best["enabled"] = bool(best["val_ale"] + args.min_val_improvement_mm < best["base_val_ale"])
        thresholds[str(lm)] = best
    return thresholds


def apply_gate(sample_ids, base, stage3, expert, feature_rows, probabilities, thresholds):
    final = base.copy()
    source = {}
    sample_to_pos = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    for row, prob in zip(feature_rows, probabilities):
        lm = int(row["landmark"])
        config = thresholds.get(str(lm), {})
        if not config.get("enabled", False):
            continue
        if float(prob) >= float(config["threshold"]):
            pos = sample_to_pos[row["sample_id"]]
            final[pos, lm] = stage3[pos, lm]
            source[(row["sample_id"], lm)] = "stage3_gate"
    errors = np.linalg.norm(final - expert, axis=-1)
    return final, errors, source


def write_prediction_rows(path, stage3_rows, sample_ids, final, expert, errors, source):
    sample_to_pos = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    out = []
    for row in stage3_rows:
        lm = int(row["landmark"])
        sample_id = row["sample_id"]
        pos = sample_to_pos[sample_id]
        item = dict(row)
        for axis, value in zip(("x", "y", "z"), final[pos, lm]):
            item[f"gate_final_{axis}"] = float(value)
        item["gate_source"] = source.get((sample_id, lm), "base")
        item["gate_final_error"] = float(errors[pos, lm])
        out.append(item)
    write_rows(path, out)


def main():
    parser = argparse.ArgumentParser(description="Confidence gate for AGH-Former Stage3 core20 refinements.")
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-landmarks", default="2,10,11,12,13,16,19,20")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--threshold-min", type=float, default=0.25)
    parser.add_argument("--threshold-max", type=float, default=0.85)
    parser.add_argument("--threshold-steps", type=int, default=25)
    parser.add_argument("--min-val-improvement-mm", type=float, default=0.0)
    args = parser.parse_args()

    stage3_dir = Path(args.stage3_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_landmarks = [int(v.strip()) for v in args.target_landmarks.split(",") if v.strip()]
    val_rows_raw = read_rows(stage3_dir / "stage3_predictions_val.csv")
    test_rows_raw = read_rows(stage3_dir / "stage3_predictions_test.csv")
    val_ids, val_meta, val_base, val_stage3, val_expert, val_base_errors, val_stage3_errors, val_feature_rows = parse_rows(
        val_rows_raw, target_landmarks
    )
    test_ids, test_meta, test_base, test_stage3, test_expert, test_base_errors, test_stage3_errors, test_feature_rows = parse_rows(
        test_rows_raw, target_landmarks
    )
    x_val, y_val, sw_val, names, schema = featurize(val_feature_rows, target_landmarks)
    mean, std = standardize_fit(x_val)
    x_val_std = standardize_apply(x_val, mean, std)
    w, b = train_logistic_regression(x_val_std, y_val, sw_val, lr=args.lr, epochs=args.epochs, l2=args.l2)
    val_prob = sigmoid(x_val_std @ w + b)
    x_test = featurize_with_schema(test_feature_rows, schema)
    test_prob = sigmoid(standardize_apply(x_test, mean, std) @ w + b)
    thresholds = choose_thresholds(val_feature_rows, val_prob, args)
    val_final, val_errors, val_source = apply_gate(val_ids, val_base, val_stage3, val_expert, val_feature_rows, val_prob, thresholds)
    test_final, test_errors, test_source = apply_gate(test_ids, test_base, test_stage3, test_expert, test_feature_rows, test_prob, thresholds)
    write_prediction_rows(output_dir / "gated_predictions_val.csv", val_rows_raw, val_ids, val_final, val_expert, val_errors, val_source)
    write_prediction_rows(output_dir / "gated_predictions_test.csv", test_rows_raw, test_ids, test_final, test_expert, test_errors, test_source)
    feature_importance = [
        {"feature": name, "weight": float(weight)}
        for name, weight in sorted(zip(names, w), key=lambda item: abs(item[1]), reverse=True)
    ]
    metrics = {
        "method": "logistic confidence gate for Stage3 corrections",
        "stage3_run_dir": str(stage3_dir),
        "target_landmarks": target_landmarks,
        "base_validation": summarize(val_base_errors),
        "stage3_all_validation": summarize(val_stage3_errors),
        "gate_validation": summarize(val_errors),
        "base_test": summarize(test_base_errors),
        "stage3_all_test": summarize(test_stage3_errors),
        "gate_test": summarize(test_errors),
        "base_target_test": summarize_any(test_base_errors[:, target_landmarks]),
        "gate_target_test": summarize_any(test_errors[:, target_landmarks]),
        "thresholds": thresholds,
        "classifier_validation": confusion_metrics(val_prob, y_val, 0.5),
    }
    (output_dir / "metrics_gate.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "gate_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (output_dir / "feature_importance.json").write_text(json.dumps(feature_importance, indent=2), encoding="utf-8")
    print(f"Base test ALE: {metrics['base_test']['ale']:.4f}", flush=True)
    print(f"Stage3 all test ALE: {metrics['stage3_all_test']['ale']:.4f}", flush=True)
    print(f"Gate test ALE: {metrics['gate_test']['ale']:.4f}", flush=True)
    print(f"Gate test median: {metrics['gate_test']['median']:.4f}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
