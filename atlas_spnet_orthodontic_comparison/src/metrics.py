import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CORE20 = [idx for idx in range(23) if idx not in {0, 21, 22}]
HARD = [0, 21, 22]


def summarize_errors(errors):
    arr = np.asarray(errors, dtype=np.float64)
    flat = arr.reshape(-1)
    out = {
        "ale": float(flat.mean()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p75": float(np.percentile(flat, 75)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
    }
    for t in (2.0, 3.0, 4.0):
        out[f"sdr_at_{int(t)}mm"] = float((flat <= t).mean())
    if arr.ndim == 2:
        out["per_landmark_ale"] = arr.mean(axis=0).astype(float).tolist()
        out["per_landmark_std"] = arr.std(axis=0).astype(float).tolist()
        out["per_landmark_median"] = np.median(arr, axis=0).astype(float).tolist()
        out["per_sample_ale"] = arr.mean(axis=1).astype(float).tolist()
    return out


def bootstrap_ci(errors, n_boot=2000, seed=42):
    arr = np.asarray(errors, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, arr.shape[0], size=arr.shape[0])
        values.append(float(arr[idx].mean()))
    values = np.asarray(values)
    return {"mean": float(arr.mean()), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}


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


def landmark_rows(errors):
    rows = []
    arr = np.asarray(errors)
    for lm_idx in range(arr.shape[1]):
        lm = arr[:, lm_idx]
        rows.append(
            {
                "landmark": lm_idx,
                "mean": float(lm.mean()),
                "std": float(lm.std()),
                "median": float(np.median(lm)),
                "p90": float(np.percentile(lm, 90)),
                "p95": float(np.percentile(lm, 95)),
                "max": float(lm.max()),
                "sdr_at_2mm": float((lm <= 2.0).mean()),
                "sdr_at_3mm": float((lm <= 3.0).mean()),
                "sdr_at_4mm": float((lm <= 4.0).mean()),
            }
        )
    return rows


def group_rows(samples, sample_indices, errors):
    grouped = defaultdict(list)
    for pos, sample_idx in enumerate(sample_indices):
        sample = samples[sample_idx]
        grouped[("class", sample.class_name)].extend(errors[pos].tolist())
        grouped[("gender", sample.gender)].extend(errors[pos].tolist())
    rows = []
    for (group_type, group), vals in sorted(grouped.items()):
        vals = np.asarray(vals, dtype=np.float64)
        rows.append(
            {
                "group_type": group_type,
                "group": group,
                "n_points": int(vals.size),
                "ale": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(np.median(vals)),
                "sdr_at_2mm": float((vals <= 2.0).mean()),
                "sdr_at_3mm": float((vals <= 3.0).mean()),
                "sdr_at_4mm": float((vals <= 4.0).mean()),
            }
        )
    return rows


def write_predictions(path, samples, sample_indices, pred, expert, errors, confidence):
    rows = []
    for pos, sample_idx in enumerate(sample_indices):
        sample = samples[sample_idx]
        for lm_idx in range(23):
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "class": sample.class_name,
                    "gender": sample.gender,
                    "subject_id": sample.subject_id,
                    "landmark": lm_idx,
                    "expert_x": float(expert[pos, lm_idx, 0]),
                    "expert_y": float(expert[pos, lm_idx, 1]),
                    "expert_z": float(expert[pos, lm_idx, 2]),
                    "pred_x": float(pred[pos, lm_idx, 0]),
                    "pred_y": float(pred[pos, lm_idx, 1]),
                    "pred_z": float(pred[pos, lm_idx, 2]),
                    "localization_error": float(errors[pos, lm_idx]),
                    "confidence": float(confidence[pos, lm_idx]),
                }
            )
    write_rows(path, rows)


def write_outliers(path, samples, sample_indices, errors):
    rows = []
    for pos, sample_idx in enumerate(sample_indices):
        sample = samples[sample_idx]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "class": sample.class_name,
                "gender": sample.gender,
                "subject_id": sample.subject_id,
                "sample_ale": float(errors[pos].mean()),
                "sample_max": float(errors[pos].max()),
                "worst_landmark": int(errors[pos].argmax()),
            }
        )
    write_rows(path, sorted(rows, key=lambda row: row["sample_ale"], reverse=True))


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
