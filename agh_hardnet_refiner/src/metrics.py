import csv
import json
from pathlib import Path

import numpy as np

from .data import CORE20, HARD_LANDMARKS


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_errors(errors):
    arr = np.asarray(errors, dtype=np.float64).reshape(-1)
    return {
        "ale": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "sdr_at_2mm": float((arr <= 2.0).mean()),
        "sdr_at_3mm": float((arr <= 3.0).mean()),
        "sdr_at_4mm": float((arr <= 4.0).mean()),
    }


def landmark_metrics(errors):
    rows = []
    for lm in range(errors.shape[1]):
        s = summarize_errors(errors[:, lm])
        rows.append({"landmark": lm, **s})
    return rows


def combined_metrics(errors):
    return {
        "overall": summarize_errors(errors),
        "core20": summarize_errors(errors[:, CORE20]),
        "hard_landmarks": summarize_errors(errors[:, HARD_LANDMARKS]),
        "per_landmark_ale": [float(x) for x in errors.mean(axis=0)],
    }


def oracle_summary(rows):
    out = {}
    for radius in sorted({float(row["radius_mm"]) for row in rows}):
        subset = [row for row in rows if float(row["radius_mm"]) == radius]
        by_lm = {}
        for lm in HARD_LANDMARKS:
            lm_rows = [row for row in subset if int(row["landmark"]) == lm]
            by_lm[str(lm)] = {
                "oracle_ale": float(np.mean([float(row["oracle_error"]) for row in lm_rows])),
                "oracle_pck_at_2mm": float(np.mean([float(row["oracle_pck_at_2mm"]) for row in lm_rows])),
                "oracle_pck_at_3mm": float(np.mean([float(row["oracle_pck_at_3mm"]) for row in lm_rows])),
                "base_ale": float(np.mean([float(row["base_error"]) for row in lm_rows])),
                "surface_nearest_ale": float(np.mean([float(row["surface_nearest_error"]) for row in lm_rows])),
            }
        out[str(radius)] = {
            "overall": {
                "oracle_ale": float(np.mean([float(row["oracle_error"]) for row in subset])),
                "oracle_pck_at_2mm": float(np.mean([float(row["oracle_pck_at_2mm"]) for row in subset])),
                "oracle_pck_at_3mm": float(np.mean([float(row["oracle_pck_at_3mm"]) for row in subset])),
                "base_ale": float(np.mean([float(row["base_error"]) for row in subset])),
                "surface_nearest_ale": float(np.mean([float(row["surface_nearest_error"]) for row in subset])),
            },
            "by_landmark": by_lm,
        }
    return out
