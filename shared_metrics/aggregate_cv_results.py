#!/usr/bin/env python3
"""Aggregate out-of-fold landmark predictions without treating landmarks as patients."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ERROR_COLUMNS = ("error", "localization_error")


def read_rows(path, fold):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["fold"] = fold
    return rows


def error_of(row):
    for column in ERROR_COLUMNS:
        if column in row and row[column] not in (None, ""):
            return float(row[column])
    raise KeyError(f"Prediction row has none of the supported error columns: {ERROR_COLUMNS}")


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    result = {
        "n": int(arr.size),
        "ale": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }
    for threshold in (2.0, 2.5, 3.0, 4.0):
        key = ("%g" % threshold).replace(".", "_")
        result[f"sdr_at_{key}mm"] = float(np.mean(arr <= threshold))
    return result


def patient_bootstrap(rows, iterations, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(error_of(row))
    patient_ale = np.asarray([np.mean(values) for values in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(patient_ale), size=(iterations, len(patient_ale)))
    distribution = patient_ale[draws].mean(axis=1)
    return {
        "unit": "patient",
        "iterations": iterations,
        "ale": float(patient_ale.mean()),
        "ci95": np.percentile(distribution, [2.5, 97.5]).astype(float).tolist(),
    }


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate completed CV fold predictions.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--prediction-name", default="predictions_test.csv")
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = []
    fold_rows = []
    completed = []
    for fold in range(1, args.folds + 1):
        path = run_dir / f"fold_{fold}" / args.prediction_name
        if not path.exists():
            if args.allow_partial:
                continue
            raise FileNotFoundError(f"Missing fold prediction file: {path}")
        current = read_rows(path, fold)
        rows.extend(current)
        completed.append(fold)
        metrics = summarize([error_of(row) for row in current])
        core20 = [
            error_of(row) for row in current if 1 <= int(row["landmark"]) <= 20
        ]
        hard3 = [
            error_of(row) for row in current if int(row["landmark"]) in (0, 21, 22)
        ]
        metadata = {}
        metrics_path = run_dir / f"fold_{fold}" / "metrics.json"
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metadata = {
                "parameter_count": payload.get("parameter_count"),
                "training_seconds": payload.get("training_seconds"),
                "best_epoch": payload.get("best_epoch"),
            }
        fold_rows.append(
            {
                "fold": fold,
                **metrics,
                "core20_ale": summarize(core20)["ale"],
                "hard3_ale": summarize(hard3)["ale"],
                **metadata,
            }
        )

    counts = Counter(row["sample_id"] for row in rows)
    invalid = {sample_id: count for sample_id, count in counts.items() if count != 23}
    if invalid:
        raise ValueError(f"Every OOF sample must have 23 prediction rows; invalid={invalid}")
    if len(completed) == args.folds and len(counts) != 300:
        raise ValueError(f"Expected 300 unique outer-test samples, found {len(counts)}")

    all_errors = [error_of(row) for row in rows]
    core20_errors = [
        error_of(row) for row in rows if 1 <= int(row["landmark"]) <= 20
    ]
    hard3_errors = [
        error_of(row) for row in rows if int(row["landmark"]) in (0, 21, 22)
    ]
    landmark_groups = defaultdict(list)
    class_groups = defaultdict(list)
    gender_groups = defaultdict(list)
    for row in rows:
        landmark_groups[int(row["landmark"])].append(error_of(row))
        class_groups[row["class"]].append(error_of(row))
        gender_groups[row["gender"]].append(error_of(row))

    fold_ales = np.asarray([row["ale"] for row in fold_rows], dtype=np.float64)
    summary = {
        "model": args.model,
        "complete": len(completed) == args.folds,
        "completed_folds": completed,
        "n_samples": len(counts),
        "n_predictions": len(rows),
        "pooled": summarize(all_errors),
        "core20": summarize(core20_errors),
        "hard3": summarize(hard3_errors),
        "fold_mean_ale": float(fold_ales.mean()),
        "fold_population_std_ale": float(fold_ales.std()),
        "fold_sample_std_ale": float(fold_ales.std(ddof=1)) if len(fold_ales) > 1 else 0.0,
        "patient_bootstrap": patient_bootstrap(rows, args.bootstrap_iters, args.seed),
    }
    parameter_counts = [row["parameter_count"] for row in fold_rows if row.get("parameter_count")]
    training_times = [row["training_seconds"] for row in fold_rows if row.get("training_seconds")]
    if parameter_counts:
        summary["parameter_count"] = int(max(parameter_counts))
    if training_times:
        summary["training_seconds_total"] = float(sum(training_times))
        summary["training_seconds_mean_per_fold"] = float(np.mean(training_times))
    (run_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(run_dir / "summary_fold_metrics.csv", fold_rows)
    write_csv(
        run_dir / "summary_landmark_metrics.csv",
        ({"landmark": landmark, **summarize(values)} for landmark, values in sorted(landmark_groups.items())),
    )
    write_csv(
        run_dir / "summary_class_metrics.csv",
        ({"class": key, **summarize(values)} for key, values in sorted(class_groups.items())),
    )
    write_csv(
        run_dir / "summary_gender_metrics.csv",
        ({"gender": key, **summarize(values)} for key, values in sorted(gender_groups.items())),
    )
    write_csv(run_dir / "pooled_predictions_test.csv", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
