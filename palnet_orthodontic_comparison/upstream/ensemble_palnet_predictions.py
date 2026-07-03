import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent / "shared_metrics" / "orthodontic_analysis.py").exists():
        sys.path.append(str(parent))
        break

from shared_metrics.orthodontic_analysis import build_error_analysis, write_analysis_csvs


@dataclass(frozen=True)
class PredictionSample:
    sample_id: str
    class_name: str
    gender: str
    subject_id: int


def prediction_columns(fieldnames):
    candidates = []
    for name in fieldnames:
        if name.endswith("_x") and name != "expert_x":
            prefix = name[:-2]
            if f"{prefix}_y" in fieldnames and f"{prefix}_z" in fieldnames:
                candidates.append(prefix)
    if not candidates:
        raise ValueError(f"Could not find prediction x/y/z columns in {fieldnames}")
    return candidates[0]


def load_prediction_csv(path):
    rows = {}
    samples = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        pred_prefix = prediction_columns(reader.fieldnames)
        for row in reader:
            key = (row["sample_id"], int(row["landmark"]))
            expert = np.asarray([float(row["expert_x"]), float(row["expert_y"]), float(row["expert_z"])], dtype=np.float64)
            pred = np.asarray(
                [float(row[f"{pred_prefix}_x"]), float(row[f"{pred_prefix}_y"]), float(row[f"{pred_prefix}_z"])],
                dtype=np.float64,
            )
            rows[key] = (expert, pred)
            samples.setdefault(
                row["sample_id"],
                PredictionSample(row["sample_id"], row["class"], row["gender"], int(float(row["subject_id"]))),
            )
    return rows, samples


def main():
    parser = argparse.ArgumentParser(description="Average multiple PAL-Net prediction CSV files and report ensemble ALE.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction CSVs from matching test splits.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix", default="ensemble_test")
    args = parser.parse_args()

    loaded = [load_prediction_csv(path) for path in args.predictions]
    key_sets = [set(rows) for rows, _ in loaded]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Prediction CSVs do not contain the same sample_id/landmark keys")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = sorted({key[0] for key in key_sets[0]})
    sample_meta = loaded[0][1]
    samples = [sample_meta[sample_id] for sample_id in sample_ids]
    y_true = []
    y_pred = []
    for sample_id in sample_ids:
        true_rows = []
        pred_rows = []
        for lm_idx in range(23):
            key = (sample_id, lm_idx)
            expert = loaded[0][0][key][0]
            preds = np.stack([rows[key][1] for rows, _ in loaded], axis=0)
            true_rows.append(expert)
            pred_rows.append(preds.mean(axis=0))
        y_true.append(true_rows)
        y_pred.append(pred_rows)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = np.linalg.norm(y_pred - y_true, axis=-1)

    out_csv = output_dir / f"predictions_{args.suffix}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "class",
                "gender",
                "subject_id",
                "landmark",
                "expert_x",
                "expert_y",
                "expert_z",
                "palnet_ensemble_x",
                "palnet_ensemble_y",
                "palnet_ensemble_z",
                "localization_error",
            ]
        )
        for sample, truth, pred, sample_errors in zip(samples, y_true, y_pred, errors):
            for lm_idx in range(23):
                writer.writerow(
                    [
                        sample.sample_id,
                        sample.class_name,
                        sample.gender,
                        sample.subject_id,
                        lm_idx,
                        *truth[lm_idx].tolist(),
                        *pred[lm_idx].tolist(),
                        float(sample_errors[lm_idx]),
                    ]
                )

    analysis = build_error_analysis(samples, errors)
    metrics = {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "unit": "dataset coordinate unit",
        "clinical_threshold_unit": "mm",
        "model": "PAL-Net prediction CSV ensemble",
        "prediction_files": [str(path) for path in args.predictions],
        "palnet_ensemble": {
            "ale": float(errors.mean()),
            "std": float(errors.std()),
            "median": float(np.median(errors)),
            "max": float(errors.max()),
            "per_landmark_ale": errors.mean(axis=0).astype(float).tolist(),
            "per_sample_ale": errors.mean(axis=1).astype(float).tolist(),
        },
        **analysis,
    }
    (output_dir / f"metrics_{args.suffix}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_analysis_csvs(output_dir, analysis, suffix=args.suffix)
    print(f"Ensemble ALE: {metrics['palnet_ensemble']['ale']:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
