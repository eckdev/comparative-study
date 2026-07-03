import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared_metrics.orthodontic_analysis import build_error_analysis, write_analysis_csvs


@dataclass(frozen=True)
class PredictionSample:
    sample_id: str
    class_name: str
    gender: str
    subject_id: int


def load_prediction_errors(path):
    rows_by_sample = {}
    metadata = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row["sample_id"]
            landmark = int(row["landmark"])
            rows_by_sample.setdefault(sample_id, {})[landmark] = float(row["localization_error"])
            metadata.setdefault(
                sample_id,
                PredictionSample(
                    sample_id=sample_id,
                    class_name=row["class"],
                    gender=row["gender"],
                    subject_id=int(float(row["subject_id"])),
                ),
            )

    sample_ids = sorted(rows_by_sample)
    samples = [metadata[sample_id] for sample_id in sample_ids]
    errors = []
    for sample_id in sample_ids:
        landmarks = rows_by_sample[sample_id]
        if sorted(landmarks) != list(range(23)):
            raise ValueError(f"{sample_id} does not contain all 23 landmark errors")
        errors.append([landmarks[idx] for idx in range(23)])
    return samples, np.asarray(errors, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Add landmark, PCK, class, gender, and difficult-landmark analysis to an existing run.")
    parser.add_argument("--predictions", required=True, help="Path to predictions_test.csv or predictions_eval_test.csv.")
    parser.add_argument("--output-dir", default=None, help="Run directory. Defaults to the predictions file parent.")
    parser.add_argument("--suffix", default="test", help="Output suffix, for example test or eval_test.")
    parser.add_argument("--metrics-json", default=None, help="Optional metrics JSON to update. Defaults to metrics.json in output-dir when present.")
    args = parser.parse_args()

    predictions = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions.parent
    samples, errors = load_prediction_errors(predictions)
    analysis = build_error_analysis(samples, errors)
    write_analysis_csvs(output_dir, analysis, suffix=args.suffix)

    metrics_path = Path(args.metrics_json) if args.metrics_json else output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["clinical_threshold_unit"] = "mm"
        metrics.update(analysis)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote analysis CSVs to {output_dir}")
    if metrics_path.exists():
        print(f"Updated {metrics_path}")


if __name__ == "__main__":
    main()
