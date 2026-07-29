import argparse
import csv
from pathlib import Path

import numpy as np

from src.metrics import combined_metrics, landmark_metrics, write_csv, write_json


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return (row["sample_id"], int(row["landmark"]))


def row_error(row, prefix):
    expert = np.array([float(row[f"expert_{axis}"]) for axis in "xyz"], dtype=np.float64)
    pred = np.array([float(row[f"{prefix}_{axis}"]) for axis in "xyz"], dtype=np.float64)
    return float(np.linalg.norm(pred - expert))


def merge_split(base_path, specialist_paths, output_path):
    rows = read_rows(base_path)
    by_key = {key(row): row for row in rows}
    for specialist_path in specialist_paths:
        for row in read_rows(specialist_path):
            if row.get("is_hard_refined", "False") != "True":
                continue
            target = by_key[key(row)]
            for axis in "xyz":
                target[f"final_{axis}"] = row[f"final_{axis}"]
            target["confidence"] = row.get("confidence", target.get("confidence", "0"))
            target["is_hard_refined"] = "True"
    for row in rows:
        row["final_error"] = row_error(row, "final")
        row["base_error"] = row_error(row, "base")
    write_csv(output_path, rows)
    return rows


def metrics_from_rows(rows):
    sample_ids = sorted({row["sample_id"] for row in rows})
    index = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    errors = np.zeros((len(sample_ids), 23), dtype=np.float64)
    for row in rows:
        errors[index[row["sample_id"]], int(row["landmark"])] = float(row["final_error"])
    return errors, combined_metrics(errors)


def main():
    parser = argparse.ArgumentParser(description="Merge AGH-HardNet specialist prediction CSVs.")
    parser.add_argument("--base-dir", type=Path, required=True, help="Run directory used as base predictions.")
    parser.add_argument("--specialist-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for split in ("val", "test"):
        base_path = args.base_dir / f"predictions_{split}.csv"
        specialist_paths = [path / f"predictions_{split}.csv" for path in args.specialist_dirs]
        rows = merge_split(base_path, specialist_paths, args.output_dir / f"predictions_{split}.csv")
        errors, split_metrics = metrics_from_rows(rows)
        write_csv(args.output_dir / f"landmark_metrics_{split}.csv", landmark_metrics(errors))
        metrics[split] = split_metrics
    write_json(
        args.output_dir / "metrics_merged_specialists.json",
        {
            "base_dir": str(args.base_dir),
            "specialist_dirs": [str(path) for path in args.specialist_dirs],
            "metrics": metrics,
        },
    )
    print("Merged specialist test ALE:", f"{metrics['test']['overall']['ale']:.4f}", flush=True)
    print("Merged specialist hard3 ALE:", f"{metrics['test']['hard_landmarks']['ale']:.4f}", flush=True)
    print("Results saved to:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
