#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, NUM_LANDMARKS
    from all23_rgb_geodesic_cascade.metrics import (
        bootstrap_ci, group_rows, landmark_rows, localization_errors, summarize, write_csv,
    )
else:
    from .anatomy import CORE20, HARD3, NUM_LANDMARKS
    from .metrics import bootstrap_ci, group_rows, landmark_rows, localization_errors, summarize, write_csv


def load(path, prefix):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids, prediction, expert, metadata = [], {}, {}, {}
    for row in rows:
        sample_id = row["sample_id"]
        landmark = int(row["landmark"])
        if sample_id not in prediction:
            ids.append(sample_id)
            prediction[sample_id] = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            expert[sample_id] = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            metadata[sample_id] = row
        prediction[sample_id][landmark] = [float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")]
        expert[sample_id][landmark] = [float(row[f"expert_{axis}"]) for axis in ("x", "y", "z")]
    return ids, np.stack([prediction[key] for key in ids]), np.stack([expert[key] for key in ids]), metadata


def main():
    parser = argparse.ArgumentParser(description="E0 existing-prediction baseline metrics")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="stacked")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        ids, prediction, expert, metadata = load(
            Path(args.prediction_dir) / f"predictions_{split}.csv", args.prefix
        )
        errors = localization_errors(prediction, expert)
        metrics = {
            "model": "E0 existing stacker baseline",
            "overall": summarize(errors),
            "core20": summarize(errors[:, CORE20]),
            "hard3": summarize(errors[:, HARD3]),
            "bootstrap_ale": bootstrap_ci(errors, args.bootstrap_iters, args.seed),
            "source": str(Path(args.prediction_dir) / f"predictions_{split}.csv"),
            "prefix": args.prefix,
        }
        (output_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_csv(output_dir / f"landmark_metrics_{split}.csv", landmark_rows(errors))
        classes = [metadata[sample_id].get("class", "") for sample_id in ids]
        genders = [metadata[sample_id].get("gender", "") for sample_id in ids]
        write_csv(output_dir / f"group_metrics_{split}.csv", group_rows(errors, classes, genders))
    print(f"E0 test ALE: {metrics['overall']['ale']:.4f}", flush=True)


if __name__ == "__main__":
    main()
