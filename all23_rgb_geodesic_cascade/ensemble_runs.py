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
    from all23_rgb_geodesic_cascade.metrics import bootstrap_ci, localization_errors, summarize, write_csv
else:
    from .anatomy import CORE20, HARD3, NUM_LANDMARKS
    from .metrics import bootstrap_ci, localization_errors, summarize, write_csv


def read_predictions(path):
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
        prediction[sample_id][landmark] = [float(row[f"prediction_{axis}"]) for axis in ("x", "y", "z")]
        expert[sample_id][landmark] = [float(row[f"expert_{axis}"]) for axis in ("x", "y", "z")]
    return ids, np.stack([prediction[key] for key in ids]), np.stack([expert[key] for key in ids]), metadata


def main():
    parser = argparse.ArgumentParser(description="Validation-locked three-seed All-23 ensemble")
    parser.add_argument("--run-dirs", nargs=3, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--required-validation-ale", type=float, default=2.05)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for split in ("val", "test"):
        sources = [read_predictions(Path(path) / f"predictions_{split}.csv") for path in args.run_dirs]
        ids, experts, metadata = sources[0][0], sources[0][2], sources[0][3]
        for source in sources[1:]:
            if source[0] != ids or not np.allclose(source[2], experts, atol=1e-4):
                raise ValueError(f"Seed prediction rows disagree for split={split}")
        stack = np.stack([source[1] for source in sources])
        loaded[split] = {
            "ids": ids,
            "expert": experts,
            "metadata": metadata,
            "mean": stack.mean(axis=0),
            "median": np.median(stack, axis=0),
        }
    val_scores = {
        mode: float(localization_errors(loaded["val"][mode], loaded["val"]["expert"]).mean())
        for mode in ("mean", "median")
    }
    individual_val_scores = [
        float(localization_errors(source[1], source[2]).mean())
        for source in [read_predictions(Path(path) / "predictions_val.csv") for path in args.run_dirs]
    ]
    if min(individual_val_scores) > args.required_validation_ale and not args.force:
        raise RuntimeError(
            f"Best seed validation ALE={min(individual_val_scores):.4f} is above "
            f"the E8 gate {args.required_validation_ale:.4f}. Use --force only for diagnostic analysis."
        )
    selected = min(val_scores, key=val_scores.get)
    test_prediction = loaded["test"][selected]
    test_expert = loaded["test"]["expert"]
    errors = localization_errors(test_prediction, test_expert)
    metrics = {
        "selected_on_validation": selected,
        "validation_candidates": val_scores,
        "individual_validation_ale": individual_val_scores,
        "overall": summarize(errors),
        "core20": summarize(errors[:, CORE20]),
        "hard3": summarize(errors[:, HARD3]),
        "bootstrap_ale": bootstrap_ci(errors, args.bootstrap_iters, args.seed),
        "run_dirs": args.run_dirs,
    }
    (output_dir / "metrics_ensemble.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rows = []
    for sample_index, sample_id in enumerate(loaded["test"]["ids"]):
        meta = loaded["test"]["metadata"][sample_id]
        for landmark in range(NUM_LANDMARKS):
            row = {
                "sample_id": sample_id,
                "class": meta.get("class", ""),
                "gender": meta.get("gender", ""),
                "subject_id": meta.get("subject_id", ""),
                "landmark": landmark,
            }
            for name, values in (("expert", test_expert), ("prediction", test_prediction)):
                for axis_index, axis in enumerate(("x", "y", "z")):
                    row[f"{name}_{axis}"] = float(values[sample_index, landmark, axis_index])
            row["error"] = float(errors[sample_index, landmark])
            rows.append(row)
    write_csv(output_dir / "predictions_test.csv", rows)
    print(f"Selected ensemble: {selected}", flush=True)
    print(f"All-23 ALE: {metrics['overall']['ale']:.4f}", flush=True)


if __name__ == "__main__":
    main()
