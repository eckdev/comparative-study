import csv
import json
from pathlib import Path

import numpy as np

from .anatomy import CORE20, HARD3, LANDMARK_NAMES, NUM_LANDMARKS, landmark_group


def localization_errors(prediction, expert):
    return np.linalg.norm(np.asarray(prediction) - np.asarray(expert), axis=-1)


def summarize(errors):
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    result = {
        "n": int(len(values)),
        "ale": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }
    for threshold in (2.0, 3.0, 4.0):
        result[f"sdr_at_{int(threshold)}mm"] = float(np.mean(values <= threshold))
    return result


def bootstrap_ci(errors, iterations=2000, seed=42):
    matrix = np.asarray(errors, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        sample_indices = rng.integers(0, len(matrix), size=len(matrix))
        estimates[index] = matrix[sample_indices].mean()
    return {
        "ale": float(matrix.mean()),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
    }


def bootstrap_delta(base_errors, final_errors, iterations=2000, seed=42):
    base = np.asarray(base_errors, dtype=np.float64)
    final = np.asarray(final_errors, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        sample_indices = rng.integers(0, len(base), size=len(base))
        values[index] = final[sample_indices].mean() - base[sample_indices].mean()
    return {
        "delta_ale": float(final.mean() - base.mean()),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "probability_improved": float(np.mean(values < 0)),
    }


def landmark_rows(errors):
    matrix = np.asarray(errors)
    rows = []
    for index in range(NUM_LANDMARKS):
        values = matrix[:, index]
        rows.append(
            {
                "landmark": index,
                "name": LANDMARK_NAMES[index],
                "group": landmark_group(index),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "sdr_at_2mm": float(np.mean(values <= 2.0)),
                "sdr_at_3mm": float(np.mean(values <= 3.0)),
                "sdr_at_4mm": float(np.mean(values <= 4.0)),
            }
        )
    return rows


def group_rows(errors, classes, genders):
    errors = np.asarray(errors)
    rows = []
    groups = {"overall:all": np.arange(len(errors))}
    for value in sorted(set(classes)):
        groups[f"class:{value}"] = np.flatnonzero(np.asarray(classes) == value)
    for value in sorted(set(genders)):
        groups[f"gender:{value}"] = np.flatnonzero(np.asarray(genders) == value)
    for name, indices in groups.items():
        scope, group = name.split(":", 1)
        rows.append({"scope": scope, "group": group, "n_samples": int(len(indices)), **summarize(errors[indices])})
    return rows


def calibrate_confidence(log_var, errors):
    log_var = np.asarray(log_var, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    scales = {}
    for group in ("midline", "bilateral", "hard3"):
        indices = [index for index in range(NUM_LANDMARKS) if landmark_group(index) == group]
        ratio = errors[:, indices] ** 2 / np.exp(log_var[:, indices])
        scales[group] = float(np.clip(np.mean(ratio), 1e-3, 1e3))
    return scales


def confidence_scores(log_var, calibration):
    log_var = np.asarray(log_var, dtype=np.float64)
    output = np.zeros_like(log_var)
    for index in range(NUM_LANDMARKS):
        variance = np.exp(log_var[:, index]) * calibration[landmark_group(index)]
        output[:, index] = np.exp(-np.sqrt(variance) / 2.0)
    return output


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(outputs, confidence):
    rows = []
    for sample_index, sample_id in enumerate(outputs["sample_ids"]):
        for landmark in range(NUM_LANDMARKS):
            row = {
                "sample_id": sample_id,
                "class": outputs["classes"][sample_index],
                "gender": outputs["genders"][sample_index],
                "subject_id": int(outputs["subject_ids"][sample_index]),
                "landmark": landmark,
                "landmark_name": LANDMARK_NAMES[landmark],
            }
            coordinate_names = ["expert", "coarse", "refined", "prediction"]
            if "stage1_coarse" in outputs:
                coordinate_names.append("stage1_coarse")
            if "neural_prediction" in outputs:
                coordinate_names.append("neural_prediction")
            for name in coordinate_names:
                for axis_index, axis in enumerate(("x", "y", "z")):
                    row[f"{name}_{axis}"] = float(outputs[name][sample_index, landmark, axis_index])
            row["error"] = float(outputs["errors"][sample_index, landmark])
            row["confidence"] = float(confidence[sample_index, landmark])
            row["refinement_alpha"] = float(
                outputs["refinement_alpha"][sample_index, landmark]
            )
            row["fusion_alpha"] = float(
                outputs.get("fusion_alpha", np.zeros_like(outputs["refinement_alpha"]))[
                    sample_index, landmark
                ]
            )
            row["oracle_error"] = float(outputs["oracle"][sample_index, landmark])
            rows.append(row)
    return rows


def save_evaluation(output_dir, split_name, outputs, calibration, bootstrap_iterations=2000, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = outputs["errors"]
    refined_errors = outputs["refined_errors"]
    coarse_errors = localization_errors(outputs["coarse"], outputs["expert"])
    confidence = confidence_scores(outputs["log_var"], calibration)
    metrics = {
        "overall": summarize(errors),
        "core20": summarize(errors[:, CORE20]),
        "hard3": summarize(errors[:, HARD3]),
        "ungated_refined_overall": summarize(refined_errors),
        "ungated_refined_core20": summarize(refined_errors[:, CORE20]),
        "ungated_refined_hard3": summarize(refined_errors[:, HARD3]),
        "coarse_overall": summarize(coarse_errors),
        "coarse_core20": summarize(coarse_errors[:, CORE20]),
        "coarse_hard3": summarize(coarse_errors[:, HARD3]),
        "oracle_overall": summarize(outputs["oracle"]),
        "bootstrap_ale": bootstrap_ci(errors, bootstrap_iterations, seed),
        "bootstrap_vs_coarse": bootstrap_delta(coarse_errors, errors, bootstrap_iterations, seed),
        "bootstrap_vs_ungated_refined": bootstrap_delta(
            refined_errors, errors, bootstrap_iterations, seed
        ),
        "refinement_gate": {
            "mean_alpha": float(np.mean(outputs["refinement_alpha"])),
            "core20_mean_alpha": float(np.mean(outputs["refinement_alpha"][:, CORE20])),
            "hard3_mean_alpha": float(np.mean(outputs["refinement_alpha"][:, HARD3])),
            "refined_better_fraction": float(np.mean(refined_errors < coarse_errors)),
            "core20_refined_better_fraction": float(
                np.mean(refined_errors[:, CORE20] < coarse_errors[:, CORE20])
            ),
            "hard3_refined_better_fraction": float(
                np.mean(refined_errors[:, HARD3] < coarse_errors[:, HARD3])
            ),
        },
        "confidence_calibration": calibration,
    }
    if "stage1_coarse" in outputs:
        stage1_errors = localization_errors(outputs["stage1_coarse"], outputs["expert"])
        metrics["stage1_coarse_overall"] = summarize(stage1_errors)
        metrics["stage1_coarse_core20"] = summarize(stage1_errors[:, CORE20])
        metrics["stage1_coarse_hard3"] = summarize(stage1_errors[:, HARD3])
    if "fusion_alpha" in outputs:
        metrics["coarse_fusion"] = {
            "mean_alpha": float(np.mean(outputs["fusion_alpha"])),
            "core20_mean_alpha": float(np.mean(outputs["fusion_alpha"][:, CORE20])),
            "hard3_mean_alpha": float(np.mean(outputs["fusion_alpha"][:, HARD3])),
        }
    (output_dir / f"metrics_{split_name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(output_dir / f"landmark_metrics_{split_name}.csv", landmark_rows(errors))
    write_csv(output_dir / f"group_metrics_{split_name}.csv", group_rows(errors, outputs["classes"], outputs["genders"]))
    write_csv(output_dir / f"predictions_{split_name}.csv", prediction_rows(outputs, confidence))
    worst = np.argsort(errors.mean(axis=1))[::-1]
    write_csv(
        output_dir / f"outlier_samples_{split_name}.csv",
        [
            {
                "sample_id": outputs["sample_ids"][index],
                "class": outputs["classes"][index],
                "gender": outputs["genders"][index],
                "ale": float(errors[index].mean()),
                "median": float(np.median(errors[index])),
                "max": float(errors[index].max()),
            }
            for index in worst
        ],
    )
    return metrics
