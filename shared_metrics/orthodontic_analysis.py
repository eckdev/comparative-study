import csv
from pathlib import Path

import numpy as np


DEFAULT_THRESHOLDS_MM = (2.0, 2.5, 3.0)
CLASS_LABELS = {
    "Class1": "Class I",
    "Class2": "Class II",
    "Class3": "Class III",
}
GENDER_LABELS = {
    "men": "male",
    "women": "female",
}


def metric_key(prefix, threshold):
    text = ("%g" % threshold).replace(".", "_")
    return f"{prefix}_{text}mm"


def summarize_error_array(errors, thresholds=DEFAULT_THRESHOLDS_MM):
    arr = np.asarray(errors, dtype=np.float64).reshape(-1)
    summary = {
        "n_points": int(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "ale": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }
    for threshold in thresholds:
        within = arr <= threshold
        summary[metric_key("pck_at", threshold)] = float(within.mean()) if arr.size else 0.0
        summary[metric_key("fail_rate_gt", threshold)] = float((~within).mean()) if arr.size else 0.0
        summary[metric_key("n_within", threshold)] = int(within.sum())
    return summary


def class_label(class_name):
    return CLASS_LABELS.get(class_name, class_name)


def gender_label(gender):
    return GENDER_LABELS.get(gender, gender)


def landmark_name(index):
    return f"landmark_{index:02d}"


def build_error_analysis(samples, errors, thresholds=DEFAULT_THRESHOLDS_MM):
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2:
        raise ValueError(f"errors must have shape [n_samples, n_landmarks], got {errors.shape}")
    if len(samples) != errors.shape[0]:
        raise ValueError(f"samples/errors length mismatch: {len(samples)} vs {errors.shape[0]}")

    landmark_rows = []
    for lm_idx in range(errors.shape[1]):
        landmark_rows.append(
            {
                "landmark": int(lm_idx),
                "landmark_name": landmark_name(lm_idx),
                **summarize_error_array(errors[:, lm_idx], thresholds),
            }
        )

    class_rows = []
    for class_name in sorted({sample.class_name for sample in samples}):
        idxs = [idx for idx, sample in enumerate(samples) if sample.class_name == class_name]
        class_rows.append(
            {
                "class_key": class_name,
                "class": class_label(class_name),
                "n_samples": int(len(idxs)),
                **summarize_error_array(errors[idxs], thresholds),
            }
        )

    gender_rows = []
    for gender in sorted({sample.gender for sample in samples}):
        idxs = [idx for idx, sample in enumerate(samples) if sample.gender == gender]
        gender_rows.append(
            {
                "gender_key": gender,
                "gender": gender_label(gender),
                "n_samples": int(len(idxs)),
                **summarize_error_array(errors[idxs], thresholds),
            }
        )

    class_gender_rows = []
    for class_name, gender in sorted({(sample.class_name, sample.gender) for sample in samples}):
        idxs = [
            idx
            for idx, sample in enumerate(samples)
            if sample.class_name == class_name and sample.gender == gender
        ]
        class_gender_rows.append(
            {
                "class_key": class_name,
                "class": class_label(class_name),
                "gender_key": gender,
                "gender": gender_label(gender),
                "n_samples": int(len(idxs)),
                **summarize_error_array(errors[idxs], thresholds),
            }
        )

    clinical_thresholds = []
    scopes = [("overall", "all", "all", list(range(errors.shape[0])))]
    scopes.extend(
        ("class", row["class_key"], row["class"], [i for i, s in enumerate(samples) if s.class_name == row["class_key"]])
        for row in class_rows
    )
    scopes.extend(
        ("gender", row["gender_key"], row["gender"], [i for i, s in enumerate(samples) if s.gender == row["gender_key"]])
        for row in gender_rows
    )

    for scope, subgroup_key, subgroup, idxs in scopes:
        arr = errors[idxs].reshape(-1)
        for threshold in thresholds:
            within = arr <= threshold
            clinical_thresholds.append(
                {
                    "scope": scope,
                    "subgroup_key": subgroup_key,
                    "subgroup": subgroup,
                    "threshold_mm": float(threshold),
                    "n_points": int(arr.size),
                    "n_within_threshold": int(within.sum()),
                    "pck": float(within.mean()) if arr.size else 0.0,
                    "fail_rate": float((~within).mean()) if arr.size else 0.0,
                }
            )

    difficult_landmarks = sorted(
        landmark_rows,
        key=lambda row: (row["mean"], row["median"], row["max"]),
        reverse=True,
    )
    difficult_landmarks = [
        {
            "rank": rank,
            **row,
        }
        for rank, row in enumerate(difficult_landmarks, start=1)
    ]

    return {
        "thresholds_mm": [float(t) for t in thresholds],
        "overall_threshold_performance": summarize_error_array(errors, thresholds),
        "landmark_error_analysis": landmark_rows,
        "clinical_threshold_analysis": clinical_thresholds,
        "class_performance": class_rows,
        "gender_performance": gender_rows,
        "class_gender_performance": class_gender_rows,
        "difficult_landmark_analysis": {
            "ranking_metric": "mean_error_desc_then_median_then_max",
            "top_landmarks": difficult_landmarks[:5],
            "all_landmarks": difficult_landmarks,
        },
    }


def write_dict_rows_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_csvs(output_dir, analysis, suffix="test"):
    output_dir = Path(output_dir)
    write_dict_rows_csv(output_dir / f"landmark_metrics_{suffix}.csv", analysis["landmark_error_analysis"])
    write_dict_rows_csv(output_dir / f"clinical_thresholds_{suffix}.csv", analysis["clinical_threshold_analysis"])
    write_dict_rows_csv(output_dir / f"class_metrics_{suffix}.csv", analysis["class_performance"])
    write_dict_rows_csv(output_dir / f"gender_metrics_{suffix}.csv", analysis["gender_performance"])
    write_dict_rows_csv(output_dir / f"class_gender_metrics_{suffix}.csv", analysis["class_gender_performance"])
    write_dict_rows_csv(output_dir / f"difficult_landmarks_{suffix}.csv", analysis["difficult_landmark_analysis"]["all_landmarks"])
