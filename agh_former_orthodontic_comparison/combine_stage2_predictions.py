import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_landmarks(value):
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
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


def summarize(errors):
    arr = np.asarray(errors, dtype=np.float64)
    flat = arr.reshape(-1)
    out = {
        "ale": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "max": float(flat.max()),
        "per_landmark_ale": arr.mean(axis=0).astype(float).tolist(),
        "per_landmark_median": np.median(arr, axis=0).astype(float).tolist(),
        "per_sample_ale": arr.mean(axis=1).astype(float).tolist(),
    }
    for threshold in (2.0, 2.5, 3.0):
        key = ("%g" % threshold).replace(".", "_")
        out[f"pck_at_{key}mm"] = float((flat <= threshold).mean())
    return out


def load_prediction_table(run_dir, split):
    rows = read_rows(Path(run_dir) / f"refined_predictions_{split}.csv")
    return {(row["sample_id"], int(row["landmark"])): row for row in rows}


def coord(row, prefix):
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")], dtype=np.float64)


def build_combined(base_dir, specialist_dir, split, replace_landmarks, source_prefix):
    base = load_prediction_table(base_dir, split)
    specialist = load_prediction_table(specialist_dir, split)
    sample_ids = sorted({sample_id for sample_id, _ in base})
    sample_to_pos = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    errors = np.zeros((len(sample_ids), 23), dtype=np.float64)
    rows = []
    replace_set = set(replace_landmarks)
    for sample_id in sample_ids:
        for lm_idx in range(23):
            key = (sample_id, lm_idx)
            row = dict(base[key])
            chosen = specialist[key] if lm_idx in replace_set and key in specialist else base[key]
            pred = coord(chosen, source_prefix)
            expert = coord(base[key], "expert")
            error = float(np.linalg.norm(pred - expert))
            errors[sample_to_pos[sample_id], lm_idx] = error
            for axis, value in zip(("x", "y", "z"), pred):
                row[f"combined_{axis}"] = float(value)
            row["combined_source"] = "specialist" if lm_idx in replace_set and key in specialist else "base"
            row["combined_localization_error"] = error
            rows.append(row)
    return rows, errors


def grouped_metrics(rows):
    groups = {}
    landmark_groups = {}
    for row in rows:
        error = float(row["combined_localization_error"])
        groups.setdefault(("class", row["class"]), []).append(error)
        groups.setdefault(("gender", row["gender"]), []).append(error)
        groups.setdefault(("class_gender", f"{row['class']}|{row['gender']}"), []).append(error)
        landmark_groups.setdefault(int(row["landmark"]), []).append(error)

    group_rows = []
    for (scope, key), values in sorted(groups.items()):
        arr = np.asarray(values, dtype=np.float64)
        group_rows.append(
            {
                "scope": scope,
                "group": key,
                "n_points": len(values),
                "ale": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "pck_at_2mm": float((arr <= 2.0).mean()),
                "pck_at_2_5mm": float((arr <= 2.5).mean()),
                "pck_at_3mm": float((arr <= 3.0).mean()),
            }
        )

    landmark_rows = []
    for lm_idx, values in sorted(landmark_groups.items()):
        arr = np.asarray(values, dtype=np.float64)
        landmark_rows.append(
            {
                "landmark": lm_idx,
                "n_points": len(values),
                "ale": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "max": float(arr.max()),
                "pck_at_2mm": float((arr <= 2.0).mean()),
                "pck_at_2_5mm": float((arr <= 2.5).mean()),
                "pck_at_3mm": float((arr <= 3.0).mean()),
            }
        )
    return group_rows, landmark_rows


def main():
    parser = argparse.ArgumentParser(description="Combine a base Stage2 run with a hard-landmark specialist Stage2 run.")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--specialist-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replace-landmarks", default="0,21,22")
    parser.add_argument("--source-prefix", choices=["stage2_raw", "stage2_snapped"], default="stage2_raw")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replace_landmarks = parse_landmarks(args.replace_landmarks)
    metrics = {
        "method": "base Stage2 predictions with selected landmarks replaced by specialist Stage2 predictions",
        "base_run_dir": str(args.base_run_dir),
        "specialist_run_dir": str(args.specialist_run_dir),
        "replace_landmarks": replace_landmarks,
        "source_prefix": args.source_prefix,
    }
    for split in ("val", "test"):
        rows, errors = build_combined(args.base_run_dir, args.specialist_run_dir, split, replace_landmarks, args.source_prefix)
        write_rows(output_dir / f"combined_predictions_{split}.csv", rows)
        group_rows, landmark_rows = grouped_metrics(rows)
        write_rows(output_dir / f"combined_group_metrics_{split}.csv", group_rows)
        write_rows(output_dir / f"combined_landmark_metrics_{split}.csv", landmark_rows)
        metrics[split] = summarize(errors)
    (output_dir / "metrics_combined.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Validation combined ALE: {metrics['val']['ale']:.4f}", flush=True)
    print(f"Test combined ALE: {metrics['test']['ale']:.4f}", flush=True)
    print(f"Test combined median: {metrics['test']['median']:.4f}", flush=True)
    print(f"Results saved to: {output_dir / 'metrics_combined.json'}", flush=True)


if __name__ == "__main__":
    main()
