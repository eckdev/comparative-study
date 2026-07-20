import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
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


def load_split(run_dir, split, stage1_center="snapped", stage2_source="raw"):
    stage1_rows = read_rows(Path(run_dir) / f"stage1_predictions_{split}.csv")
    stage2_rows = read_rows(Path(run_dir) / f"refined_predictions_{split}.csv")
    stage1 = {(r["sample_id"], int(r["landmark"])): r for r in stage1_rows}
    stage2 = {(r["sample_id"], int(r["landmark"])): r for r in stage2_rows}
    keys = sorted(stage2.keys())
    sample_ids = sorted({sample_id for sample_id, _ in keys})
    sample_to_pos = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    centers = np.zeros((len(sample_ids), 23, 3), dtype=np.float64)
    refined = np.zeros_like(centers)
    expert = np.zeros_like(centers)
    meta = {}
    for sample_id, lm_idx in keys:
        if (sample_id, lm_idx) not in stage1:
            raise ValueError(f"Missing Stage1 row for {sample_id} landmark {lm_idx}")
        s1 = stage1[(sample_id, lm_idx)]
        s2 = stage2[(sample_id, lm_idx)]
        pos = sample_to_pos[sample_id]
        center_prefix = f"stage1_{stage1_center}"
        stage2_prefix = f"stage2_{stage2_source}"
        centers[pos, lm_idx] = [float(s1[f"{center_prefix}_{axis}"]) for axis in ("x", "y", "z")]
        refined[pos, lm_idx] = [float(s2[f"{stage2_prefix}_{axis}"]) for axis in ("x", "y", "z")]
        expert[pos, lm_idx] = [float(s2[f"expert_{axis}"]) for axis in ("x", "y", "z")]
        meta[sample_id] = {
            "sample_id": sample_id,
            "class": s2.get("class", ""),
            "gender": s2.get("gender", ""),
            "subject_id": s2.get("subject_id", ""),
        }
    return sample_ids, meta, centers, refined, expert


def learn_landmark_alphas(centers, refined, expert, alpha_grid):
    deltas = refined - centers
    alphas = np.zeros(23, dtype=np.float64)
    val_errors = np.zeros(23, dtype=np.float64)
    for lm_idx in range(23):
        best_alpha = 0.0
        best_error = float("inf")
        for alpha in alpha_grid:
            pred = centers[:, lm_idx] + alpha * deltas[:, lm_idx]
            err = np.linalg.norm(pred - expert[:, lm_idx], axis=-1).mean()
            if err < best_error:
                best_error = float(err)
                best_alpha = float(alpha)
        alphas[lm_idx] = best_alpha
        val_errors[lm_idx] = best_error
    return alphas, val_errors


def apply_alphas(centers, refined, alphas):
    return centers + alphas[None, :, None] * (refined - centers)


def build_prediction_rows(sample_ids, meta, pred, expert, errors, alphas):
    rows = []
    for sample_pos, sample_id in enumerate(sample_ids):
        item = meta[sample_id]
        for lm_idx in range(23):
            rows.append(
                {
                    "sample_id": sample_id,
                    "class": item["class"],
                    "gender": item["gender"],
                    "subject_id": item["subject_id"],
                    "landmark": lm_idx,
                    "alpha": float(alphas[lm_idx]),
                    "expert_x": float(expert[sample_pos, lm_idx, 0]),
                    "expert_y": float(expert[sample_pos, lm_idx, 1]),
                    "expert_z": float(expert[sample_pos, lm_idx, 2]),
                    "calibrated_x": float(pred[sample_pos, lm_idx, 0]),
                    "calibrated_y": float(pred[sample_pos, lm_idx, 1]),
                    "calibrated_z": float(pred[sample_pos, lm_idx, 2]),
                    "calibrated_localization_error": float(errors[sample_pos, lm_idx]),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Validate-only delta calibration for AGH-Former Stage2 outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage1-center", choices=["raw", "snapped"], default="snapped")
    parser.add_argument("--stage2-source", choices=["raw", "snapped"], default="raw")
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.25)
    parser.add_argument("--alpha-step", type=float, default=0.025)
    parser.add_argument("--output-suffix", default="delta_calibrated")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    alpha_grid = np.arange(args.alpha_min, args.alpha_max + args.alpha_step * 0.5, args.alpha_step)
    val_ids, val_meta, val_centers, val_refined, val_expert = load_split(
        run_dir, "val", stage1_center=args.stage1_center, stage2_source=args.stage2_source
    )
    test_ids, test_meta, test_centers, test_refined, test_expert = load_split(
        run_dir, "test", stage1_center=args.stage1_center, stage2_source=args.stage2_source
    )

    alphas, val_lm_errors = learn_landmark_alphas(val_centers, val_refined, val_expert, alpha_grid)
    val_pred = apply_alphas(val_centers, val_refined, alphas)
    test_pred = apply_alphas(test_centers, test_refined, alphas)
    val_errors = np.linalg.norm(val_pred - val_expert, axis=-1)
    test_errors = np.linalg.norm(test_pred - test_expert, axis=-1)

    metrics = {
        "method": "validation-selected per-landmark delta calibration",
        "run_dir": str(run_dir),
        "stage1_center": args.stage1_center,
        "stage2_source": args.stage2_source,
        "alpha_grid": alpha_grid.astype(float).tolist(),
        "landmark_alphas": alphas.astype(float).tolist(),
        "landmark_val_selected_ale": val_lm_errors.astype(float).tolist(),
        "validation": summarize(val_errors),
        "test": summarize(test_errors),
    }
    metrics_path = run_dir / f"metrics_{args.output_suffix}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_rows(
        run_dir / f"predictions_test_{args.output_suffix}.csv",
        build_prediction_rows(test_ids, test_meta, test_pred, test_expert, test_errors, alphas),
    )
    print(f"Validation calibrated ALE: {metrics['validation']['ale']:.4f}", flush=True)
    print(f"Test calibrated ALE: {metrics['test']['ale']:.4f}", flush=True)
    print(f"Test calibrated median: {metrics['test']['median']:.4f}", flush=True)
    print(f"Results saved to: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
