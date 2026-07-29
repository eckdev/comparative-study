import argparse
import csv
import json
from pathlib import Path

import numpy as np


HARD_LANDMARKS = (0, 21, 22)
CORE20 = tuple(i for i in range(23) if i not in HARD_LANDMARKS)


def parse_float_grid(value):
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


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


def infer_prefix(row, preferred):
    if f"{preferred}_x" in row:
        return preferred
    for prefix in ("final", "shape_prior", "stage2_raw", "stage2_snapped", "base", "pred"):
        if f"{prefix}_x" in row:
            return prefix
    raise KeyError(f"Could not infer prediction prefix from columns: {sorted(row)}")


def load_base_prediction(prediction_dir, split, source_prefix):
    path = Path(prediction_dir) / f"refined_predictions_{split}.csv"
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"Empty prediction CSV: {path}")
    prefix = infer_prefix(rows[0], source_prefix)
    sample_ids = []
    pred = {}
    expert = {}
    metadata = {}
    for row in rows:
        sample_id = row["sample_id"]
        lm_idx = int(row["landmark"])
        if sample_id not in pred:
            sample_ids.append(sample_id)
            pred[sample_id] = np.zeros((23, 3), dtype=np.float64)
            expert[sample_id] = np.zeros((23, 3), dtype=np.float64)
            metadata[sample_id] = {
                "sample_id": sample_id,
                "class": row.get("class", ""),
                "gender": row.get("gender", ""),
                "subject_id": row.get("subject_id", ""),
            }
        pred[sample_id][lm_idx] = [float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")]
        expert[sample_id][lm_idx] = [float(row[f"expert_{axis}"]) for axis in ("x", "y", "z")]
    return {
        "path": str(path),
        "prefix": prefix,
        "sample_ids": sample_ids,
        "metadata": metadata,
        "pred": np.stack([pred[sample_id] for sample_id in sample_ids]),
        "expert": np.stack([expert[sample_id] for sample_id in sample_ids]),
    }


def load_candidate_prediction(candidate_dir, split, source_prefix):
    path = Path(candidate_dir) / f"predictions_{split}.csv"
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"Empty candidate CSV: {path}")
    prefix = infer_prefix(rows[0], source_prefix)
    sample_ids = []
    pred = {}
    for row in rows:
        sample_id = row["sample_id"]
        lm_idx = int(row["landmark"])
        if sample_id not in pred:
            sample_ids.append(sample_id)
            pred[sample_id] = np.zeros((23, 3), dtype=np.float64)
        pred[sample_id][lm_idx] = [float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")]
    return {
        "path": str(path),
        "prefix": prefix,
        "sample_ids": sample_ids,
        "pred": np.stack([pred[sample_id] for sample_id in sample_ids]),
    }


def summarize(errors):
    arr = np.asarray(errors, dtype=np.float64)
    flat = arr.reshape(-1)
    out = {
        "ale": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "p75": float(np.percentile(flat, 75)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
        "sdr_at_2mm": float((flat <= 2.0).mean()),
        "sdr_at_3mm": float((flat <= 3.0).mean()),
        "sdr_at_4mm": float((flat <= 4.0).mean()),
    }
    if arr.ndim == 2:
        out["per_landmark_ale"] = arr.mean(axis=0).astype(float).tolist()
        out["per_landmark_median"] = np.median(arr, axis=0).astype(float).tolist()
    return out


def bootstrap_delta(base_errors, stacked_errors, n_boot, seed):
    rng = np.random.default_rng(seed)
    base = np.asarray(base_errors, dtype=np.float64)
    stacked = np.asarray(stacked_errors, dtype=np.float64)
    sample_count = base.shape[0]
    out = {}
    for name, landmarks in {
        "all23": tuple(range(23)),
        "core20": CORE20,
        "hard3": HARD_LANDMARKS,
    }.items():
        deltas = []
        pck2 = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, sample_count, size=sample_count)
            b = base[idx][:, landmarks].reshape(-1)
            s = stacked[idx][:, landmarks].reshape(-1)
            deltas.append(float(s.mean() - b.mean()))
            pck2.append(float((s <= 2.0).mean() - (b <= 2.0).mean()))
        deltas = np.asarray(deltas)
        pck2 = np.asarray(pck2)
        out[name] = {
            "base_ale": float(base[:, landmarks].mean()),
            "stacked_ale": float(stacked[:, landmarks].mean()),
            "delta_ale": float(stacked[:, landmarks].mean() - base[:, landmarks].mean()),
            "delta_ale_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "probability_stacked_better": float((deltas < 0).mean()),
            "delta_sdr_at_2mm": float((stacked[:, landmarks].reshape(-1) <= 2.0).mean() - (base[:, landmarks].reshape(-1) <= 2.0).mean()),
            "delta_sdr_at_2mm_ci95": [float(np.percentile(pck2, 2.5)), float(np.percentile(pck2, 97.5))],
        }
    return out


def fit_stacker(val_base, val_expert, val_candidates, l2, shrinkage, coef_clip):
    n_methods = len(val_candidates)
    pred = val_base.copy()
    coefs = np.zeros((23, n_methods), dtype=np.float64)
    for lm_idx in range(23):
        residual_features = np.stack(
            [(candidate[:, lm_idx] - val_base[:, lm_idx]).reshape(len(val_base), 3) for candidate in val_candidates],
            axis=2,
        )
        x = residual_features.reshape(-1, n_methods)
        y = (val_expert[:, lm_idx] - val_base[:, lm_idx]).reshape(-1)
        reg = float(l2) * np.eye(n_methods, dtype=np.float64)
        coef = np.linalg.solve(x.T @ x + reg, x.T @ y)
        coef = np.clip(coef, -float(coef_clip), float(coef_clip))
        coefs[lm_idx] = coef
        pred[:, lm_idx] = val_base[:, lm_idx] + float(shrinkage) * (x @ coef).reshape(len(val_base), 3)
    return pred, coefs


def apply_stacker(base, candidates, coefs, shrinkage):
    pred = base.copy()
    for lm_idx in range(23):
        residual_features = np.stack(
            [(candidate[:, lm_idx] - base[:, lm_idx]).reshape(len(base), 3) for candidate in candidates],
            axis=2,
        )
        x = residual_features.reshape(-1, len(candidates))
        pred[:, lm_idx] = base[:, lm_idx] + float(shrinkage) * (x @ coefs[lm_idx]).reshape(len(base), 3)
    return pred


def write_prediction_csv(path, split, base, stacked, errors):
    rows = []
    expert = split["expert"]
    for sample_pos, sample_id in enumerate(split["sample_ids"]):
        meta = split["metadata"][sample_id]
        for lm_idx in range(23):
            rows.append(
                {
                    "sample_id": sample_id,
                    "class": meta.get("class", ""),
                    "gender": meta.get("gender", ""),
                    "subject_id": meta.get("subject_id", ""),
                    "landmark": lm_idx,
                    "expert_x": float(expert[sample_pos, lm_idx, 0]),
                    "expert_y": float(expert[sample_pos, lm_idx, 1]),
                    "expert_z": float(expert[sample_pos, lm_idx, 2]),
                    "base_x": float(base[sample_pos, lm_idx, 0]),
                    "base_y": float(base[sample_pos, lm_idx, 1]),
                    "base_z": float(base[sample_pos, lm_idx, 2]),
                    "stacked_x": float(stacked[sample_pos, lm_idx, 0]),
                    "stacked_y": float(stacked[sample_pos, lm_idx, 1]),
                    "stacked_z": float(stacked[sample_pos, lm_idx, 2]),
                    "stacked_error": float(errors[sample_pos, lm_idx]),
                }
            )
    write_rows(path, rows)


def write_landmark_metrics(path, base_errors, stacked_errors):
    rows = []
    for lm_idx in range(23):
        base = base_errors[:, lm_idx]
        stacked = stacked_errors[:, lm_idx]
        rows.append(
            {
                "landmark": lm_idx,
                "base_ale": float(base.mean()),
                "stacked_ale": float(stacked.mean()),
                "delta_ale": float(stacked.mean() - base.mean()),
                "base_median": float(np.median(base)),
                "stacked_median": float(np.median(stacked)),
                "base_sdr_at_2mm": float((base <= 2.0).mean()),
                "stacked_sdr_at_2mm": float((stacked <= 2.0).mean()),
                "base_sdr_at_3mm": float((base <= 3.0).mean()),
                "stacked_sdr_at_3mm": float((stacked <= 3.0).mean()),
                "improved_count": int((stacked < base).sum()),
                "worsened_count": int((stacked > base).sum()),
                "n": int(len(base)),
            }
        )
    write_rows(path, rows)


def main():
    parser = argparse.ArgumentParser(description="Validation-selected residual stacker for shape-prior variants.")
    parser.add_argument("--base-prediction-dir", required=True)
    parser.add_argument("--candidate-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-source-prefix", default="stage2_raw")
    parser.add_argument("--candidate-source-prefix", default="final")
    parser.add_argument("--l2-grid", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--shrinkage-grid", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--coef-clip", type=float, default=1.0)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val = load_base_prediction(args.base_prediction_dir, "val", args.base_source_prefix)
    test = load_base_prediction(args.base_prediction_dir, "test", args.base_source_prefix)
    val_candidates = []
    test_candidates = []
    candidate_info = []
    for candidate_dir in [Path(path) for path in args.candidate_dirs]:
        val_candidate = load_candidate_prediction(candidate_dir, "val", args.candidate_source_prefix)
        test_candidate = load_candidate_prediction(candidate_dir, "test", args.candidate_source_prefix)
        if val_candidate["sample_ids"] != val["sample_ids"]:
            raise ValueError(f"Validation sample order mismatch: {candidate_dir}")
        if test_candidate["sample_ids"] != test["sample_ids"]:
            raise ValueError(f"Test sample order mismatch: {candidate_dir}")
        val_candidates.append(val_candidate["pred"])
        test_candidates.append(test_candidate["pred"])
        candidate_info.append(
            {
                "dir": str(candidate_dir),
                "val_path": val_candidate["path"],
                "test_path": test_candidate["path"],
                "prefix": val_candidate["prefix"],
            }
        )

    best = None
    sweep_rows = []
    for l2 in parse_float_grid(args.l2_grid):
        for shrinkage in parse_float_grid(args.shrinkage_grid):
            val_pred, coefs = fit_stacker(
                val["pred"],
                val["expert"],
                val_candidates,
                l2=l2,
                shrinkage=shrinkage,
                coef_clip=args.coef_clip,
            )
            val_errors = np.linalg.norm(val_pred - val["expert"], axis=-1)
            score = float(val_errors.mean())
            sweep_rows.append({"l2": l2, "shrinkage": shrinkage, "validation_ale": score})
            if best is None or score < best["score"]:
                best = {"score": score, "l2": l2, "shrinkage": shrinkage, "coefs": coefs}

    val_pred = apply_stacker(val["pred"], val_candidates, best["coefs"], best["shrinkage"])
    test_pred = apply_stacker(test["pred"], test_candidates, best["coefs"], best["shrinkage"])
    val_base_errors = np.linalg.norm(val["pred"] - val["expert"], axis=-1)
    test_base_errors = np.linalg.norm(test["pred"] - test["expert"], axis=-1)
    val_errors = np.linalg.norm(val_pred - val["expert"], axis=-1)
    test_errors = np.linalg.norm(test_pred - test["expert"], axis=-1)

    write_prediction_csv(output_dir / "predictions_val.csv", val, val["pred"], val_pred, val_errors)
    write_prediction_csv(output_dir / "predictions_test.csv", test, test["pred"], test_pred, test_errors)
    write_landmark_metrics(output_dir / "landmark_metrics_val.csv", val_base_errors, val_errors)
    write_landmark_metrics(output_dir / "landmark_metrics_test.csv", test_base_errors, test_errors)
    write_rows(output_dir / "sweep_validation.csv", sweep_rows)
    bootstrap = bootstrap_delta(test_base_errors, test_errors, args.bootstrap_iters, args.seed)
    metrics = {
        "model": "Shape-prior residual stacker",
        "base_prediction_dir": str(args.base_prediction_dir),
        "base_val_path": val["path"],
        "base_test_path": test["path"],
        "candidate_info": candidate_info,
        "best_l2": float(best["l2"]),
        "best_shrinkage": float(best["shrinkage"]),
        "coef_clip": float(args.coef_clip),
        "base_validation": summarize(val_base_errors),
        "stacked_validation": summarize(val_errors),
        "base_test": summarize(test_base_errors),
        "stacked_test": summarize(test_errors),
        "base_core20_test": summarize(test_base_errors[:, CORE20]),
        "stacked_core20_test": summarize(test_errors[:, CORE20]),
        "base_hard3_test": summarize(test_base_errors[:, HARD_LANDMARKS]),
        "stacked_hard3_test": summarize(test_errors[:, HARD_LANDMARKS]),
        "bootstrap": bootstrap,
        "coefficients": {
            str(lm_idx): {
                Path(args.candidate_dirs[col_idx]).name: float(best["coefs"][lm_idx, col_idx])
                for col_idx in range(len(args.candidate_dirs))
            }
            for lm_idx in range(23)
        },
    }
    (output_dir / "metrics_shape_prior_stacker.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "config_shape_prior_stacker.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"Base ALE: {metrics['base_test']['ale']:.4f}", flush=True)
    print(f"Stacked ALE: {metrics['stacked_test']['ale']:.4f}", flush=True)
    print(f"Stacked median: {metrics['stacked_test']['median']:.4f}", flush=True)
    print(f"Core20 base/stacked ALE: {metrics['base_core20_test']['ale']:.4f} -> {metrics['stacked_core20_test']['ale']:.4f}", flush=True)
    print(f"Hard3 base/stacked ALE: {metrics['base_hard3_test']['ale']:.4f} -> {metrics['stacked_hard3_test']['ale']:.4f}", flush=True)
    ci = metrics["bootstrap"]["all23"]["delta_ale_ci95"]
    prob = metrics["bootstrap"]["all23"]["probability_stacked_better"]
    print(f"Stacked ALE delta CI95: [{ci[0]:.4f}, {ci[1]:.4f}], P(improved)={prob:.3f}", flush=True)
    print(f"Best l2={best['l2']} shrinkage={best['shrinkage']}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
