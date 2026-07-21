import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


def parse_ints(value):
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


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


def coord(row, prefix):
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")], dtype=np.float64)


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


def load_stage_tables(run_dir, split):
    stage1_rows = read_rows(Path(run_dir) / f"stage1_predictions_{split}.csv")
    stage2_rows = read_rows(Path(run_dir) / f"refined_predictions_{split}.csv")
    stage1 = {(row["sample_id"], int(row["landmark"])): row for row in stage1_rows}
    stage2 = {(row["sample_id"], int(row["landmark"])): row for row in stage2_rows}
    return stage1, stage2


def split_sample_ids(stage2_table):
    return sorted({sample_id for sample_id, _ in stage2_table})


def table_to_arrays(stage2_table, sample_ids, source_prefix="stage2_raw"):
    sample_to_pos = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    pred = np.zeros((len(sample_ids), 23, 3), dtype=np.float64)
    expert = np.zeros_like(pred)
    meta = {}
    for (sample_id, lm_idx), row in stage2_table.items():
        pos = sample_to_pos[sample_id]
        pred[pos, lm_idx] = coord(row, source_prefix)
        expert[pos, lm_idx] = coord(row, "expert")
        meta[sample_id] = {
            "sample_id": sample_id,
            "class": row.get("class", ""),
            "gender": row.get("gender", ""),
            "subject_id": row.get("subject_id", ""),
        }
    return pred, expert, meta


def load_train_experts(run_dir):
    rows = read_rows(Path(run_dir) / "stage1_predictions_train.csv")
    sample_ids = sorted({row["sample_id"] for row in rows})
    sample_to_pos = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    experts = np.zeros((len(sample_ids), 23, 3), dtype=np.float64)
    for row in rows:
        experts[sample_to_pos[row["sample_id"]], int(row["landmark"])] = coord(row, "expert")
    return experts


def build_distance_priors(train_experts, hard_landmarks, reliable_landmarks, num_neighbors):
    priors = {}
    for lm_idx in hard_landmarks:
        lm_priors = []
        for nb_idx in reliable_landmarks:
            distances = np.linalg.norm(train_experts[:, lm_idx] - train_experts[:, nb_idx], axis=1)
            lm_priors.append(
                {
                    "neighbor": int(nb_idx),
                    "mean": float(distances.mean()),
                    "std": float(max(distances.std(), 0.75)),
                }
            )
        lm_priors.sort(key=lambda item: item["mean"])
        priors[int(lm_idx)] = lm_priors[: int(num_neighbors)]
    return priors


def residual_covariances(val_pred, val_expert, hard_landmarks, min_variance):
    covariances = {}
    for lm_idx in hard_landmarks:
        residuals = val_expert[:, lm_idx] - val_pred[:, lm_idx]
        cov = np.cov(residuals.T)
        if cov.shape != (3, 3) or not np.isfinite(cov).all():
            cov = np.eye(3, dtype=np.float64)
        cov = cov + np.eye(3, dtype=np.float64) * float(min_variance)
        covariances[int(lm_idx)] = {
            "cov": cov,
            "inv": np.linalg.pinv(cov),
            "diag": np.diag(cov).astype(float).tolist(),
        }
    return covariances


def load_dataset(args, split_payload):
    from run_orthodontic_aghformer import AGHFormerDataset, ids_to_indices

    dataset = AGHFormerDataset(
        root_dir=args.data_root,
        cache_dir=Path(args.base_run_dir) / "stage1_point_cache",
        num_points=args.surface_points,
        heatmap_sigma=args.heatmap_sigma,
        use_normals=True,
        use_local_geometry=True,
        local_geometry_k=16,
        transformation_dir=args.transformation_dir,
        seed=args.seed,
    )
    val_idx = ids_to_indices(dataset, split_payload["val"])
    test_idx = ids_to_indices(dataset, split_payload["test"])
    return dataset, {"val": val_idx, "test": test_idx}


def candidate_points(points, base_point, stage1_point, radius, stage1_radius, max_candidates):
    dist_base = np.linalg.norm(points - base_point[None, :], axis=1)
    dist_stage1 = np.linalg.norm(points - stage1_point[None, :], axis=1)
    mask = (dist_base <= float(radius)) | (dist_stage1 <= float(stage1_radius))
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.arange(len(points))
    if len(idx) > int(max_candidates):
        combined = np.minimum(dist_base[idx], dist_stage1[idx])
        keep = np.argpartition(combined, int(max_candidates) - 1)[: int(max_candidates)]
        idx = idx[keep]
    return points[idx].astype(np.float64)


def candidate_feature_matrix(
    candidates,
    base_point,
    stage1_point,
    all_pred,
    lm_idx,
    priors,
    covariance,
    pair_landmark=None,
    pair_prior=None,
):
    delta = candidates - base_point[None, :]
    model_score = np.einsum("ni,ij,nj->n", delta, covariance["inv"], delta)
    stage1_score = (np.linalg.norm(candidates - stage1_point[None, :], axis=1) / 8.0) ** 2
    isotropic_score = (np.linalg.norm(delta, axis=1) / 6.0) ** 2

    anatomical_scores = []
    for prior in priors:
        nb_idx = int(prior["neighbor"])
        distance = np.linalg.norm(candidates - all_pred[nb_idx][None, :], axis=1)
        z = (distance - float(prior["mean"])) / float(prior["std"])
        anatomical_scores.append(np.clip(z * z, 0.0, 25.0))
    anatomy_score = np.mean(np.stack(anatomical_scores, axis=0), axis=0) if anatomical_scores else np.zeros(len(candidates))

    pair_score = np.zeros(len(candidates), dtype=np.float64)
    if pair_landmark is not None and pair_prior is not None:
        pair_distance = np.linalg.norm(candidates - all_pred[int(pair_landmark)][None, :], axis=1)
        pair_z = (pair_distance - float(pair_prior["mean"])) / float(pair_prior["std"])
        pair_score = np.clip(pair_z * pair_z, 0.0, 25.0)

    return {
        "model": model_score,
        "stage1": stage1_score,
        "isotropic": isotropic_score,
        "anatomy": anatomy_score,
        "pair": pair_score,
    }


def train_pair_priors(train_experts, pair_map):
    out = {}
    for lm_idx, pair_idx in pair_map.items():
        distances = np.linalg.norm(train_experts[:, lm_idx] - train_experts[:, pair_idx], axis=1)
        out[int(lm_idx)] = {"mean": float(distances.mean()), "std": float(max(distances.std(), 0.75))}
    return out


def generate_split_candidates(
    split,
    dataset,
    split_indices,
    stage1_table,
    stage2_table,
    hard_landmarks,
    priors,
    covariances,
    pair_map,
    pair_priors,
    args,
):
    sample_ids = split_sample_ids(stage2_table)
    sample_to_idx = {dataset.metadata(idx).sample_id: idx for idx in split_indices}
    pred_array, expert_array, _ = table_to_arrays(stage2_table, sample_ids, source_prefix=args.source_prefix)
    candidate_bank = {}
    for sample_id in sample_ids:
        data = dataset[sample_to_idx[sample_id]]
        points = data["points_world"].numpy().astype(np.float64)
        pos = sample_ids.index(sample_id)
        for lm_idx in hard_landmarks:
            s1_row = stage1_table[(sample_id, int(lm_idx))]
            stage1_point = coord(s1_row, f"stage1_{args.stage1_center}")
            base_point = pred_array[pos, int(lm_idx)]
            candidates = candidate_points(
                points,
                base_point,
                stage1_point,
                radius=args.candidate_radius_mm,
                stage1_radius=args.stage1_radius_mm,
                max_candidates=args.max_candidates,
            )
            pair_idx = pair_map.get(int(lm_idx))
            features = candidate_feature_matrix(
                candidates,
                base_point,
                stage1_point,
                pred_array[pos],
                int(lm_idx),
                priors[int(lm_idx)],
                covariances[int(lm_idx)],
                pair_landmark=pair_idx,
                pair_prior=pair_priors.get(int(lm_idx)) if pair_idx is not None else None,
            )
            candidate_bank[(sample_id, int(lm_idx))] = {
                "points": candidates,
                "features": features,
                "base_point": base_point,
                "expert": expert_array[pos, int(lm_idx)],
            }
    return sample_ids, pred_array, expert_array, candidate_bank


def score_candidates(features, weights):
    score = np.zeros_like(next(iter(features.values())), dtype=np.float64)
    for key, weight in weights.items():
        score = score + float(weight) * features[key]
    return score


def select_predictions(candidate_bank, sample_ids, lm_idx, weights):
    selected = {}
    for sample_id in sample_ids:
        item = candidate_bank[(sample_id, int(lm_idx))]
        scores = score_candidates(item["features"], weights)
        best_idx = int(np.argmin(scores))
        selected[sample_id] = {
            "point": item["points"][best_idx],
            "score": float(scores[best_idx]),
            "candidate_index": best_idx,
        }
    return selected


def evaluate_selected(selected, candidate_bank, sample_ids, lm_idx):
    errors = []
    for sample_id in sample_ids:
        expert = candidate_bank[(sample_id, int(lm_idx))]["expert"]
        errors.append(float(np.linalg.norm(selected[sample_id]["point"] - expert)))
    return float(np.mean(errors)), errors


def learn_weights(val_bank, val_sample_ids, hard_landmarks, base_val_errors, args):
    grids = {
        "model": [float(v) for v in args.model_weight_grid.split(",")],
        "anatomy": [float(v) for v in args.anatomy_weight_grid.split(",")],
        "stage1": [float(v) for v in args.stage1_weight_grid.split(",")],
        "pair": [float(v) for v in args.pair_weight_grid.split(",")],
        "isotropic": [float(v) for v in args.isotropic_weight_grid.split(",")],
    }
    selected = {}
    for lm_idx in hard_landmarks:
        best = None
        for model_w, anatomy_w, stage1_w, pair_w, isotropic_w in itertools.product(
            grids["model"], grids["anatomy"], grids["stage1"], grids["pair"], grids["isotropic"]
        ):
            weights = {
                "model": model_w,
                "anatomy": anatomy_w,
                "stage1": stage1_w,
                "pair": pair_w,
                "isotropic": isotropic_w,
            }
            chosen = select_predictions(val_bank, val_sample_ids, int(lm_idx), weights)
            mean_error, _ = evaluate_selected(chosen, val_bank, val_sample_ids, int(lm_idx))
            if best is None or mean_error < best["val_ale"]:
                best = {"weights": weights, "val_ale": mean_error}
        base_error = float(base_val_errors[:, int(lm_idx)].mean())
        best["base_val_ale"] = base_error
        best["enabled"] = bool(best["val_ale"] + float(args.min_val_improvement_mm) < base_error)
        selected[int(lm_idx)] = best
    return selected


def build_output_rows(stage2_table, sample_ids, combined_pred, expert, source_map):
    rows = []
    sample_to_pos = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    for sample_id in sample_ids:
        pos = sample_to_pos[sample_id]
        for lm_idx in range(23):
            row = dict(stage2_table[(sample_id, lm_idx)])
            pred = combined_pred[pos, lm_idx]
            err = float(np.linalg.norm(pred - expert[pos, lm_idx]))
            for axis, value in zip(("x", "y", "z"), pred):
                row[f"hard_postprocess_{axis}"] = float(value)
            row["hard_postprocess_source"] = source_map.get((sample_id, lm_idx), "base")
            row["hard_postprocess_localization_error"] = err
            rows.append(row)
    return rows


def group_rows(rows):
    groups = {}
    by_landmark = {}
    for row in rows:
        err = float(row["hard_postprocess_localization_error"])
        groups.setdefault(("class", row["class"]), []).append(err)
        groups.setdefault(("gender", row["gender"]), []).append(err)
        groups.setdefault(("class_gender", f"{row['class']}|{row['gender']}"), []).append(err)
        by_landmark.setdefault(int(row["landmark"]), []).append(err)

    grouped = []
    for (scope, group), values in sorted(groups.items()):
        arr = np.asarray(values, dtype=np.float64)
        grouped.append(
            {
                "scope": scope,
                "group": group,
                "n_points": len(values),
                "ale": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "pck_at_2mm": float((arr <= 2.0).mean()),
                "pck_at_2_5mm": float((arr <= 2.5).mean()),
                "pck_at_3mm": float((arr <= 3.0).mean()),
            }
        )

    landmarks = []
    for lm_idx, values in sorted(by_landmark.items()):
        arr = np.asarray(values, dtype=np.float64)
        landmarks.append(
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
    return grouped, landmarks


def apply_postprocess(sample_ids, base_pred, expert, candidate_bank, learned, hard_landmarks):
    combined = base_pred.copy()
    source_map = {}
    sample_to_pos = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    hard_errors = {}
    for lm_idx in hard_landmarks:
        config = learned[int(lm_idx)]
        if not config["enabled"]:
            continue
        selected = select_predictions(candidate_bank, sample_ids, int(lm_idx), config["weights"])
        errors = []
        for sample_id in sample_ids:
            pos = sample_to_pos[sample_id]
            point = selected[sample_id]["point"]
            combined[pos, int(lm_idx)] = point
            source_map[(sample_id, int(lm_idx))] = "hard_candidate"
            errors.append(float(np.linalg.norm(point - expert[pos, int(lm_idx)])))
        hard_errors[int(lm_idx)] = errors
    errors = np.linalg.norm(combined - expert, axis=-1)
    return combined, errors, source_map, hard_errors


def main():
    parser = argparse.ArgumentParser(description="Validation-selected hard landmark postprocess for AGH-Former Stage2 outputs.")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits-json", default=None)
    parser.add_argument("--transformation-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hard-landmarks", default="0,21,22")
    parser.add_argument("--stage1-center", choices=["raw", "snapped"], default="snapped")
    parser.add_argument("--source-prefix", choices=["stage2_raw", "stage2_snapped"], default="stage2_raw")
    parser.add_argument("--surface-points", type=int, default=12000)
    parser.add_argument("--heatmap-sigma", type=float, default=5.0)
    parser.add_argument("--candidate-radius-mm", type=float, default=18.0)
    parser.add_argument("--stage1-radius-mm", type=float, default=22.0)
    parser.add_argument("--max-candidates", type=int, default=384)
    parser.add_argument("--num-neighbors", type=int, default=8)
    parser.add_argument("--min-cov-variance", type=float, default=1.0)
    parser.add_argument("--min-val-improvement-mm", type=float, default=0.0)
    parser.add_argument("--model-weight-grid", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--anatomy-weight-grid", default="0.25,0.5,1.0,2.0,4.0")
    parser.add_argument("--stage1-weight-grid", default="0.0,0.25,0.5")
    parser.add_argument("--pair-weight-grid", default="0.0,0.25,0.5")
    parser.add_argument("--isotropic-weight-grid", default="0.0,0.25,0.5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_run_dir = Path(args.base_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hard_landmarks = parse_ints(args.hard_landmarks)
    pair_map = {21: 22, 22: 21}
    reliable_landmarks = [idx for idx in range(23) if idx not in set(hard_landmarks)]

    split_path = Path(args.splits_json) if args.splits_json else base_run_dir / "splits.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    dataset, split_indices = load_dataset(args, split_payload)

    train_experts = load_train_experts(base_run_dir)
    priors = build_distance_priors(train_experts, hard_landmarks, reliable_landmarks, args.num_neighbors)
    pair_priors = train_pair_priors(train_experts, pair_map)

    val_stage1, val_stage2 = load_stage_tables(base_run_dir, "val")
    test_stage1, test_stage2 = load_stage_tables(base_run_dir, "test")
    val_sample_ids = split_sample_ids(val_stage2)
    test_sample_ids = split_sample_ids(test_stage2)
    val_base_pred, val_expert, _ = table_to_arrays(val_stage2, val_sample_ids, source_prefix=args.source_prefix)
    test_base_pred, test_expert, _ = table_to_arrays(test_stage2, test_sample_ids, source_prefix=args.source_prefix)
    val_base_errors = np.linalg.norm(val_base_pred - val_expert, axis=-1)
    test_base_errors = np.linalg.norm(test_base_pred - test_expert, axis=-1)
    covariances = residual_covariances(val_base_pred, val_expert, hard_landmarks, args.min_cov_variance)

    val_sample_ids, _, _, val_bank = generate_split_candidates(
        "val",
        dataset,
        split_indices["val"],
        val_stage1,
        val_stage2,
        hard_landmarks,
        priors,
        covariances,
        pair_map,
        pair_priors,
        args,
    )
    test_sample_ids, _, _, test_bank = generate_split_candidates(
        "test",
        dataset,
        split_indices["test"],
        test_stage1,
        test_stage2,
        hard_landmarks,
        priors,
        covariances,
        pair_map,
        pair_priors,
        args,
    )

    learned = learn_weights(val_bank, val_sample_ids, hard_landmarks, val_base_errors, args)
    val_combined, val_errors, val_sources, val_hard_errors = apply_postprocess(
        val_sample_ids, val_base_pred, val_expert, val_bank, learned, hard_landmarks
    )
    test_combined, test_errors, test_sources, test_hard_errors = apply_postprocess(
        test_sample_ids, test_base_pred, test_expert, test_bank, learned, hard_landmarks
    )

    val_rows = build_output_rows(val_stage2, val_sample_ids, val_combined, val_expert, val_sources)
    test_rows = build_output_rows(test_stage2, test_sample_ids, test_combined, test_expert, test_sources)
    write_rows(output_dir / "hard_postprocess_predictions_val.csv", val_rows)
    write_rows(output_dir / "hard_postprocess_predictions_test.csv", test_rows)
    val_grouped, val_landmarks = group_rows(val_rows)
    test_grouped, test_landmarks = group_rows(test_rows)
    write_rows(output_dir / "hard_postprocess_group_metrics_val.csv", val_grouped)
    write_rows(output_dir / "hard_postprocess_group_metrics_test.csv", test_grouped)
    write_rows(output_dir / "hard_postprocess_landmark_metrics_val.csv", val_landmarks)
    write_rows(output_dir / "hard_postprocess_landmark_metrics_test.csv", test_landmarks)

    metrics = {
        "method": "validation-selected hard landmark candidate postprocess",
        "base_run_dir": str(base_run_dir),
        "hard_landmarks": hard_landmarks,
        "source_prefix": args.source_prefix,
        "candidate_radius_mm": args.candidate_radius_mm,
        "stage1_radius_mm": args.stage1_radius_mm,
        "max_candidates": args.max_candidates,
        "num_neighbors": args.num_neighbors,
        "distance_priors": priors,
        "residual_covariance_diag": {str(k): v["diag"] for k, v in covariances.items()},
        "learned_landmark_configs": learned,
        "base_validation": summarize(val_base_errors),
        "base_test": summarize(test_base_errors),
        "postprocess_validation": summarize(val_errors),
        "postprocess_test": summarize(test_errors),
        "hard_landmark_validation_ale": {
            str(lm_idx): float(np.mean(val_hard_errors.get(int(lm_idx), val_base_errors[:, int(lm_idx)])))
            for lm_idx in hard_landmarks
        },
        "hard_landmark_test_ale": {
            str(lm_idx): float(np.mean(test_hard_errors.get(int(lm_idx), test_base_errors[:, int(lm_idx)])))
            for lm_idx in hard_landmarks
        },
    }
    (output_dir / "metrics_hard_postprocess.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Base validation ALE: {metrics['base_validation']['ale']:.4f}", flush=True)
    print(f"Postprocess validation ALE: {metrics['postprocess_validation']['ale']:.4f}", flush=True)
    print(f"Base test ALE: {metrics['base_test']['ale']:.4f}", flush=True)
    print(f"Postprocess test ALE: {metrics['postprocess_test']['ale']:.4f}", flush=True)
    print(f"Postprocess test median: {metrics['postprocess_test']['median']:.4f}", flush=True)
    print(f"Results saved to: {output_dir / 'metrics_hard_postprocess.json'}", flush=True)


if __name__ == "__main__":
    main()
