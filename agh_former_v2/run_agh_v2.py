#!/usr/bin/env python3
"""AGH-Former v2: global stage + one all-23 multi-view local refiner.

Phase 1 reuses the archived AGH-Former vNext artefacts for the global stage:
the Stage-1 OOF centres, alignment and mesh caches come from the shared
preprocessing root, and the Stage-2 fusion output ("coarse_*" columns of the
archived vNext predictions) is the incoming estimate. Everything after that is
new: a Procrustes shape prior and the All23-MVSC refiner with a cross-fitted
OOF gate. The vNext local head, its gate, the Hard3 dual-view specialist and
the Core20 specialist are not used.

Modes
  --validation-only : fit on outer-train, lock the policy on outer validation;
                      no test file is opened.
  --final-test      : reuse the locked policy and predict the test split with
                      label-free ROIs; expert test landmarks are read only after
                      the predictions are written.
"""

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agh_former_vnext_orthodontic_comparison.run_aghformer_vnext import (
    build_parser as build_vnext_parser,
    configure_vnext,
)
from all23_mvsc_refinement import (
    All23MVSCConfig,
    ProcrustesShapePrior,
    apply_policy,
    calibrate_policy,
    fit_or_load_all23_refinement,
    write_policy,
)
from all23_mvsc_refinement.anatomy import ANATOMICAL_GROUPS, LANDMARK_TYPES
from all23_mvsc_refinement.refiner import dataset_shapes, roi_cache_dir
from all23_rgb_geodesic_cascade.alignment import apply_transform
from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, LANDMARK_NAMES, NUM_LANDMARKS
from all23_rgb_geodesic_cascade.data import discover_samples, read_landmarks
from all23_rgb_geodesic_cascade.metrics import (
    bootstrap_ci,
    bootstrap_delta,
    landmark_rows,
    summarize,
    write_csv,
)
from all23_rgb_geodesic_cascade.run_all23_rgb_geodesic import (
    fold_output_dir,
    make_cv_splits,
    prepare_fold,
    resolve_device,
    resolve_precision,
    set_seed,
    validate_preprocessing_root,
)


VERSION = "AGH-Former-v2.1"
STAGE_A_COLUMN = "coarse"
SHARED_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "shared_splits/orthodontic_5fold_192_48_60_seed42.json"
)


def build_parser():
    parser = build_vnext_parser()
    parser.description = "AGH-Former v2 (global stage + single all-23 refiner)"
    parser.add_argument(
        "--stage-a-run",
        required=True,
        help="Archived vNext CV run directory with fold_k/predictions_{val,test}.csv",
    )
    parser.add_argument("--stage-a-column", default=STAGE_A_COLUMN)
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Local disk for memory-mapped candidate patches (deleted after use)",
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Predict the test split with a policy locked by a prior validation run",
    )
    parser.add_argument("--all23-epochs", type=int, default=120)
    parser.add_argument("--all23-min-epochs", type=int, default=30)
    parser.add_argument("--all23-patience", type=int, default=20)
    parser.add_argument("--all23-batch-size", type=int, default=32)
    parser.add_argument("--all23-image-size", type=int, default=96)
    parser.add_argument("--all23-width", type=int, default=48)
    parser.add_argument("--all23-lr", type=float, default=3e-4)
    parser.add_argument("--all23-max-iterations", type=int, default=2)
    parser.add_argument("--all23-gate-folds", type=int, default=5)
    parser.add_argument("--all23-gate-epochs", type=int, default=300)
    parser.add_argument("--all23-mc-samples", type=int, default=8)
    parser.add_argument("--no-all23-ensemble", dest="all23_ensemble", action="store_false")
    parser.add_argument(
        "--refiner-source-run",
        default=None,
        help=(
            "Earlier v2 run whose trained refiners (fold_k/all23_mvsc) are reused "
            "without retraining; evidence, gates and policies are rebuilt here"
        ),
    )
    parser.add_argument(
        "--no-gate-evidence-prior",
        dest="gate_evidence_prior",
        action="store_false",
        help="Do not pass gate-evidence centres through the selected shape prior",
    )
    parser.add_argument(
        "--no-all23-jaw-profile", dest="all23_jaw_profile", action="store_false"
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="Recompute validation_gate.json from existing predictions_val.csv files",
    )
    parser.add_argument(
        "--skip-manifest-check",
        action="store_true",
        help="Smoke runs only: do not require folds to equal the shared manifest",
    )
    parser.add_argument("--no-v2-shape-prior", dest="v2_shape_prior", action="store_false")
    parser.add_argument("--v2-shape-prior-min-gain-mm", type=float, default=0.01)
    parser.set_defaults(
        all23_jaw_profile=True,
        v2_shape_prior=True,
        all23_ensemble=True,
        gate_evidence_prior=True,
    )
    return parser


def all23_config_from_args(args):
    return All23MVSCConfig(
        epochs=args.all23_epochs,
        min_epochs=args.all23_min_epochs,
        patience=args.all23_patience,
        batch_size=args.all23_batch_size,
        image_size=args.all23_image_size,
        width=args.all23_width,
        lr=args.all23_lr,
        max_iterations=args.all23_max_iterations,
        gate_folds=args.all23_gate_folds,
        gate_epochs=args.all23_gate_epochs,
        ensemble=args.all23_ensemble,
        mc_samples=args.all23_mc_samples,
        use_jaw_profile=args.all23_jaw_profile,
        bootstrap_iters=args.bootstrap_iters,
        mixed_precision=bool(args.mixed_precision),
        amp_dtype=args.amp_dtype,
        seed=args.seed,
    )


def verify_against_shared_manifest(fold_index, splits, manifest_path=SHARED_MANIFEST):
    """The vNext runner derives folds itself; confirm they equal the shared manifest."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {"checked": False, "reason": f"missing {manifest_path}"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold = next(row for row in manifest["folds"] if int(row["fold"]) == int(fold_index))
    mismatched = [
        name for name in ("train", "val", "test") if sorted(fold[name]) != sorted(splits[name])
    ]
    if mismatched:
        raise RuntimeError(
            f"Fold {fold_index} differs from the shared manifest in {mismatched}"
        )
    return {
        "checked": True,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }


def read_stage_a(path, sample_ids, column=STAGE_A_COLUMN, read_expert=False):
    """Read only the requested coordinate columns of an archived prediction CSV."""
    wanted = set(sample_ids)
    coordinates = {sample_id: np.full((NUM_LANDMARKS, 3), np.nan) for sample_id in wanted}
    experts = {sample_id: np.full((NUM_LANDMARKS, 3), np.nan) for sample_id in wanted}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if sample_id not in wanted:
                continue
            landmark = int(row["landmark"])
            coordinates[sample_id][landmark] = [
                float(row[f"{column}_{axis}"]) for axis in "xyz"
            ]
            if read_expert:
                experts[sample_id][landmark] = [
                    float(row[f"expert_{axis}"]) for axis in "xyz"
                ]
    incomplete = [key for key, value in coordinates.items() if not np.isfinite(value).all()]
    if incomplete:
        raise RuntimeError(f"Stage A file {path} misses rows for {incomplete[:5]}")
    coordinates = {key: value.astype(np.float32) for key, value in coordinates.items()}
    if read_expert:
        return coordinates, {key: value.astype(np.float32) for key, value in experts.items()}
    return coordinates


def score_rows(sample_ids, splits_meta, expert, columns, extras):
    rows = []
    for sample_index, sample_id in enumerate(sample_ids):
        meta = splits_meta[sample_id]
        for landmark in range(NUM_LANDMARKS):
            row = {
                "sample_id": sample_id,
                "class": meta["class"],
                "gender": meta["gender"],
                "subject_id": meta["subject_id"],
                "landmark": landmark,
                "landmark_name": LANDMARK_NAMES[landmark],
            }
            for name, values in [("expert", expert)] + list(columns.items()):
                for axis_index, axis in enumerate("xyz"):
                    row[f"{name}_{axis}"] = float(values[sample_index, landmark, axis_index])
            row["error"] = float(
                np.linalg.norm(columns["prediction"][sample_index, landmark] - expert[sample_index, landmark])
            )
            for name, values in extras.items():
                row[name] = float(values[sample_index, landmark])
            rows.append(row)
    return rows


def stage_metrics(expert, columns):
    result = {}
    for name, values in columns.items():
        errors = np.linalg.norm(values - expert, axis=-1)
        result[name] = {
            "overall": summarize(errors),
            "core20": summarize(errors[:, list(CORE20)]),
            "hard3": summarize(errors[:, list(HARD3)]),
            "types": {
                key: summarize(errors[:, list(landmarks)])
                for key, landmarks in LANDMARK_TYPES.items()
            },
        }
    return result


def save_split(output_dir, split_name, sample_ids, meta, expert, columns, extras, seed, iterations):
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = np.linalg.norm(columns["prediction"] - expert, axis=-1)
    metrics = {
        "split": split_name,
        "overall": summarize(errors),
        "core20": summarize(errors[:, list(CORE20)]),
        "hard3": summarize(errors[:, list(HARD3)]),
        "bootstrap_ale": bootstrap_ci(errors, iterations, seed),
        "bootstrap_vs_stage_a": bootstrap_delta(
            np.linalg.norm(columns["stage_a"] - expert, axis=-1), errors, iterations, seed
        ),
        "stages": stage_metrics(expert, columns),
        "groups": {
            name: summarize(errors[:, list(landmarks)])
            for name, landmarks in ANATOMICAL_GROUPS.items()
        },
    }
    (output_dir / f"metrics_{split_name}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / f"landmark_metrics_{split_name}.csv", landmark_rows(errors))
    write_csv(
        output_dir / f"predictions_{split_name}.csv",
        score_rows(sample_ids, meta, expert, columns, extras),
    )
    return metrics


def sample_meta(dataset):
    return {
        sample.sample_id: {
            "class": sample.class_name,
            "gender": sample.gender,
            "subject_id": int(sample.subject_id),
        }
        for sample in dataset.samples
    }


def test_experts_after_prediction(dataset, transforms):
    """Read test landmark files only after predictions have been written."""
    return {
        sample.sample_id: apply_transform(
            read_landmarks(sample.landmark_path), transforms[sample.sample_id]
        ).astype(np.float32)
        for sample in dataset.samples
    }


def fit_v2_shape_prior(args, train_dataset, base, expert, sample_ids, fold_dir):
    if not args.v2_shape_prior:
        return None, {"enabled": False}
    shapes, train_ids, _, _ = dataset_shapes(train_dataset)
    prior = ProcrustesShapePrior().fit(shapes, train_ids)
    prior.calibrate(base, expert, sample_ids)
    base_ale = float(np.linalg.norm(base - expert, axis=-1).mean())
    candidate_ale = float(np.linalg.norm(prior.transform(base) - expert, axis=-1).mean())
    accepted = base_ale - candidate_ale >= float(args.v2_shape_prior_min_gain_mm)
    if not accepted:
        prior.reject()
    report = {
        "enabled": True,
        "accepted": bool(accepted),
        "validation_base_ale": base_ale,
        "validation_candidate_ale": candidate_ale,
        "selected": prior.config,
        "fit_sample_count": len(train_ids),
    }
    prior.save(fold_dir / "shape_prior.json")
    return prior, report


def refinement_extras(applied):
    return {
        name: applied[name]
        for name in (
            "gate_alpha",
            "effective_alpha",
            "confidence",
            "mc_spread",
            "ensemble_spread",
        )
    }


def check_locked_validation(path, sample_ids, prediction, tolerance_mm=0.01):
    """The final-test run must reproduce the validation predictions it locked."""
    if not Path(path).exists():
        raise RuntimeError(f"Locked validation predictions are missing: {path}")
    with np.load(path, allow_pickle=False) as record:
        locked_ids = record["sample_ids"].tolist()
        locked = record["prediction"]
    if locked_ids != list(sample_ids):
        raise RuntimeError("Validation participants differ from the locked run")
    drift = float(np.linalg.norm(np.asarray(prediction) - locked, axis=-1).max())
    if drift > tolerance_mm:
        raise RuntimeError(
            f"Validation predictions drifted {drift:.4f} mm from the locked run; "
            "the pipeline is not the one that was validated"
        )
    return drift


def policy_digest(policy):
    payload = {key: policy[key] for key in ("iterations", "alpha_caps", "accepted")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def run_fold(samples, splits, args, fold_index, fold_dir, preprocessing_dir, device):
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest_check = (
        {"checked": False, "reason": "--skip-manifest-check"}
        if args.skip_manifest_check
        else verify_against_shared_manifest(fold_index, splits)
    )
    datasets, _ = prepare_fold(
        samples,
        splits,
        args,
        fold_dir,
        device,
        enforce_oracle_gate=False,
        preprocessing_dir=preprocessing_dir,
    )
    transforms = datasets["val"].transforms
    config = all23_config_from_args(args)
    stage_a_dir = Path(args.stage_a_run) / f"fold_{fold_index}"
    refine_dir = fold_dir / "all23_mvsc"
    scratch = Path(args.scratch_dir) / f"fold_{fold_index}" if args.scratch_dir else None
    policy_path = refine_dir / "all23_policy.json"
    prior_state_path = fold_dir / "v2_lock.json"

    # Outer validation: Stage A, shape prior, refiner, policy.
    val_dataset = datasets["val"]
    val_ids = [sample.sample_id for sample in val_dataset.samples]
    val_stage_a, archived_expert = read_stage_a(
        stage_a_dir / "predictions_val.csv", val_ids, args.stage_a_column, read_expert=True
    )
    val_shapes, val_shape_ids, _, _ = dataset_shapes(val_dataset)
    val_expert_by_id = dict(zip(val_shape_ids, val_shapes))
    frame_drift = max(
        float(np.abs(archived_expert[sample_id] - val_expert_by_id[sample_id]).max())
        for sample_id in val_ids
    )
    if frame_drift > 1e-2:
        raise RuntimeError(
            f"Archived Stage A frame differs from the preprocessing frame ({frame_drift:.4f} mm)"
        )
    val_base_a = np.stack([val_stage_a[sample_id] for sample_id in val_ids])
    val_expert = np.stack([val_expert_by_id[sample_id] for sample_id in val_ids])
    prior, prior_report = fit_v2_shape_prior(
        args, datasets["train"], val_base_a, val_expert, val_ids, fold_dir
    )
    val_base = prior.transform(val_base_a) if prior else val_base_a.copy()

    started = time.time()
    model_dir = (
        Path(args.refiner_source_run) / f"fold_{fold_index}" / "all23_mvsc"
        if args.refiner_source_run
        else None
    )
    incoming_prior_config = (
        {key: prior.config[key] for key in ("components", "l2", "alpha_core", "alpha_hard")}
        if prior is not None and args.gate_evidence_prior and prior_report.get("accepted")
        else None
    )
    refinement = fit_or_load_all23_refinement(
        datasets["train"],
        refine_dir,
        config,
        device,
        scratch_dir=scratch,
        model_dir=model_dir,
        incoming_prior_config=incoming_prior_config,
    )
    inference = {
        "stage_a_column": args.stage_a_column,
        "ensemble_members": refinement.reports["ensemble_members"],
        "mc_samples": config.mc_samples,
        "model_dir": refinement.reports["model_dir"],
        "gate_evidence_prior": incoming_prior_config,
    }
    val_result = refinement.predict(
        val_dataset,
        {sample_id: val_base[index] for index, sample_id in enumerate(val_ids)},
        scratch_dir=scratch,
        cache_dir=roi_cache_dir(scratch, refine_dir / "roi_cache"),
        label="All23-MVSC validation",
    )
    if args.final_test:
        if not policy_path.exists() or not prior_state_path.exists():
            raise RuntimeError(
                f"--final-test needs the locked validation policy: {policy_path}"
            )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        lock = json.loads(prior_state_path.read_text(encoding="utf-8"))
        if lock["policy_digest"] != policy_digest(policy):
            raise RuntimeError("Locked policy file was modified after validation")
        if lock["shape_prior"].get("selected") != prior_report.get("selected"):
            raise RuntimeError("Shape-prior selection differs from the locked validation run")
        if lock.get("inference") != json.loads(json.dumps(inference)):
            raise RuntimeError("Inference settings differ from the locked validation run")
    else:
        policy = calibrate_policy(val_result, val_expert, config)
        write_policy(refine_dir, policy)
    applied = apply_policy(val_result, policy)
    val_columns = {
        "stage_a": val_base_a,
        "prior": val_base,
        **{
            f"iter{key}": value
            for key, value in applied["iteration_proposals"].items()
        },
        "prediction": applied["prediction"],
    }
    val_extras = refinement_extras(applied)
    val_metrics = save_split(
        fold_dir, "val", val_ids, sample_meta(val_dataset), val_expert,
        val_columns, val_extras, args.seed, args.bootstrap_iters,
    )
    lock = {
        "version": VERSION,
        "policy_digest": policy_digest(policy),
        "shape_prior": prior_report,
        "policy": {key: policy[key] for key in ("accepted", "iterations", "alpha_caps")},
        "manifest_check": manifest_check,
        "frame_drift_mm": frame_drift,
        "inference": inference,
        "test_labels_consumed": False,
    }
    locked_predictions_path = fold_dir / "locked_validation_predictions.npz"
    if not args.final_test:
        prior_state_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        np.savez_compressed(
            locked_predictions_path,
            sample_ids=np.asarray(val_ids),
            prediction=applied["prediction"],
        )
    else:
        check_locked_validation(locked_predictions_path, val_ids, applied["prediction"])
    print(
        f"Fold {fold_index} validation: stage A={val_metrics['stages']['stage_a']['overall']['ale']:.4f} "
        f"prior={val_metrics['stages']['prior']['overall']['ale']:.4f} "
        f"final={val_metrics['overall']['ale']:.4f} (core20={val_metrics['core20']['ale']:.4f}, "
        f"hard3={val_metrics['hard3']['ale']:.4f}); policy={lock['policy']}",
        flush=True,
    )
    summary = {
        "fold": fold_index,
        "validation": val_metrics,
        "lock": lock,
        "refinement_reports": refinement.reports.get("gates"),
        "training_seconds": time.time() - started,
    }
    if not args.final_test:
        (fold_dir / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return summary

    # Test: label-free ROIs, predictions written before experts are read.
    test_dataset = datasets["test"]
    test_dataset.include_expert = False
    test_ids = [sample.sample_id for sample in test_dataset.samples]
    test_stage_a = read_stage_a(
        stage_a_dir / "predictions_test.csv", test_ids, args.stage_a_column
    )
    test_base_a = np.stack([test_stage_a[sample_id] for sample_id in test_ids])
    test_base = prior.transform(test_base_a) if prior else test_base_a.copy()
    test_result = refinement.predict(
        test_dataset,
        {sample_id: test_base[index] for index, sample_id in enumerate(test_ids)},
        scratch_dir=scratch,
        cache_dir=roi_cache_dir(scratch, refine_dir / "roi_cache_test"),
        label="All23-MVSC test",
    )
    test_applied = apply_policy(test_result, policy)
    np.savez_compressed(
        fold_dir / "test_predictions_before_scoring.npz",
        sample_ids=np.asarray(test_ids),
        prediction=test_applied["prediction"],
    )
    test_expert_by_id = test_experts_after_prediction(test_dataset, transforms)
    test_expert = np.stack([test_expert_by_id[sample_id] for sample_id in test_ids])
    test_columns = {
        "stage_a": test_base_a,
        "prior": test_base,
        **{
            f"iter{key}": value
            for key, value in test_applied["iteration_proposals"].items()
        },
        "prediction": test_applied["prediction"],
    }
    test_metrics = save_split(
        fold_dir, "test", test_ids, sample_meta(test_dataset), test_expert, test_columns,
        refinement_extras(test_applied),
        args.seed, args.bootstrap_iters,
    )
    summary["test"] = test_metrics
    summary["test_protocol"] = {
        "roi_built_without_labels": True,
        "predictions_saved_before_expert_read": True,
        "policy_locked_before_test": True,
    }
    (fold_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"Fold {fold_index} test: final={test_metrics['overall']['ale']:.4f} "
        f"(core20={test_metrics['core20']['ale']:.4f}, hard3={test_metrics['hard3']['ale']:.4f})",
        flush=True,
    )
    return summary


def pooled_csv_errors(paths_by_fold, column_prefix=None):
    """Pool per-fold prediction errors keyed by (fold, sample_id).

    Validation partitions are drawn independently inside each outer-train set,
    so one participant can appear in several folds; each fold-prediction is a
    separate row and the participant id is returned for cluster resampling.
    """
    errors = {}
    for fold, path in paths_by_fold.items():
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if column_prefix is None:
                    value = float(row["error"])
                else:
                    prediction = np.asarray([float(row[f"{column_prefix}_{axis}"]) for axis in "xyz"])
                    expert = np.asarray([float(row[f"expert_{axis}"]) for axis in "xyz"])
                    value = float(np.linalg.norm(prediction - expert))
                errors.setdefault((int(fold), row["sample_id"]), np.full(NUM_LANDMARKS, np.nan))[
                    int(row["landmark"])
                ] = value
    keys = sorted(errors)
    matrix = np.stack([errors[key] for key in keys])
    if not np.isfinite(matrix).all():
        raise RuntimeError("Pooled validation errors are incomplete")
    return keys, matrix, [sample_id for _, sample_id in keys]


def cluster_bootstrap_delta(base_errors, final_errors, clusters, iterations=2000, seed=42):
    """Bootstrap of mean(final) - mean(base) resampling whole participants.

    All rows (fold-predictions) of a resampled participant enter together, so
    participants that were validated in several folds are not treated as
    independent observations.
    """
    base = np.asarray(base_errors, dtype=np.float64).reshape(len(clusters), -1)
    final = np.asarray(final_errors, dtype=np.float64).reshape(len(clusters), -1)
    names = sorted(set(clusters))
    index = {name: position for position, name in enumerate(names)}
    labels = np.asarray([index[name] for name in clusters])
    width = base.shape[1]
    base_sum = np.bincount(labels, weights=base.sum(axis=1), minlength=len(names))
    final_sum = np.bincount(labels, weights=final.sum(axis=1), minlength=len(names))
    counts = np.bincount(labels, minlength=len(names)) * width
    rng = np.random.default_rng(seed)
    values = np.empty(int(iterations), dtype=np.float64)
    for iteration in range(int(iterations)):
        drawn = np.bincount(rng.integers(0, len(names), size=len(names)), minlength=len(names))
        total = float((drawn * counts).sum())
        values[iteration] = ((drawn * final_sum).sum() - (drawn * base_sum).sum()) / total
    return {
        "delta_ale": float(final.mean() - base.mean()),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "probability_improved": float(np.mean(values < 0)),
        "clusters": len(names),
        "rows": len(clusters),
    }


def validation_gate(output_dir, stage_a_run, folds, seed, iterations=2000):
    """Compare pooled v2 validation errors with the archived vNext validation errors."""
    v2_keys, v2, participants = pooled_csv_errors(
        {fold: Path(output_dir) / f"fold_{fold}" / "predictions_val.csv" for fold in folds}
    )
    vnext_keys, vnext, _ = pooled_csv_errors(
        {fold: Path(stage_a_run) / f"fold_{fold}" / "predictions_val.csv" for fold in folds}
    )
    if v2_keys != vnext_keys:
        raise RuntimeError("v2 and vNext validation fold-predictions differ")
    delta = {
        "all23": cluster_bootstrap_delta(vnext, v2, participants, iterations, seed),
        "core20": cluster_bootstrap_delta(
            vnext[:, list(CORE20)], v2[:, list(CORE20)], participants, iterations, seed
        ),
        "hard3": cluster_bootstrap_delta(
            vnext[:, list(HARD3)], v2[:, list(HARD3)], participants, iterations, seed
        ),
    }
    checks = {
        "all23_not_worse": float(v2.mean()) <= float(vnext.mean()),
        "hard3_regression_at_most_0p10": float(v2[:, list(HARD3)].mean())
        - float(vnext[:, list(HARD3)].mean())
        <= 0.10,
        "core20_regression_at_most_0p05": float(v2[:, list(CORE20)].mean())
        - float(vnext[:, list(CORE20)].mean())
        <= 0.05,
    }
    gate = {
        "version": VERSION,
        "folds": list(folds),
        "predictions": len(v2_keys),
        "unique_participants": len(set(participants)),
        "v2_validation": {
            "all23": float(v2.mean()),
            "core20": float(v2[:, list(CORE20)].mean()),
            "hard3": float(v2[:, list(HARD3)].mean()),
        },
        "vnext_validation": {
            "all23": float(vnext.mean()),
            "core20": float(vnext[:, list(CORE20)].mean()),
            "hard3": float(vnext[:, list(HARD3)].mean()),
        },
        "bootstrap_v2_minus_vnext": delta,
        "checks": checks,
        "run_final_test": bool(all(checks.values())),
        "test_labels_consumed": False,
    }
    (Path(output_dir) / "validation_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    return gate


def aggregate_test(output_dir, summaries):
    rows = []
    for summary in summaries:
        test = summary["test"]
        rows.append(
            {
                "fold": summary["fold"],
                "overall_ale": test["overall"]["ale"],
                "core20_ale": test["core20"]["ale"],
                "hard3_ale": test["hard3"]["ale"],
                "stage_a_ale": test["stages"]["stage_a"]["overall"]["ale"],
                "prior_ale": test["stages"]["prior"]["overall"]["ale"],
                "sdr_at_2mm": test["overall"]["sdr_at_2mm"],
            }
        )
    write_csv(Path(output_dir) / "summary_fold_metrics.csv", rows)
    values = np.asarray([row["overall_ale"] for row in rows])
    payload = {
        "model": VERSION,
        "fold_count": len(rows),
        "fold_mean_ale": float(values.mean()),
        "fold_std_ale": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "folds": rows,
    }
    (Path(output_dir) / "summary_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main():
    args = configure_vnext(build_parser().parse_args())
    set_seed(args.seed)
    device = resolve_device(args.device)
    resolve_precision(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.final_test and (args.validation_only or args.gate_only):
        raise ValueError("--final-test cannot be combined with --validation-only/--gate-only")
    if args.gate_only:
        folds = (
            [int(value) for value in str(args.fold_indices).split(",") if value.strip()]
            if args.fold_indices
            else list(range(1, args.folds + 1))
        )
        gate = validation_gate(
            output_dir, args.stage_a_run, folds, args.seed, args.bootstrap_iters
        )
        shown = (
            "predictions",
            "unique_participants",
            "v2_validation",
            "vnext_validation",
            "checks",
            "run_final_test",
        )
        print(json.dumps({key: gate[key] for key in shown}, indent=2), flush=True)
        return
    samples = discover_samples(args.data_root)
    preprocessing_root = (
        validate_preprocessing_root(args.preprocessing_root, args)
        if args.preprocessing_root
        else None
    )
    if args.protocol != "cv":
        raise ValueError("AGH-Former v2 phase 1 supports only --protocol cv")
    folds = list(enumerate(make_cv_splits(samples, args.folds, args.val_fraction, args.seed), start=1))
    if args.fold_indices:
        requested = {int(value) for value in str(args.fold_indices).split(",") if value.strip()}
        folds = [item for item in folds if item[0] in requested]
    if args.final_test:
        gate_path = output_dir / "validation_gate.json"
        if not gate_path.exists():
            raise RuntimeError(f"Validation gate is missing: {gate_path}")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("run_final_test", False):
            raise RuntimeError(f"Validation gate failed; final test is blocked: {gate['checks']}")
        if sorted(gate["folds"]) != sorted(index for index, _ in folds):
            raise RuntimeError("Final test folds differ from the validated folds")
    (output_dir / ("experiment_config_test.json" if args.final_test else "experiment_config.json")).write_text(
        json.dumps({**vars(args), "version": VERSION}, indent=2, default=str), encoding="utf-8"
    )
    summaries = []
    for fold_index, splits in folds:
        print(
            f"\nAGH-Former v2 fold {fold_index}: "
            f"train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}",
            flush=True,
        )
        fold_dir = fold_output_dir(output_dir, args, 1, fold_index)
        preprocessing_dir = (
            fold_output_dir(preprocessing_root, args, 1, fold_index)
            if preprocessing_root
            else None
        )
        summaries.append(
            run_fold(samples, splits, args, fold_index, fold_dir, preprocessing_dir, device)
        )
    if args.final_test:
        summary = aggregate_test(output_dir, summaries)
        print(
            f"\nAGH-Former v2 test fold ALE: {summary['fold_mean_ale']:.4f} +/- {summary['fold_std_ale']:.4f}",
            flush=True,
        )
    elif len(folds) == args.folds:
        gate = validation_gate(
            output_dir, args.stage_a_run, [index for index, _ in folds], args.seed, args.bootstrap_iters
        )
        print(f"\nValidation gate: {json.dumps(gate['checks'])} -> run_final_test={gate['run_final_test']}", flush=True)


if __name__ == "__main__":
    main()
