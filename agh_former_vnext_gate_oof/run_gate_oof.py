#!/usr/bin/env python3
"""Nested Stage 2 OOF evidence for a model-independent AGH-vNext gate."""

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agh_former_vnext_orthodontic_comparison.run_aghformer_vnext import (
    configure_vnext, model_from_args, vnext_signature,
)
from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, NUM_LANDMARKS
from all23_rgb_geodesic_cascade.data import assert_disjoint_splits, discover_samples
from all23_rgb_geodesic_cascade.losses import LossWeights
from all23_rgb_geodesic_cascade.run_all23_rgb_geodesic import (
    make_loader, prepare_fold, resolve_device, set_seed,
)
from all23_rgb_geodesic_cascade.train import fit_model
from agh_former_vnext_gate_oof.analysis import (
    audit_archived, paired_comparison, write_json,
)
from agh_former_vnext_gate_oof.evidence import (
    attach_aligned_expert, collect_evidence, inference_dataset,
    load_outer_preprocessing,
)
from agh_former_vnext_gate_oof.gate import (
    Evidence, fit_gate, gate_predictions, load_gate,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def dataset_source_fingerprint(samples):
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda row: row.sample_id):
        for path in (sample.mesh_path, sample.landmark_path):
            path = Path(path)
            stat = path.stat()
            digest.update(
                f"{sample.sample_id}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
            )
    return digest.hexdigest()


def inner_partitions(samples, train_ids, folds, val_fraction, seed):
    by_id = {sample.sample_id: sample for sample in samples}
    ids = np.asarray(train_ids)
    strata = np.asarray([f"{by_id[sample_id].class_name}_{by_id[sample_id].gender}" for sample_id in ids])
    partitioner = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    partitions = []
    for index, (fit_idx, holdout_idx) in enumerate(partitioner.split(ids, strata), 1):
        selector = StratifiedShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=seed + 1000 + index,
        )
        train_local, val_local = next(selector.split(fit_idx, strata[fit_idx]))
        split = {
            "train": ids[fit_idx[train_local]].tolist(),
            "val": ids[fit_idx[val_local]].tolist(),
            "test": ids[holdout_idx].tolist(),
        }
        assert_disjoint_splits(split)
        partitions.append(split)
    holdouts = [sample_id for split in partitions for sample_id in split["test"]]
    if len(holdouts) != len(train_ids) or set(holdouts) != set(train_ids):
        raise ValueError("Nested Stage 2 holdouts do not cover outer-train exactly once")
    return partitions


def source_fold(source_root, fold, expected_ids):
    directory = Path(source_root) / f"fold_{fold}"
    report = json.loads((directory / "split_and_leakage_report.json").read_text(encoding="utf-8"))
    splits = report["splits"]
    assert_disjoint_splits(splits)
    if set(sum(splits.values(), [])) != expected_ids:
        raise ValueError(f"Source fold {fold} does not cover the current dataset")
    checkpoint_path = directory / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not checkpoint.get("stage2_signature"):
        raise ValueError(f"Source fold {fold} checkpoint has no Stage 2 signature")
    gate_path = directory / "best_gate.pth"
    gate_result = json.loads((directory / "gate_training.json").read_text(encoding="utf-8"))
    if gate_result.get("accepted", False) and not gate_path.exists():
        raise FileNotFoundError(gate_path)
    return directory, splits, checkpoint, checkpoint_path, gate_path, gate_result


def args_from_checkpoint(checkpoint, *, seed, inner_epochs=0, stage1_fixed_epochs=0):
    args = argparse.Namespace(**copy.deepcopy(checkpoint["args"]))
    configure_vnext(args)
    args.seed = seed
    args.preprocessing_root = None
    args.resume_stage2 = True
    args.force_stage2_retrain = False
    args.separate_gate_training = True
    if inner_epochs:
        args.epochs = inner_epochs
        args.min_epochs = min(args.min_epochs, inner_epochs)
        args.patience = min(args.patience, inner_epochs)
        args.scheduler_start_epoch = min(args.scheduler_start_epoch, inner_epochs)
    if stage1_fixed_epochs:
        args.stage1_oof_fixed_epochs = stage1_fixed_epochs
        args.stage1_epochs = max(stage1_fixed_epochs, 1)
        args.stage1_min_epochs = min(args.stage1_min_epochs, args.stage1_epochs)
    return args


def loss_weights(args):
    return LossWeights(
        heatmap=args.heatmap_weight,
        region=args.region_weight,
        coordinate=args.coordinate_weight,
        coarse=args.coarse_weight,
        anatomy=args.anatomy_weight,
        symmetry=args.symmetry_weight,
        uncertainty=args.uncertainty_weight,
        clinical=args.clinical_weight,
        gate=args.gate_weight,
        hard_landmark=args.hard_landmark_weight,
        hard_rank=args.hard_rank_weight,
    )


def collect_outer_evidence(
    samples, splits, checkpoint, gate_path, gate_result,
    preprocessing_dir, fold_dir, device, split_name, *, attach_labels,
):
    args = args_from_checkpoint(checkpoint, seed=checkpoint["args"]["seed"])
    transforms, records, normalizer, coarse, scales = load_outer_preprocessing(
        samples, splits, preprocessing_dir,
        roi_expansion=args.registration_roi_expansion,
    )
    if (
        not np.allclose(normalizer["mean"], checkpoint["normalizer"]["mean"], atol=1e-6)
        or not np.allclose(normalizer["std"], checkpoint["normalizer"]["std"], atol=1e-6)
    ):
        raise ValueError("Archived checkpoint and outer preprocessing normalizers differ")
    model = model_from_args(args, normalizer, device)
    model.load_state_dict(checkpoint["model_state"])
    if gate_result.get("accepted", False):
        gate_checkpoint = torch.load(gate_path, map_location=device, weights_only=False)
        if gate_checkpoint.get("stage2_signature") != checkpoint.get("stage2_signature"):
            raise ValueError("Archived token gate was trained for another Stage 2 model")
        model.load_refinement_gate_state_dict(gate_checkpoint["gate_state"])
    dataset = inference_dataset(
        samples, splits[split_name], records, transforms, coarse, normalizer,
        scales, args, fold_dir / "outer_roi",
    )
    loader = make_loader(dataset, args.eval_batch_size, False, args)
    evidence = collect_evidence(model, loader, device, args, normalizer, use_tta=args.tta_validation)
    if not gate_result.get("accepted", False):
        evidence.token_alpha[:] = 1.0
    if attach_labels:
        attach_aligned_expert(evidence, samples, transforms)
    return evidence


def fit_inner_stage2(samples, split, args, fold_dir, device, data_fingerprint, seed):
    args = copy.deepcopy(args)
    args.seed = seed
    args.output_dir = str(fold_dir)
    args.stage2_signature = signature({
        "stage2": vnext_signature(args, split), "mesh_fingerprint": data_fingerprint,
    })
    set_seed(seed)
    datasets, normalizer = prepare_fold(
        samples, split, args, fold_dir, device,
        enforce_oracle_gate=False,
    )
    # The inner holdout is never used as a training or checkpoint-validation target.
    datasets["test"].include_expert = False
    loaders = {
        name: make_loader(dataset, args.eval_batch_size if name == "test" else args.batch_size,
                          name == "train", args)
        for name, dataset in datasets.items()
    }
    model = model_from_args(args, normalizer, device)
    model.set_refinement_gate_trainable(False)
    fit_model(model, loaders["train"], loaders["val"], device,
              args, loss_weights(args), normalizer, fold_dir)
    evidence = collect_evidence(
        model, loaders["test"], device, args, normalizer,
        use_tta=args.tta_validation,
    )
    evidence.save(fold_dir / "holdout_predictions_unlabeled.npz")
    attach_aligned_expert(evidence, samples, datasets["test"].transforms)
    return evidence


def build_oof_evidence(samples, splits, checkpoint, work_dir, device, options, data_fingerprint):
    partitions = inner_partitions(
        samples, splits["train"], options.inner_folds,
        options.inner_val_fraction, options.seed,
    )
    pieces = []
    for index, split in enumerate(partitions, 1):
        inner_dir = Path(work_dir) / f"inner_fold_{index}"
        inner_dir.mkdir(parents=True, exist_ok=True)
        inner_seed = options.seed + 100 * options.outer_fold + index
        args = args_from_checkpoint(
            checkpoint, seed=inner_seed,
            inner_epochs=options.inner_stage2_epochs,
            stage1_fixed_epochs=options.inner_stage1_fixed_epochs,
        )
        expected = signature({
            "version": 1,
            "split": split,
            "data_fingerprint": data_fingerprint,
            "args": vnext_signature(args, split),
            "source_checkpoint": options.source_checkpoint_sha,
            "seed": inner_seed,
        })
        manifest_path = inner_dir / "evidence_manifest.json"
        evidence_path = inner_dir / "holdout_evidence.npz"
        if evidence_path.exists() and not manifest_path.exists():
            checkpoint_path = inner_dir / "best_model.pth"
            if not checkpoint_path.exists():
                raise RuntimeError(f"Unattributed inner evidence without a checkpoint: {inner_dir}")
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            expected_stage2 = signature({
                "stage2": vnext_signature(args, split),
                "mesh_fingerprint": data_fingerprint,
            })
            evidence = Evidence.load(evidence_path).validate(require_expert=True)
            if saved.get("stage2_signature") != expected_stage2 or set(evidence.sample_ids) != set(split["test"]):
                raise ValueError(f"Cannot recover stale inner evidence: {inner_dir}")
            write_json(manifest_path, {
                "signature": expected, "fit_ids": split["train"],
                "checkpoint_selection_ids": split["val"],
                "holdout_ids": split["test"],
                "holdout_labels_read_after_prediction": True,
                "stage1_fit_scope": "inner-train OOF; final Stage 1 uses inner train/validation, never holdout",
                "evidence_sha256": file_sha256(evidence_path),
            })
        if manifest_path.exists() and evidence_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") != expected or manifest.get("evidence_sha256") != file_sha256(evidence_path):
                raise ValueError(f"Stale inner-fold evidence; choose a new output directory: {inner_dir}")
            evidence = Evidence.load(evidence_path).validate(require_expert=True)
            if set(evidence.sample_ids) != set(split["test"]):
                raise ValueError("Cached inner-fold evidence contains different participants")
            print(f"Inner Stage 2 {index}/{len(partitions)} cached", flush=True)
        else:
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("signature") != expected:
                    raise ValueError(f"Stale inner-fold cache: {inner_dir}")
            print(
                f"Inner Stage 2 {index}/{len(partitions)} "
                f"train/val/holdout={len(split['train'])}/{len(split['val'])}/{len(split['test'])}",
                flush=True,
            )
            write_json(inner_dir / "inner_split.json", split)
            evidence = fit_inner_stage2(
                samples, split, args, inner_dir, device, data_fingerprint, inner_seed,
            )
            evidence.save(evidence_path)
            write_json(manifest_path, {
                "signature": expected, "fit_ids": split["train"],
                "checkpoint_selection_ids": split["val"],
                "holdout_ids": split["test"],
                "holdout_labels_read_after_prediction": True,
                "stage1_fit_scope": "inner-train OOF; final Stage 1 uses inner train/validation, never holdout",
                "evidence_sha256": file_sha256(evidence_path),
            })
        pieces.append(evidence)
    by_id = {sample_id: (piece, index) for piece in pieces
             for index, sample_id in enumerate(piece.sample_ids)}
    if set(by_id) != set(splits["train"]) or sum(map(len, (piece.sample_ids for piece in pieces))) != len(by_id):
        raise ValueError("OOF evidence does not cover each outer-training participant exactly once")
    ordering = splits["train"]
    return Evidence(
        sample_ids=ordering,
        **{
            name: np.stack([getattr(by_id[sample_id][0], name)[by_id[sample_id][1]]
                            for sample_id in ordering])
            for name in ("center", "local", "diagnostics", "token_alpha", "expert")
        },
    ).validate(require_expert=True)


def concatenate_evidence(pieces):
    return Evidence(
        sample_ids=[sample_id for piece in pieces for sample_id in piece.sample_ids],
        **{
            name: np.concatenate([getattr(piece, name) for piece in pieces], axis=0)
            for name in ("center", "local", "diagnostics", "token_alpha", "expert")
        },
    ).validate(require_expert=True)


def _prediction(evidence, alpha):
    return evidence.center + alpha[..., None] * (evidence.local - evidence.center)


def compare_policies(evidence, in_sample_gate, oof_gate, device, seed):
    models = {
        "alpha0_fusion": evidence.center,
        "alpha1_local": evidence.local,
        "original_token_gate": _prediction(evidence, evidence.token_alpha),
    }
    for name, (model, mean, std, selected) in (
        ("insample_diagnostic_gate", in_sample_gate),
        ("oof_diagnostic_gate", oof_gate),
    ):
        raw, alpha = gate_predictions(evidence, model, mean, std, device, accepted=True)
        models[name] = raw
        models[f"{name}_validation_selected"] = (
            raw if selected else evidence.local
        )
    reports = {
        name: paired_comparison(
            evidence.expert, evidence.local, prediction, evidence.sample_ids,
            seed=seed,
        )
        for name, prediction in models.items()
    }
    return models, reports


def _load_or_fit_gate(name, train, validation, fold_dir, device, seed, epochs, patience, key):
    gate_dir = Path(fold_dir) / name
    manifest_path = gate_dir / "manifest.json"
    selection_path = gate_dir / "gate_selection.json"
    checkpoint_path = gate_dir / "best_gate.pth"
    if manifest_path.exists() and selection_path.exists() and checkpoint_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != key:
            raise ValueError(f"Cached gate configuration differs; use a new output directory: {gate_dir}")
        model, mean, std = load_gate(checkpoint_path, device)
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        print(f"{name} cached: best epoch={selected['best_epoch']}", flush=True)
    else:
        if any(path.exists() for path in (manifest_path, selection_path)):
            raise RuntimeError(f"Incomplete gate cache: {gate_dir}")
        model, mean, std, selected = fit_gate(
            train, validation, gate_dir, device,
            seed=seed, epochs=epochs, patience=patience,
        )
        write_json(manifest_path, {"signature": key, "train_ids": train.sample_ids})
    return model, mean, std, bool(selected["accepted_on_validation"])


def pilot_decision(report):
    hard = report["hard3"]
    core = report["core20"]
    checks = {
        "hard3_gain_at_least_0p10_mm": hard["gain_mm"] >= 0.10,
        "core20_regression_at_most_0p05_mm": core["gain_mm"] >= -0.05,
        "hard3_p95_not_worse": hard["candidate"]["p95"] <= hard["incoming"]["p95"],
    }
    return {"run_full_cv": all(checks.values()), "checks": checks}


def final_decision(report):
    hard = report["hard3"]
    core = report["core20"]
    checks = {
        "hard3_gain_at_least_0p20_mm": hard["gain_mm"] >= 0.20,
        "hard3_ci95_lower_above_zero": hard["gain_ci95"][0] > 0.0,
        "hard3_p95_not_worse": hard["candidate"]["p95"] <= hard["incoming"]["p95"],
        "core20_regression_at_most_0p05_mm": core["gain_mm"] >= -0.05,
    }
    return {
        "gate_as_main_mechanism_supported": all(checks.values()),
        "checks": checks,
        "interpretation": (
            "Internal-development evidence only; not independent external validation."
        ),
    }


def _write_prediction_csv(path, evidence, models):
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "landmark"]
        for name in ("expert", *models):
            fields.extend(f"{name}_{axis}" for axis in "xyz")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_index, sample_id in enumerate(evidence.sample_ids):
            for landmark in range(NUM_LANDMARKS):
                row = {"sample_id": sample_id, "landmark": landmark}
                for name, coordinates in (("expert", evidence.expert), *models.items()):
                    for axis, value in zip("xyz", coordinates[sample_index, landmark]):
                        row[f"{name}_{axis}"] = float(value)
                writer.writerow(row)


def _archived_rows(path):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["sample_id"], int(row["landmark"]))
            if key in rows:
                raise ValueError(f"Duplicate archived validation row: {key}")
            rows[key] = row
    return rows


def _archived_coordinates(rows, sample_ids, prefix):
    return np.asarray([
        [[float(rows[(sample_id, landmark)][f"{prefix}_{axis}"]) for axis in "xyz"]
         for landmark in range(NUM_LANDMARKS)]
        for sample_id in sample_ids
    ], dtype=np.float32)


def assert_archived_stage1_parity(coarse, path, sample_ids, tolerance_mm=1e-4):
    rows = _archived_rows(path)
    expected = {(sample_id, landmark) for sample_id in sample_ids
                for landmark in range(NUM_LANDMARKS)}
    if set(rows) != expected:
        raise ValueError("Archived Stage 1 validation rows differ from this fold")
    archived = _archived_coordinates(rows, sample_ids, "stage1_coarse")
    current = np.stack([coarse[sample_id] for sample_id in sample_ids])
    maximum = float(np.linalg.norm(archived - current, axis=-1).max())
    if maximum > tolerance_mm:
        raise ValueError(
            f"Archived Stage 1 centers differ by {maximum:.4f} mm; "
            "the coarse cache cannot be reused for this experiment"
        )
    return maximum


def assert_archived_validation_parity(evidence, path, tolerance_mm=0.05, report_path=None):
    """Audit replay drift without mistaking rare mesh-vertex flips for a new model."""
    evidence.validate(require_expert=True)
    rows = _archived_rows(path)
    expected = {
        (sample_id, landmark)
        for sample_id in evidence.sample_ids for landmark in range(NUM_LANDMARKS)
    }
    if set(rows) != expected:
        raise ValueError("Archived validation sample/landmark rows differ from this fold")
    archived_expert = _archived_coordinates(rows, evidence.sample_ids, "expert")
    expert_max = float(np.linalg.norm(archived_expert - evidence.expert, axis=-1).max())
    if expert_max > 1e-3:
        raise ValueError(
            f"Archived validation expert coordinates differ by {expert_max:.4f} mm; "
            "landmarks or alignment changed"
        )

    comparisons = {}
    for name, prefix, predicted in (
        ("fusion", "coarse", evidence.center),
        ("local", "refined", evidence.local),
        ("token_gate", "neural_prediction", _prediction(evidence, evidence.token_alpha)),
    ):
        source = _archived_coordinates(rows, evidence.sample_ids, prefix)
        drift = np.linalg.norm(source - predicted, axis=-1)
        worst_sample, worst_landmark = np.unravel_index(int(np.argmax(drift)), drift.shape)
        comparisons[name] = {
            "mean_mm": float(drift.mean()),
            "p95_mm": float(np.percentile(drift, 95)),
            "max_mm": float(drift.max()),
            "count_above_0p05mm": int(np.count_nonzero(drift > tolerance_mm)),
            "fraction_above_0p05mm": float(np.mean(drift > tolerance_mm)),
            "worst_sample_id": evidence.sample_ids[worst_sample],
            "worst_landmark": int(worst_landmark),
        }

    source_prediction = _archived_coordinates(rows, evidence.sample_ids, "neural_prediction")
    current_prediction = _prediction(evidence, evidence.token_alpha)
    source_errors = np.linalg.norm(source_prediction - evidence.expert, axis=-1)
    current_errors = np.linalg.norm(current_prediction - evidence.expert, axis=-1)
    clinical = {}
    for name, indices in (("overall", range(NUM_LANDMARKS)),
                          ("core20", CORE20), ("hard3", HARD3)):
        clinical[name] = {
            "archived_ale_mm": float(source_errors[:, indices].mean()),
            "replayed_ale_mm": float(current_errors[:, indices].mean()),
            "delta_mm": float(current_errors[:, indices].mean() - source_errors[:, indices].mean()),
            "archived_p95_mm": float(np.percentile(source_errors[:, indices], 95)),
            "replayed_p95_mm": float(np.percentile(current_errors[:, indices], 95)),
        }
        clinical[name]["p95_delta_mm"] = (
            clinical[name]["replayed_p95_mm"] - clinical[name]["archived_p95_mm"]
        )

    # Small logit differences can move discrete mesh decoding to another vertex.
    limits = {
        "fusion_mean_mm": 0.02, "fusion_p95_mm": 0.05, "fusion_max_mm": 0.35,
        "local_or_gate_mean_mm": 0.03, "local_or_gate_p95_mm": 0.05,
        "local_or_gate_max_mm": 3.0, "local_or_gate_fraction_above_0p05mm": 0.02,
        "overall_ale_delta_mm": 0.02, "core20_ale_delta_mm": 0.02,
        "hard3_ale_delta_mm": 0.05,
    }
    issues = []
    fusion = comparisons["fusion"]
    for statistic, limit in (("mean_mm", limits["fusion_mean_mm"]),
                             ("p95_mm", limits["fusion_p95_mm"]),
                             ("max_mm", limits["fusion_max_mm"])):
        if fusion[statistic] > limit:
            issues.append(f"fusion {statistic}={fusion[statistic]:.4f}>{limit:.4f}")
    for name in ("local", "token_gate"):
        row = comparisons[name]
        for statistic, limit in (
            ("mean_mm", limits["local_or_gate_mean_mm"]),
            ("p95_mm", limits["local_or_gate_p95_mm"]),
            ("max_mm", limits["local_or_gate_max_mm"]),
            ("fraction_above_0p05mm", limits["local_or_gate_fraction_above_0p05mm"]),
        ):
            if row[statistic] > limit:
                issues.append(f"{name} {statistic}={row[statistic]:.4f}>{limit:.4f}")
    for name in ("overall", "core20", "hard3"):
        limit = limits[f"{name}_ale_delta_mm"]
        if abs(clinical[name]["delta_mm"]) > limit:
            issues.append(f"{name} ALE delta={clinical[name]['delta_mm']:.4f}, limit={limit:.4f}")
    exact = all(row["max_mm"] <= tolerance_mm for row in comparisons.values())
    report = {
        "status": "rejected" if issues else ("exact" if exact else "bounded_numeric_drift"),
        "expert_max_difference_mm": expert_max,
        "coordinates": comparisons,
        "clinical": clinical,
        "landmark_ale_delta_mm": (current_errors - source_errors).mean(axis=0).tolist(),
        "guardrails": limits,
        "issues": issues,
        "note": "Bounded drift is not exact coordinate parity; all gate policies use the same replayed evidence.",
    }
    if report_path is not None:
        write_json(report_path, report)
    if issues:
        raise ValueError("Archived Stage 2 replay differs beyond guardrails: " + "; ".join(issues))
    return report


def run_outer_fold(fold, samples, options, device, *, validation_only):
    all_ids = {sample.sample_id for sample in samples}
    source_dir, splits, checkpoint, checkpoint_path, token_gate_path, token_gate_report = source_fold(
        options.source_run_root, fold, all_ids
    )
    options.outer_fold = fold
    options.source_checkpoint_sha = file_sha256(checkpoint_path)
    preprocessing_dir = options.preprocessing_root / f"fold_{fold}"
    fold_dir = options.output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    transforms, records, normalizer, coarse, scales = load_outer_preprocessing(
        samples, splits, preprocessing_dir,
        roi_expansion=checkpoint["args"]["registration_roi_expansion"],
    )
    stage1_parity = assert_archived_stage1_parity(
        coarse, source_dir / "predictions_val.csv", splits["val"]
    )
    del records, normalizer, coarse, scales
    fingerprint = dataset_source_fingerprint(samples)
    run_key = signature({
        "version": 2, "source_checkpoint_sha": options.source_checkpoint_sha,
        "source_gate_sha": file_sha256(token_gate_path) if token_gate_path.exists() else None,
        "archived_validation_sha": file_sha256(source_dir / "predictions_val.csv"),
        "stage1_train_sha": file_sha256(
            preprocessing_dir / "stage1_global_coarse" / "oof_predictions_train.csv"
        ),
        "stage1_validation_sha": file_sha256(
            preprocessing_dir / "stage1_global_coarse" / "predictions_val.csv"
        ),
        "stage1_test_sha": file_sha256(
            preprocessing_dir / "stage1_global_coarse" / "predictions_test.csv"
        ),
        "normalization_sha": file_sha256(preprocessing_dir / "normalization.json"),
        "alignment_sha": file_sha256(preprocessing_dir / "alignment" / "mesh_only_transforms.npz"),
        "data_fingerprint": fingerprint, "splits": splits,
        "inner_folds": options.inner_folds, "inner_val_fraction": options.inner_val_fraction,
        "inner_stage2_epochs": options.inner_stage2_epochs,
        "inner_stage1_fixed_epochs": options.inner_stage1_fixed_epochs,
        "gate_epochs": options.gate_epochs, "gate_patience": options.gate_patience,
        "seed": options.seed,
    })
    identity_path = fold_dir / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8"))["signature"] != run_key:
            raise ValueError(f"Run identity changed: {fold_dir}; use a new output directory")
    else:
        write_json(identity_path, {
            "signature": run_key,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": options.source_checkpoint_sha,
            "source_gate": str(token_gate_path),
            "preprocessing_source": str(preprocessing_dir),
            "stage1_validation_max_difference_mm": stage1_parity,
            "splits": splits,
        })
    def outer_evidence(split_name, labels):
        path = fold_dir / f"{split_name}_evidence.npz"
        manifest_path = fold_dir / f"{split_name}_evidence_manifest.json"
        if path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["signature"] != run_key or manifest["sha256"] != file_sha256(path):
                raise ValueError(f"Stale outer evidence: {path}")
            evidence = Evidence.load(path)
            if set(evidence.sample_ids) != set(splits[split_name]):
                raise ValueError(f"Outer evidence sample mismatch: {path}")
            return evidence.validate(require_expert=labels)
        if path.exists() and not manifest_path.exists():
            evidence = Evidence.load(path).validate(require_expert=labels)
            if set(evidence.sample_ids) != set(splits[split_name]):
                raise ValueError(f"Cannot recover stale outer evidence: {path}")
            write_json(manifest_path, {"signature": run_key, "sha256": file_sha256(path)})
            return evidence
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["signature"] != run_key:
                raise ValueError(f"Stale outer evidence manifest: {manifest_path}")
        evidence = collect_outer_evidence(
            samples, splits, checkpoint, token_gate_path, token_gate_report,
            preprocessing_dir, fold_dir, device, split_name, attach_labels=False,
        )
        if labels:
            # The test split reaches this line only after the validation policy is locked.
            attach_aligned_expert(evidence, samples, transforms)
        evidence.save(path)
        write_json(manifest_path, {"signature": run_key, "sha256": file_sha256(path)})
        return evidence

    print(f"Outer fold {fold}: collecting fixed Stage 2 validation evidence", flush=True)
    validation = outer_evidence("val", labels=True)
    parity = assert_archived_validation_parity(
        validation, source_dir / "predictions_val.csv",
        report_path=fold_dir / "archived_stage2_parity.json",
    )
    print(
        f"Archived Stage 2 replay: {parity['status']}; "
        f"fusion max={parity['coordinates']['fusion']['max_mm']:.4f} mm, "
        f"local max={parity['coordinates']['local']['max_mm']:.4f} mm, "
        f"Hard3 ALE delta={parity['clinical']['hard3']['delta_mm']:+.4f} mm",
        flush=True,
    )
    train_insample = outer_evidence("train", labels=True)
    oof = build_oof_evidence(
        samples, splits, checkpoint, fold_dir / "stage2_oof", device,
        options, fingerprint,
    )
    oof.save(fold_dir / "stage2_oof_train_evidence.npz")
    if oof.sample_ids != splits["train"]:
        raise ValueError("OOF evidence ordering differs from outer train")
    oof_gate = _load_or_fit_gate(
        "oof_diagnostic_gate", oof, validation, fold_dir, device,
        options.seed + fold, options.gate_epochs, options.gate_patience,
        signature({"run": run_key, "train_mode": "oof"}),
    )
    insample_gate = _load_or_fit_gate(
        "insample_diagnostic_gate", train_insample, validation, fold_dir, device,
        options.seed + fold, options.gate_epochs, options.gate_patience,
        signature({"run": run_key, "train_mode": "insample"}),
    )
    val_models, val_reports = compare_policies(
        validation, insample_gate, oof_gate, device, options.seed + fold,
    )
    chosen = val_reports["oof_diagnostic_gate"]
    decision = pilot_decision(chosen)
    decision["oof_gate_accepted_on_validation"] = oof_gate[3]
    decision["validation_hard3_gain_mm"] = chosen["hard3"]["gain_mm"]
    decision["source_is_internal_development_cv"] = True
    write_json(fold_dir / "validation_policy_comparison.json", val_reports)
    _write_prediction_csv(fold_dir / "predictions_val.csv", validation, val_models)
    write_json(fold_dir / "selection_policy.json", decision)
    print(
        f"Fold {fold} validation: OOF gate hard3 gain={chosen['hard3']['gain_mm']:.4f} "
        f"core20 gain={chosen['core20']['gain_mm']:.4f} "
        f"p95 change={chosen['hard3']['candidate']['p95'] - chosen['hard3']['incoming']['p95']:.4f} "
        f"pilot_pass={decision['run_full_cv']}", flush=True,
    )
    if validation_only:
        return validation, val_models, val_reports
    print(f"Fold {fold}: policy locked; evaluating held-out test", flush=True)
    test = outer_evidence("test", labels=True)
    test_models, test_reports = compare_policies(
        test, insample_gate, oof_gate, device, options.seed + 100 + fold,
    )
    write_json(fold_dir / "test_policy_comparison.json", test_reports)
    _write_prediction_csv(fold_dir / "predictions_test.csv", test, test_models)
    return test, test_models, test_reports


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("gate_oof_preflight", "gate_oof_fold1", "gate_oof_cv"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--preprocessing-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true", help="Permit slow nested training without CUDA")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--inner-val-fraction", type=float, default=0.20)
    parser.add_argument("--inner-stage2-epochs", type=int, default=0)
    parser.add_argument("--inner-stage1-fixed-epochs", type=int, default=0)
    parser.add_argument("--gate-epochs", type=int, default=100)
    parser.add_argument("--gate-patience", type=int, default=15)
    return parser


def main():
    options = build_parser().parse_args()
    if options.inner_folds < 2 or not 0.0 < options.inner_val_fraction < 0.5:
        raise ValueError("Invalid inner-fold configuration")
    samples = discover_samples(options.data_root)
    all_ids = {sample.sample_id for sample in samples}
    device = resolve_device(options.device)
    set_seed(options.seed)
    print(f"Device: {device}; paired samples: {len(samples)}", flush=True)
    if options.preset != "gate_oof_preflight" and device.type != "cuda" and not options.allow_cpu:
        raise RuntimeError(
            "Nested Gate-OOF training requires a CUDA GPU by default. "
            "Use Colab A100 or explicitly pass --allow-cpu for a slow local run."
        )
    if options.preset == "gate_oof_preflight":
        for fold in range(1, 6):
            source_dir, splits, checkpoint, _, _, _ = source_fold(options.source_run_root, fold, all_ids)
            _, _, _, coarse, _ = load_outer_preprocessing(
                samples, splits, options.preprocessing_root / f"fold_{fold}",
                roi_expansion=checkpoint["args"]["registration_roi_expansion"],
            )
            stage1_difference = assert_archived_stage1_parity(
                coarse, source_dir / "predictions_val.csv", splits["val"]
            )
            partitions = inner_partitions(
                samples, splits["train"], options.inner_folds,
                options.inner_val_fraction, options.seed,
            )
            print(
                f"Preflight fold {fold}: {len(partitions)} inner fits; "
                f"outer train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}; "
                f"Stage 1 max difference={stage1_difference:.6f} mm",
                flush=True,
            )
        options.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(options.output_dir / "historical_gate_audit.json", audit_archived(options.source_run_root))
        print("Preflight passed; historical Stage 2 audit saved", flush=True)
        return
    if options.preset == "gate_oof_cv":
        pilot_path = options.output_dir / "fold_1" / "selection_policy.json"
        if not pilot_path.exists() or not json.loads(pilot_path.read_text(encoding="utf-8")).get("run_full_cv"):
            raise RuntimeError("Fold 1 validation pilot did not pass; full CV is blocked")
        folds = range(1, 6)
    else:
        folds = (1,)
    results = []
    for fold in folds:
        result = run_outer_fold(
            fold, samples, options, device,
            validation_only=options.preset == "gate_oof_fold1",
        )
        results.append(result)
    if options.preset == "gate_oof_fold1":
        return
    evidences = [result[0] for result in results]
    pooled = concatenate_evidence(evidences)
    selected = np.concatenate([
        result[1]["oof_diagnostic_gate_validation_selected"] for result in results
    ], axis=0)
    overall = paired_comparison(
        pooled.expert, pooled.local, selected, pooled.sample_ids,
        seed=options.seed, iterations=5000,
    )
    decision = final_decision(overall)
    write_json(options.output_dir / "summary_metrics.json", {
        "comparison": overall, "decision": decision,
        "fold_count": len(results),
        "evaluation_type": "internal development cross-validation",
    })
    print(
        f"Five-fold OOF gate: Hard3 gain={overall['hard3']['gain_mm']:.4f} mm; "
        f"main-mechanism claim supported={decision['gate_as_main_mechanism_supported']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
