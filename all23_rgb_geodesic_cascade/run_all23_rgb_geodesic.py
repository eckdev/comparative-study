#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from all23_rgb_geodesic_cascade.alignment import build_label_free_alignment
    from all23_rgb_geodesic_cascade.anatomy import (
        ANATOMICAL_EDGES, CORE20, HARD3, LANDMARK_NAMES, MIDLINE, SYMMETRY_PAIRS,
    )
    from all23_rgb_geodesic_cascade.data import (
        RGBGeodesicDataset, assert_disjoint_splits, collate_graphs, discover_samples,
        fit_feature_normalizer, load_coarse_predictions, load_mesh, load_split_file,
        prepare_mesh_record, read_landmarks, resolve_legacy_matrix, convert_legacy_coordinates,
        build_roi_cache,
    )
    from all23_rgb_geodesic_cascade.losses import LossWeights
    from all23_rgb_geodesic_cascade.metrics import calibrate_confidence, save_evaluation, write_csv
    from all23_rgb_geodesic_cascade.model import All23RGBGeodesicCascade
    from all23_rgb_geodesic_cascade.stage1 import generate_oof_stage1_predictions
    from all23_rgb_geodesic_cascade.train import (
        apply_refinement_calibration, collect_outputs, fit_model,
        fit_refinement_calibration, fit_separate_refinement_gate,
    )
else:
    from .alignment import build_label_free_alignment
    from .anatomy import ANATOMICAL_EDGES, CORE20, HARD3, LANDMARK_NAMES, MIDLINE, SYMMETRY_PAIRS
    from .data import (
        RGBGeodesicDataset, assert_disjoint_splits, collate_graphs, discover_samples,
        fit_feature_normalizer, load_coarse_predictions, load_mesh, load_split_file,
        prepare_mesh_record, read_landmarks, resolve_legacy_matrix, convert_legacy_coordinates,
        build_roi_cache,
    )
    from .losses import LossWeights
    from .metrics import calibrate_confidence, save_evaluation, write_csv
    from .model import All23RGBGeodesicCascade
    from .stage1 import generate_oof_stage1_predictions
    from .train import (
        apply_refinement_calibration, collect_outputs, fit_model,
        fit_refinement_calibration, fit_separate_refinement_gate,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_precision(args, device):
    args.mixed_precision = bool(args.mixed_precision and device.type == "cuda")
    if not args.mixed_precision:
        args.amp_dtype = "float32"
        return
    if args.amp_dtype == "auto":
        supports_bfloat16 = bool(
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        )
        args.amp_dtype = "bfloat16" if supports_bfloat16 else "float16"
    if args.amp_dtype == "bfloat16":
        supports_bfloat16 = bool(
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        )
        if not supports_bfloat16:
            raise RuntimeError(
                "--amp-dtype=bfloat16 was requested, but this CUDA device does not support BF16. "
                "Use --amp-dtype=float16 or disable --mixed-precision."
            )


def make_cv_splits(samples, folds, val_fraction, seed):
    ids = np.asarray([sample.sample_id for sample in samples])
    strata = np.asarray([f"{sample.class_name}_{sample.gender}" for sample in samples])
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = []
    for fold_index, (train_val, test) in enumerate(outer.split(ids, strata)):
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=seed + fold_index
        )
        train_local, val_local = next(inner.split(train_val, strata[train_val]))
        result.append(
            {
                "train": ids[train_val[train_local]].tolist(),
                "val": ids[train_val[val_local]].tolist(),
                "test": ids[test].tolist(),
            }
        )
    return result


def limit_splits(splits, maximum):
    if not maximum:
        return splits
    per_split = max(1, int(maximum) // 3)
    return {name: list(values[:per_split]) for name, values in splits.items()}


def load_oof_manifest(path):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_external_cv_provenance(splits, manifest):
    if manifest is None:
        raise ValueError(
            "External coarse predictions in CV require --external-coarse-oof-manifest. "
            "Use --coarse-source train_template for a fully leakage-free run."
        )
    entries = manifest.get("samples", manifest)
    for sample_id in set(sum((splits[name] for name in splits), [])):
        entry = entries.get(sample_id)
        if entry is None or not entry.get("out_of_fold", False):
            raise ValueError(f"Missing verified OOF provenance for {sample_id}")


def experiment_settings(args):
    level = args.experiment.upper()
    order = {f"E{index}": index for index in range(1, 11)}
    if level == "FULL":
        level = "E9"
    if level not in order:
        raise ValueError("--experiment must be E1..E10 or FULL")
    stage = order[level]
    args.use_rgb = stage >= 2
    args.use_local_refiner = stage >= 4
    args.use_anatomical_attention = stage >= 5
    args.use_specialized_heads = stage >= 6
    args.use_refinement_gate = stage >= 8
    args.use_hard_candidate_ranker = stage >= 9
    args.use_e10_rankers = stage >= 10
    args.separate_gate_training = stage >= 10
    if stage < 3:
        args.global_blocks = 1
    if stage < 5:
        args.anatomy_weight = 0.0
        args.symmetry_weight = 0.0
    if stage >= 7:
        args.coordinate_mode = "mse_over_mesh"
        args.tta = True
        args.tta_validation = True
    if stage >= 9:
        if args.hard_landmark_weight == 1.5:
            args.hard_landmark_weight = 4.0
        if args.hard_rank_weight == 0.0:
            args.hard_rank_weight = 1.0
        if args.gate_weight == 0.10:
            args.gate_weight = 0.5
        if args.checkpoint_metric == "overall":
            args.checkpoint_metric = "balanced"
        if args.refinement_calibration == "none":
            args.refinement_calibration = "group_scale"
    if stage >= 10:
        args.hard_rank_mode = "soft_listwise"
        args.gate_weight = 0.0
        args.refinement_calibration = "none"
    return stage


def legacy_transforms(samples, legacy_root):
    if not legacy_root:
        raise ValueError("--alignment legacy requires --legacy-transformation-dir")
    return {sample.sample_id: resolve_legacy_matrix(legacy_root, sample) for sample in samples}


def oracle_gate_status(validation, args):
    checks = {
        "overall": validation["ale"] <= args.max_val_oracle_ale,
        "hard3": validation["hard3_ale"] <= args.max_val_hard3_oracle_ale,
        "p95": validation["p95"] <= args.max_val_oracle_p95,
        "max": validation["max"] <= args.max_val_oracle_max,
        "sample_max": validation["sample_ale_max"] <= args.max_val_sample_oracle_ale,
    }
    return {**checks, "passed": all(checks.values())}


def label_free_registration_roi_scales(alignment_report, train_ids, max_expansion=0.5):
    """Expand uncertain ROIs using a threshold fitted only on train registrations."""
    rows = alignment_report.get("samples", [])
    if not rows or max_expansion <= 0:
        return {row.get("sample_id"): 1.0 for row in rows}, {
            "enabled": False,
            "fit_sample_ids": list(train_ids),
        }
    by_id = {row["sample_id"]: row for row in rows}
    train_values = np.asarray(
        [by_id[sample_id]["robust_symmetric_median_mm"] for sample_id in train_ids],
        dtype=np.float64,
    )
    center = float(np.median(train_values))
    mad = float(np.median(np.abs(train_values - center)))
    robust_sigma = max(1.4826 * mad, 0.25)
    threshold = center + 4.0 * robust_sigma
    scales = {}
    flagged = []
    for sample_id, row in by_id.items():
        residual = float(row["robust_symmetric_median_mm"])
        excess = max(0.0, residual - threshold) / max(threshold, 1e-6)
        scale = 1.0 + min(float(max_expansion), excess)
        scales[sample_id] = scale
        row["adaptive_roi_scale"] = scale
        row["registration_outlier"] = bool(scale > 1.0)
        if scale > 1.0:
            flagged.append(sample_id)
    return scales, {
        "enabled": True,
        "fit_sample_ids": list(train_ids),
        "train_median_mm": center,
        "train_mad_mm": mad,
        "threshold_mm": threshold,
        "max_expansion": float(max_expansion),
        "flagged_sample_ids": sorted(flagged),
    }


def prepare_fold(
    samples,
    splits,
    args,
    fold_dir,
    device,
    enforce_oracle_gate=True,
    preprocessing_dir=None,
):
    fold_dir = Path(fold_dir)
    artifact_dir = Path(preprocessing_dir) if preprocessing_dir else fold_dir
    fold_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    assert_disjoint_splits(splits)
    by_id = {sample.sample_id: sample for sample in samples}
    selected_ids = list(dict.fromkeys(splits["train"] + splits["val"] + splits["test"]))
    selected_samples = [by_id[sample_id] for sample_id in selected_ids if sample_id in by_id]
    missing = set(selected_ids) - set(by_id)
    if missing:
        raise ValueError(f"Split references missing dataset samples: {sorted(missing)[:5]}")

    alignment_dir = artifact_dir / "alignment"
    if args.alignment == "mesh_icp":
        transforms, alignment_report = build_label_free_alignment(
            selected_samples,
            splits["train"],
            alignment_dir,
            load_mesh,
            sample_points=args.icp_points,
            icp_iterations=args.icp_iterations,
            atlas_size=args.atlas_size,
            atlas_iterations=args.atlas_iterations,
            registration_candidates=args.registration_candidates,
            registration_restarts=args.registration_restarts,
            registration_trim_quantile=args.registration_trim_quantile,
            seed=args.seed,
        )
    else:
        transforms = legacy_transforms(selected_samples, args.legacy_transformation_dir)
        alignment_report = {
            "method": "legacy_expert_landmark_procrustes",
            "uses_expert_landmarks": True,
            "publication_safe": False,
        }
        alignment_dir.mkdir(parents=True, exist_ok=True)
        (alignment_dir / "alignment_report.json").write_text(json.dumps(alignment_report, indent=2), encoding="utf-8")

    registration_radius_scales, registration_scale_report = label_free_registration_roi_scales(
        alignment_report,
        splits["train"],
        args.registration_roi_expansion,
    )
    alignment_report["adaptive_roi"] = registration_scale_report
    if args.alignment == "mesh_icp":
        (alignment_dir / "alignment_report.json").write_text(
            json.dumps(alignment_report, indent=2), encoding="utf-8"
        )

    records = {}
    for sample in tqdm(selected_samples, desc="mesh cache", disable=args.no_tqdm):
        records[sample.sample_id] = prepare_mesh_record(
            sample,
            transforms[sample.sample_id],
            artifact_dir / "mesh_cache",
            args.max_vertices,
            include_landmarks=sample.sample_id not in set(splits["test"]),
        )
    normalizer = fit_feature_normalizer(
        selected_samples,
        records,
        splits["train"],
        artifact_dir / "normalization.json",
        seed=args.seed,
    )

    train_landmarks = []
    for sample_id in splits["train"]:
        with np.load(records[sample_id]) as record:
            train_landmarks.append(record["landmarks"].copy())
    train_template = np.mean(np.stack(train_landmarks), axis=0).astype(np.float32)
    np.save(artifact_dir / "train_template_landmarks.npy", train_template)
    coarse_target = {}
    coarse_sources = {"mode": args.coarse_source}
    if args.coarse_source == "stage1_oof":
        coarse_target, stage1_report = generate_oof_stage1_predictions(
            selected_samples,
            splits,
            records,
            transforms,
            normalizer,
            artifact_dir,
            device,
            args,
        )
        coarse_sources = {
            "mode": "stage1_nested_oof",
            "report": stage1_report,
        }
    elif args.coarse_source == "external":
        external, coarse_sources = load_coarse_predictions(
            args.base_prediction_dir,
            args.initial_prediction_dir,
            args.base_prefix,
            args.initial_prefix,
            require_train=args.train_center_mode == "external",
        )
        for sample in selected_samples:
            if sample.sample_id not in external and sample.sample_id not in splits["train"]:
                raise KeyError(f"External coarse predictions miss {sample.sample_id}")
            if sample.sample_id not in external:
                continue
            legacy = resolve_legacy_matrix(args.legacy_transformation_dir, sample)
            coarse_target[sample.sample_id] = convert_legacy_coordinates(
                external[sample.sample_id], legacy, transforms[sample.sample_id]
            ).astype(np.float32)
        if args.train_center_mode == "template":
            for sample_id in splits["train"]:
                coarse_target[sample_id] = train_template.copy()
            coarse_sources["train_override"] = "train_template"
    else:
        coarse_target = {sample_id: train_template.copy() for sample_id in selected_ids}
        coarse_sources = {"mode": "train_template", "fit_sample_ids": list(splits["train"])}

    if args.train_center_mode == "synthetic" and args.coarse_source != "stage1_oof":
        for sample_id in splits["train"]:
            with np.load(records[sample_id]) as record:
                expert = record["landmarks"].astype(np.float32)
            stable = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(args.seed + stable)
            sigma = np.full((23, 1), args.synthetic_core_sigma_mm, dtype=np.float32)
            sigma[[0, 21, 22]] = args.synthetic_hard_sigma_mm
            noise = rng.normal(0.0, 1.0, size=(23, 3)).astype(np.float32) * sigma
            limit = np.where(np.isin(np.arange(23), [0, 21, 22]), 12.0, 8.0)[:, None]
            norm = np.linalg.norm(noise, axis=1, keepdims=True)
            noise *= np.minimum(1.0, limit / np.maximum(norm, 1e-6))
            coarse_target[sample_id] = expert + noise
        coarse_sources["train_override"] = {
            "mode": "expert_plus_deterministic_synthetic_error",
            "core_sigma_mm": args.synthetic_core_sigma_mm,
            "hard3_sigma_mm": args.synthetic_hard_sigma_mm,
            "note": "Train labels only; no validation/test labels or in-sample model predictions are used.",
        }

    precompute_samples = [sample for sample in selected_samples if sample.sample_id not in set(splits["test"])]
    print("Precomputing train/validation geodesic ROIs...", flush=True)
    oracle_by_split = {"train": [], "val": []}
    oracle_samples = {"train": [], "val": []}
    for offset, sample in enumerate(tqdm(precompute_samples, desc="geodesic ROI", disable=args.no_tqdm)):
        roi_path = build_roi_cache(
            records[sample.sample_id],
            coarse_target[sample.sample_id],
            args.roi_points,
            artifact_dir / "cache" / "roi",
            sample.sample_id,
            args.seed,
            radius_scale=(
                args.roi_radius_scale
                * registration_radius_scales.get(sample.sample_id, 1.0)
            ),
            roi_mode=args.roi_mode,
            euclidean_radius_scale=args.roi_euclidean_scale,
            multi_seed_count=args.roi_multi_seeds,
        )
        split_name = "train" if sample.sample_id in set(splits["train"]) else "val"
        with np.load(roi_path) as roi_record:
            sample_oracle = roi_record["oracle_error"].copy()
            oracle_by_split[split_name].append(sample_oracle)
            oracle_samples[split_name].append(
                {
                    "split": split_name,
                    "sample_id": sample.sample_id,
                    "oracle_ale": float(sample_oracle.mean()),
                    "oracle_p95": float(np.percentile(sample_oracle, 95)),
                    "oracle_max": float(sample_oracle.max()),
                }
            )
        if args.no_tqdm and ((offset + 1) % 10 == 0 or offset + 1 == len(precompute_samples)):
            print(f"Geodesic ROI {offset + 1}/{len(precompute_samples)}", flush=True)
    oracle_report = {}
    for split_name, rows in oracle_by_split.items():
        values = np.stack(rows) if rows else np.empty((0, 23), dtype=np.float32)
        oracle_report[split_name] = {
            "ale": float(values.mean()) if values.size else None,
            "core20_ale": float(values[:, CORE20].mean()) if values.size else None,
            "hard3_ale": float(values[:, HARD3].mean()) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "fraction_above_2mm": float(np.mean(values > 2.0)) if values.size else None,
            "per_landmark_ale": values.mean(axis=0).tolist() if values.size else None,
            "sample_ale_p95": float(
                np.percentile([row["oracle_ale"] for row in oracle_samples[split_name]], 95)
            ) if values.size else None,
            "sample_ale_max": float(
                max(row["oracle_ale"] for row in oracle_samples[split_name])
            ) if values.size else None,
            "fraction_samples_above_limit": float(
                np.mean(
                    [
                        row["oracle_ale"] > args.max_val_sample_oracle_ale
                        for row in oracle_samples[split_name]
                    ]
                )
            ) if values.size else None,
        }
    write_csv(
        artifact_dir / "candidate_oracle_samples_pretrain.csv",
        oracle_samples["train"] + oracle_samples["val"],
    )
    (artifact_dir / "candidate_oracle_pretrain.json").write_text(
        json.dumps(oracle_report, indent=2), encoding="utf-8"
    )
    val_oracle = oracle_report["val"]["ale"]
    val_hard3_oracle = oracle_report["val"]["hard3_ale"]
    gate = oracle_gate_status(oracle_report["val"], args)
    if enforce_oracle_gate and not gate["passed"] and not args.skip_oracle_gate:
        raise RuntimeError(
            "Validation candidate oracle failed: "
            f"overall={val_oracle:.4f} (gate {args.max_val_oracle_ale:.4f}), "
            f"hard3={val_hard3_oracle:.4f} (gate {args.max_val_hard3_oracle_ale:.4f}), "
            f"p95={oracle_report['val']['p95']:.4f} (gate {args.max_val_oracle_p95:.4f}), "
            f"max={oracle_report['val']['max']:.4f} (gate {args.max_val_oracle_max:.4f}), "
            f"sample_max={oracle_report['val']['sample_ale_max']:.4f} "
            f"(gate {args.max_val_sample_oracle_ale:.4f}). "
            "Increase ROI coverage before training."
        )

    common = {
        "samples": selected_samples,
        "records": records,
        "transforms": transforms,
        "coarse_predictions": coarse_target,
        "legacy_transform_dir": None,
        "normalizer": normalizer,
        "cache_dir": artifact_dir / "cache",
        "roi_points": args.roi_points,
        "roi_radius_scale": args.roi_radius_scale,
        "roi_mode": args.roi_mode,
        "roi_euclidean_scale": args.roi_euclidean_scale,
        "roi_multi_seeds": args.roi_multi_seeds,
        "sample_radius_scales": registration_radius_scales,
        "use_rgb": args.use_rgb,
        "coarse_in_target_space": True,
        "memory_cache": not args.no_memory_cache,
        "seed": args.seed,
    }
    datasets = {
        "train": RGBGeodesicDataset(
            sample_ids=splits["train"], training=True,
            rotation_degrees=args.rotation_degrees,
            point_noise_mm=args.point_noise_mm,
            rgb_noise=args.rgb_noise,
            point_dropout=args.point_dropout,
            mirror_probability=args.mirror_probability,
            center_jitter_mm=args.center_jitter_mm,
            **common,
        ),
        "val": RGBGeodesicDataset(sample_ids=splits["val"], training=False, **common),
        "test": RGBGeodesicDataset(sample_ids=splits["test"], training=False, **common),
    }
    split_report = {
        "splits": splits,
        "overlap": {
            "train_val": sorted(set(splits["train"]) & set(splits["val"])),
            "train_test": sorted(set(splits["train"]) & set(splits["test"])),
            "val_test": sorted(set(splits["val"]) & set(splits["test"])),
        },
        "coarse_sources": coarse_sources,
        "alignment": alignment_report,
        "test_label_protocol": {
            "stored_in_pretraining_mesh_cache": False,
            "test_roi_targets_precomputed_before_checkpoint_selection": False,
            "consumed_only_after_validation_checkpoint_and_confidence_lock": True,
        },
    }
    (fold_dir / "split_and_leakage_report.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")
    if artifact_dir != fold_dir:
        (fold_dir / "preprocessing_source.json").write_text(
            json.dumps(
                {
                    "source": str(artifact_dir),
                    "reused_stage1_alignment_and_roi": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (fold_dir / "normalization.json").write_text(
            json.dumps(normalizer, indent=2), encoding="utf-8"
        )
        (fold_dir / "candidate_oracle_pretrain.json").write_text(
            json.dumps(oracle_report, indent=2), encoding="utf-8"
        )
        output_alignment = fold_dir / "alignment"
        output_alignment.mkdir(parents=True, exist_ok=True)
        (output_alignment / "alignment_report.json").write_text(
            json.dumps(alignment_report, indent=2), encoding="utf-8"
        )
    return datasets, normalizer


def make_loader(dataset, batch_size, shuffle, args):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.num_workers > 0),
        collate_fn=collate_graphs,
        generator=generator,
    )


def fold_output_dir(output_dir, args, repeat, fold_index):
    if args.protocol == "fixed":
        return output_dir
    if args.cv_repeats == 1:
        return output_dir / f"fold_{fold_index}"
    return output_dir / f"repeat_{repeat}" / f"fold_{fold_index}"


def stage2_signature(args, splits):
    keys = (
        "experiment", "seed", "width", "global_blocks", "heads", "dropout",
        "coordinate_mode", "coordinate_topk", "coordinate_temperature",
        "hard_coordinate_topk", "hard_coordinate_temperature", "roi_points",
        "roi_radius_scale", "roi_mode", "roi_euclidean_scale", "roi_multi_seeds",
        "use_rgb", "use_local_refiner", "use_anatomical_attention",
        "use_specialized_heads", "use_refinement_gate", "use_hard_candidate_ranker",
        "rotation_degrees", "center_jitter_mm", "point_noise_mm", "rgb_noise",
        "point_dropout", "mirror_probability", "heatmap_weight", "region_weight", "coordinate_weight",
        "coarse_weight", "anatomy_weight", "symmetry_weight", "uncertainty_weight",
        "clinical_weight", "gate_weight", "hard_landmark_weight", "hard_rank_weight",
        "checkpoint_metric", "checkpoint_hard3_weight", "refinement_calibration",
        "epochs", "lr", "weight_decay",
        "batch_size", "min_epochs", "patience", "scheduler_start_epoch",
        "lr_warmup_epochs", "coarse_source", "train_center_mode", "alignment",
    )
    if getattr(args, "use_e10_rankers", False):
        keys += (
            "gonion_pair_topk", "hard_rank_mode", "hard_rank_sigma_lm0",
            "hard_rank_sigma_gonion", "hard_negative_count", "hard_negative_margin",
            "hard_negative_weight", "use_e10_rankers", "separate_gate_training",
            "gate_stage_epochs", "gate_stage_min_epochs", "gate_stage_patience",
            "gate_stage_lr", "gate_stage_weight_decay",
            "gate_stage_scheduler_patience",
        )
    payload = {
        "args": {key: getattr(args, key) for key in keys},
        "splits": {name: list(values) for name, values in splits.items()},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_completed_fold(fold_dir, args, splits):
    if not args.resume_stage2 or args.force_stage2_retrain:
        return None
    summary_path = Path(fold_dir) / "run_summary.json"
    required = (
        Path(fold_dir) / "metrics_val.json",
        Path(fold_dir) / "metrics_test.json",
        Path(fold_dir) / "predictions_test.csv",
    )
    if not summary_path.exists() or not all(path.exists() for path in required):
        return None
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = stage2_signature(args, splits)
    if result.get("stage2_signature") != expected:
        return None
    print(f"Completed Stage 2 fold cached: {fold_dir}", flush=True)
    return result


def validate_preprocessing_root(root, args):
    root = Path(root)
    config_path = root / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"--preprocessing-root has no experiment_config.json: {root}"
        )
    cached = json.loads(config_path.read_text(encoding="utf-8"))
    keys = (
        "protocol", "folds", "cv_repeats", "val_fraction", "seed",
        "coarse_source", "train_center_mode", "alignment", "icp_points",
        "icp_iterations", "atlas_size", "atlas_iterations",
        "registration_candidates", "registration_restarts",
        "registration_trim_quantile", "registration_roi_expansion",
        "max_vertices", "roi_points", "roi_radius_scale", "roi_mode",
        "roi_euclidean_scale", "roi_multi_seeds", "stage1_oof_folds",
        "stage1_oof_mode", "stage1_oof_fixed_epochs",
        "stage1_oof_template_alpha", "stage1_inner_val_fraction",
        "stage1_width", "stage1_blocks", "heads", "dropout",
        "stage1_topk", "stage1_temperature", "stage1_epochs",
        "stage1_min_epochs", "stage1_patience", "stage1_scheduler_patience",
        "stage1_scheduler_start_epoch", "stage1_lr_warmup_epochs", "stage1_lr",
        "stage1_batch_size", "weight_decay", "grad_clip", "min_delta",
        "stage1_rotation_degrees", "point_noise_mm", "rgb_noise", "point_dropout",
        "use_rgb", "mixed_precision", "amp_dtype",
    )
    mismatches = []
    for key in keys:
        current = getattr(args, key)
        previous = cached.get(key)
        if previous != current:
            mismatches.append(f"{key}: cached={previous!r}, current={current!r}")
    if mismatches:
        raise RuntimeError(
            "Preprocessing cache is incompatible with the current run:\n- "
            + "\n- ".join(mismatches)
        )
    return root


def run_fold(samples, splits, args, fold_dir, device, preprocessing_dir=None):
    fold_dir.mkdir(parents=True, exist_ok=True)
    args.stage2_signature = stage2_signature(args, splits)
    datasets, normalizer = prepare_fold(
        samples,
        splits,
        args,
        fold_dir,
        device,
        preprocessing_dir=preprocessing_dir,
    )
    loaders = {
        "train": make_loader(datasets["train"], args.batch_size, True, args),
        "val": make_loader(datasets["val"], args.eval_batch_size, False, args),
        "test": make_loader(datasets["test"], args.eval_batch_size, False, args),
    }
    model = All23RGBGeodesicCascade(
        input_dim=len(normalizer["mean"]),
        width=args.width,
        global_blocks=args.global_blocks,
        heads=args.heads,
        dropout=args.dropout,
        coordinate_topk=args.coordinate_topk,
        coordinate_temperature=args.coordinate_temperature,
        use_anatomical_attention=args.use_anatomical_attention,
        use_specialized_heads=args.use_specialized_heads,
        use_local_refiner=args.use_local_refiner,
        use_refinement_gate=args.use_refinement_gate,
        use_hard_candidate_ranker=args.use_hard_candidate_ranker,
        use_e10_rankers=args.use_e10_rankers,
        hard_coordinate_topk=args.hard_coordinate_topk,
        hard_coordinate_temperature=args.hard_coordinate_temperature,
        gonion_pair_topk=args.gonion_pair_topk,
        roi_radius_scale=args.roi_radius_scale,
    ).to(device)
    if args.separate_gate_training:
        model.set_refinement_gate_trainable(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameter_count:,}", flush=True)
    loss_weights = LossWeights(
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
    training = fit_model(model, loaders["train"], loaders["val"], device, args, loss_weights, normalizer, fold_dir)
    if training["best_validation_ale"] > args.max_stage2_val_ale:
        raise RuntimeError(
            f"Stage 2 best validation ALE={training['best_validation_ale']:.4f} exceeds "
            f"--max-stage2-val-ale={args.max_stage2_val_ale:.4f}. Test labels remain sealed."
        )
    force_refined_evaluation = False
    if args.separate_gate_training:
        gate_training = fit_separate_refinement_gate(
            model,
            loaders["train"],
            loaders["val"],
            device,
            args,
            normalizer,
            fold_dir,
        )
        training["separate_gate"] = gate_training
        force_refined_evaluation = not gate_training["accepted"]
    validation = collect_outputs(
        model,
        loaders["val"],
        device,
        args,
        normalizer,
        use_tta=args.tta,
        force_refined=force_refined_evaluation,
    )
    refinement_calibration = fit_refinement_calibration(
        validation, args.refinement_calibration
    )
    validation = apply_refinement_calibration(validation, refinement_calibration)
    (fold_dir / "refinement_calibration.json").write_text(
        json.dumps(refinement_calibration, indent=2), encoding="utf-8"
    )
    calibration = calibrate_confidence(validation["log_var"], validation["errors"])
    validation_metrics = save_evaluation(
        fold_dir, "val", validation, calibration, args.bootstrap_iters, args.seed
    )
    if args.validation_only:
        result = {
            "stage2_signature": args.stage2_signature,
            "evaluation_policy": (
                "refined_only" if force_refined_evaluation else "sample_specific_gate"
            ),
            "refinement_calibration": refinement_calibration,
            "training": training,
            "parameter_count": parameter_count,
            "validation": validation_metrics,
            "test": None,
        }
        (fold_dir / "validation_only_summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print("\nValidation-only evaluation; test labels remain sealed", flush=True)
        print(f"All-23 validation ALE: {validation_metrics['overall']['ale']:.4f}", flush=True)
        print(f"Core20 validation ALE: {validation_metrics['core20']['ale']:.4f}", flush=True)
        print(f"Hard3 validation ALE: {validation_metrics['hard3']['ale']:.4f}", flush=True)
        return result
    # Test labels are consumed only after checkpoint and confidence calibration are locked on validation.
    print("Checkpoint locked. Preparing test ROIs and consuming test labels for final evaluation...", flush=True)
    for index in tqdm(range(len(datasets["test"])), desc="test ROI", disable=args.no_tqdm):
        datasets["test"][index]
        if args.no_tqdm and ((index + 1) % 10 == 0 or index + 1 == len(datasets["test"])):
            print(f"Test ROI {index + 1}/{len(datasets['test'])}", flush=True)
    test = collect_outputs(
        model,
        loaders["test"],
        device,
        args,
        normalizer,
        use_tta=args.tta,
        force_refined=force_refined_evaluation,
    )
    test = apply_refinement_calibration(test, refinement_calibration)
    test_metrics = save_evaluation(fold_dir, "test", test, calibration, args.bootstrap_iters, args.seed)
    result = {
        "stage2_signature": args.stage2_signature,
        "evaluation_policy": (
            "refined_only" if force_refined_evaluation else "sample_specific_gate"
        ),
        "refinement_calibration": refinement_calibration,
        "training": training,
        "parameter_count": parameter_count,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    (fold_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"All-23 ALE: {test_metrics['overall']['ale']:.4f}", flush=True)
    print(f"Core20 ALE: {test_metrics['core20']['ale']:.4f}", flush=True)
    print(f"Hard3 ALE: {test_metrics['hard3']['ale']:.4f}", flush=True)
    print(f"Median: {test_metrics['overall']['median']:.4f}", flush=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="All-23 RGB-geodesic coarse-to-fine landmark cascade")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits-json", default=None)
    parser.add_argument("--protocol", choices=("fixed", "cv"), default="fixed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument(
        "--fold-indices",
        default=None,
        help="Comma-separated outer fold numbers to run, for example 1 or 1,2",
    )
    parser.add_argument(
        "--preprocessing-root",
        default=None,
        help="Compatible prior run whose alignment, Stage 1 and ROI caches are reused",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--coarse-source",
        choices=("external", "train_template", "stage1_oof"),
        default="external",
    )
    parser.add_argument("--base-prediction-dir", default=None)
    parser.add_argument("--initial-prediction-dir", default=None)
    parser.add_argument("--base-prefix", default="stage2_raw")
    parser.add_argument("--initial-prefix", default="stacked")
    parser.add_argument("--train-center-mode", choices=("synthetic", "template", "external"), default="synthetic")
    parser.add_argument("--synthetic-core-sigma-mm", type=float, default=1.3)
    parser.add_argument("--synthetic-hard-sigma-mm", type=float, default=2.5)
    parser.add_argument("--external-coarse-oof-manifest", default=None)
    parser.add_argument("--alignment", choices=("mesh_icp", "legacy"), default="mesh_icp")
    parser.add_argument("--legacy-transformation-dir", default=None)
    parser.add_argument("--icp-points", type=int, default=4096)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--atlas-size", type=int, default=8)
    parser.add_argument("--atlas-iterations", type=int, default=2)
    parser.add_argument("--registration-candidates", type=int, default=3)
    parser.add_argument("--registration-restarts", type=int, default=2)
    parser.add_argument("--registration-trim-quantile", type=float, default=0.8)
    parser.add_argument("--registration-roi-expansion", type=float, default=0.5)
    parser.add_argument("--max-vertices", type=int, default=50000)
    parser.add_argument("--roi-points", type=int, default=512)
    parser.add_argument("--roi-radius-scale", type=float, default=1.0)
    parser.add_argument("--roi-mode", choices=("geodesic", "hybrid"), default="hybrid")
    parser.add_argument("--roi-euclidean-scale", type=float, default=1.25)
    parser.add_argument("--roi-multi-seeds", type=int, default=3)
    parser.add_argument("--max-val-oracle-ale", type=float, default=1.5)
    parser.add_argument("--max-val-hard3-oracle-ale", type=float, default=2.5)
    parser.add_argument("--max-val-oracle-p95", type=float, default=2.0)
    parser.add_argument("--max-val-oracle-max", type=float, default=15.0)
    parser.add_argument("--max-val-sample-oracle-ale", type=float, default=2.0)
    parser.add_argument("--skip-oracle-gate", action="store_true")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--global-blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-start-epoch", type=int, default=60)
    parser.add_argument("--lr-warmup-epochs", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-nonfinite-fraction", type=float, default=0.02)
    parser.add_argument("--max-amp-overflow-fraction", type=float, default=0.10)
    parser.add_argument("--coordinate-mode", choices=("topk", "mse_over_mesh"), default="topk")
    parser.add_argument("--coordinate-topk", type=int, default=30)
    parser.add_argument("--coordinate-temperature", type=float, default=0.75)
    parser.add_argument("--hard-coordinate-topk", type=int, default=8)
    parser.add_argument("--hard-coordinate-temperature", type=float, default=0.35)
    parser.add_argument("--gonion-pair-topk", type=int, default=32)
    parser.add_argument(
        "--hard-rank-mode",
        choices=("nearest_ce", "soft_listwise"),
        default="nearest_ce",
    )
    parser.add_argument("--hard-rank-sigma-lm0", type=float, default=3.0)
    parser.add_argument("--hard-rank-sigma-gonion", type=float, default=4.0)
    parser.add_argument("--hard-negative-count", type=int, default=16)
    parser.add_argument("--hard-negative-margin", type=float, default=0.5)
    parser.add_argument("--hard-negative-weight", type=float, default=0.25)
    parser.add_argument("--rotation-degrees", type=float, default=10.0)
    parser.add_argument("--center-jitter-mm", type=float, default=0.5)
    parser.add_argument("--point-noise-mm", type=float, default=0.1)
    parser.add_argument("--rgb-noise", type=float, default=0.05)
    parser.add_argument("--point-dropout", type=float, default=0.03)
    parser.add_argument("--mirror-probability", type=float, default=0.0)
    parser.add_argument("--heatmap-weight", type=float, default=1.0)
    parser.add_argument("--region-weight", type=float, default=0.25)
    parser.add_argument("--region-positive-weight", type=float, default=20.0)
    parser.add_argument("--coordinate-weight", type=float, default=0.5)
    parser.add_argument("--coarse-weight", type=float, default=0.2)
    parser.add_argument("--anatomy-weight", type=float, default=0.08)
    parser.add_argument("--symmetry-weight", type=float, default=0.04)
    parser.add_argument("--uncertainty-weight", type=float, default=0.02)
    parser.add_argument("--clinical-weight", type=float, default=0.05)
    parser.add_argument("--gate-weight", type=float, default=0.10)
    parser.add_argument("--gate-warmup-epochs", type=int, default=30)
    parser.add_argument("--gate-stage-epochs", type=int, default=40)
    parser.add_argument("--gate-stage-min-epochs", type=int, default=10)
    parser.add_argument("--gate-stage-patience", type=int, default=10)
    parser.add_argument("--gate-stage-scheduler-patience", type=int, default=4)
    parser.add_argument("--gate-stage-lr", type=float, default=1e-3)
    parser.add_argument("--gate-stage-weight-decay", type=float, default=1e-4)
    parser.add_argument("--hard-landmark-weight", type=float, default=1.5)
    parser.add_argument("--hard-rank-weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-metric", choices=("overall", "balanced"), default="overall"
    )
    parser.add_argument("--checkpoint-hard3-weight", type=float, default=0.35)
    parser.add_argument(
        "--refinement-calibration",
        choices=("none", "group_scale"),
        default="none",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-resume-stage2", dest="resume_stage2", action="store_false")
    parser.add_argument("--force-stage2-retrain", action="store_true")
    parser.set_defaults(resume_stage2=True)
    parser.add_argument("--experiment", default="FULL")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-validation", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--amp-dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    parser.add_argument("--stage1-oof-folds", type=int, default=5)
    parser.add_argument(
        "--stage1-oof-mode",
        choices=("fixed_epoch", "nested_early_stop"),
        default="fixed_epoch",
    )
    parser.add_argument("--stage1-oof-fixed-epochs", type=int, default=120)
    parser.add_argument(
        "--stage1-oof-template-alpha",
        default="auto",
        help="auto fits one shared alpha on outer-train OOF labels; otherwise use [0,1]",
    )
    parser.add_argument("--stage1-inner-val-fraction", type=float, default=0.2)
    parser.add_argument("--stage1-width", type=int, default=96)
    parser.add_argument("--stage1-blocks", type=int, default=3)
    parser.add_argument("--stage1-topk", type=int, default=50)
    parser.add_argument("--stage1-temperature", type=float, default=0.75)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage1-min-epochs", type=int, default=60)
    parser.add_argument("--stage1-patience", type=int, default=20)
    parser.add_argument("--stage1-scheduler-patience", type=int, default=8)
    parser.add_argument("--stage1-scheduler-start-epoch", type=int, default=40)
    parser.add_argument("--stage1-lr-warmup-epochs", type=int, default=5)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage1-batch-size", type=int, default=1)
    parser.add_argument("--stage1-eval-batch-size", type=int, default=1)
    parser.add_argument("--stage1-rotation-degrees", type=float, default=8.0)
    parser.add_argument("--max-stage1-val-ale", type=float, default=6.0)
    parser.add_argument("--max-stage1-oof-ale", type=float, default=6.0)
    parser.add_argument("--max-stage1-oof-p95", type=float, default=12.0)
    parser.add_argument("--max-stage1-oof-val-gap", type=float, default=1.5)
    parser.add_argument("--max-stage2-val-ale", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tqdm", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    stage = experiment_settings(args)
    device = resolve_device(args.device)
    resolve_precision(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_root = (
        validate_preprocessing_root(args.preprocessing_root, args)
        if args.preprocessing_root
        else None
    )
    samples = discover_samples(args.data_root)
    print(f"Device: {device}; samples={len(samples)}; experiment=E{stage}", flush=True)
    print(
        f"Precision: {'AMP ' + args.amp_dtype if args.mixed_precision else 'float32'}",
        flush=True,
    )
    print(f"Alignment: {args.alignment}; coarse source: {args.coarse_source}", flush=True)

    anatomy = {
        "landmarks": list(enumerate(LANDMARK_NAMES)),
        "midline": list(MIDLINE),
        "symmetry_pairs": [list(pair) for pair in SYMMETRY_PAIRS],
        "anatomical_edges": [list(edge) for edge in ANATOMICAL_EDGES],
    }
    (output_dir / "anatomy_schema.json").write_text(json.dumps(anatomy, indent=2), encoding="utf-8")
    (output_dir / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    if args.protocol == "fixed":
        if not args.splits_json:
            raise ValueError("--protocol fixed requires --splits-json")
        fold_specs = [(1, 1, limit_splits(load_split_file(args.splits_json), args.max_samples))]
    else:
        fold_specs = []
        for repeat in range(1, args.cv_repeats + 1):
            repeat_folds = make_cv_splits(samples, args.folds, args.val_fraction, args.seed + (repeat - 1) * 1009)
            for fold_index, split in enumerate(repeat_folds, start=1):
                fold_specs.append((repeat, fold_index, limit_splits(split, args.max_samples)))
        if args.coarse_source == "external":
            manifest = load_oof_manifest(args.external_coarse_oof_manifest)
            for _, _, split in fold_specs:
                validate_external_cv_provenance(split, manifest)

    if args.fold_indices:
        requested_folds = {
            int(value.strip())
            for value in str(args.fold_indices).split(",")
            if value.strip()
        }
        available_folds = {fold_index for _, fold_index, _ in fold_specs}
        unknown = requested_folds - available_folds
        if unknown:
            raise ValueError(f"Unknown --fold-indices: {sorted(unknown)}")
        fold_specs = [
            spec for spec in fold_specs if spec[1] in requested_folds
        ]

    summaries = []
    started = time.time()
    if args.preflight_only:
        reports = []
        for run_index, (repeat, fold_index, splits) in enumerate(fold_specs, start=1):
            fold_dir = fold_output_dir(output_dir, args, repeat, fold_index)
            preprocessing_dir = (
                fold_output_dir(preprocessing_root, args, repeat, fold_index)
                if preprocessing_root
                else None
            )
            fold_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nPreflight {run_index}/{len(fold_specs)} repeat={repeat} fold={fold_index}", flush=True)
            prepare_fold(
                samples,
                splits,
                args,
                fold_dir,
                device,
                enforce_oracle_gate=False,
                preprocessing_dir=preprocessing_dir,
            )
            oracle = json.loads((fold_dir / "candidate_oracle_pretrain.json").read_text(encoding="utf-8"))
            alignment = json.loads(
                (fold_dir / "alignment" / "alignment_report.json").read_text(
                    encoding="utf-8"
                )
            )
            adaptive_roi = alignment.get("adaptive_roi", {})
            stage1_metrics = None
            if args.coarse_source == "stage1_oof":
                stage1_dir = preprocessing_dir if preprocessing_dir else fold_dir
                stage1_metrics = json.loads(
                    (stage1_dir / "stage1_global_coarse" / "metrics_train_val.json").read_text(
                        encoding="utf-8"
                    )
                )
            gate = oracle_gate_status(oracle["val"], args)
            reports.append(
                {
                    "repeat": repeat,
                    "fold": fold_index,
                    "train_oracle_ale": oracle["train"]["ale"],
                    "validation_oracle_ale": oracle["val"]["ale"],
                    "validation_core20_oracle_ale": oracle["val"]["core20_ale"],
                    "validation_hard3_oracle_ale": oracle["val"]["hard3_ale"],
                    "validation_oracle_p95": oracle["val"]["p95"],
                    "validation_oracle_max": oracle["val"]["max"],
                    "validation_sample_oracle_p95": oracle["val"]["sample_ale_p95"],
                    "validation_sample_oracle_max": oracle["val"]["sample_ale_max"],
                    "stage1_train_oof_ale": (
                        stage1_metrics["train_oof"]["overall_ale"] if stage1_metrics else None
                    ),
                    "stage1_validation_ale": (
                        stage1_metrics["validation"]["overall_ale"] if stage1_metrics else None
                    ),
                    "stage1_oof_p95": (
                        stage1_metrics["train_oof"]["p95"] if stage1_metrics else None
                    ),
                    "stage1_oof_validation_gap": (
                        abs(
                            stage1_metrics["train_oof"]["overall_ale"]
                            - stage1_metrics["validation"]["overall_ale"]
                        )
                        if stage1_metrics else None
                    ),
                    "stage1_oof_validation_signed_gap": (
                        stage1_metrics["train_oof"]["overall_ale"]
                        - stage1_metrics["validation"]["overall_ale"]
                        if stage1_metrics else None
                    ),
                    "registration_outlier_count": len(
                        adaptive_roi.get("flagged_sample_ids", [])
                    ),
                    "registration_train_median_mm": adaptive_roi.get(
                        "train_median_mm"
                    ),
                    "overall_gate_pass": gate["overall"],
                    "hard3_gate_pass": gate["hard3"],
                    "tail_gate_pass": gate["p95"] and gate["max"] and gate["sample_max"],
                    "passed": gate["passed"],
                }
            )
            stage1_text = (
                f"stage1_val={stage1_metrics['validation']['overall_ale']:.4f}, "
                if stage1_metrics else ""
            )
            print(
                f"Fold {fold_index} candidate oracle: overall={oracle['val']['ale']:.4f}, "
                f"core20={oracle['val']['core20_ale']:.4f}, hard3={oracle['val']['hard3_ale']:.4f}, "
                f"{stage1_text}"
                f"sample_max={oracle['val']['sample_ale_max']:.4f}, "
                f"passed={gate['passed']}",
                flush=True,
            )
        write_csv(output_dir / "preflight_oracle_summary.csv", reports)
        all_passed = all(report["passed"] for report in reports)
        (output_dir / "preflight_complete.json").write_text(
            json.dumps({"complete": all_passed, "folds": reports}, indent=2), encoding="utf-8"
        )
        print(
            f"\nPreflight completed for {len(reports)} folds; passed={all_passed}. "
            "Stage 1 was trained/cached; Stage 2 training was not started.",
            flush=True,
        )
        if not all_passed and not args.skip_oracle_gate:
            raise RuntimeError(
                "CV preflight failed. Inspect preflight_oracle_summary.csv and increase ROI "
                "coverage before starting training."
            )
        return
    for run_index, (repeat, fold_index, splits) in enumerate(fold_specs, start=1):
        fold_dir = fold_output_dir(output_dir, args, repeat, fold_index)
        preprocessing_dir = (
            fold_output_dir(preprocessing_root, args, repeat, fold_index)
            if preprocessing_root
            else None
        )
        print(
            f"\nRun {run_index}/{len(fold_specs)} repeat={repeat} fold={fold_index} train/val/test="
            f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}",
            flush=True,
        )
        result = load_completed_fold(fold_dir, args, splits)
        if result is None:
            result = run_fold(
                samples,
                splits,
                args,
                fold_dir,
                device,
                preprocessing_dir=preprocessing_dir,
            )
        metric_scope = result["validation"] if args.validation_only else result["test"]
        summaries.append(
            {
                "fold": fold_index,
                "repeat": repeat,
                "ale": metric_scope["overall"]["ale"],
                "median": metric_scope["overall"]["median"],
                "std": metric_scope["overall"]["std"],
                "core20_ale": metric_scope["core20"]["ale"],
                "hard3_ale": metric_scope["hard3"]["ale"],
                "training_seconds": result["training"]["training_seconds"],
                "parameter_count": result["parameter_count"],
            }
        )
    if args.validation_only:
        write_csv(output_dir / "summary_validation_metrics.csv", summaries)
        validation_summary = {
            "protocol": args.protocol,
            "validation_only": True,
            "folds": summaries,
            "validation_ale_mean": float(np.mean([row["ale"] for row in summaries])),
            "validation_ale_std": float(np.std([row["ale"] for row in summaries])),
            "total_seconds": float(time.time() - started),
            "test_labels_consumed": False,
        }
        (output_dir / "summary_validation_metrics.json").write_text(
            json.dumps(validation_summary, indent=2), encoding="utf-8"
        )
        print(
            f"\nValidation ALE: {validation_summary['validation_ale_mean']:.4f} +/- "
            f"{validation_summary['validation_ale_std']:.4f}",
            flush=True,
        )
        print("Test labels were not consumed.", flush=True)
        return
    write_csv(output_dir / "summary_fold_metrics.csv", summaries)
    aggregate = {
        "protocol": args.protocol,
        "folds": summaries,
        "fold_ale_mean": float(np.mean([row["ale"] for row in summaries])),
        "fold_ale_std": float(np.std([row["ale"] for row in summaries])),
        "total_seconds": float(time.time() - started),
        "publication_safe_alignment": args.alignment == "mesh_icp",
        "publication_safe_pipeline": args.alignment == "mesh_icp"
        and args.coarse_source in ("train_template", "stage1_oof"),
        "publication_safety_note": (
            "External AGH/stacker centers inherit their original expert-landmark Procrustes preprocessing; "
            "use train_template CV for the primary publication result."
            if args.coarse_source == "external"
            else (
                "Train centers are nested OOF Stage 1 predictions; validation/test centers come "
                "from the outer-train-only Stage 1 model. No test expert landmark is used."
                if args.coarse_source == "stage1_oof"
                else "No validation/test expert landmark is used for alignment or coarse-center construction."
            )
        ),
        "primary_target_met": float(np.mean([row["ale"] for row in summaries])) < 2.0,
    }
    (output_dir / "summary_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\nFold ALE: {aggregate['fold_ale_mean']:.4f} +/- {aggregate['fold_ale_std']:.4f}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
