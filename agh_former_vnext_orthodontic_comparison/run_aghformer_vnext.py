#!/usr/bin/env python3
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agh_former_vnext_orthodontic_comparison.model import AGHFormerVNext
from agh_former_vnext_orthodontic_comparison.shape_prior import TrainOnlyShapePrior
from all23_rgb_geodesic_cascade.anatomy import (
    ANATOMICAL_EDGES,
    CORE20,
    HARD3,
    LANDMARK_NAMES,
    MIDLINE,
    SYMMETRY_PAIRS,
)
from all23_rgb_geodesic_cascade.data import discover_samples, load_split_file
from all23_rgb_geodesic_cascade.losses import LossWeights
from all23_rgb_geodesic_cascade.metrics import (
    calibrate_confidence,
    save_evaluation,
    write_csv,
)
from all23_rgb_geodesic_cascade.run_all23_rgb_geodesic import (
    build_parser as build_base_parser,
    fold_output_dir,
    limit_splits,
    make_cv_splits,
    make_loader,
    prepare_fold,
    resolve_device,
    resolve_precision,
    set_seed,
    validate_preprocessing_root,
)
from all23_rgb_geodesic_cascade.train import (
    apply_refinement_calibration,
    collect_outputs,
    fit_model,
    fit_refinement_calibration,
    fit_separate_refinement_gate,
)


def configure_vnext(args):
    args.experiment = "AGH_VNEXT"
    args.use_rgb = True
    args.use_local_refiner = True
    args.use_anatomical_attention = True
    args.use_specialized_heads = True
    args.use_refinement_gate = True
    args.use_hard_candidate_ranker = True
    args.use_e10_rankers = True
    args.separate_gate_training = True
    args.hard_rank_mode = "soft_listwise"
    return args


def build_parser():
    parser = build_base_parser()
    parser.description = "Leakage-free AGH-Former vNext with OOF coarse fusion"
    parser.set_defaults(
        protocol="cv",
        coarse_source="stage1_oof",
        train_center_mode="external",
        alignment="mesh_icp",
        experiment="AGH_VNEXT",
        coordinate_mode="mse_over_mesh",
        tta=True,
        tta_validation=True,
        hard_landmark_weight=2.0,
        hard_rank_weight=0.5,
        gate_weight=0.0,
        mirror_probability=0.5,
        checkpoint_metric="balanced",
        checkpoint_hard3_weight=0.25,
        refinement_calibration="none",
        stage1_oof_mode="fixed_epoch",
    )
    parser.add_argument("--token-blocks", type=int, default=2)
    parser.add_argument("--token-surface-points", type=int, default=4096)
    parser.add_argument("--fusion-residual-limit-mm", type=float, default=8.0)
    parser.add_argument("--fusion-hard-residual-limit-mm", type=float, default=15.0)
    parser.add_argument("--no-shape-prior", dest="shape_prior", action="store_false")
    parser.set_defaults(shape_prior=True)
    parser.add_argument("--shape-prior-components", default="6,10,15,20,30")
    parser.add_argument("--shape-prior-l2", default="0.1,1,10,100")
    parser.add_argument("--shape-prior-max-core-regression-mm", type=float, default=0.03)
    parser.add_argument("--shape-prior-min-val-gain-mm", type=float, default=0.01)
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument(
        "--no-tta-validation", dest="tta_validation", action="store_false"
    )
    return parser


def vnext_signature(args, splits):
    ignored = {
        "output_dir",
        "preprocessing_root",
        "fold_indices",
        "validation_only",
        "preflight_only",
        "no_tqdm",
        "resume_stage2",
        "force_stage2_retrain",
        "checkpoint_every",
    }
    payload = {
        "pipeline_version": 2,
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in ignored and isinstance(value, (str, int, float, bool, type(None)))
        },
        "splits": {name: list(values) for name, values in splits.items()},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_shapes_from_dataset(dataset):
    shapes = []
    sample_ids = []
    for sample in dataset.samples:
        with np.load(dataset.records[sample.sample_id]) as record:
            if "landmarks" not in record.files:
                raise RuntimeError(f"Outer-train record has no landmarks: {sample.sample_id}")
            shapes.append(record["landmarks"].astype(np.float32))
        sample_ids.append(sample.sample_id)
    return np.stack(shapes), sample_ids


def apply_shape_prior(outputs, prior):
    result = dict(outputs)
    result["neural_prediction"] = outputs["prediction"].copy()
    prediction = prior.transform(outputs["prediction"])
    result["prediction"] = prediction
    result["errors"] = np.linalg.norm(prediction - outputs["expert"], axis=-1)
    return result


def fit_shape_prior(args, dataset, validation, fold_dir):
    if not args.shape_prior:
        return None, {"enabled": False, "reason": "disabled_by_argument"}
    train_shapes, train_ids = train_shapes_from_dataset(dataset)
    prior = TrainOnlyShapePrior(
        component_grid=[int(value) for value in args.shape_prior_components.split(",") if value.strip()],
        l2_grid=[float(value) for value in args.shape_prior_l2.split(",") if value.strip()],
    ).fit(train_shapes, train_ids)
    selected = prior.calibrate(
        validation["prediction"],
        validation["expert"],
        validation["sample_ids"],
        args.shape_prior_max_core_regression_mm,
    )
    base_ale = float(validation["errors"].mean())
    candidate = prior.transform(validation["prediction"])
    candidate_ale = float(np.linalg.norm(candidate - validation["expert"], axis=-1).mean())
    gain = base_ale - candidate_ale
    accepted = gain >= float(args.shape_prior_min_val_gain_mm)
    if not accepted:
        prior.config = {
            **selected,
            "alpha_core": 0.0,
            "alpha_hard": 0.0,
            "validation_rejected": True,
        }
    report = {
        "enabled": True,
        "accepted": accepted,
        "base_validation_ale": base_ale,
        "candidate_validation_ale": candidate_ale,
        "validation_gain_mm": gain,
        "minimum_gain_mm": float(args.shape_prior_min_val_gain_mm),
        "selected": prior.config,
        "fit_sample_count": len(train_ids),
        "fit_sample_ids": train_ids,
        "validation_sample_ids": list(validation["sample_ids"]),
        "uses_test_labels": False,
    }
    prior.save(Path(fold_dir) / "shape_prior_full_report.json")
    write_csv(
        Path(fold_dir) / "shape_prior_validation_sweep.csv",
        prior.calibration_rows,
    )
    (Path(fold_dir) / "shape_prior_selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return prior, report


def model_from_args(args, normalizer, device):
    return AGHFormerVNext(
        input_dim=len(normalizer["mean"]),
        width=args.width,
        global_blocks=args.global_blocks,
        heads=args.heads,
        dropout=args.dropout,
        coordinate_topk=args.coordinate_topk,
        coordinate_temperature=args.coordinate_temperature,
        use_anatomical_attention=True,
        use_specialized_heads=True,
        use_local_refiner=True,
        use_refinement_gate=True,
        use_hard_candidate_ranker=True,
        use_e10_rankers=True,
        hard_coordinate_topk=args.hard_coordinate_topk,
        hard_coordinate_temperature=args.hard_coordinate_temperature,
        gonion_pair_topk=args.gonion_pair_topk,
        roi_radius_scale=args.roi_radius_scale,
        token_blocks=args.token_blocks,
        token_surface_points=args.token_surface_points,
        fusion_residual_limit_mm=args.fusion_residual_limit_mm,
        fusion_hard_residual_limit_mm=args.fusion_hard_residual_limit_mm,
    ).to(device)


def run_fold(samples, splits, args, fold_dir, device, preprocessing_dir=None):
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    args.stage2_signature = vnext_signature(args, splits)
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
    model = model_from_args(args, normalizer, device)
    model.set_refinement_gate_trainable(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"AGH-Former vNext parameters: {parameter_count:,}", flush=True)
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
    training = fit_model(
        model,
        loaders["train"],
        loaders["val"],
        device,
        args,
        loss_weights,
        normalizer,
        fold_dir,
    )
    if training["best_validation_ale"] > args.max_stage2_val_ale:
        raise RuntimeError(
            f"Stage 2 validation ALE={training['best_validation_ale']:.4f} exceeds "
            f"--max-stage2-val-ale={args.max_stage2_val_ale:.4f}; test remains sealed."
        )
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
    force_refined = not gate_training["accepted"]
    validation = collect_outputs(
        model,
        loaders["val"],
        device,
        args,
        normalizer,
        use_tta=args.tta_validation,
        force_refined=force_refined,
    )
    refinement_calibration = fit_refinement_calibration(
        validation, args.refinement_calibration
    )
    validation = apply_refinement_calibration(validation, refinement_calibration)
    neural_calibration = calibrate_confidence(
        validation["log_var"], validation["errors"]
    )
    neural_validation_metrics = save_evaluation(
        fold_dir / "neural_only",
        "val",
        validation,
        neural_calibration,
        args.bootstrap_iters,
        args.seed,
    )
    prior, prior_report = fit_shape_prior(args, datasets["train"], validation, fold_dir)
    if prior is not None:
        validation = apply_shape_prior(validation, prior)
    calibration = calibrate_confidence(validation["log_var"], validation["errors"])
    validation_metrics = save_evaluation(
        fold_dir,
        "val",
        validation,
        calibration,
        args.bootstrap_iters,
        args.seed,
    )
    if args.validation_only:
        result = {
            "stage2_signature": args.stage2_signature,
            "parameter_count": parameter_count,
            "training": training,
            "shape_prior": prior_report,
            "neural_validation": neural_validation_metrics,
            "validation": validation_metrics,
            "test": None,
        }
        (fold_dir / "validation_only_summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print("\nValidation-only; test labels remain sealed", flush=True)
        print(f"All-23 validation ALE: {validation_metrics['overall']['ale']:.4f}", flush=True)
        print(f"Core20 validation ALE: {validation_metrics['core20']['ale']:.4f}", flush=True)
        print(f"Hard3 validation ALE: {validation_metrics['hard3']['ale']:.4f}", flush=True)
        return result

    print("Checkpoint and validation policies locked. Preparing final test ROIs...", flush=True)
    for index in range(len(datasets["test"])):
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
        force_refined=force_refined,
    )
    test = apply_refinement_calibration(test, refinement_calibration)
    neural_test_metrics = save_evaluation(
        fold_dir / "neural_only",
        "test",
        test,
        neural_calibration,
        args.bootstrap_iters,
        args.seed,
    )
    if prior is not None:
        test = apply_shape_prior(test, prior)
    test_metrics = save_evaluation(
        fold_dir,
        "test",
        test,
        calibration,
        args.bootstrap_iters,
        args.seed,
    )
    result = {
        "stage2_signature": args.stage2_signature,
        "parameter_count": parameter_count,
        "training": training,
        "shape_prior": prior_report,
        "neural_validation": neural_validation_metrics,
        "neural_test": neural_test_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    (fold_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"AGH vNext All-23 ALE: {test_metrics['overall']['ale']:.4f}", flush=True)
    print(f"Core20 ALE: {test_metrics['core20']['ale']:.4f}", flush=True)
    print(f"Hard3 ALE: {test_metrics['hard3']['ale']:.4f}", flush=True)
    print(f"Median: {test_metrics['overall']['median']:.4f}", flush=True)
    return result


def aggregate(output_dir, summaries, elapsed):
    rows = []
    for item in summaries:
        metrics = item["result"].get("test") or item["result"]["validation"]
        rows.append(
            {
                "repeat": item["repeat"],
                "fold": item["fold"],
                "split": "test" if item["result"].get("test") else "validation",
                "overall_ale": metrics["overall"]["ale"],
                "median": metrics["overall"]["median"],
                "std": metrics["overall"]["std"],
                "core20_ale": metrics["core20"]["ale"],
                "hard3_ale": metrics["hard3"]["ale"],
                "sdr_at_2mm": metrics["overall"]["sdr_at_2mm"],
                "sdr_at_3mm": metrics["overall"]["sdr_at_3mm"],
                "sdr_at_4mm": metrics["overall"]["sdr_at_4mm"],
                "parameter_count": item["result"]["parameter_count"],
                "training_seconds": float(
                    item["result"]["training"].get("training_seconds", 0.0)
                    + item["result"]["training"].get("separate_gate", {}).get(
                        "training_seconds", 0.0
                    )
                ),
            }
        )
    write_csv(Path(output_dir) / "summary_fold_metrics.csv", rows)
    overall = np.asarray([row["overall_ale"] for row in rows], dtype=np.float64)
    payload = {
        "model": "AGH-Former vNext",
        "fold_count": len(rows),
        "fold_mean_ale": float(overall.mean()),
        "fold_std_ale": float(overall.std()),
        "elapsed_seconds": float(elapsed),
        "folds": rows,
    }
    (Path(output_dir) / "summary_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def load_completed_fold(fold_dir, args, splits):
    if not args.resume_stage2 or args.force_stage2_retrain or args.validation_only:
        return None
    fold_dir = Path(fold_dir)
    summary_path = fold_dir / "run_summary.json"
    required = (
        fold_dir / "metrics_val.json",
        fold_dir / "metrics_test.json",
        fold_dir / "predictions_test.csv",
    )
    if not summary_path.exists() or not all(path.exists() for path in required):
        return None
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    if result.get("stage2_signature") != vnext_signature(args, splits):
        return None
    print(f"Completed AGH vNext fold cached: {fold_dir}", flush=True)
    return result


def main():
    args = configure_vnext(build_parser().parse_args())
    set_seed(args.seed)
    device = resolve_device(args.device)
    resolve_precision(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(args.data_root)
    preprocessing_root = (
        validate_preprocessing_root(args.preprocessing_root, args)
        if args.preprocessing_root
        else None
    )
    anatomy = {
        "landmarks": list(enumerate(LANDMARK_NAMES)),
        "midline": list(MIDLINE),
        "symmetry_pairs": [list(pair) for pair in SYMMETRY_PAIRS],
        "anatomical_edges": [list(edge) for edge in ANATOMICAL_EDGES],
    }
    (output_dir / "anatomy_schema.json").write_text(
        json.dumps(anatomy, indent=2), encoding="utf-8"
    )
    (output_dir / "experiment_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(
        f"Device: {device}; samples={len(samples)}; model=AGH-Former vNext",
        flush=True,
    )
    print(
        f"Alignment: {args.alignment}; coarse source: {args.coarse_source}; "
        f"shape prior: {args.shape_prior}",
        flush=True,
    )

    if args.protocol == "fixed":
        if not args.splits_json:
            raise ValueError("--protocol fixed requires --splits-json")
        fold_specs = [(1, 1, limit_splits(load_split_file(args.splits_json), args.max_samples))]
    else:
        fold_specs = []
        for repeat in range(1, args.cv_repeats + 1):
            for fold_index, split in enumerate(
                make_cv_splits(
                    samples,
                    args.folds,
                    args.val_fraction,
                    args.seed + (repeat - 1) * 1009,
                ),
                start=1,
            ):
                fold_specs.append(
                    (repeat, fold_index, limit_splits(split, args.max_samples))
                )
    if args.fold_indices:
        requested = {
            int(value) for value in str(args.fold_indices).split(",") if value.strip()
        }
        fold_specs = [item for item in fold_specs if item[1] in requested]
    if not fold_specs:
        raise ValueError("No folds selected")

    started = time.time()
    summaries = []
    for run_index, (repeat, fold_index, splits) in enumerate(fold_specs, start=1):
        fold_dir = fold_output_dir(output_dir, args, repeat, fold_index)
        preprocessing_dir = (
            fold_output_dir(preprocessing_root, args, repeat, fold_index)
            if preprocessing_root
            else None
        )
        print(
            f"\nRun {run_index}/{len(fold_specs)} repeat={repeat} fold={fold_index} "
            f"train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}",
            flush=True,
        )
        if args.preflight_only:
            prepare_fold(
                samples,
                splits,
                args,
                fold_dir,
                device,
                enforce_oracle_gate=False,
                preprocessing_dir=preprocessing_dir,
            )
            print(f"Preflight fold {fold_index} complete", flush=True)
            continue
        result = load_completed_fold(fold_dir, args, splits)
        if result is None:
            result = run_fold(
                samples,
                splits,
                args,
                fold_dir,
                device,
                preprocessing_dir,
            )
        summaries.append(
            {"repeat": repeat, "fold": fold_index, "result": result}
        )
    if args.preflight_only:
        (output_dir / "preflight_complete.json").write_text(
            json.dumps({"complete": True, "fold_count": len(fold_specs)}, indent=2),
            encoding="utf-8",
        )
        return
    summary = aggregate(output_dir, summaries, time.time() - started)
    print(
        f"\nFold ALE: {summary['fold_mean_ale']:.4f} +/- "
        f"{summary['fold_std_ale']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
