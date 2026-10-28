"""Component ablation of the multi-view refiner (Colab, GPU).

Stage 1, Stage 2 and the shape prior are read from the archived outputs of the
reported runs, so every variant starts from the same estimates and only the
refiner changes. Every variant is applied with the locked per-type policy of
the final model (iteration count and on/off switch per landmark type).

Inference-only variants reuse the six trained refiners of each fold:
  I0_full           six-member ensemble from the shape-prior output (reference)
  I1_single         full-fit refiner only
  I2_no_colour      I0 with every colour channel and colour feature set to zero
                    (the refiners were trained with this on 10% of samples)
  I3_from_stage1    I0 started from the Stage 1 output (no Stage 2, no shape prior)
  I4_from_stage2    I0 started from the Stage 2 output (no shape prior)
Retrained variants train one full-fit refiner per fold on the training
partition, with the published training settings and one switch changed:
  R0_retrained      no switch (reference for R1-R5 and an estimate of retraining noise)
  R1_geometry_only  no images
  R2_no_context     three fine views only (the per-view visibility flags of the
                    geometric descriptor are kept)
  R3_mean_fusion    equal view weights instead of view attention
  R4_image_only     geometric descriptor set to zero
  R5_no_colour      colour channels and features set to zero in training and inference

Outputs per variant and fold, below --out:
  <variant>/fold_k/predictions_test.csv  aligned frame of fold k; expert, base and
                                         prediction coordinates
  <variant>/fold_k/report.json
  <variant>/fold_k/all23_mvsc/           trained refiner (retrained variants)
Variants whose report exists are skipped, so an interrupted run can resume.

Colab:
    python agh_former_v2/run_ablation.py --folds 1 --variants I0_full,I1_single,R0_retrained
    python agh_former_v2/run_ablation.py --folds 1,2,3,4,5 --variants all
List the variants and their refiner signatures without data or GPU:
    python agh_former_v2/run_ablation.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agh_former_v2.measure_cost import (  # noqa: E402
    DRIVE,
    PROPOSED_RUN,
    fold_test_ids,
    proposed_args,
    read_archived,
)

INFERENCE_VARIANTS = {
    "I0_full": {"base": "prior", "ensemble": True, "inference": {}},
    "I1_single": {"base": "prior", "ensemble": False, "inference": {}},
    "I2_no_colour": {"base": "prior", "ensemble": True, "inference": {"drop_colour": True}},
    "I3_from_stage1": {"base": "stage1", "ensemble": True, "inference": {}},
    "I4_from_stage2": {"base": "stage2", "ensemble": True, "inference": {}},
}
RETRAINED_VARIANTS = {
    "R0_retrained": {},
    "R1_geometry_only": {"use_images": False},
    "R2_no_context": {"view_subset": (0, 1, 2)},
    "R3_mean_fusion": {"view_fusion": "mean"},
    "R4_image_only": {"use_geometry": False},
    "R5_no_colour": {"drop_colour": True},
}
VARIANTS = [*INFERENCE_VARIANTS, *RETRAINED_VARIANTS]
MAX_FRAME_DRIFT_MM = 1e-2


def retrained_config(config, name):
    """One full-fit refiner with the published training settings and one switch."""
    return replace(
        config, ensemble=False, use_gate=False, mc_samples=0, **RETRAINED_VARIANTS[name]
    )


def max_difference(first, second):
    return float(max(np.linalg.norm(first[key] - second[key], axis=-1).max() for key in first))


def write_predictions(path, sample_ids, expert, base, prediction):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "landmark"]
            + [f"{name}_{axis}" for name in ("expert", "base", "prediction") for axis in "xyz"]
            + ["error"]
        )
        for index, sample_id in enumerate(sample_ids):
            for landmark in range(23):
                truth = expert[sample_id][landmark]
                guess = prediction[index, landmark]
                writer.writerow(
                    [sample_id, landmark, *truth, *base[sample_id][landmark], *guess,
                     float(np.linalg.norm(guess - truth))]
                )


def run_fold(fold, variants, device, out_root, scratch_root):
    from agh_former_v2.run_agh_v2 import all23_config_from_args
    from all23_mvsc_refinement import apply_policy, fit_or_load_all23_refinement
    from all23_rgb_geodesic_cascade.data import discover_samples
    from all23_rgb_geodesic_cascade.run_all23_rgb_geodesic import (
        fold_output_dir,
        make_cv_splits,
        prepare_fold,
        validate_preprocessing_root,
    )

    pending = [
        name for name in variants
        if not (out_root / name / f"fold_{fold}" / "report.json").exists()
    ]
    if not pending:
        print(f"Fold {fold}: all requested variants done", flush=True)
        return
    args = proposed_args(device)
    samples = discover_samples(args.data_root)
    splits = make_cv_splits(samples, args.folds, args.val_fraction, args.seed)[fold - 1]
    if sorted(splits["test"]) != sorted(fold_test_ids(fold)):
        raise RuntimeError("Fold test partition differs from the shared manifest")
    preprocessing_dir = fold_output_dir(
        validate_preprocessing_root(args.preprocessing_root, args), args, 1, fold
    )
    scratch = scratch_root / f"fold_{fold}"
    datasets, _ = prepare_fold(
        samples, splits, args, scratch / "fold", device,
        enforce_oracle_gate=False, preprocessing_dir=preprocessing_dir,
    )
    test_dataset = datasets["test"]
    test_dataset.include_expert = False
    sample_ids = [sample.sample_id for sample in test_dataset.samples]

    final_csv = PROPOSED_RUN / f"fold_{fold}" / "predictions_test.csv"
    stage_a_csv = Path(args.stage_a_run) / f"fold_{fold}" / "predictions_test.csv"
    expert = read_archived(final_csv, sample_ids, "expert")
    drift = max_difference(read_archived(stage_a_csv, sample_ids, "expert"), expert)
    if drift > MAX_FRAME_DRIFT_MM:
        raise RuntimeError(f"Stage A and final exports use different frames ({drift:.4f} mm)")
    bases = {
        "prior": read_archived(final_csv, sample_ids, "prior"),
        "stage2": read_archived(final_csv, sample_ids, "stage_a"),
        "stage1": read_archived(stage_a_csv, sample_ids, "stage1_coarse"),
    }
    archived_final = read_archived(final_csv, sample_ids, "prediction")
    policy = json.loads(
        (PROPOSED_RUN / f"fold_{fold}" / "all23_mvsc" / "all23_policy.json").read_text(
            encoding="utf-8"
        )
    )
    config = all23_config_from_args(args)
    published = {}

    for name in pending:
        started = time.time()
        fold_out = out_root / name / f"fold_{fold}"
        if name in INFERENCE_VARIANTS:
            spec = INFERENCE_VARIANTS[name]
            if spec["ensemble"] not in published:
                published[spec["ensemble"]] = fit_or_load_all23_refinement(
                    datasets["train"], scratch / f"published_ensemble_{spec['ensemble']}",
                    replace(config, ensemble=spec["ensemble"]), device,
                    scratch_dir=scratch / "refiner_scratch",
                    model_dir=Path(args.refiner_source_run) / f"fold_{fold}" / "all23_mvsc",
                )
            refinement = published[spec["ensemble"]]
            base_name = spec["base"]
            switches = spec["inference"]
            # Inference-only switches: the networks were loaded with the published
            # configuration, so its signature check is unaffected.
            original = refinement.ensemble.config
            refinement.ensemble.config = replace(original, **switches)
            try:
                result = refinement.predict(
                    test_dataset, bases[base_name], scratch_dir=scratch / "render",
                    cache_dir=scratch / "roi_cache_test", label=name,
                )
            finally:
                refinement.ensemble.config = original
            training_seconds = None
        else:
            switches = RETRAINED_VARIANTS[name]
            refinement = fit_or_load_all23_refinement(
                datasets["train"], fold_out / "all23_mvsc", retrained_config(config, name),
                device, scratch_dir=scratch / "refiner_scratch",
            )
            base_name = "prior"
            result = refinement.predict(
                test_dataset, bases["prior"], scratch_dir=scratch / "render",
                cache_dir=scratch / "roi_cache_test", label=name,
            )
            training_seconds = refinement.reports["full_model"].get("training_seconds")
        prediction = apply_policy(result, policy)["prediction"]
        if list(result["sample_ids"]) != sample_ids:
            raise RuntimeError("Refiner output order differs from the test partition")
        write_predictions(
            fold_out / "predictions_test.csv", sample_ids, expert, bases[base_name], prediction
        )
        truth = np.stack([expert[sample_id] for sample_id in sample_ids])
        report = {
            "variant": name,
            "fold": fold,
            "base": base_name,
            "switches": {key: list(value) if isinstance(value, tuple) else value
                         for key, value in switches.items()},
            "ensemble_members": len(refinement.ensemble.models),
            "policy": {key: policy[key] for key in ("iterations", "alpha_caps")},
            "ale": float(np.linalg.norm(prediction - truth, axis=-1).mean()),
            "max_difference_to_archived_final_mm": max_difference(
                {sample_id: prediction[index] for index, sample_id in enumerate(sample_ids)},
                archived_final,
            ),
            "training_seconds": training_seconds,
            "elapsed_seconds": time.time() - started,
            "test_labels_used_for_training": False,
        }
        (fold_out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"Fold {fold} {name}: ALE {report['ale']:.4f} mm, "
            f"max difference to archived final {report['max_difference_to_archived_final_mm']:.3f} mm, "
            f"training {training_seconds} s, elapsed {report['elapsed_seconds']:.0f} s",
            flush=True,
        )


def dry_run():
    from all23_mvsc_refinement.refiner import All23MVSCConfig, config_signature

    config = All23MVSCConfig(use_gate=False, prior_covariates=())
    print(f"published  {config_signature(config, ['A'])[:12]}")
    for name, spec in INFERENCE_VARIANTS.items():
        print(f"{name:<17} inference only, base={spec['base']}, ensemble={spec['ensemble']}, "
              f"switches={spec['inference']}")
    for name in RETRAINED_VARIANTS:
        print(f"{name:<17} retrained, signature "
              f"{config_signature(retrained_config(config, name), ['A'])[:12]}, "
              f"switches={RETRAINED_VARIANTS[name]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--out", default=str(DRIVE / "ablation"))
    parser.add_argument("--scratch", default="/content/ablation_scratch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
        return
    variants = VARIANTS if args.variants == "all" else args.variants.split(",")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; choose from {VARIANTS}")
    if not torch.cuda.is_available():
        raise RuntimeError("The ablation runs on the GPU used for the experiments")
    device = torch.device("cuda")
    torch.manual_seed(42)
    np.random.seed(42)
    for fold in (int(value) for value in args.folds.split(",")):
        run_fold(fold, variants, device, Path(args.out), Path(args.scratch))


if __name__ == "__main__":
    main()
