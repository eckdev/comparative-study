#!/usr/bin/env python3
"""Create the canonical stratified outer-CV manifest used by every model."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from create_common_splits import discover_samples


def split_group_counts(samples_by_id, split):
    counts = Counter()
    for sample_id in split:
        sample = samples_by_id[sample_id]
        counts[f"{sample.class_name}_{sample.gender}"] += 1
    return dict(sorted(counts.items()))


def validate_folds(sample_ids, folds):
    expected = set(sample_ids)
    test_counts = Counter()
    for item in folds:
        split_sets = {name: set(item[name]) for name in ("train", "val", "test")}
        if split_sets["train"] & split_sets["val"]:
            raise ValueError(f"Fold {item['fold']} has train/val overlap")
        if split_sets["train"] & split_sets["test"]:
            raise ValueError(f"Fold {item['fold']} has train/test overlap")
        if split_sets["val"] & split_sets["test"]:
            raise ValueError(f"Fold {item['fold']} has val/test overlap")
        if set.union(*split_sets.values()) != expected:
            raise ValueError(f"Fold {item['fold']} does not cover the complete dataset")
        test_counts.update(item["test"])
    if set(test_counts) != expected or set(test_counts.values()) != {1}:
        raise ValueError("Every sample must occur in exactly one outer test fold")


def make_cv_manifest(samples, folds, val_fraction, seed):
    ids = np.asarray([sample.sample_id for sample in samples])
    strata = np.asarray([f"{sample.class_name}_{sample.gender}" for sample in samples])
    samples_by_id = {sample.sample_id: sample for sample in samples}
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = []
    for fold_index, (train_val, test) in enumerate(outer.split(ids, strata), start=1):
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=seed + fold_index - 1,
        )
        train_local, val_local = next(inner.split(train_val, strata[train_val]))
        split = {
            "repeat": 1,
            "fold": fold_index,
            "train": ids[train_val[train_local]].tolist(),
            "val": ids[train_val[val_local]].tolist(),
            "test": ids[test].tolist(),
        }
        split["counts"] = {name: len(split[name]) for name in ("train", "val", "test")}
        split["groups"] = {
            name: split_group_counts(samples_by_id, split[name])
            for name in ("train", "val", "test")
        }
        result.append(split)
    validate_folds(ids.tolist(), result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Create the shared orthodontic 5-fold CV manifest.")
    parser.add_argument("--data-root", default="data/dataset")
    parser.add_argument(
        "--output",
        default="shared_splits/orthodontic_5fold_192_48_60_seed42.json",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples, missing = discover_samples(args.data_root)
    # Match the historical AGH/All-23 discovery order exactly. The original
    # pipeline sorted PLY filenames lexicographically inside class/gender.
    samples = sorted(
        samples,
        key=lambda sample: (
            sample.class_name,
            0 if sample.gender == "men" else 1,
            str(sample.subject_id),
        ),
    )
    if not samples:
        raise RuntimeError(f"No paired samples found under {args.data_root}")
    if missing:
        raise RuntimeError(f"Found {len(missing)} meshes without landmarks; manifest was not written")

    folds = make_cv_manifest(samples, args.folds, args.val_fraction, args.seed)
    payload = {
        "name": f"orthodontic_{args.folds}fold_192_48_60_seed{args.seed}",
        "protocol": "stratified_outer_cv_with_fixed_inner_validation",
        "seed": args.seed,
        "n_total": len(samples),
        "n_folds": args.folds,
        "val_fraction_within_outer_train": args.val_fraction,
        "stratification": "class_gender",
        "patient_key": "sample_id",
        "patient_identity_assumption": (
            "Class-specific identifiers are distinct patients; confirm this with the data custodian."
        ),
        "outer_test_once": True,
        "folds": folds,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Samples/folds: {len(samples)}/{len(folds)}")
    for item in folds:
        print(
            f"Fold {item['fold']}: "
            f"{item['counts']['train']}/{item['counts']['val']}/{item['counts']['test']}"
        )


if __name__ == "__main__":
    main()
