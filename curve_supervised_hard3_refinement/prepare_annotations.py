#!/usr/bin/env python3
"""Create a blank, versioned curve-annotation manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from all23_rgb_geodesic_cascade.data import discover_samples

from curve_supervised_hard3_refinement.annotations import blank_manifest


def _eligible_ids(samples, split_report, split_name):
    if split_report is None:
        return {sample.sample_id for sample in samples}
    payload = json.loads(Path(split_report).read_text(encoding="utf-8"))
    try:
        return {str(value) for value in payload["splits"][split_name]}
    except KeyError as error:
        raise ValueError(
            f"Split report has no splits.{split_name}: {split_report}"
        ) from error


def _balanced_selection(samples, eligible, count, seed):
    rows = [sample for sample in samples if sample.sample_id in eligible]
    if count is None or int(count) >= len(rows):
        return rows
    if int(count) <= 0:
        raise ValueError("--sample-count must be positive")
    grouped = defaultdict(list)
    for sample in rows:
        grouped[(sample.class_name, sample.gender)].append(sample)
    rng = np.random.default_rng(seed)
    for values in grouped.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(grouped)
    while len(selected) < int(count):
        progressed = False
        for key in keys:
            if grouped[key] and len(selected) < int(count):
                selected.append(grouped[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _write_tasks(path, samples):
    task_path = path.with_name(f"{path.stem}_tasks.csv")
    with task_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "class", "gender", "mesh_path"),
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "class": sample.class_name,
                    "gender": sample.gender,
                    "mesh_path": str(Path(sample.mesh_path).resolve()),
                }
            )
    return task_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-report")
    parser.add_argument(
        "--split-name", choices=("train", "val", "test"), default="train"
    )
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    samples = discover_samples(args.data_root)
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite an existing annotation manifest: {output}. "
            "Use a new path or pass --force only for an intentionally empty file."
        )
    eligible = _eligible_ids(samples, args.split_report, args.split_name)
    selected = _balanced_selection(samples, eligible, args.sample_count, args.seed)
    payload = blank_manifest(sample.sample_id for sample in selected)
    payload["selection"] = {
        "seed": args.seed,
        "split_report": str(args.split_report) if args.split_report else None,
        "split_name": args.split_name if args.split_report else None,
        "eligible_samples": len(eligible),
        "selected_samples": len(selected),
        "balanced_by": ["class", "gender"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    task_path = _write_tasks(output, selected)
    group_counts = defaultdict(int)
    for sample in selected:
        group_counts[f"{sample.class_name}|{sample.gender}"] += 1
    print(f"Blank manifest for {len(selected)} samples: {output}", flush=True)
    print(f"Annotation task table: {task_path}", flush=True)
    print(f"Selected groups: {dict(sorted(group_counts.items()))}", flush=True)


if __name__ == "__main__":
    main()
