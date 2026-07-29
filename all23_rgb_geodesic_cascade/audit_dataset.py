#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from all23_rgb_geodesic_cascade.anatomy import SYMMETRY_PAIRS
    from all23_rgb_geodesic_cascade.data import (
        assert_disjoint_splits, discover_samples, load_mesh, load_split_file, read_landmarks,
    )
    from all23_rgb_geodesic_cascade.metrics import summarize, write_csv
else:
    from .anatomy import SYMMETRY_PAIRS
    from .data import assert_disjoint_splits, discover_samples, load_mesh, load_split_file, read_landmarks
    from .metrics import summarize, write_csv


def main():
    parser = argparse.ArgumentParser(description="Dataset coordinate/index/RGB/surface audit")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits-json", default=None)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(args.data_root)
    rows, surface_distances, vertex_counts, color_missing = [], [], [], []
    pair_signs = {f"{left}-{right}": [] for left, right in SYMMETRY_PAIRS}
    subject_keys = {}
    for sample in samples:
        landmarks = read_landmarks(sample.landmark_path)
        mesh = load_mesh(sample.mesh_path)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        colors = getattr(mesh.visual, "vertex_colors", None)
        has_rgb = colors is not None and len(colors) == len(vertices)
        if not has_rgb:
            color_missing.append(sample.sample_id)
        distances = cKDTree(vertices).query(landmarks, k=1, workers=-1)[0]
        surface_distances.append(distances)
        vertex_counts.append(len(vertices))
        for left, right in SYMMETRY_PAIRS:
            pair_signs[f"{left}-{right}"].append(float(np.sign(landmarks[left, 0] - landmarks[right, 0])))
        prefix = "F" if sample.gender == "women" else "M"
        subject_keys.setdefault(f"{prefix}{sample.subject_id}", []).append(sample.sample_id)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "class": sample.class_name,
                "gender": sample.gender,
                "subject_id": sample.subject_id,
                "vertices": len(vertices),
                "faces": len(mesh.faces),
                "has_rgb": has_rgb,
                "landmark_surface_mean_mm": float(distances.mean()),
                "landmark_surface_max_mm": float(distances.max()),
            }
        )
    surface_distances = np.stack(surface_distances)
    split_report = None
    if args.splits_json:
        splits = load_split_file(args.splits_json)
        assert_disjoint_splits(splits)
        split_report = {name: len(values) for name, values in splits.items()}
    report = {
        "samples": len(samples),
        "landmarks_per_sample": 23,
        "missing_rgb_sample_ids": color_missing,
        "vertex_count": {
            "min": int(np.min(vertex_counts)),
            "median": float(np.median(vertex_counts)),
            "p95": float(np.percentile(vertex_counts, 95)),
            "max": int(np.max(vertex_counts)),
        },
        "landmark_to_surface_mm": summarize(surface_distances),
        "bilateral_x_order": {
            pair: {
                "positive_fraction": float(np.mean(np.asarray(signs) > 0)),
                "negative_fraction": float(np.mean(np.asarray(signs) < 0)),
                "zero_fraction": float(np.mean(np.asarray(signs) == 0)),
            }
            for pair, signs in pair_signs.items()
        },
        "reused_gender_subject_keys": {
            key: value for key, value in subject_keys.items() if len(value) > 1
        },
        "split_counts": split_report,
        "note": "Repeated gender_subject numbers across classes are reported, not assumed to be the same patient.",
    }
    write_csv(output_dir / "dataset_audit_samples.csv", rows)
    (output_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
