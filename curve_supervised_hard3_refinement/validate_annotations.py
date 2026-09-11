#!/usr/bin/env python3
"""Validate Curve-H3 polylines against their raw meshes and train landmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from all23_rgb_geodesic_cascade.data import (
    discover_samples,
    load_mesh,
    read_landmarks,
)

from curve_supervised_hard3_refinement.annotations import (
    CURVE_KEYS,
    LANDMARK_FOR_CURVE,
    CurveAnnotationStore,
)
from curve_supervised_hard3_refinement.targets import point_to_polyline_distance


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def validate_annotation_store(
    samples,
    store,
    minimum_annotated_samples=0,
    maximum_surface_distance_mm=5.0,
    maximum_landmark_distance_mm=5.0,
    recommended_segment_length_mm=8.0,
):
    sample_by_id = {sample.sample_id: sample for sample in samples}
    unknown = sorted(set(store.samples) - set(sample_by_id))
    rows, surface_distances, landmark_distances, segment_lengths = [], [], [], []
    fully_annotated = 0
    for sample_id, annotation in sorted(store.samples.items()):
        if sample_id not in sample_by_id or not annotation.curves:
            continue
        sample = sample_by_id[sample_id]
        vertices = np.asarray(load_mesh(sample.mesh_path).vertices, dtype=np.float32)
        tree = cKDTree(vertices)
        landmarks = read_landmarks(sample.landmark_path)
        complete = all(key in annotation.curves for key in CURVE_KEYS)
        fully_annotated += int(complete)
        for curve_key, points in sorted(annotation.curves.items()):
            nearest = np.asarray(tree.query(points, k=1)[0], dtype=np.float64)
            landmark = LANDMARK_FOR_CURVE[curve_key]
            landmark_distance = float(
                point_to_polyline_distance(landmarks[landmark][None], points)[0]
            )
            segments = np.linalg.norm(np.diff(points, axis=0), axis=-1).astype(
                np.float64
            )
            surface_distances.extend(nearest.tolist())
            landmark_distances.append(landmark_distance)
            segment_lengths.extend(segments.tolist())
            rows.append(
                {
                    "sample_id": sample_id,
                    "curve": curve_key,
                    "point_count": int(len(points)),
                    "surface_distance_mm": _summary(nearest),
                    "landmark_to_curve_mm": landmark_distance,
                    "segment_length_mm": _summary(segments),
                }
            )

    surface = _summary(surface_distances)
    landmark = _summary(landmark_distances)
    segments = _summary(segment_lengths)
    failures = []
    warnings = []
    if unknown:
        failures.append(f"manifest contains {len(unknown)} unknown sample IDs")
    if fully_annotated < int(minimum_annotated_samples):
        failures.append(
            f"fully annotated samples {fully_annotated} < required "
            f"{int(minimum_annotated_samples)}"
        )
    if surface and surface["max"] > float(maximum_surface_distance_mm):
        failures.append(
            f"maximum curve-to-mesh distance {surface['max']:.3f} mm exceeds "
            f"{float(maximum_surface_distance_mm):.3f} mm"
        )
    if landmark and landmark["max"] > float(maximum_landmark_distance_mm):
        failures.append(
            f"maximum landmark-to-curve distance {landmark['max']:.3f} mm exceeds "
            f"{float(maximum_landmark_distance_mm):.3f} mm"
        )
    if segments and segments["p95"] > float(recommended_segment_length_mm):
        warnings.append(
            f"curve segment p95 {segments['p95']:.3f} mm exceeds the recommended "
            f"{float(recommended_segment_length_mm):.3f} mm sampling interval"
        )
    return {
        "version": "Curve-H3-annotation-QA-v1",
        "manifest": str(store.source_path) if store.source_path else None,
        "dataset_samples": len(samples),
        "manifest_samples": len(store.samples),
        "fully_annotated_samples": int(fully_annotated),
        "curve_rows": len(rows),
        "surface_distance_mm": surface,
        "landmark_to_curve_mm": landmark,
        "segment_length_mm": segments,
        "unknown_sample_ids": unknown,
        "failures": failures,
        "warnings": warnings,
        "passed": not failures,
        "rows": rows,
    }


def validate_manifest(
    data_root,
    manifest_path,
    minimum_annotated_samples=0,
    maximum_surface_distance_mm=5.0,
    maximum_landmark_distance_mm=5.0,
    recommended_segment_length_mm=8.0,
):
    return validate_annotation_store(
        discover_samples(data_root),
        CurveAnnotationStore.load(manifest_path),
        minimum_annotated_samples,
        maximum_surface_distance_mm,
        maximum_landmark_distance_mm,
        recommended_segment_length_mm,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--minimum-annotated-samples", type=int, default=0)
    parser.add_argument("--maximum-surface-distance-mm", type=float, default=5.0)
    parser.add_argument("--maximum-landmark-distance-mm", type=float, default=5.0)
    parser.add_argument("--recommended-segment-length-mm", type=float, default=8.0)
    args = parser.parse_args()
    report = validate_manifest(
        args.data_root,
        args.manifest,
        args.minimum_annotated_samples,
        args.maximum_surface_distance_mm,
        args.maximum_landmark_distance_mm,
        args.recommended_segment_length_mm,
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.manifest).with_name(f"{Path(args.manifest).stem}_qa.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Curve QA: complete={report['fully_annotated_samples']} "
        f"passed={report['passed']} report={output}",
        flush=True,
    )
    if report["surface_distance_mm"]:
        print(
            "Curve-to-mesh: "
            f"mean={report['surface_distance_mm']['mean']:.3f} "
            f"p95={report['surface_distance_mm']['p95']:.3f} "
            f"max={report['surface_distance_mm']['max']:.3f} mm",
            flush=True,
        )
    if report["landmark_to_curve_mm"]:
        print(
            "Landmark-to-curve: "
            f"mean={report['landmark_to_curve_mm']['mean']:.3f} "
            f"p95={report['landmark_to_curve_mm']['p95']:.3f} "
            f"max={report['landmark_to_curve_mm']['max']:.3f} mm",
            flush=True,
        )
    for warning in report["warnings"]:
        print(f"Warning: {warning}", flush=True)
    if report["failures"]:
        raise RuntimeError("; ".join(report["failures"]))


if __name__ == "__main__":
    main()
