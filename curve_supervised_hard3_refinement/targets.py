"""Curve-distance targets for annotated and point-supervised development runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from all23_rgb_geodesic_cascade.anatomy import HARD3

from .annotations import CURVE_KEYS, CurveAnnotationStore


@dataclass(frozen=True)
class CurveTargets:
    curve_distance: np.ndarray
    point_distance: np.ndarray
    source: np.ndarray
    annotation_report: dict


def point_to_polyline_distance(points, polyline):
    points = np.asarray(points, dtype=np.float32)
    polyline = np.asarray(polyline, dtype=np.float32)
    starts = polyline[:-1]
    segments = polyline[1:] - starts
    denominator = np.maximum(np.sum(segments * segments, axis=-1), 1e-8)
    relative = points[:, None] - starts[None]
    fraction = np.clip(
        np.sum(relative * segments[None], axis=-1) / denominator[None], 0.0, 1.0
    )
    projection = starts[None] + fraction[..., None] * segments[None]
    return np.linalg.norm(points[:, None] - projection, axis=-1).min(axis=1)


def _principal_curve_distance(points, expert, landmark, radius_mm, half_length_mm):
    """Build a local tangent tube for smoke/ablation runs without curve labels."""
    points = np.asarray(points, dtype=np.float32)
    expert = np.asarray(expert, dtype=np.float32)
    delta = points - expert
    radial = np.linalg.norm(delta, axis=-1)
    local = delta[radial <= float(radius_mm)]
    if len(local) < 6:
        local = delta[np.argsort(radial)[: min(24, len(points))]]
    covariance = local.T @ local / max(len(local), 1)
    _, vectors = np.linalg.eigh(covariance)
    tangent_candidates = vectors[:, -2:]
    # Hairline is predominantly lateral. Gonion follows the inferior/profile
    # contour, so the tangent with the strongest vertical-depth component wins.
    preferred = np.asarray(
        [1.0, 0.0, 0.0] if landmark == 0 else [0.0, 1.0, 1.0],
        dtype=np.float32,
    )
    preferred /= np.linalg.norm(preferred)
    tangent = tangent_candidates[:, np.argmax(np.abs(tangent_candidates.T @ preferred))]
    projection = delta @ tangent
    clipped = np.clip(projection, -float(half_length_mm), float(half_length_mm))
    closest = clipped[:, None] * tangent[None]
    return np.linalg.norm(delta - closest, axis=-1).astype(np.float32)


def _repeat_distance(points, expert, repeats):
    targets = [np.asarray(expert, dtype=np.float32)]
    if repeats is not None and len(repeats):
        targets.extend(np.asarray(repeats, dtype=np.float32))
    target = np.stack(targets)
    # A mixture target retains disagreement instead of collapsing repeated
    # annotations to an artificial mean that may lie away from the surface.
    return np.linalg.norm(points[:, None] - target[None], axis=-1).min(axis=1)


def build_curve_targets(
    dataset,
    candidate_set,
    annotation_store: CurveAnnotationStore,
    allow_pseudo_curves=True,
    pseudo_radius_mm=12.0,
    pseudo_half_length_mm=20.0,
):
    sample_lookup = {sample.sample_id: sample for sample in dataset.samples}
    sample_rows = []
    point_rows = []
    source_rows = []
    for row, sample_id in enumerate(candidate_set.sample_ids):
        sample = sample_lookup[sample_id]
        annotation = annotation_store.transformed(
            sample_id, dataset.transforms[sample_id]
        )
        curves, point_distances, sources = [], [], []
        for local_index, (landmark, curve_key) in enumerate(zip(HARD3, CURVE_KEYS)):
            points = candidate_set.points[row, local_index]
            valid = candidate_set.mask[row, local_index]
            if curve_key in annotation.curves:
                curve_distance = point_to_polyline_distance(
                    points, annotation.curves[curve_key]
                )
                source = 1
            elif allow_pseudo_curves:
                curve_distance = _principal_curve_distance(
                    points,
                    candidate_set.expert[row, local_index],
                    landmark,
                    pseudo_radius_mm,
                    pseudo_half_length_mm,
                )
                source = 0
            else:
                raise ValueError(
                    f"Missing {curve_key} annotation for training sample {sample_id}"
                )
            repeat = annotation.repeats.get(landmark)
            point_distance = _repeat_distance(
                points, candidate_set.expert[row, local_index], repeat
            )
            curve_distance = np.asarray(curve_distance, dtype=np.float32)
            point_distance = np.asarray(point_distance, dtype=np.float32)
            curve_distance[~valid] = np.inf
            point_distance[~valid] = np.inf
            curves.append(curve_distance)
            point_distances.append(point_distance)
            sources.append(source)
        sample_rows.append(curves)
        point_rows.append(point_distances)
        source_rows.append(sources)
    coverage = annotation_store.coverage(candidate_set.sample_ids)
    source = np.asarray(source_rows, dtype=np.int8)
    coverage.update(
        {
            "real_curve_targets": int(source.sum()),
            "pseudo_curve_targets": int(source.size - source.sum()),
            "publication_ready": bool(np.all(source == 1)),
            "annotation_manifest": (
                str(annotation_store.source_path)
                if annotation_store.source_path is not None
                else None
            ),
            "annotation_sha256": annotation_store.source_hash,
        }
    )
    return CurveTargets(
        np.asarray(sample_rows, dtype=np.float32),
        np.asarray(point_rows, dtype=np.float32),
        source,
        coverage,
    )
