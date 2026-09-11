"""Validated curve annotations in raw-mesh millimetre coordinates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from all23_rgb_geodesic_cascade.alignment import apply_transform


CURVE_KEYS = ("hairline", "jaw_left", "jaw_right")
LANDMARK_FOR_CURVE = {"hairline": 0, "jaw_left": 21, "jaw_right": 22}


def _point_rows(values, label, minimum):
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 3 or len(rows) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} XYZ points")
    if not np.isfinite(rows).all():
        raise ValueError(f"{label} contains non-finite coordinates")
    return rows


@dataclass(frozen=True)
class CurveAnnotation:
    curves: dict[str, np.ndarray]
    repeats: dict[int, np.ndarray]


class CurveAnnotationStore:
    """Load optional publication annotations without exposing evaluation labels."""

    version = 1

    def __init__(self, samples=None, source_path=None, source_hash="none"):
        self.samples = dict(samples or {})
        self.source_path = Path(source_path) if source_path else None
        self.source_hash = str(source_hash)

    @classmethod
    def empty(cls):
        return cls()

    @classmethod
    def load(cls, path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != cls.version:
            raise ValueError(
                f"Unsupported curve annotation version: {payload.get('version')}"
            )
        if payload.get("coordinate_space") != "raw_mesh_mm":
            raise ValueError("curve annotations must use raw_mesh_mm coordinates")
        raw_samples = payload.get("samples")
        if not isinstance(raw_samples, dict):
            raise ValueError("curve annotation manifest requires a samples object")
        parsed = {}
        for sample_id, values in raw_samples.items():
            curves = {}
            for key, points in values.get("curves", {}).items():
                if key not in CURVE_KEYS:
                    raise ValueError(f"Unknown curve key for {sample_id}: {key}")
                curves[key] = _point_rows(points, f"{sample_id}.{key}", 2)
            repeats = {}
            for landmark, points in values.get("repeat_landmarks", {}).items():
                index = int(landmark)
                if index not in (0, 21, 22):
                    raise ValueError(
                        f"Repeat landmark for {sample_id} must be LM0, LM21 or LM22"
                    )
                repeats[index] = _point_rows(
                    points, f"{sample_id}.repeat_landmarks.{index}", 1
                )
            parsed[str(sample_id)] = CurveAnnotation(curves, repeats)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(parsed, path, digest)

    def transformed(self, sample_id, matrix):
        annotation = self.samples.get(str(sample_id))
        if annotation is None:
            return CurveAnnotation({}, {})
        return CurveAnnotation(
            {
                key: apply_transform(points, matrix).astype(np.float32)
                for key, points in annotation.curves.items()
            },
            {
                landmark: apply_transform(points, matrix).astype(np.float32)
                for landmark, points in annotation.repeats.items()
            },
        )

    def subset(self, sample_ids):
        selected = {
            str(sample_id): self.samples[str(sample_id)]
            for sample_id in sample_ids
            if str(sample_id) in self.samples
        }
        payload = {
            sample_id: {
                "curves": {
                    key: points.tolist()
                    for key, points in sorted(annotation.curves.items())
                },
                "repeat_landmarks": {
                    str(index): points.tolist()
                    for index, points in sorted(annotation.repeats.items())
                },
            }
            for sample_id, annotation in sorted(selected.items())
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return CurveAnnotationStore(selected, self.source_path, digest)

    def coverage(self, sample_ids):
        ids = [str(value) for value in sample_ids]
        by_curve = {
            key: sum(
                key in self.samples.get(sample_id, CurveAnnotation({}, {})).curves
                for sample_id in ids
            )
            for key in CURVE_KEYS
        }
        repeat_samples = sum(
            bool(self.samples.get(sample_id, CurveAnnotation({}, {})).repeats)
            for sample_id in ids
        )
        return {
            "sample_count": len(ids),
            "annotated_by_curve": by_curve,
            "fully_curve_annotated": sum(
                all(
                    key in self.samples.get(sample_id, CurveAnnotation({}, {})).curves
                    for key in CURVE_KEYS
                )
                for sample_id in ids
            ),
            "repeat_annotated_samples": repeat_samples,
        }


def blank_manifest(sample_ids):
    return {
        "version": CurveAnnotationStore.version,
        "coordinate_space": "raw_mesh_mm",
        "description": (
            "Polylines are ordered XYZ samples in the untransformed PLY coordinate "
            "system. Evaluation-fold curves must remain empty during development."
        ),
        "samples": {
            str(sample_id): {"curves": {}, "repeat_landmarks": {}}
            for sample_id in sample_ids
        },
    }
