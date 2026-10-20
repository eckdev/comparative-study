"""Procrustes-aligned statistical shape prior fitted on outer-train shapes.

The vNext prior (agh_former_vnext_orthodontic_comparison/shape_prior.py) ran
PCA on absolute scanner coordinates, so residual head pose was mixed into the
shape modes. Here every training shape is rigidly aligned to a generalized
Procrustes mean first; each prediction is aligned with its Core20 points,
projected, and mapped back with the inverse rigid transform. Scale is not
removed, matching the millimetre-scale evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, NUM_LANDMARKS


def kabsch(source, target):
    """Rotation R and translation t minimizing ||source @ R + t - target||."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    translation = target_center - source_center @ rotation
    return rotation, translation


def generalized_procrustes(shapes, iterations=10, anchor_indices=None):
    shapes = np.asarray(shapes, dtype=np.float64)
    anchors = list(anchor_indices if anchor_indices is not None else range(shapes.shape[1]))
    mean = shapes[0].copy()
    aligned = shapes.copy()
    for _ in range(int(iterations)):
        for index, shape in enumerate(shapes):
            rotation, translation = kabsch(shape[anchors], mean[anchors])
            aligned[index] = shape @ rotation + translation
        updated = aligned.mean(axis=0)
        updated -= updated[anchors].mean(axis=0)
        if np.abs(updated - mean).max() < 1e-6:
            mean = updated
            break
        mean = updated
    return aligned, mean


class ProcrustesShapePrior:
    version = 1

    def __init__(
        self,
        component_grid=(6, 12, 20),
        l2_grid=(1.0, 10.0, 100.0),
        alpha_core_grid=(0.0, 0.15, 0.3),
        alpha_hard_grid=(0.0, 0.25, 0.5, 0.75),
    ):
        self.component_grid = tuple(int(value) for value in component_grid)
        self.l2_grid = tuple(float(value) for value in l2_grid)
        self.alpha_core_grid = tuple(float(value) for value in alpha_core_grid)
        self.alpha_hard_grid = tuple(float(value) for value in alpha_hard_grid)
        self.config = None
        self.fitted = False

    def fit(self, train_shapes, fit_sample_ids=None):
        shapes = np.asarray(train_shapes, dtype=np.float64)
        if shapes.ndim != 3 or shapes.shape[1:] != (NUM_LANDMARKS, 3):
            raise ValueError("train_shapes must have shape [N, 23, 3]")
        aligned, self.mean_shape = generalized_procrustes(shapes, anchor_indices=CORE20)
        flat = aligned.reshape(len(aligned), -1)
        self.mean = flat.mean(axis=0)
        centered = flat - self.mean
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        self.components = vh
        self.core_columns = np.asarray(
            [3 * landmark + axis for landmark in CORE20 for axis in range(3)], dtype=np.int64
        )
        self.hard_columns = np.asarray(
            [3 * landmark + axis for landmark in HARD3 for axis in range(3)], dtype=np.int64
        )
        design = np.concatenate(
            [np.ones((len(centered), 1)), centered[:, self.core_columns]], axis=1
        )
        self.conditional_weights = {}
        for l2 in self.l2_grid:
            regularizer = float(l2) * np.eye(design.shape[1])
            regularizer[0, 0] = 0.0
            self.conditional_weights[float(l2)] = np.linalg.solve(
                design.T @ design + regularizer, design.T @ centered[:, self.hard_columns]
            )
        self.fit_sample_ids = list(fit_sample_ids or [])
        self.fitted = True
        return self

    def _align(self, prediction):
        rotations, translations, aligned = [], [], []
        for shape in np.asarray(prediction, dtype=np.float64):
            rotation, translation = kabsch(shape[list(CORE20)], self.mean_shape[list(CORE20)])
            rotations.append(rotation)
            translations.append(translation)
            aligned.append(shape @ rotation + translation)
        return np.stack(aligned), np.stack(rotations), np.stack(translations)

    def _candidate(self, prediction, components, l2, alpha_core, alpha_hard):
        prediction = np.asarray(prediction, dtype=np.float64)
        aligned, rotations, translations = self._align(prediction)
        centered = aligned.reshape(len(aligned), -1) - self.mean
        basis = self.components[: min(int(components), len(self.components))]
        projected = (centered @ basis.T) @ basis
        design = np.concatenate(
            [np.ones((len(projected), 1)), projected[:, self.core_columns]], axis=1
        )
        projected[:, self.hard_columns] = design @ self.conditional_weights[float(l2)]
        prior_aligned = (projected + self.mean).reshape(-1, NUM_LANDMARKS, 3)
        # Inverse rigid transform: x = (y - t) @ R^T.
        prior = np.einsum(
            "nlj,nkj->nlk", prior_aligned - translations[:, None], rotations
        )
        result = prediction.copy()
        result[:, CORE20] += float(alpha_core) * (prior[:, CORE20] - result[:, CORE20])
        result[:, HARD3] += float(alpha_hard) * (prior[:, HARD3] - result[:, HARD3])
        return result.astype(np.float32)

    def calibrate(self, prediction, expert, sample_ids=None, max_core_regression_mm=0.03):
        if not self.fitted:
            raise RuntimeError("fit must be called before calibrate")
        prediction = np.asarray(prediction, dtype=np.float64)
        expert = np.asarray(expert, dtype=np.float64)
        base_core = float(np.linalg.norm(prediction - expert, axis=-1)[:, CORE20].mean())
        rows = []
        for components in sorted({min(value, len(self.components)) for value in self.component_grid}):
            for l2 in self.l2_grid:
                for alpha_core in self.alpha_core_grid:
                    for alpha_hard in self.alpha_hard_grid:
                        candidate = self._candidate(prediction, components, l2, alpha_core, alpha_hard)
                        errors = np.linalg.norm(candidate - expert, axis=-1)
                        core = float(errors[:, CORE20].mean())
                        rows.append(
                            {
                                "components": int(components),
                                "l2": float(l2),
                                "alpha_core": float(alpha_core),
                                "alpha_hard": float(alpha_hard),
                                "overall_ale": float(errors.mean()),
                                "core20_ale": core,
                                "hard3_ale": float(errors[:, HARD3].mean()),
                                "p95": float(np.percentile(errors, 95)),
                                "core_constraint_passed": bool(
                                    core <= base_core + float(max_core_regression_mm)
                                ),
                            }
                        )
        eligible = [row for row in rows if row["core_constraint_passed"]] or rows
        self.config = min(
            eligible,
            key=lambda row: (
                row["overall_ale"],
                row["alpha_core"] + row["alpha_hard"],
                row["components"],
            ),
        )
        self.validation_sample_ids = list(sample_ids or [])
        self.calibration_rows = rows
        return dict(self.config)

    def reject(self):
        self.config = {
            **(self.config or {}),
            "components": self.component_grid[0],
            "l2": self.l2_grid[0],
            "alpha_core": 0.0,
            "alpha_hard": 0.0,
            "validation_rejected": True,
        }

    def transform(self, prediction):
        if self.config is None:
            raise RuntimeError("calibrate must be called before transform")
        return self._candidate(
            prediction,
            self.config["components"],
            self.config["l2"],
            self.config["alpha_core"],
            self.config["alpha_hard"],
        )

    def report(self):
        return {
            "version": self.version,
            "method": "generalized Procrustes (rigid, Core20 anchors) + PCA + Core20-to-Hard3 ridge",
            "uses_test_labels": False,
            "fit_sample_ids": self.fit_sample_ids,
            "validation_sample_ids": getattr(self, "validation_sample_ids", []),
            "selected": self.config,
            "sweep": getattr(self, "calibration_rows", []),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
