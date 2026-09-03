import json
from pathlib import Path

import numpy as np

from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, NUM_LANDMARKS


class TrainOnlyShapePrior:
    """Statistical shape projection fitted only from outer-train expert shapes."""

    def __init__(self, component_grid=(6, 10, 15, 20, 30), l2_grid=(0.1, 1.0, 10.0, 100.0)):
        self.component_grid = tuple(int(value) for value in component_grid)
        self.l2_grid = tuple(float(value) for value in l2_grid)
        self.fitted = False
        self.config = None

    def fit(self, train_shapes, fit_sample_ids=None):
        shapes = np.asarray(train_shapes, dtype=np.float64)
        if shapes.ndim != 3 or shapes.shape[1:] != (NUM_LANDMARKS, 3):
            raise ValueError("train_shapes must have shape [N, 23, 3]")
        flat = shapes.reshape(len(shapes), -1)
        self.mean = flat.mean(axis=0)
        self.std = np.maximum(flat.std(axis=0), 1.0)
        normalized = (flat - self.mean) / self.std
        _, _, vh = np.linalg.svd(normalized, full_matrices=False)
        self.components = vh.astype(np.float64)

        core_columns = np.asarray(
            [3 * landmark + axis for landmark in CORE20 for axis in range(3)],
            dtype=np.int64,
        )
        hard_columns = np.asarray(
            [3 * landmark + axis for landmark in HARD3 for axis in range(3)],
            dtype=np.int64,
        )
        self.core_columns = core_columns
        self.hard_columns = hard_columns
        core = normalized[:, core_columns]
        hard = normalized[:, hard_columns]
        design = np.concatenate([np.ones((len(core), 1)), core], axis=1)
        self.conditional_weights = {}
        for l2 in self.l2_grid:
            regularizer = float(l2) * np.eye(design.shape[1])
            regularizer[0, 0] = 0.0
            weights = np.linalg.solve(
                design.T @ design + regularizer,
                design.T @ hard,
            )
            self.conditional_weights[float(l2)] = weights
        self.fit_sample_ids = list(fit_sample_ids or [])
        self.fitted = True
        return self

    def _normalized(self, prediction):
        values = np.asarray(prediction, dtype=np.float64).reshape(len(prediction), -1)
        return (values - self.mean) / self.std

    def _pca_projection(self, normalized, components):
        count = min(int(components), len(self.components))
        basis = self.components[:count]
        return (normalized @ basis.T) @ basis

    def _conditional_projection(self, normalized, l2):
        output = normalized.copy()
        design = np.concatenate(
            [np.ones((len(normalized), 1)), normalized[:, self.core_columns]], axis=1
        )
        output[:, self.hard_columns] = design @ self.conditional_weights[float(l2)]
        return output

    def _candidate(self, prediction, components, l2, alpha_core, alpha_hard):
        normalized = self._normalized(prediction)
        projected = self._pca_projection(normalized, components)
        projected = self._conditional_projection(projected, l2)
        prior = (projected * self.std + self.mean).reshape(-1, NUM_LANDMARKS, 3)
        result = np.asarray(prediction, dtype=np.float64).copy()
        result[:, CORE20] += float(alpha_core) * (
            prior[:, CORE20] - result[:, CORE20]
        )
        result[:, HARD3] += float(alpha_hard) * (
            prior[:, HARD3] - result[:, HARD3]
        )
        return result.astype(np.float32)

    def calibrate(
        self,
        validation_prediction,
        validation_expert,
        validation_sample_ids=None,
        max_core_regression_mm=0.03,
    ):
        if not self.fitted:
            raise RuntimeError("fit must be called before calibrate")
        prediction = np.asarray(validation_prediction, dtype=np.float64)
        expert = np.asarray(validation_expert, dtype=np.float64)
        base_errors = np.linalg.norm(prediction - expert, axis=-1)
        base_core = float(base_errors[:, CORE20].mean())
        component_values = sorted(
            {min(value, len(self.components)) for value in self.component_grid if value > 0}
        )
        alpha_core_values = (0.0, 0.1, 0.2, 0.3, 0.5)
        alpha_hard_values = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
        rows = []
        for components in component_values:
            for l2 in self.l2_grid:
                for alpha_core in alpha_core_values:
                    for alpha_hard in alpha_hard_values:
                        candidate = self._candidate(
                            prediction,
                            components,
                            l2,
                            alpha_core,
                            alpha_hard,
                        )
                        errors = np.linalg.norm(candidate - expert, axis=-1)
                        core = float(errors[:, CORE20].mean())
                        row = {
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
                        rows.append(row)
        eligible = [row for row in rows if row["core_constraint_passed"]]
        if not eligible:
            eligible = rows
        self.config = min(
            eligible,
            key=lambda row: (row["overall_ale"], row["p95"], row["hard3_ale"]),
        )
        self.validation_sample_ids = list(validation_sample_ids or [])
        self.calibration_rows = rows
        return dict(self.config)

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
            "method": "outer-train statistical PCA plus Core20-to-Hard3 conditional ridge",
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
