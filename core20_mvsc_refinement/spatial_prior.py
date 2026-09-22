"""Leakage-free leave-one-landmark-out conditional spatial prior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .anatomy import CORE20_INDICES


@dataclass
class SpatialPriorConfig:
    folds: int = 5
    l2_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    covariance_shrinkage: float = 0.25
    covariance_floor_mm: float = 0.25
    seed: int = 42


def _folds_from_strata(strata, folds, seed):
    """Deterministic stratified folds which also work for tiny smoke datasets."""
    strata = np.asarray(strata, dtype=object)
    count = max(2, min(int(folds), len(strata)))
    assignments = [[] for _ in range(count)]
    rng = np.random.default_rng(seed)
    cursor = 0
    for value in sorted(set(strata.tolist())):
        indices = np.flatnonzero(strata == value)
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            assignments[(cursor + offset) % count].append(int(index))
        cursor = (cursor + len(indices)) % count
    # Empty folds are possible when a very small smoke sample has many strata.
    return [np.asarray(rows, dtype=np.int64) for rows in assignments if rows]


def _ridge_fit(values, targets, l2):
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    penalty = np.eye(values.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    system = values.T @ values + penalty
    right = values.T @ targets
    try:
        return np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(system) @ right


class ConditionalSpatialPrior:
    """Predict each Core20 point from the other 19 points and metadata.

    The outer-training sample itself is excluded when its OOF prior mean is
    generated. Full-fit parameters are used only for validation/test inference.
    """

    version = 2

    def __init__(self, config=None):
        self.config = config or SpatialPriorConfig()
        self.parameters = {}
        self.fit_sample_ids = []
        self.class_values = []
        self.gender_values = []
        self.selected_l2 = {}
        self.oof_means = {}
        self.crossfit_parameters = {}
        self.oof_ale = None

    def _raw_design(self, shapes, target_landmark, classes, genders):
        shapes = np.asarray(shapes, dtype=np.float64)
        core = shapes[:, CORE20_INDICES]
        local_target = CORE20_INDICES.index(int(target_landmark))
        keep = [index for index in range(len(CORE20_INDICES)) if index != local_target]
        context = core[:, keep]
        origin = context.mean(axis=1)
        centered = (context - origin[:, None]).reshape(len(shapes), -1)
        class_features = np.stack(
            [np.asarray(classes) == value for value in self.class_values], axis=1
        ).astype(np.float64)
        gender_features = np.stack(
            [np.asarray(genders) == value for value in self.gender_values], axis=1
        ).astype(np.float64)
        design = np.concatenate([centered, class_features, gender_features], axis=1)
        target = shapes[:, int(target_landmark)] - origin
        return design, target, origin

    @staticmethod
    def _normalize(values, mean=None, scale=None):
        values = np.asarray(values, dtype=np.float64)
        if mean is None:
            mean = values.mean(axis=0)
        if scale is None:
            scale = values.std(axis=0)
        scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-6)
        normalized = (values - mean) / scale
        return (
            np.concatenate([np.ones((len(values), 1)), normalized], axis=1),
            mean,
            scale,
        )

    def fit(self, shapes, sample_ids, classes, genders):
        shapes = np.asarray(shapes, dtype=np.float64)
        if shapes.ndim != 3 or shapes.shape[1:] != (23, 3):
            raise ValueError("Spatial prior expects shapes with [N,23,3]")
        if len(shapes) < 2:
            raise ValueError(
                "Spatial prior requires at least two outer-training samples"
            )
        self.fit_sample_ids = list(sample_ids)
        self.class_values = sorted(set(map(str, classes)))
        self.gender_values = sorted(set(map(str, genders)))
        strata = [
            f"{class_name}|{gender}" for class_name, gender in zip(classes, genders)
        ]
        fold_rows = _folds_from_strata(strata, self.config.folds, self.config.seed)
        all_indices = np.arange(len(shapes), dtype=np.int64)
        oof_shape = np.zeros((len(shapes), len(CORE20_INDICES), 3), dtype=np.float64)

        for local_index, landmark in enumerate(CORE20_INDICES):
            raw, target, origins = self._raw_design(shapes, landmark, classes, genders)
            sweep = []
            for l2 in self.config.l2_grid:
                prediction = np.zeros_like(target)
                covered = np.zeros(len(shapes), dtype=np.bool_)
                for validation_index in fold_rows:
                    train_index = np.setdiff1d(
                        all_indices, validation_index, assume_unique=True
                    )
                    if len(train_index) == 0:
                        continue
                    x_train, mean, scale = self._normalize(raw[train_index])
                    coefficients = _ridge_fit(x_train, target[train_index], l2)
                    x_validation, _, _ = self._normalize(
                        raw[validation_index], mean, scale
                    )
                    prediction[validation_index] = x_validation @ coefficients
                    covered[validation_index] = True
                if not covered.all():
                    prediction[~covered] = target[~covered].mean(axis=0)
                ale = float(np.linalg.norm(prediction - target, axis=-1).mean())
                sweep.append((ale, float(l2), prediction))
            _, selected_l2, selected_prediction = min(
                sweep, key=lambda row: (row[0], row[1])
            )
            self.selected_l2[str(landmark)] = selected_l2
            oof_shape[:, local_index] = selected_prediction + origins

            crossfit_rows = []
            for validation_index in fold_rows:
                train_index = np.setdiff1d(
                    all_indices, validation_index, assume_unique=True
                )
                if len(train_index) == 0:
                    continue
                x_train, fold_mean, fold_scale = self._normalize(raw[train_index])
                fold_coefficients = _ridge_fit(
                    x_train, target[train_index], selected_l2
                )
                crossfit_rows.append(
                    {
                        "validation_sample_ids": [
                            self.fit_sample_ids[index] for index in validation_index
                        ],
                        "mean": fold_mean,
                        "scale": fold_scale,
                        "coefficients": fold_coefficients,
                    }
                )
            self.crossfit_parameters[str(landmark)] = crossfit_rows

            x_full, mean, scale = self._normalize(raw)
            coefficients = _ridge_fit(x_full, target, selected_l2)
            # OOF residuals avoid an unrealistically narrow prior covariance.
            residual = target - selected_prediction
            if len(residual) > 3:
                covariance = np.cov(residual, rowvar=False)
            else:
                covariance = np.eye(3, dtype=np.float64)
            covariance = np.atleast_2d(covariance).astype(np.float64)
            diagonal = np.diag(np.diag(covariance))
            amount = float(self.config.covariance_shrinkage)
            covariance = (1.0 - amount) * covariance + amount * diagonal
            covariance += np.eye(3) * float(self.config.covariance_floor_mm) ** 2
            self.parameters[str(landmark)] = {
                "mean": mean,
                "scale": scale,
                "coefficients": coefficients,
                "covariance": covariance,
            }

        self.oof_means = {
            sample_id: oof_shape[index].astype(np.float32)
            for index, sample_id in enumerate(self.fit_sample_ids)
        }
        expert_core = shapes[:, CORE20_INDICES]
        self.oof_ale = float(np.linalg.norm(oof_shape - expert_core, axis=-1).mean())
        return self

    def oof_predict_from_shapes(self, shapes, sample_ids, classes, genders):
        """Apply each sample's held-out prior to noisy/coarse shape context."""
        shapes = np.asarray(shapes, dtype=np.float64)
        sample_ids = list(sample_ids)
        if set(sample_ids) != set(self.fit_sample_ids):
            missing = sorted(set(self.fit_sample_ids) - set(sample_ids))
            extra = sorted(set(sample_ids) - set(self.fit_sample_ids))
            raise ValueError(
                "OOF contextual prediction requires the fitted outer-train samples; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        output = np.zeros((len(shapes), len(CORE20_INDICES), 3), dtype=np.float64)
        covered = np.zeros((len(shapes), len(CORE20_INDICES)), dtype=np.bool_)
        for local_index, landmark in enumerate(CORE20_INDICES):
            raw, _, origins = self._raw_design(shapes, landmark, classes, genders)
            for parameters in self.crossfit_parameters[str(landmark)]:
                rows = np.asarray(
                    [
                        lookup[sample_id]
                        for sample_id in parameters["validation_sample_ids"]
                    ],
                    dtype=np.int64,
                )
                design, _, _ = self._normalize(
                    raw[rows], parameters["mean"], parameters["scale"]
                )
                output[rows, local_index] = (
                    design @ parameters["coefficients"] + origins[rows]
                )
                covered[rows, local_index] = True
        if not covered.all():
            missing = np.argwhere(~covered)
            raise RuntimeError(
                f"Spatial-prior crossfit coverage is incomplete: {missing[:5].tolist()}"
            )
        return output.astype(np.float32)

    def predict(self, shapes, classes, genders):
        shapes = np.asarray(shapes, dtype=np.float64)
        means, covariances = [], []
        for landmark in CORE20_INDICES:
            raw, _, origins = self._raw_design(shapes, landmark, classes, genders)
            parameters = self.parameters[str(landmark)]
            design, _, _ = self._normalize(raw, parameters["mean"], parameters["scale"])
            relative = design @ parameters["coefficients"]
            means.append(relative + origins)
            covariances.append(
                np.repeat(parameters["covariance"][None], len(shapes), axis=0)
            )
        return (
            np.stack(means, axis=1).astype(np.float32),
            np.stack(covariances, axis=1).astype(np.float32),
        )

    def oof_prediction(self, sample_ids):
        missing = [
            sample_id for sample_id in sample_ids if sample_id not in self.oof_means
        ]
        if missing:
            raise KeyError(f"Spatial-prior OOF means miss samples: {missing[:5]}")
        return np.stack([self.oof_means[sample_id] for sample_id in sample_ids])

    def report(self):
        return {
            "version": self.version,
            "method": "leave_one_landmark_out_conditional_gaussian_ridge",
            "folds": int(self.config.folds),
            "l2_grid": list(map(float, self.config.l2_grid)),
            "selected_l2": self.selected_l2,
            "oof_ale_mm": self.oof_ale,
            "fit_sample_ids": list(self.fit_sample_ids),
            "class_values": self.class_values,
            "gender_values": self.gender_values,
            "uses_validation_labels": False,
            "uses_test_labels": False,
            "training_context": "cross_fitted_model_prediction",
        }

    def save(self, path):
        payload = {
            **self.report(),
            "config": {
                "folds": self.config.folds,
                "l2_grid": list(self.config.l2_grid),
                "covariance_shrinkage": self.config.covariance_shrinkage,
                "covariance_floor_mm": self.config.covariance_floor_mm,
                "seed": self.config.seed,
            },
            "parameters": {
                landmark: {
                    name: np.asarray(value).tolist() for name, value in rows.items()
                }
                for landmark, rows in self.parameters.items()
            },
            "crossfit_parameters": {
                landmark: [
                    {
                        name: (
                            value
                            if name == "validation_sample_ids"
                            else np.asarray(value).tolist()
                        )
                        for name, value in row.items()
                    }
                    for row in rows
                ]
                for landmark, rows in self.crossfit_parameters.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        instance = cls(SpatialPriorConfig(**payload["config"]))
        instance.fit_sample_ids = list(payload["fit_sample_ids"])
        instance.class_values = list(payload["class_values"])
        instance.gender_values = list(payload["gender_values"])
        instance.selected_l2 = dict(payload["selected_l2"])
        instance.oof_ale = payload.get("oof_ale_mm")
        instance.parameters = {
            landmark: {
                name: np.asarray(value, dtype=np.float64)
                for name, value in rows.items()
            }
            for landmark, rows in payload["parameters"].items()
        }
        instance.crossfit_parameters = {
            landmark: [
                {
                    name: (
                        list(value)
                        if name == "validation_sample_ids"
                        else np.asarray(value, dtype=np.float64)
                    )
                    for name, value in row.items()
                }
                for row in rows
            ]
            for landmark, rows in payload.get("crossfit_parameters", {}).items()
        }
        return instance
