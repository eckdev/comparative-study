"""Train-only local atlas prior conditioned on the reliable Core20 configuration."""

from __future__ import annotations

import numpy as np

from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, NUM_LANDMARKS


class TrainOnlyLocalHard3Atlas:
    def __init__(self, neighbors=8, temperature=2.0):
        self.neighbors = int(neighbors)
        self.temperature = float(temperature)
        self.fitted = False

    def fit(self, expert_shapes, sample_ids):
        shapes = np.asarray(expert_shapes, dtype=np.float32)
        if shapes.ndim != 3 or shapes.shape[1:] != (NUM_LANDMARKS, 3):
            raise ValueError("expert_shapes must have shape [N,23,3]")
        if len(shapes) < 2:
            raise ValueError("At least two outer-train shapes are required")
        self.shapes = shapes
        self.sample_ids = list(sample_ids)
        core = shapes[:, list(CORE20)].reshape(len(shapes), -1)
        self.core_mean = core.mean(axis=0).astype(np.float32)
        self.core_scale = np.maximum(core.std(axis=0), 1.0).astype(np.float32)
        self.core = ((core - self.core_mean) / self.core_scale).astype(np.float32)
        self.fitted = True
        return self

    def predict(self, prediction, sample_ids=None):
        if not self.fitted:
            raise RuntimeError("fit must be called before predict")
        prediction = np.asarray(prediction, dtype=np.float32)
        query = prediction[:, list(CORE20)].reshape(len(prediction), -1)
        query = (query - self.core_mean) / self.core_scale
        distance = np.mean((query[:, None] - self.core[None]) ** 2, axis=-1)
        if sample_ids is not None:
            lookup = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
            for row, sample_id in enumerate(sample_ids):
                if sample_id in lookup:
                    distance[row, lookup[sample_id]] = np.inf
        count = min(max(self.neighbors, 1), len(self.shapes) - 1 if sample_ids else len(self.shapes))
        indices = np.argpartition(distance, count - 1, axis=1)[:, :count]
        selected_distance = np.take_along_axis(distance, indices, axis=1)
        centered = selected_distance - np.min(selected_distance, axis=1, keepdims=True)
        weights = np.exp(-centered / max(self.temperature, 1e-4))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        hard = self.shapes[indices][:, :, list(HARD3)]
        mean = np.sum(weights[:, :, None, None] * hard, axis=1)
        dispersion = np.sqrt(
            np.sum(
                weights[:, :, None]
                * np.sum((hard - mean[:, None]) ** 2, axis=-1),
                axis=1,
            )
        )
        return {
            "prediction": mean.astype(np.float32),
            "dispersion": dispersion.astype(np.float32),
            "neighbor_indices": indices.astype(np.int64),
            "neighbor_weights": weights.astype(np.float32),
        }

    def state_dict(self):
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted atlas")
        return {
            "neighbors": self.neighbors,
            "temperature": self.temperature,
            "shapes": self.shapes,
            "sample_ids": self.sample_ids,
            "core_mean": self.core_mean,
            "core_scale": self.core_scale,
            "core": self.core,
        }

    @classmethod
    def from_state_dict(cls, state):
        atlas = cls(state["neighbors"], state["temperature"])
        atlas.shapes = np.asarray(state["shapes"], dtype=np.float32)
        atlas.sample_ids = list(state["sample_ids"])
        atlas.core_mean = np.asarray(state["core_mean"], dtype=np.float32)
        atlas.core_scale = np.asarray(state["core_scale"], dtype=np.float32)
        atlas.core = np.asarray(state["core"], dtype=np.float32)
        atlas.fitted = True
        return atlas
