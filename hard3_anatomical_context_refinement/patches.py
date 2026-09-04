"""Canonical RGB-depth patch construction without inference-label leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, sobel

from agh_former_vnext_orthodontic_comparison.hard3_structured import _canonical_frame
from all23_rgb_geodesic_cascade.anatomy import HARD3, heatmap_sigma_mm, roi_radius_mm


@dataclass
class DualViewCandidateSet:
    sample_ids: list[str]
    strata: list[str]
    images: np.ndarray
    targets: np.ndarray
    grids: np.ndarray
    points: np.ndarray
    mask: np.ndarray
    expert: np.ndarray
    expert_full: np.ndarray
    target_distance: np.ndarray
    target_view_mask: np.ndarray

    def __len__(self):
        return len(self.sample_ids)


def _side_sign(landmark, coarse, origin, frame):
    if landmark == 0:
        return 1.0
    anchors = (13, 17, 19) if landmark == 21 else (16, 18, 20)
    lateral = float(((coarse[list(anchors)].mean(axis=0) - origin) @ frame)[0])
    return 1.0 if lateral >= 0.0 else -1.0


def _robust_scale(values, minimum=1e-4):
    valid = np.asarray(values, dtype=np.float32)
    median = np.median(valid)
    scale = np.percentile(valid, 75) - np.percentile(valid, 25)
    return median, max(float(scale), minimum)


def _fill_sparse(image, occupied, maximum_distance=3.0):
    if occupied.all() or not occupied.any():
        return image
    distance, indices = distance_transform_edt(~occupied, return_indices=True)
    fill = (~occupied) & (distance <= float(maximum_distance))
    image[:, fill] = image[:, indices[0][fill], indices[1][fill]]
    # Last channel distinguishes measured pixels from short-range interpolation.
    image[-1, fill] = np.exp(-distance[fill] / 1.5)
    return image


def _contour_channels(image, occupied, maximum_fill_distance=3.0):
    """Encode image position and the projected surface boundary explicitly."""
    size = image.shape[-1]
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    denominator = max(size - 1, 1)
    coordinate_u = 2.0 * xx / denominator - 1.0
    coordinate_v = 1.0 - 2.0 * yy / denominator

    if occupied.any():
        distance_to_measurement = distance_transform_edt(~occupied)
        support = binary_fill_holes(
            distance_to_measurement <= float(maximum_fill_distance)
        )
        inside = distance_transform_edt(support)
        outside = distance_transform_edt(~support)
        signed_distance = np.clip((inside - outside) / 8.0, -1.0, 1.0)

        # Depth is the third channel from the end before engineered channels
        # are appended: [... point features, depth, view_id, occupancy].
        depth = image[-3]
        gradient = np.hypot(sobel(depth, axis=0), sobel(depth, axis=1))
        valid_gradient = gradient[support]
        scale = (
            max(float(np.percentile(valid_gradient, 95)), 1e-4)
            if valid_gradient.size
            else 1.0
        )
        depth_gradient = np.clip(gradient / scale, 0.0, 4.0)
        depth_gradient[~support] = 0.0
    else:
        signed_distance = np.full((size, size), -1.0, dtype=np.float32)
        depth_gradient = np.zeros((size, size), dtype=np.float32)

    return np.stack(
        [coordinate_u, coordinate_v, signed_distance, depth_gradient], axis=0
    ).astype(np.float32)


def _rasterize(point_features, u, v, depth, radius, image_size, view_code):
    size = int(image_size)
    column = np.rint((u / radius + 1.0) * 0.5 * (size - 1)).astype(np.int64)
    row = np.rint((1.0 - v / radius) * 0.5 * (size - 1)).astype(np.int64)
    valid = (column >= 0) & (column < size) & (row >= 0) & (row < size)
    column, row = column[valid], row[valid]
    values = point_features[valid]
    depth = depth[valid]

    channels = values.shape[1] + 3
    image = np.zeros((channels, size, size), dtype=np.float32)
    counts = np.zeros((size, size), dtype=np.float32)
    occupied = counts > 0
    if len(row):
        for channel in range(values.shape[1]):
            np.add.at(image[channel], (row, column), values[:, channel])
        np.add.at(image[-3], (row, column), np.clip(depth / radius, -2.0, 2.0))
        np.add.at(counts, (row, column), 1.0)
        occupied = counts > 0
        image[:-2, occupied] /= counts[occupied]
        image[-2, occupied] = float(view_code)
        image[-1, occupied] = 1.0
        image = _fill_sparse(image, occupied)
    contour = _contour_channels(image, occupied)
    # Keep occupancy as the final channel for downstream diagnostics.
    return np.concatenate([image[:-1], contour, image[-1:]], axis=0)


def _target_heatmap(expert_relative, axes, radius, image_size, sigma_mm):
    size = int(image_size)
    u = float(expert_relative[axes[0]])
    v = float(expert_relative[axes[1]])
    x = (u / radius + 1.0) * 0.5 * (size - 1)
    y = (1.0 - v / radius) * 0.5 * (size - 1)
    yy, xx = np.mgrid[:size, :size]
    sigma_pixels = max(float(sigma_mm) * (size - 1) / (2.0 * radius), 0.75)
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma_pixels**2)).astype(
        np.float32
    )


def render_item(item, normalizer_mean, normalizer_std, image_size=64, radius_scale=1.0, centers=None):
    points = item["points"].numpy().astype(np.float32)
    normalized = item["features"].numpy().astype(np.float32)
    raw = normalized * np.asarray(normalizer_std, dtype=np.float32) + np.asarray(
        normalizer_mean, dtype=np.float32
    )
    coarse = item["coarse"].numpy().astype(np.float32)
    centers = coarse if centers is None else np.asarray(centers, dtype=np.float32)
    expert_full = item["expert"].numpy().astype(np.float32)
    expert = expert_full[list(HARD3)]
    roi_index = item["roi_index"].numpy().astype(np.int64)[list(HARD3)]
    roi_mask = item["roi_mask"].numpy().astype(bool)[list(HARD3)]

    origin, frame, _ = _canonical_frame(centers)
    images, targets, grids, target_view_masks = [], [], [], []
    candidate_points = points[roi_index]
    for local_index, landmark in enumerate(HARD3):
        indices = roi_index[local_index]
        mask = roi_mask[local_index]
        selected = points[indices]
        side = _side_sign(landmark, centers, origin, frame)
        relative = (selected - centers[landmark]) @ frame
        expert_relative = (expert_full[landmark] - centers[landmark]) @ frame
        normal = raw[indices, 9:12] @ frame
        if landmark in (21, 22):
            relative[:, 0] *= side
            expert_relative[0] *= side
            normal[:, 0] *= side

        rgb = np.clip(raw[indices, 3:6], 0.0, 1.0)
        contrast = raw[indices, 6:9]
        intensity = rgb.mean(axis=1, keepdims=True)
        chroma = rgb.max(axis=1, keepdims=True) - rgb.min(axis=1, keepdims=True)
        _, curvature_scale = _robust_scale(raw[indices[mask], 13])
        curvature = np.clip(raw[indices, 13:14] / curvature_scale, 0.0, 8.0)
        density_median, density_scale = _robust_scale(raw[indices[mask], 12])
        density = np.clip(
            (raw[indices, 12:13] - density_median) / density_scale, -4.0, 4.0
        )
        per_point = np.concatenate(
            [rgb, contrast, normal, intensity, chroma, curvature, density], axis=1
        ).astype(np.float32)
        per_point[~mask] = 0.0
        radius = float(roi_radius_mm(landmark)) * float(radius_scale)
        # A candidate must be represented in both raster views. Otherwise
        # grid_sample would assign an artificial zero logit to an off-frame point.
        projected = np.all(np.abs(relative) <= radius, axis=1)
        mask = mask & projected
        if not np.any(mask):
            available = roi_mask[local_index]
            nearest = np.argmin(
                np.where(
                    available,
                    np.linalg.norm(relative, axis=1),
                    np.inf,
                )
            )
            mask[nearest] = True
        roi_mask[local_index] = mask
        landmark_images, landmark_targets, landmark_grids, landmark_target_masks = [], [], [], []
        # Frontal view (lateral/vertical) and side view (depth/vertical).
        for view_code, axes in enumerate(((0, 1, 2), (2, 1, 0))):
            landmark_images.append(
                _rasterize(
                    per_point[mask],
                    relative[mask, axes[0]],
                    relative[mask, axes[1]],
                    relative[mask, axes[2]],
                    radius,
                    image_size,
                    -1.0 if view_code == 0 else 1.0,
                )
            )
            landmark_targets.append(
                _target_heatmap(
                    expert_relative,
                    axes,
                    radius,
                    image_size,
                    heatmap_sigma_mm(landmark),
                )
            )
            landmark_target_masks.append(
                abs(float(expert_relative[axes[0]])) <= radius
                and abs(float(expert_relative[axes[1]])) <= radius
            )
            grid = np.stack(
                [relative[:, axes[0]] / radius, -relative[:, axes[1]] / radius],
                axis=-1,
            )
            landmark_grids.append(np.clip(grid, -2.0, 2.0).astype(np.float32))
        images.append(landmark_images)
        targets.append(landmark_targets)
        grids.append(landmark_grids)
        target_view_masks.append(landmark_target_masks)

    target_distance = np.linalg.norm(candidate_points - expert[:, None], axis=-1)
    target_distance[~roi_mask] = np.inf
    return (
        np.asarray(images, dtype=np.float16),
        np.asarray(targets, dtype=np.float16),
        np.asarray(grids, dtype=np.float32),
        candidate_points.astype(np.float32),
        roi_mask,
        expert,
        expert_full,
        target_distance.astype(np.float32),
        np.asarray(target_view_masks, dtype=np.bool_),
    )


def extract_dual_view_set(dataset, image_size=64, radius_scale=1.0, centers_by_id=None, label="Hard3 patches"):
    previous_training = dataset.training
    dataset.training = False
    rows = [[] for _ in range(9)]
    sample_ids, strata = [], []
    try:
        for index in range(len(dataset)):
            item = dataset[index]
            centers = None
            if centers_by_id is not None:
                centers = centers_by_id[item["sample_id"]]
            rendered = render_item(
                item,
                dataset.mean,
                dataset.std,
                image_size=image_size,
                radius_scale=radius_scale,
                centers=centers,
            )
            for destination, value in zip(rows, rendered):
                destination.append(value)
            sample_ids.append(item["sample_id"])
            strata.append(f"{item['class']}|{item['gender']}")
            if (index + 1) % 20 == 0 or index + 1 == len(dataset):
                print(f"{label} {index + 1}/{len(dataset)}", flush=True)
    finally:
        dataset.training = previous_training
    return DualViewCandidateSet(
        sample_ids=sample_ids,
        strata=strata,
        images=np.stack(rows[0]),
        targets=np.stack(rows[1]),
        grids=np.stack(rows[2]),
        points=np.stack(rows[3]),
        mask=np.stack(rows[4]),
        expert=np.stack(rows[5]),
        expert_full=np.stack(rows[6]),
        target_distance=np.stack(rows[7]),
        target_view_mask=np.stack(rows[8]),
    )
