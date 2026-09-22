"""Three-view RGB-D/geometric candidate construction for Core20."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from agh_former_vnext_orthodontic_comparison.hard3_structured import _canonical_frame
from all23_rgb_geodesic_cascade.anatomy import heatmap_sigma_mm, roi_radius_mm

from .anatomy import (
    CORE20_INDICES,
    LEFT_LANDMARKS,
    RIGHT_LANDMARKS,
    core20_group_index,
)


IMAGE_CHANNELS = (
    "red",
    "green",
    "blue",
    "normal_u",
    "normal_v",
    "normal_depth",
    "curvature_k8",
    "curvature_k24",
    "curvature_k64",
    "rgb_gradient_k8",
    "rgb_gradient_k24",
    "rgb_gradient_k64",
    "depth",
    "image_u",
    "image_v",
    "occupancy",
)

GEOMETRY_FEATURES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "global_x",
    "global_y",
    "global_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "abs_normal_x",
    "abs_normal_y",
    "abs_normal_z",
    "red",
    "green",
    "blue",
    "local_dr",
    "local_dg",
    "local_db",
    "curvature_k8",
    "curvature_k24",
    "curvature_k64",
    "rgb_gradient_k8",
    "rgb_gradient_k24",
    "rgb_gradient_k64",
    "density",
    "normalized_geodesic_center_distance",
)


@dataclass
class Core20CandidateSet:
    sample_ids: list[str]
    classes: list[str]
    genders: list[str]
    images: np.ndarray
    grids: np.ndarray
    points: np.ndarray
    features: np.ndarray
    mask: np.ndarray
    heatmap_target: np.ndarray
    target_distance: np.ndarray
    expert: np.ndarray
    expert_full: np.ndarray
    base: np.ndarray
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    prior_score: np.ndarray
    anatomy_anchors: np.ndarray
    anatomy_distances: np.ndarray
    anatomy_mask: np.ndarray
    has_rgb: np.ndarray

    def __len__(self):
        return len(self.sample_ids)


def _unit(vector, fallback):
    vector = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return vector / length


def _side_sign(landmark):
    if landmark in LEFT_LANDMARKS:
        return -1.0
    if landmark in RIGHT_LANDMARKS:
        return 1.0
    return 1.0


def _fill_sparse(image, occupied, maximum_distance=3.0):
    if not occupied.any() or occupied.all():
        return image
    distance, indices = distance_transform_edt(~occupied, return_indices=True)
    fill = (~occupied) & (distance <= maximum_distance)
    image[:, fill] = image[:, indices[0][fill], indices[1][fill]]
    image[-1, fill] = np.exp(-distance[fill] / 1.5)
    return image


def _rasterize(values, coordinates, radius, image_size):
    """Z-buffer point attributes; final channels are image XY and occupancy."""
    size = int(image_size)
    u, v, depth = coordinates.T
    column = np.rint((u / radius + 1.0) * 0.5 * (size - 1)).astype(np.int64)
    row = np.rint((1.0 - v / radius) * 0.5 * (size - 1)).astype(np.int64)
    valid = (column >= 0) & (column < size) & (row >= 0) & (row < size)
    row, column, depth, values = row[valid], column[valid], depth[valid], values[valid]
    if len(row):
        flat = row * size + column
        order = np.lexsort((depth, flat))
        ordered_flat = flat[order]
        keep = np.ones(len(order), dtype=np.bool_)
        keep[:-1] = ordered_flat[:-1] != ordered_flat[1:]
        visible = order[keep]
        row, column, depth, values = (
            row[visible],
            column[visible],
            depth[visible],
            values[visible],
        )

    image = np.zeros((values.shape[1] + 4, size, size), dtype=np.float32)
    occupied = np.zeros((size, size), dtype=np.bool_)
    if len(row):
        image[: values.shape[1], row, column] = values.T
        image[values.shape[1], row, column] = np.clip(depth / radius, -2.0, 2.0)
        occupied[row, column] = True
        image[-1, row, column] = 1.0
        image = _fill_sparse(image, occupied)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    denominator = max(size - 1, 1)
    image[-3] = 2.0 * xx / denominator - 1.0
    image[-2] = 1.0 - 2.0 * yy / denominator
    return image


def _multiscale_descriptors(points, normals, rgb, valid_mask, counts=(8, 24, 64)):
    """Normal variation and RGB gradient at three local neighborhood scales."""
    output_curvature = np.zeros((len(points), len(counts)), dtype=np.float32)
    output_gradient = np.zeros((len(points), len(counts)), dtype=np.float32)
    valid = np.flatnonzero(valid_mask)
    if len(valid) < 2:
        return output_curvature, output_gradient
    selected = points[valid]
    maximum = min(max(counts) + 1, len(valid))
    distance, neighbor = cKDTree(selected).query(selected, k=maximum)
    if maximum == 1:
        distance, neighbor = distance[:, None], neighbor[:, None]
    for column, requested in enumerate(counts):
        count = min(int(requested) + 1, maximum)
        local_neighbor = neighbor[:, 1:count]
        local_distance = distance[:, 1:count]
        if local_neighbor.shape[1] == 0:
            continue
        dot = np.sum(normals[valid, None] * normals[valid[local_neighbor]], axis=-1)
        curvature = np.mean(1.0 - np.clip(np.abs(dot), 0.0, 1.0), axis=1)
        color_delta = np.linalg.norm(
            rgb[valid, None] - rgb[valid[local_neighbor]], axis=-1
        )
        gradient = np.mean(color_delta / np.maximum(local_distance, 0.5), axis=1)
        output_curvature[valid, column] = np.clip(curvature, 0.0, 2.0)
        output_gradient[valid, column] = np.clip(gradient, 0.0, 1.0)
    return output_curvature, output_gradient


def _view_bases(relative, normal, valid_mask, landmark):
    identity = np.eye(3, dtype=np.float32)
    local = valid_mask & (np.linalg.norm(relative, axis=1) <= 8.0)
    if not local.any():
        local = valid_mask
    surface_normal = _unit(normal[local].mean(axis=0), (0.0, 0.0, 1.0))
    tangent_u = identity[:, 0] - surface_normal * float(surface_normal[0])
    tangent_u = _unit(tangent_u, (1.0, 0.0, 0.0))
    tangent_v = _unit(np.cross(surface_normal, tangent_u), (0.0, 1.0, 0.0))
    if tangent_v[1] < 0:
        tangent_v *= -1.0
    tangent_basis = np.stack([tangent_u, tangent_v, surface_normal], axis=1)

    if landmark <= 12:
        third_basis = np.stack([identity[:, 2], identity[:, 1], identity[:, 0]], axis=1)
    else:
        oblique_u = _unit(identity[:, 0] + 0.65 * identity[:, 2], (1.0, 0.0, 0.0))
        oblique_v = identity[:, 1]
        oblique_depth = _unit(np.cross(oblique_u, oblique_v), (0.0, 0.0, 1.0))
        third_basis = np.stack([oblique_u, oblique_v, oblique_depth], axis=1)
    return (identity, tangent_basis.astype(np.float32), third_basis.astype(np.float32))


def _patch_radius(landmark, radius_scale):
    if landmark in (1, 2, 10, 11, 12):
        base = 14.0
    else:
        base = 11.0
    return base * float(radius_scale)


def _mahalanobis_scores(points, mean, covariance):
    covariance = np.asarray(covariance, dtype=np.float64)
    inverse = np.linalg.pinv(covariance)
    delta = np.asarray(points, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    score = -0.5 * np.einsum("ki,ij,kj->k", delta, inverse, delta)
    return np.clip(score, -50.0, 0.0).astype(np.float32)


def render_item(
    item,
    normalizer_mean,
    normalizer_std,
    centers,
    prior_mean,
    prior_covariance,
    image_size=96,
):
    points = item["points"].numpy().astype(np.float32)
    normalized = item["features"].numpy().astype(np.float32)
    raw = normalized * np.asarray(normalizer_std, dtype=np.float32) + np.asarray(
        normalizer_mean, dtype=np.float32
    )
    centers = np.asarray(centers, dtype=np.float32)
    expert_full = item["expert"].numpy().astype(np.float32)
    expert = expert_full[list(CORE20_INDICES)]
    roi_index = item["roi_index"].numpy().astype(np.int64)[list(CORE20_INDICES)]
    roi_mask = item["roi_mask"].numpy().astype(bool)[list(CORE20_INDICES)]
    heatmap_target = (
        item["heatmap_target"].numpy().astype(np.float32)[list(CORE20_INDICES)]
    )
    center_distance = (
        item["roi_center_distance"].numpy().astype(np.float32)[list(CORE20_INDICES)]
    )
    origin, frame, face_scale = _canonical_frame(centers)

    image_rows, grid_rows, feature_rows, score_rows = [], [], [], []
    anchor_rows, anchor_distance_rows, anchor_mask_rows = [], [], []
    candidate_points = points[roi_index]
    target_distance = np.linalg.norm(candidate_points - expert[:, None], axis=-1)
    target_distance[~roi_mask] = np.inf
    for local_index, landmark in enumerate(CORE20_INDICES):
        indices = roi_index[local_index]
        mask = roi_mask[local_index]
        selected = points[indices]
        side = _side_sign(landmark)
        relative = (selected - centers[landmark]) @ frame
        global_relative = (selected - origin) @ frame
        normal = raw[indices, 9:12] @ frame
        if landmark >= 13:
            relative[:, 0] *= side
            global_relative[:, 0] *= side
            normal[:, 0] *= side
        rgb = np.clip(raw[indices, 3:6], 0.0, 1.0)
        contrast = np.clip(raw[indices, 6:9], -1.0, 1.0)
        curvature, rgb_gradient = _multiscale_descriptors(selected, normal, rgb, mask)
        radius = _patch_radius(landmark, 1.0)
        views, grids = [], []
        for basis in _view_bases(relative, normal, mask, landmark):
            view_coordinate = relative @ basis
            view_normal = normal @ basis
            point_values = np.concatenate(
                [rgb, view_normal, curvature, rgb_gradient], axis=1
            ).astype(np.float32)
            point_values[~mask] = 0.0
            views.append(
                _rasterize(
                    point_values[mask], view_coordinate[mask], radius, image_size
                )
            )
            grids.append(
                np.clip(
                    np.stack(
                        [
                            view_coordinate[:, 0] / radius,
                            -view_coordinate[:, 1] / radius,
                        ],
                        axis=-1,
                    ),
                    -2.0,
                    2.0,
                ).astype(np.float32)
            )

        density = np.clip(raw[indices, 12:13], 0.0, 8.0)
        radial = center_distance[local_index, :, None] / max(
            roi_radius_mm(landmark), 1e-4
        )
        geometry = np.concatenate(
            [
                relative / max(roi_radius_mm(landmark), 1e-4),
                global_relative / max(face_scale, 1e-4),
                normal,
                np.abs(normal),
                rgb,
                contrast,
                curvature,
                rgb_gradient,
                density,
                radial,
            ],
            axis=1,
        ).astype(np.float32)
        geometry[~mask] = 0.0
        score = _mahalanobis_scores(
            selected, prior_mean[local_index], prior_covariance[local_index]
        )
        score[~mask] = -50.0

        from .anatomy import ANCHORS

        anchors = list(ANCHORS[landmark])[:4]
        padded_anchors = np.zeros((4, 3), dtype=np.float32)
        padded_distances = np.zeros(4, dtype=np.float32)
        padded_mask = np.zeros(4, dtype=np.bool_)
        padded_anchors[: len(anchors)] = expert_full[anchors]
        padded_distances[: len(anchors)] = np.linalg.norm(
            expert_full[landmark] - expert_full[anchors], axis=-1
        )
        padded_mask[: len(anchors)] = True

        image_rows.append(views)
        grid_rows.append(grids)
        feature_rows.append(geometry)
        score_rows.append(score)
        anchor_rows.append(padded_anchors)
        anchor_distance_rows.append(padded_distances)
        anchor_mask_rows.append(padded_mask)

    return {
        "images": np.asarray(image_rows, dtype=np.float16),
        "grids": np.asarray(grid_rows, dtype=np.float32),
        "points": candidate_points.astype(np.float32),
        "features": np.asarray(feature_rows, dtype=np.float32),
        "mask": roi_mask,
        "heatmap_target": heatmap_target,
        "target_distance": target_distance.astype(np.float32),
        "expert": expert,
        "expert_full": expert_full,
        "base": centers[list(CORE20_INDICES)],
        "prior_mean": np.asarray(prior_mean, dtype=np.float32),
        "prior_covariance": np.asarray(prior_covariance, dtype=np.float32),
        "prior_score": np.asarray(score_rows, dtype=np.float32),
        "anatomy_anchors": np.asarray(anchor_rows, dtype=np.float32),
        "anatomy_distances": np.asarray(anchor_distance_rows, dtype=np.float32),
        "anatomy_mask": np.asarray(anchor_mask_rows, dtype=np.bool_),
    }


def extract_core20_set(
    dataset,
    centers_by_id,
    prior_means_by_id,
    prior_covariances_by_id,
    image_size=96,
    label="Core20 MVSC patches",
):
    missing = [
        sample.sample_id
        for sample in dataset.samples
        if sample.sample_id not in centers_by_id
        or sample.sample_id not in prior_means_by_id
        or sample.sample_id not in prior_covariances_by_id
    ]
    if missing:
        raise KeyError(f"Core20 MVSC inputs miss samples: {missing[:5]}")
    working = copy.copy(dataset)
    working.coarse_predictions = {
        sample.sample_id: np.asarray(centers_by_id[sample.sample_id], dtype=np.float32)
        for sample in dataset.samples
    }
    working.coarse_in_target_space = True
    working.training = False
    working._roi_memory = {}
    rows = {
        name: []
        for name in (
            "images",
            "grids",
            "points",
            "features",
            "mask",
            "heatmap_target",
            "target_distance",
            "expert",
            "expert_full",
            "base",
            "prior_mean",
            "prior_covariance",
            "prior_score",
            "anatomy_anchors",
            "anatomy_distances",
            "anatomy_mask",
        )
    }
    sample_ids, classes, genders, has_rgb = [], [], [], []
    for index in range(len(working)):
        item = working[index]
        sample_id = item["sample_id"]
        rendered = render_item(
            item,
            working.mean,
            working.std,
            centers_by_id[sample_id],
            prior_means_by_id[sample_id],
            prior_covariances_by_id[sample_id],
            image_size=image_size,
        )
        for name, value in rendered.items():
            if name == "target_distance":
                finite = np.isfinite(value[rendered["mask"]]).all()
            elif np.issubdtype(np.asarray(value).dtype, np.floating):
                finite = np.isfinite(value).all()
            else:
                finite = True
            if not finite:
                raise RuntimeError(
                    f"Non-finite Core20-MVSC {name} values for {sample_id}"
                )
        for name, value in rendered.items():
            rows[name].append(value)
        sample_ids.append(sample_id)
        classes.append(item["class"])
        genders.append(item["gender"])
        with np.load(working.records[sample_id]) as record:
            has_rgb.append(
                bool(record["has_rgb"][0]) if "has_rgb" in record.files else False
            )
        if (index + 1) % 10 == 0 or index + 1 == len(working):
            print(f"{label} {index + 1}/{len(working)}", flush=True)
    result = Core20CandidateSet(
        sample_ids=sample_ids,
        classes=classes,
        genders=genders,
        has_rgb=np.asarray(has_rgb, dtype=np.bool_),
        **{name: np.stack(values) for name, values in rows.items()},
    )
    if result.images.shape[3] != len(IMAGE_CHANNELS):
        raise AssertionError("Core20 image-channel contract changed unexpectedly")
    if result.features.shape[-1] != len(GEOMETRY_FEATURES):
        raise AssertionError("Core20 geometry-feature contract changed unexpectedly")
    return result


def group_indices():
    return np.asarray(
        [core20_group_index(landmark) for landmark in CORE20_INDICES],
        dtype=np.int64,
    )
