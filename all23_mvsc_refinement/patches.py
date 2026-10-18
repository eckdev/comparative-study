"""Four-view RGB-D/geometric candidate construction for all 23 landmarks.

Differences from Core20-MVSC (core20_mvsc_refinement/patches.py):
- every landmark, including Trichion and both Gonia, is rendered;
- three fine views keep local detail and a fourth context view always covers
  the complete candidate ROI, so no candidate receives all-zero image features;
- per-view visibility bits tell the scorer which views actually see a candidate;
- silhouette and depth-gradient channels expose contour evidence;
- Gonion candidates receive the global jaw-silhouette descriptors of the Hard3
  dual-view refiner.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, sobel
from scipy.spatial import cKDTree

from agh_former_vnext_orthodontic_comparison.hard3_structured import _canonical_frame
from all23_rgb_geodesic_cascade.anatomy import NUM_LANDMARKS, roi_radius_mm
from hard3_anatomical_context_refinement.patches import (
    _build_jaw_contour_profile,
    _sample_global_jaw_contour,
)

from .anatomy import (
    ALL23_INDICES,
    ANCHORS,
    LEFT_LANDMARKS,
    RIGHT_LANDMARKS,
    anatomical_group_index,
    fine_view_radius_mm,
    landmark_type,
    landmark_type_index,
)


VIEW_NAMES = ("fine_frontal", "fine_tangent", "fine_profile_or_oblique", "context")
POINT_CHANNELS = (
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
)
IMAGE_CHANNELS = POINT_CHANNELS + (
    "depth",
    "image_u",
    "image_v",
    "occupancy",
    "signed_silhouette",
    "depth_gradient",
)
JAW_FEATURE_COUNT = 48
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
) + tuple(f"visible_{name}" for name in VIEW_NAMES) + tuple(
    f"jaw_contour_{index}" for index in range(JAW_FEATURE_COUNT)
)
# Column ranges zeroed by the geometry-only (no RGB) training fallback.
RGB_IMAGE_CHANNELS = (0, 1, 2, 9, 10, 11)
RGB_GEOMETRY_FEATURES = tuple(range(12, 18)) + (21, 22, 23)

_ARRAY_FIELDS = (
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
# Large arrays are written to disk-backed .npy memmaps when a storage directory
# is supplied; 23 landmarks x 4 views do not fit comfortably in Colab RAM.
_MEMMAP_FIELDS = ("images", "grids", "features")


@dataclass
class All23CandidateSet:
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
    has_expert: bool = True

    def __len__(self):
        return len(self.sample_ids)

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        values = {
            name: np.asarray(getattr(self, name))[indices] for name in _ARRAY_FIELDS
        }
        return All23CandidateSet(
            sample_ids=[self.sample_ids[index] for index in indices],
            classes=[self.classes[index] for index in indices],
            genders=[self.genders[index] for index in indices],
            has_rgb=np.asarray(self.has_rgb)[indices],
            has_expert=self.has_expert,
            **values,
        )


def _unit(vector, fallback):
    vector = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return vector / length


def side_sign(landmark):
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
    return image, fill, distance


def _contour_channels(depth, occupied, maximum_fill_distance=3.0):
    """Signed silhouette distance and depth-gradient magnitude of one view."""
    size = depth.shape[-1]
    if not occupied.any():
        return (
            np.full((size, size), -1.0, dtype=np.float32),
            np.zeros((size, size), dtype=np.float32),
        )
    support = binary_fill_holes(
        distance_transform_edt(~occupied) <= float(maximum_fill_distance)
    )
    inside = distance_transform_edt(support)
    outside = distance_transform_edt(~support)
    signed = np.clip((inside - outside) / 8.0, -1.0, 1.0).astype(np.float32)
    gradient = np.hypot(sobel(depth, axis=0), sobel(depth, axis=1))
    valid = gradient[support]
    scale = max(float(np.percentile(valid, 95)), 1e-4) if valid.size else 1.0
    gradient = np.clip(gradient / scale, 0.0, 4.0).astype(np.float32)
    gradient[~support] = 0.0
    return signed, gradient


def rasterize(values, coordinates, radius, image_size):
    """Z-buffer point attributes into the IMAGE_CHANNELS layout."""
    size = int(image_size)
    point_channels = values.shape[1]
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

    image = np.zeros((point_channels + 6, size, size), dtype=np.float32)
    depth_channel = point_channels
    occupancy_channel = point_channels + 3
    occupied = np.zeros((size, size), dtype=np.bool_)
    if len(row):
        image[:point_channels, row, column] = values.T
        image[depth_channel, row, column] = np.clip(depth / radius, -2.0, 2.0)
        occupied[row, column] = True
        image[occupancy_channel, row, column] = 1.0
        filled = _fill_sparse(image[: occupancy_channel + 1], occupied)
        if isinstance(filled, tuple):
            _, fill, distance = filled
            image[occupancy_channel, fill] = np.exp(-distance[fill] / 1.5)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    denominator = max(size - 1, 1)
    image[point_channels + 1] = 2.0 * xx / denominator - 1.0
    image[point_channels + 2] = 1.0 - 2.0 * yy / denominator
    signed, gradient = _contour_channels(image[depth_channel], occupied)
    image[point_channels + 4] = signed
    image[point_channels + 5] = gradient
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


def _profile_basis():
    identity = np.eye(3, dtype=np.float32)
    return np.stack([identity[:, 2], identity[:, 1], identity[:, 0]], axis=1)


def view_bases(relative, normal, valid_mask, landmark):
    """Fine frontal, fine tangent, fine profile/oblique and context bases."""
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

    if landmark <= 12 or landmark in (21, 22):
        # Midline points and the mandibular angle are read best in profile;
        # x was already mirrored so lateral points look outwards.
        third_basis = _profile_basis()
    else:
        oblique_u = _unit(identity[:, 0] + 0.65 * identity[:, 2], (1.0, 0.0, 0.0))
        oblique_v = identity[:, 1]
        oblique_depth = _unit(np.cross(oblique_u, oblique_v), (0.0, 0.0, 1.0))
        third_basis = np.stack([oblique_u, oblique_v, oblique_depth], axis=1)
    context_basis = _profile_basis() if landmark_type(landmark) == "contour" else identity
    return (
        identity,
        tangent_basis.astype(np.float32),
        third_basis.astype(np.float32),
        context_basis.astype(np.float32),
    )


def context_radius(view_coordinate, mask, minimum):
    """Half width that keeps every valid candidate inside the context image."""
    if not np.any(mask):
        return float(minimum)
    extent = float(np.abs(view_coordinate[mask, :2]).max())
    return max(float(minimum), 1.02 * extent)


def _mahalanobis_scores(points, mean, covariance):
    inverse = np.linalg.pinv(np.asarray(covariance, dtype=np.float64))
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
    use_jaw_profile=True,
):
    points = item["points"].numpy().astype(np.float32)
    normalized = item["features"].numpy().astype(np.float32)
    raw = normalized * np.asarray(normalizer_std, dtype=np.float32) + np.asarray(
        normalizer_mean, dtype=np.float32
    )
    centers = np.asarray(centers, dtype=np.float32)
    expert_full = item["expert"].numpy().astype(np.float32)
    roi_index = item["roi_index"].numpy().astype(np.int64)
    roi_mask = item["roi_mask"].numpy().astype(bool)
    heatmap_target = item["heatmap_target"].numpy().astype(np.float32)
    center_distance = item["roi_center_distance"].numpy().astype(np.float32)
    origin, frame, face_scale = _canonical_frame(centers)

    image_rows, grid_rows, feature_rows, score_rows = [], [], [], []
    anchor_rows, anchor_distance_rows, anchor_mask_rows = [], [], []
    candidate_points = points[roi_index]
    target_distance = np.linalg.norm(candidate_points - expert_full[:, None], axis=-1)
    target_distance[~roi_mask] = np.inf
    for landmark in ALL23_INDICES:
        indices = roi_index[landmark]
        mask = roi_mask[landmark]
        selected = points[indices]
        relative = (selected - centers[landmark]) @ frame
        global_relative = (selected - origin) @ frame
        normal = raw[indices, 9:12] @ frame
        if landmark in LEFT_LANDMARKS or landmark in RIGHT_LANDMARKS:
            side = side_sign(landmark)
            relative[:, 0] *= side
            global_relative[:, 0] *= side
            normal[:, 0] *= side
        rgb = np.clip(raw[indices, 3:6], 0.0, 1.0)
        contrast = np.clip(raw[indices, 6:9], -1.0, 1.0)
        curvature, rgb_gradient = _multiscale_descriptors(selected, normal, rgb, mask)
        fine_radius = fine_view_radius_mm(landmark)
        views, grids, visibility = [], [], []
        bases = view_bases(relative, normal, mask, landmark)
        for view_index, basis in enumerate(bases):
            view_coordinate = relative @ basis
            radius = (
                context_radius(view_coordinate, mask, fine_radius)
                if view_index == len(bases) - 1
                else fine_radius
            )
            view_normal = normal @ basis
            point_values = np.concatenate(
                [rgb, view_normal, curvature, rgb_gradient], axis=1
            ).astype(np.float32)
            views.append(
                rasterize(point_values[mask], view_coordinate[mask], radius, image_size)
            )
            grid = np.stack(
                [view_coordinate[:, 0] / radius, -view_coordinate[:, 1] / radius],
                axis=-1,
            )
            visibility.append((np.abs(grid) <= 1.0).all(axis=-1) & mask)
            grids.append(np.clip(grid, -2.0, 2.0).astype(np.float32))

        density = np.clip(raw[indices, 12:13], 0.0, 8.0)
        roi_radius = max(roi_radius_mm(landmark), 1e-4)
        radial = center_distance[landmark, :, None] / roi_radius
        jaw = np.zeros((len(indices), JAW_FEATURE_COUNT), dtype=np.float32)
        if use_jaw_profile and landmark in (21, 22):
            profile, jaw_side = _build_jaw_contour_profile(
                points, centers, origin, frame, face_scale, landmark
            )
            jaw = _sample_global_jaw_contour(
                selected, profile, jaw_side, origin, frame, face_scale
            )
            jaw = np.clip(jaw, -8.0, 8.0)
        geometry = np.concatenate(
            [
                relative / roi_radius,
                global_relative / max(face_scale, 1e-4),
                normal,
                np.abs(normal),
                rgb,
                contrast,
                curvature,
                rgb_gradient,
                density,
                radial,
                np.stack(visibility, axis=1).astype(np.float32),
                jaw,
            ],
            axis=1,
        ).astype(np.float32)
        geometry[~mask] = 0.0
        score = _mahalanobis_scores(
            selected, prior_mean[landmark], prior_covariance[landmark]
        )
        score[~mask] = -50.0

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
        "features": np.asarray(feature_rows, dtype=np.float16),
        "mask": roi_mask,
        "heatmap_target": heatmap_target,
        "target_distance": target_distance.astype(np.float32),
        "expert": expert_full.copy(),
        "expert_full": expert_full,
        "base": centers.copy(),
        "prior_mean": np.asarray(prior_mean, dtype=np.float32),
        "prior_covariance": np.asarray(prior_covariance, dtype=np.float32),
        "prior_score": np.asarray(score_rows, dtype=np.float32),
        "anatomy_anchors": np.asarray(anchor_rows, dtype=np.float32),
        "anatomy_distances": np.asarray(anchor_distance_rows, dtype=np.float32),
        "anatomy_mask": np.asarray(anchor_mask_rows, dtype=np.bool_),
    }


def _allocate(name, value, count, storage_dir):
    shape = (count, *np.asarray(value).shape)
    dtype = np.asarray(value).dtype
    if storage_dir is not None and name in _MEMMAP_FIELDS:
        return np.lib.format.open_memmap(
            Path(storage_dir) / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
    return np.empty(shape, dtype=dtype)


def extract_all23_set(
    dataset,
    centers_by_id,
    prior_means_by_id,
    prior_covariances_by_id,
    image_size=96,
    use_jaw_profile=True,
    cache_dir=None,
    storage_dir=None,
    label="All23-MVSC patches",
):
    """Render candidates around the supplied centres.

    ``dataset.include_expert`` decides whether expert files are read. Test sets
    must be passed with ``include_expert=False`` so their ROIs are built
    geometry-only and no test label is opened before predictions are written.
    """
    missing = [
        sample.sample_id
        for sample in dataset.samples
        if sample.sample_id not in centers_by_id
        or sample.sample_id not in prior_means_by_id
        or sample.sample_id not in prior_covariances_by_id
    ]
    if missing:
        raise KeyError(f"All23-MVSC inputs miss samples: {missing[:5]}")
    working = copy.copy(dataset)
    working.coarse_predictions = {
        sample.sample_id: np.asarray(centers_by_id[sample.sample_id], dtype=np.float32)
        for sample in dataset.samples
    }
    working.coarse_in_target_space = True
    working.training = False
    working._roi_memory = {}
    working._record_memory = {}
    if cache_dir is not None:
        working.cache_dir = Path(cache_dir)
    if storage_dir is not None:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
    count = len(working)
    arrays = {}
    sample_ids, classes, genders, has_rgb = [], [], [], []
    for index in range(count):
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
            use_jaw_profile=use_jaw_profile,
        )
        for name, value in rendered.items():
            if name == "target_distance":
                finite = np.isfinite(value[rendered["mask"]]).all()
            elif np.issubdtype(np.asarray(value).dtype, np.floating):
                finite = np.isfinite(value).all()
            else:
                finite = True
            if not finite:
                raise RuntimeError(f"Non-finite All23-MVSC {name} values for {sample_id}")
        if not arrays:
            arrays = {
                name: _allocate(name, value, count, storage_dir)
                for name, value in rendered.items()
            }
        for name, value in rendered.items():
            arrays[name][index] = value
        sample_ids.append(sample_id)
        classes.append(item["class"])
        genders.append(item["gender"])
        with np.load(working.records[sample_id]) as record:
            has_rgb.append(
                bool(record["has_rgb"][0]) if "has_rgb" in record.files else False
            )
        if (index + 1) % 10 == 0 or index + 1 == count:
            print(f"{label} {index + 1}/{count}", flush=True)
    for name in _MEMMAP_FIELDS:
        if isinstance(arrays.get(name), np.memmap):
            arrays[name].flush()
    result = All23CandidateSet(
        sample_ids=sample_ids,
        classes=classes,
        genders=genders,
        has_rgb=np.asarray(has_rgb, dtype=np.bool_),
        has_expert=bool(getattr(dataset, "include_expert", True)),
        **arrays,
    )
    if result.images.shape[1] != NUM_LANDMARKS:
        raise AssertionError("All23 candidate set must contain 23 landmarks")
    if result.images.shape[2] != len(VIEW_NAMES):
        raise AssertionError("All23 view contract changed unexpectedly")
    if result.images.shape[3] != len(IMAGE_CHANNELS):
        raise AssertionError("All23 image-channel contract changed unexpectedly")
    if result.features.shape[-1] != len(GEOMETRY_FEATURES):
        raise AssertionError("All23 geometry-feature contract changed unexpectedly")
    return result


def type_indices():
    return np.asarray([landmark_type_index(index) for index in ALL23_INDICES], dtype=np.int64)


def group_indices():
    return np.asarray(
        [anatomical_group_index(index) for index in ALL23_INDICES], dtype=np.int64
    )
