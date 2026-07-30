import csv
import hashlib
import heapq
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .alignment import apply_transform, rotate_vectors
from .anatomy import NUM_LANDMARKS, heatmap_sigma_mm, roi_radius_mm


LANDMARK_RE = re.compile(
    r"^Point\s*#(?P<idx>\d+)\s*,\s*(?P<x>[-+0-9.eE]+)\s*,\s*"
    r"(?P<y>[-+0-9.eE]+)\s*,\s*(?P<z>[-+0-9.eE]+)\s*$"
)
NAMED_LANDMARK_RE = re.compile(
    r"^(?P<name>[^,\s]+)\s*,\s*(?P<x>[-+0-9.eE]+)\s*,\s*"
    r"(?P<y>[-+0-9.eE]+)\s*,\s*(?P<z>[-+0-9.eE]+)\s*$"
)


@dataclass(frozen=True)
class Sample:
    mesh_path: Path
    landmark_path: Path
    class_name: str
    gender: str
    subject_id: int

    @property
    def sample_id(self):
        prefix = "F" if self.gender == "women" else "M"
        return f"{self.class_name}_{prefix}{self.subject_id}"


def discover_samples(root_dir):
    root = Path(root_dir)
    samples = []
    for class_dir in sorted(root.glob("Class*")):
        if not class_dir.is_dir():
            continue
        landmark_root = class_dir / f"{class_dir.name}-Landmark"
        for gender in ("men", "women"):
            mesh_dir = class_dir / gender
            landmark_dir = landmark_root / gender
            for mesh_path in sorted(mesh_dir.glob("*.ply")) if mesh_dir.exists() else []:
                if not mesh_path.stem.isdigit():
                    continue
                subject_id = int(mesh_path.stem)
                prefix = "F" if gender == "women" else "M"
                landmark_path = landmark_dir / f"{class_dir.name}_{prefix}{subject_id}.txt"
                if landmark_path.exists():
                    samples.append(Sample(mesh_path, landmark_path, class_dir.name, gender, subject_id))
    if not samples:
        raise ValueError(f"No paired .ply/.txt samples found below {root}")
    return samples


def read_landmarks(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            match = LANDMARK_RE.match(text)
            if match:
                rows.append((int(match.group("idx")), [float(match.group(axis)) for axis in ("x", "y", "z")]))
                continue
            match = NAMED_LANDMARK_RE.match(text)
            if match:
                rows.append((len(rows), [float(match.group(axis)) for axis in ("x", "y", "z")]))
    if len(rows) != NUM_LANDMARKS or sorted(index for index, _ in rows) != list(range(NUM_LANDMARKS)):
        raise ValueError(f"{path} must contain exactly landmark indices 0..22")
    return np.asarray([xyz for _, xyz in sorted(rows)], dtype=np.float32)


def load_mesh(path):
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Mesh has no usable vertices/faces: {path}")
    return loaded


def load_split_file(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: list(payload[name]) for name in ("train", "val", "test")}


def assert_disjoint_splits(splits):
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(set(splits[left]) & set(splits[right]))
        if overlap:
            raise ValueError(f"Sample leakage between {left}/{right}: {overlap[:5]}")


def _infer_prediction_prefix(row, requested=None):
    if requested and f"{requested}_x" in row:
        return requested
    for prefix in ("stacked", "shape_prior", "final", "stage2_raw", "stage2_snapped", "pred", "base"):
        if f"{prefix}_x" in row:
            return prefix
    raise KeyError(f"Cannot infer prediction coordinate columns from {sorted(row)}")


def read_prediction_csv(path, prefix=None):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    chosen = _infer_prediction_prefix(rows[0], prefix)
    grouped = {}
    seen = {}
    for row in rows:
        sample_id = row["sample_id"]
        landmark = int(row["landmark"])
        grouped.setdefault(sample_id, np.zeros((NUM_LANDMARKS, 3), dtype=np.float32))
        seen.setdefault(sample_id, set())
        if landmark in seen[sample_id]:
            raise ValueError(f"Duplicate LM{landmark} for {sample_id} in {path}")
        seen[sample_id].add(landmark)
        grouped[sample_id][landmark] = [float(row[f"{chosen}_{axis}"]) for axis in ("x", "y", "z")]
    incomplete = [sample_id for sample_id, indices in seen.items() if indices != set(range(NUM_LANDMARKS))]
    if incomplete:
        raise ValueError(f"Incomplete prediction rows in {path}: {incomplete[:5]}")
    return grouped, chosen


def load_coarse_predictions(
    base_dir,
    initial_dir=None,
    base_prefix="stage2_raw",
    initial_prefix="stacked",
    require_train=True,
):
    """Use base train predictions and, when supplied, calibrated initial val/test predictions."""
    base_dir = Path(base_dir)
    initial_dir = Path(initial_dir) if initial_dir else None
    merged = {}
    sources = {}
    for split in ("train", "val", "test"):
        base_path = base_dir / f"refined_predictions_{split}.csv"
        if not base_path.exists() and split == "train" and not require_train:
            sources[split] = {"path": None, "prefix": None, "provenance": "train_override"}
            continue
        if not base_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {base_path}")
        source_path = base_path
        requested = base_prefix
        provenance = "base"
        if initial_dir and split in ("val", "test"):
            candidate = initial_dir / f"predictions_{split}.csv"
            if candidate.exists():
                source_path = candidate
                requested = initial_prefix
                provenance = "initial"
        values, selected = read_prediction_csv(source_path, requested)
        overlap = set(merged) & set(values)
        if overlap:
            raise ValueError(f"Prediction sample duplicated across splits: {sorted(overlap)[:5]}")
        merged.update(values)
        sources[split] = {"path": str(source_path), "prefix": selected, "provenance": provenance}
    return merged, sources


def resolve_legacy_matrix(root, sample):
    if root is None:
        return np.eye(4, dtype=np.float32)
    path = Path(root) / sample.class_name / sample.gender / f"{sample.subject_id}_transformation_matrix.npy"
    if not path.exists():
        raise FileNotFoundError(f"Legacy transform missing: {path}")
    return np.load(path).astype(np.float32)


def convert_legacy_coordinates(values, legacy_matrix, target_matrix):
    raw = apply_transform(values, np.linalg.inv(legacy_matrix))
    return apply_transform(raw, target_matrix)


def mesh_edges(faces, vertex_count):
    faces = np.asarray(faces, dtype=np.int64)
    undirected = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    undirected = np.sort(undirected, axis=1)
    undirected = np.unique(undirected, axis=0)
    directed = np.concatenate([undirected, undirected[:, ::-1]], axis=0)
    self_edges = np.arange(vertex_count, dtype=np.int64)
    directed = np.concatenate([directed, np.stack([self_edges, self_edges], axis=1)], axis=0)
    return directed.T.astype(np.int64)


def aggregate_neighbors(values, edge_index, vertex_count):
    src, dst = edge_index
    total = np.zeros((vertex_count, values.shape[1]), dtype=np.float64)
    count = np.zeros((vertex_count, 1), dtype=np.float64)
    np.add.at(total, dst, values[src])
    np.add.at(count, dst, 1.0)
    return (total / np.maximum(count, 1.0)).astype(np.float32)


def vertex_features(points, normals, rgb, edge_index):
    neighbor_rgb = aggregate_neighbors(rgb, edge_index, len(points))
    color_contrast = rgb - neighbor_rgb
    src, dst = edge_index
    edge_length = np.linalg.norm(points[src] - points[dst], axis=1, keepdims=True)
    density_total = np.zeros((len(points), 1), dtype=np.float64)
    density_count = np.zeros((len(points), 1), dtype=np.float64)
    np.add.at(density_total, dst, edge_length)
    np.add.at(density_count, dst, 1.0)
    density = density_total / np.maximum(density_count, 1.0)
    density /= max(float(np.median(density)), 1e-6)
    normal_difference = 1.0 - np.clip(np.sum(normals[src] * normals[dst], axis=1, keepdims=True), -1.0, 1.0)
    curvature_total = np.zeros((len(points), 1), dtype=np.float64)
    curvature_count = np.zeros((len(points), 1), dtype=np.float64)
    np.add.at(curvature_total, dst, normal_difference)
    np.add.at(curvature_count, dst, 1.0)
    curvature = curvature_total / np.maximum(curvature_count, 1.0)
    return np.concatenate(
        [points, rgb, color_contrast, normals, density.astype(np.float32), curvature.astype(np.float32)], axis=1
    ).astype(np.float32)


def prepare_mesh_record(sample, matrix, cache_dir, max_vertices=50000, include_landmarks=True):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_hash = hashlib.sha1(np.asarray(matrix, dtype=np.float32).tobytes()).hexdigest()[:10]
    label_tag = "labels" if include_landmarks else "geometry"
    path = cache_dir / f"{sample.sample_id}.{matrix_hash}.{label_tag}.npz"
    if path.exists():
        return path
    mesh = load_mesh(sample.mesh_path)
    points_raw = np.asarray(mesh.vertices, dtype=np.float32)
    if len(points_raw) > int(max_vertices):
        raise ValueError(
            f"{sample.sample_id} has {len(points_raw)} vertices, above --max-vertices={max_vertices}. "
            "Use a larger cap to preserve mesh topology."
        )
    points = apply_transform(points_raw, matrix)
    normals = rotate_vectors(np.asarray(mesh.vertex_normals, dtype=np.float32), matrix)
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(points):
        rgb = np.zeros((len(points), 3), dtype=np.float32)
        has_rgb = False
    else:
        rgb = np.asarray(colors, dtype=np.float32)[:, :3] / 255.0
        has_rgb = True
    edges = mesh_edges(np.asarray(mesh.faces), len(points))
    features = vertex_features(points, normals, rgb, edges)
    payload = {
        "points": points.astype(np.float32),
        "normals": normals.astype(np.float32),
        "rgb": rgb.astype(np.float32),
        "features": features,
        "edge_index": edges,
        "has_rgb": np.asarray([has_rgb], dtype=np.bool_),
    }
    if include_landmarks:
        payload["landmarks"] = apply_transform(read_landmarks(sample.landmark_path), matrix).astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def fit_feature_normalizer(samples, records, train_ids, output_path, cap_per_mesh=4096, seed=42):
    output_path = Path(output_path)
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    by_id = {sample.sample_id: sample for sample in samples}
    chunks = []
    rng = np.random.default_rng(seed)
    for sample_id in train_ids:
        with np.load(records[sample_id]) as data:
            features = data["features"].copy()
        if len(features) > cap_per_mesh:
            features = features[rng.choice(len(features), cap_per_mesh, replace=False)]
        chunks.append(features.astype(np.float64))
    values = np.concatenate(chunks, axis=0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.maximum(std, 1e-6)
    # Unit normals should preserve their physical range.
    mean[9:12] = 0.0
    std[9:12] = 1.0
    payload = {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "fit_sample_ids": list(train_ids),
        "feature_order": [
            "x", "y", "z", "r", "g", "b", "local_dr", "local_dg", "local_db",
            "nx", "ny", "nz", "density", "curvature",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def adjacency_list(edge_index, points):
    adjacency = [[] for _ in range(len(points))]
    src, dst = edge_index
    weights = np.linalg.norm(points[src] - points[dst], axis=1)
    for left, right, weight in zip(src.tolist(), dst.tolist(), weights.tolist()):
        if left != right:
            adjacency[left].append((right, float(weight)))
    return adjacency


def truncated_dijkstra(adjacency, seed, radius):
    distances = {int(seed): 0.0}
    queue = [(0.0, int(seed))]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node] or distance > radius:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate <= radius and candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    indices = np.fromiter(distances.keys(), dtype=np.int64)
    values = np.fromiter((distances[index] for index in indices), dtype=np.float32)
    return indices, values


def _select_roi(indices, distances, point_count, seed):
    order = np.argsort(distances)
    indices, distances = indices[order], distances[order]
    if len(indices) <= point_count:
        mask = np.zeros(point_count, dtype=np.bool_)
        mask[: len(indices)] = True
        pad = np.full(point_count - len(indices), indices[0], dtype=np.int64)
        return np.concatenate([indices, pad]), mask
    nearest_count = max(1, int(round(point_count * 0.65)))
    nearest = indices[:nearest_count]
    remaining = indices[nearest_count:]
    rng = np.random.default_rng(seed)
    spread = rng.choice(remaining, point_count - nearest_count, replace=False)
    selected = np.concatenate([nearest, spread])
    return selected.astype(np.int64), np.ones(point_count, dtype=np.bool_)


def build_roi_cache(
    record_path,
    coarse,
    roi_points,
    cache_dir,
    sample_id,
    seed,
    landmarks=None,
    radius_scale=1.0,
):
    radius_scale = float(radius_scale)
    digest_source = np.asarray(coarse, dtype=np.float32).tobytes() + np.float32(radius_scale).tobytes()
    digest = hashlib.sha1(digest_source).hexdigest()[:10]
    path = Path(cache_dir) / f"{sample_id}.v3.{roi_points}.r{radius_scale:g}.{digest}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    with np.load(record_path) as data:
        points = data["points"].copy()
        edge_index = data["edge_index"].copy()
        if landmarks is None:
            if "landmarks" not in data.files:
                raise ValueError(f"Landmarks must be supplied for geometry-only record: {record_path}")
            landmarks = data["landmarks"].copy()
    landmarks = np.asarray(landmarks, dtype=np.float32)
    tree = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(points)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    src, dst = edge_index
    edge_weights = np.linalg.norm(points[src] - points[dst], axis=1)
    graph = csr_matrix((edge_weights, (src, dst)), shape=(len(points), len(points)))
    coarse_seeds = tree.query(coarse, k=1)[1].astype(np.int64)
    expert_seeds = tree.query(landmarks, k=1)[1].astype(np.int64)
    max_radius = max(roi_radius_mm(index) for index in range(NUM_LANDMARKS)) * radius_scale
    distance_matrix = dijkstra(
        graph,
        directed=False,
        indices=np.concatenate([coarse_seeds, expert_seeds]),
        limit=max(90.0, max_radius * 2.0),
    ).astype(np.float32)
    roi_indices, roi_masks, targets, regions, oracles = [], [], [], [], []
    for landmark in range(NUM_LANDMARKS):
        radius = roi_radius_mm(landmark) * radius_scale
        coarse_distances = distance_matrix[landmark]
        candidates = np.flatnonzero(np.isfinite(coarse_distances) & (coarse_distances <= radius))
        geodesic = coarse_distances[candidates]
        if len(candidates) == 0:
            candidates = np.asarray([coarse_seeds[landmark]], dtype=np.int64)
            geodesic = np.asarray([0.0], dtype=np.float32)
        selected, mask = _select_roi(candidates, geodesic, roi_points, seed + landmark * 997)
        # Expert-distance heatmaps use mesh geodesics where connected, Euclidean fallback otherwise.
        selected_distances = distance_matrix[NUM_LANDMARKS + landmark, selected]
        missing = ~np.isfinite(selected_distances)
        if np.any(missing):
            selected_distances[missing] = np.linalg.norm(
                points[selected[missing]] - landmarks[landmark], axis=1
            )
        sigma = heatmap_sigma_mm(landmark)
        target = np.exp(-(selected_distances**2) / (2.0 * sigma**2)).astype(np.float32)
        target[~mask] = 0.0
        region = (selected_distances <= 3.5).astype(np.float32)
        region[~mask] = 0.0
        roi_indices.append(selected)
        roi_masks.append(mask)
        targets.append(target)
        regions.append(region)
        oracles.append(float(np.min(np.linalg.norm(points[selected[mask]] - landmarks[landmark], axis=1))))
    np.savez_compressed(
        path,
        roi_index=np.stack(roi_indices),
        roi_mask=np.stack(roi_masks),
        heatmap_target=np.stack(targets),
        region_target=np.stack(regions),
        oracle_error=np.asarray(oracles, dtype=np.float32),
    )
    return path


class RGBGeodesicDataset(Dataset):
    def __init__(
        self,
        samples,
        sample_ids,
        records,
        transforms,
        coarse_predictions,
        legacy_transform_dir,
        normalizer,
        cache_dir,
        roi_points=512,
        roi_radius_scale=1.0,
        training=False,
        rotation_degrees=0.0,
        point_noise_mm=0.0,
        rgb_noise=0.0,
        point_dropout=0.0,
        center_jitter_mm=0.0,
        use_rgb=True,
        coarse_in_target_space=False,
        memory_cache=True,
        seed=42,
    ):
        by_id = {sample.sample_id: sample for sample in samples}
        self.samples = [by_id[sample_id] for sample_id in sample_ids]
        self.records = records
        self.transforms = transforms
        self.coarse_predictions = coarse_predictions
        self.legacy_transform_dir = legacy_transform_dir
        self.mean = np.asarray(normalizer["mean"], dtype=np.float32)
        self.std = np.asarray(normalizer["std"], dtype=np.float32)
        self.cache_dir = Path(cache_dir)
        self.roi_points = int(roi_points)
        self.roi_radius_scale = float(roi_radius_scale)
        self.training = bool(training)
        self.rotation_degrees = float(rotation_degrees)
        self.point_noise_mm = float(point_noise_mm)
        self.rgb_noise = float(rgb_noise)
        self.point_dropout = float(point_dropout)
        self.center_jitter_mm = float(center_jitter_mm)
        self.use_rgb = bool(use_rgb)
        self.coarse_in_target_space = bool(coarse_in_target_space)
        self.memory_cache = bool(memory_cache)
        self._record_memory = {}
        self._roi_memory = {}
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _coarse(self, sample):
        if sample.sample_id not in self.coarse_predictions:
            raise KeyError(f"No coarse prediction for {sample.sample_id}")
        if self.coarse_in_target_space:
            return self.coarse_predictions[sample.sample_id].astype(np.float32).copy()
        legacy = resolve_legacy_matrix(self.legacy_transform_dir, sample)
        return convert_legacy_coordinates(
            self.coarse_predictions[sample.sample_id], legacy, self.transforms[sample.sample_id]
        ).astype(np.float32)

    def __getitem__(self, index):
        sample = self.samples[index]
        if sample.sample_id not in self._record_memory:
            with np.load(self.records[sample.sample_id]) as stored:
                record = {
                    "points": stored["points"].astype(np.float32),
                    "features": stored["features"].astype(np.float32),
                    "edge_index": stored["edge_index"].astype(np.int64),
                    "landmarks": stored["landmarks"].astype(np.float32) if "landmarks" in stored.files else None,
                }
            if self.memory_cache:
                self._record_memory[sample.sample_id] = record
        else:
            record = self._record_memory[sample.sample_id]
        points = record["points"].copy()
        raw_features = record["features"].copy()
        edge_index = record["edge_index"]
        if record["landmarks"] is None:
            expert = apply_transform(read_landmarks(sample.landmark_path), self.transforms[sample.sample_id]).astype(np.float32)
        else:
            expert = record["landmarks"].copy()
        coarse = self._coarse(sample)
        roi_path = build_roi_cache(
            self.records[sample.sample_id], coarse, self.roi_points,
            self.cache_dir / "roi", sample.sample_id, self.seed,
            landmarks=expert,
            radius_scale=self.roi_radius_scale,
        )
        if sample.sample_id not in self._roi_memory:
            with np.load(roi_path) as stored:
                roi = {key: stored[key].copy() for key in stored.files}
            if self.memory_cache:
                self._roi_memory[sample.sample_id] = roi
        else:
            roi = self._roi_memory[sample.sample_id]
        rng = np.random.default_rng(self.seed + index + self.epoch * 100_003)

        if self.training and self.center_jitter_mm > 0:
            coarse += rng.normal(0.0, self.center_jitter_mm, size=coarse.shape).astype(np.float32)

        if self.training and self.rotation_degrees > 0:
            from scipy.spatial.transform import Rotation

            angles = rng.uniform(-self.rotation_degrees, self.rotation_degrees, size=3)
            rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix().astype(np.float32)
            center = self.mean[:3]
            points = (points - center) @ rotation.T + center
            expert = (expert - center) @ rotation.T + center
            coarse = (coarse - center) @ rotation.T + center
            raw_features[:, :3] = points
            raw_features[:, 9:12] = raw_features[:, 9:12] @ rotation.T
        if self.training and self.point_noise_mm > 0:
            points += rng.normal(0.0, self.point_noise_mm, size=points.shape).astype(np.float32)
            raw_features[:, :3] = points
        if self.training and self.rgb_noise > 0:
            raw_features[:, 3:6] = np.clip(
                raw_features[:, 3:6] + rng.normal(0.0, self.rgb_noise, size=raw_features[:, 3:6].shape),
                0.0,
                1.0,
            )
        if not self.use_rgb:
            raw_features[:, 3:9] = 0.0

        vertex_mask = np.ones(len(points), dtype=np.bool_)
        if self.training and self.point_dropout > 0:
            vertex_mask = rng.random(len(points)) >= self.point_dropout
            # Every ROI retains a valid candidate; dropped vertices are excluded from edges and logits.
            vertex_mask[roi["roi_index"][:, 0]] = True
        features = ((raw_features - self.mean) / self.std).astype(np.float32)
        return {
            "sample_id": sample.sample_id,
            "class": sample.class_name,
            "gender": sample.gender,
            "subject_id": sample.subject_id,
            "points": torch.from_numpy(points),
            "features": torch.from_numpy(features),
            "edge_index": torch.from_numpy(edge_index),
            "vertex_mask": torch.from_numpy(vertex_mask),
            "coarse": torch.from_numpy(coarse),
            "expert": torch.from_numpy(expert),
            "roi_index": torch.from_numpy(roi["roi_index"].astype(np.int64)),
            "roi_mask": torch.from_numpy(roi["roi_mask"].astype(np.bool_)),
            "heatmap_target": torch.from_numpy(roi["heatmap_target"].astype(np.float32)),
            "region_target": torch.from_numpy(roi["region_target"].astype(np.float32)),
            "oracle_error": torch.from_numpy(roi["oracle_error"].astype(np.float32)),
        }


def collate_graphs(items):
    points, features, batches, edges = [], [], [], []
    roi_indices, roi_masks = [], []
    offset = 0
    for batch_index, item in enumerate(items):
        active = item["vertex_mask"]
        edge_index = item["edge_index"]
        edge_keep = active[edge_index[0]] & active[edge_index[1]]
        points.append(item["points"])
        features.append(item["features"])
        batches.append(torch.full((len(item["points"]),), batch_index, dtype=torch.long))
        edges.append(edge_index[:, edge_keep] + offset)
        roi_indices.append(item["roi_index"] + offset)
        roi_masks.append(item["roi_mask"] & active[item["roi_index"]])
        offset += len(item["points"])
    return {
        "sample_id": [item["sample_id"] for item in items],
        "class": [item["class"] for item in items],
        "gender": [item["gender"] for item in items],
        "subject_id": torch.tensor([item["subject_id"] for item in items], dtype=torch.long),
        "points": torch.cat(points, dim=0),
        "features": torch.cat(features, dim=0),
        "batch": torch.cat(batches, dim=0),
        "edge_index": torch.cat(edges, dim=1),
        "vertex_mask": torch.cat([item["vertex_mask"] for item in items], dim=0),
        "coarse": torch.stack([item["coarse"] for item in items]),
        "expert": torch.stack([item["expert"] for item in items]),
        "roi_index": torch.stack(roi_indices),
        "roi_mask": torch.stack(roi_masks),
        "heatmap_target": torch.stack([item["heatmap_target"] for item in items]),
        "region_target": torch.stack([item["region_target"] for item in items]),
        "oracle_error": torch.stack([item["oracle_error"] for item in items]),
    }
