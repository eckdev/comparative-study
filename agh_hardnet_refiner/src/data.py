import csv
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


HARD_LANDMARKS = (0, 21, 22)
CORE20 = tuple(i for i in range(23) if i not in HARD_LANDMARKS)
NEIGHBORS = {
    0: (1, 2, 3, 4, 5),
    21: (17, 19, 13, 20, 22),
    22: (18, 20, 16, 19, 21),
}


@dataclass(frozen=True)
class PredictionSample:
    sample_id: str
    class_name: str
    gender: str
    subject_id: int
    expert: np.ndarray
    base: np.ndarray

    @property
    def gender_dir(self):
        return "men" if self.gender in ("men", "male", "M") else "women"

    @property
    def cache_glob(self):
        return f"{self.class_name}__{self.gender_dir}__{self.subject_id}.ply.*.npz"


def parse_sample_id(sample_id):
    class_name, rest = sample_id.split("_", 1)
    gender = "men" if rest[0].upper() == "M" else "women"
    subject_id = int(rest[1:])
    return class_name, gender, subject_id


def _detect_base_prefix(fieldnames, requested):
    if requested != "auto":
        return requested
    candidates = ("final", "shape_prior", "stage2_raw", "stage2_snapped", "stage1_snapped", "stage1_raw", "pred", "base")
    for prefix in candidates:
        if all(f"{prefix}_{axis}" in fieldnames for axis in "xyz"):
            return prefix
    raise ValueError(f"Could not detect prediction coordinate prefix from columns: {fieldnames}")


def read_prediction_csv(path, base_prefix="auto"):
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        prefix = _detect_base_prefix(fieldnames, base_prefix)
        grouped = {}
        meta = {}
        for row in reader:
            sample_id = row["sample_id"]
            lm = int(row["landmark"])
            if sample_id not in grouped:
                grouped[sample_id] = {
                    "expert": np.zeros((23, 3), dtype=np.float32),
                    "base": np.zeros((23, 3), dtype=np.float32),
                }
                class_name = row.get("class") or row.get("class_name")
                gender = row.get("gender", "")
                subject_id = row.get("subject_id")
                if not class_name or not subject_id:
                    class_name, parsed_gender, parsed_subject = parse_sample_id(sample_id)
                    gender = gender or parsed_gender
                    subject_id = subject_id or parsed_subject
                meta[sample_id] = (class_name, gender, int(subject_id))
            grouped[sample_id]["expert"][lm] = [float(row[f"expert_{axis}"]) for axis in "xyz"]
            grouped[sample_id]["base"][lm] = [float(row[f"{prefix}_{axis}"]) for axis in "xyz"]
    samples = []
    for sample_id in sorted(grouped):
        class_name, gender, subject_id = meta[sample_id]
        samples.append(
            PredictionSample(
                sample_id=sample_id,
                class_name=class_name,
                gender=gender,
                subject_id=subject_id,
                expert=grouped[sample_id]["expert"],
                base=grouped[sample_id]["base"],
            )
        )
    return samples, prefix


def _read_ply_vertices_fallback(path):
    path = Path(path)
    with open(path, "rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY file without end_header: {path}")
            text = line.decode("ascii", errors="ignore").strip()
            header.append(text)
            if text == "end_header":
                break
        fmt = next((line for line in header if line.startswith("format ")), "")
        vertex_count = 0
        props = []
        in_vertex = False
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            if in_vertex and line.startswith("property "):
                props.append(line.split()[-1])
        if vertex_count <= 0:
            raise ValueError(f"No vertices found in PLY: {path}")
        xyz_idx = [props.index(axis) for axis in ("x", "y", "z")]
        if "ascii" in fmt:
            rows = []
            for _ in range(vertex_count):
                parts = handle.readline().decode("ascii", errors="ignore").strip().split()
                rows.append([float(parts[i]) for i in xyz_idx])
            return np.asarray(rows, dtype=np.float32)
        if "binary_little_endian" not in fmt:
            raise ValueError(f"Unsupported PLY format without trimesh: {fmt}")
        # Minimal common case reader: float properties in vertex block.
        if not all("float" in line for line in header if line.startswith("property ") and len(props) > 0):
            # Most orthodontic PLYs here are simple float vertex files; fail loudly otherwise.
            raise ValueError(f"Binary PLY fallback supports float vertex properties only: {path}")
        row_fmt = "<" + "f" * len(props)
        row_size = struct.calcsize(row_fmt)
        rows = np.zeros((vertex_count, 3), dtype=np.float32)
        for i in range(vertex_count):
            vals = struct.unpack(row_fmt, handle.read(row_size))
            rows[i] = [vals[j] for j in xyz_idx]
        return rows


def load_points_for_sample(sample, data_root=None, point_cache_dir=None, max_points=12000, seed=42):
    rng = np.random.default_rng(int(seed))
    if point_cache_dir:
        cache_dir = Path(point_cache_dir)
        matches = sorted(cache_dir.glob(sample.cache_glob))
        if matches:
            data = np.load(matches[0])
            points = data["points_world"].astype(np.float32)
            features = data["features"].astype(np.float32) if "features" in data.files else None
            normals = features[:, 3:6].astype(np.float32) if features is not None and features.shape[1] >= 6 else np.zeros_like(points)
            extra = features[:, 6:8].astype(np.float32) if features is not None and features.shape[1] >= 8 else np.zeros((len(points), 2), dtype=np.float32)
            return _subsample(points, normals, extra, max_points, rng)
    if data_root is None:
        raise FileNotFoundError(f"No point cache for {sample.sample_id}; provide --data-root or --point-cache-dir")
    mesh_path = Path(data_root) / sample.class_name / sample.gender_dir / f"{sample.subject_id}.ply"
    try:
        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh")
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32) if hasattr(mesh, "vertex_normals") else np.zeros_like(vertices)
    except Exception:
        vertices = _read_ply_vertices_fallback(mesh_path)
        normals = np.zeros_like(vertices)
    extra = np.zeros((len(vertices), 2), dtype=np.float32)
    return _subsample(vertices, normals, extra, max_points, rng)


def _subsample(points, normals, extra, max_points, rng):
    if max_points and len(points) > int(max_points):
        idx = rng.choice(len(points), int(max_points), replace=False)
        return points[idx].astype(np.float32), normals[idx].astype(np.float32), extra[idx].astype(np.float32)
    return points.astype(np.float32), normals.astype(np.float32), extra.astype(np.float32)


def patch_indices(points, center, radius_mm, patch_points):
    dists = np.linalg.norm(points - center[None, :], axis=1)
    idx = np.where(dists <= float(radius_mm))[0]
    if len(idx) >= int(patch_points):
        local = idx[np.argsort(dists[idx])[: int(patch_points)]]
    else:
        local = np.argsort(dists)[: int(patch_points)]
    return local.astype(np.int64), dists


class HardLandmarkDataset(Dataset):
    def __init__(
        self,
        samples,
        data_root=None,
        point_cache_dir=None,
        landmarks=HARD_LANDMARKS,
        radius_mm=45.0,
        trichion_radius_mm=50.0,
        patch_points=2048,
        max_surface_points=12000,
        sigma_mm=2.5,
        seed=42,
        max_items=None,
    ):
        self.samples = list(samples)
        self.data_root = data_root
        self.point_cache_dir = point_cache_dir
        self.landmarks = tuple(int(x) for x in landmarks)
        self.radius_mm = float(radius_mm)
        self.trichion_radius_mm = float(trichion_radius_mm)
        self.patch_points = int(patch_points)
        self.max_surface_points = int(max_surface_points)
        self.sigma_mm = float(sigma_mm)
        self.seed = int(seed)
        items = [(sample_idx, lm) for sample_idx in range(len(self.samples)) for lm in self.landmarks]
        self.items = items[: int(max_items)] if max_items else items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample_idx, lm = self.items[idx]
        sample = self.samples[sample_idx]
        points, normals, extra = load_points_for_sample(
            sample,
            data_root=self.data_root,
            point_cache_dir=self.point_cache_dir,
            max_points=self.max_surface_points,
            seed=self.seed + sample_idx,
        )
        center = sample.base[lm].astype(np.float32)
        expert = sample.expert[lm].astype(np.float32)
        radius = self.trichion_radius_mm if lm == 0 else self.radius_mm
        idxs, _ = patch_indices(points, center, radius, self.patch_points)
        patch = points[idxs]
        patch_normals = normals[idxs]
        patch_extra = extra[idxs]
        neigh = np.asarray(NEIGHBORS[lm], dtype=np.int64)
        neighbor_coords = sample.base[neigh].astype(np.float32)
        rel_center = (patch - center[None, :]) / max(radius, 1.0)
        dist_center = np.linalg.norm(patch - center[None, :], axis=1, keepdims=True) / max(radius, 1.0)
        neighbor_parts = []
        for coord in neighbor_coords:
            vec = (patch - coord[None, :]) / 80.0
            dist = np.linalg.norm(patch - coord[None, :], axis=1, keepdims=True) / 80.0
            neighbor_parts.extend([vec, dist])
        cand_features = np.concatenate([rel_center, dist_center, patch_normals, patch_extra] + neighbor_parts, axis=1).astype(np.float32)
        dist_expert = np.linalg.norm(patch - expert[None, :], axis=1)
        soft = np.exp(-(dist_expert**2) / (2.0 * self.sigma_mm**2)).astype(np.float32)
        if float(soft.sum()) <= 1e-12:
            soft[np.argmin(dist_expert)] = 1.0
        soft = soft / max(float(soft.sum()), 1e-12)
        return {
            "candidate_features": torch.tensor(cand_features, dtype=torch.float32),
            "candidate_points": torch.tensor(patch.astype(np.float32), dtype=torch.float32),
            "soft_labels": torch.tensor(soft, dtype=torch.float32),
            "base": torch.tensor(center, dtype=torch.float32),
            "expert": torch.tensor(expert, dtype=torch.float32),
            "landmark": torch.tensor(lm, dtype=torch.long),
            "sample_index": torch.tensor(sample_idx, dtype=torch.long),
        }


def oracle_rows(samples, data_root=None, point_cache_dir=None, radius_values=(30, 40, 50), patch_points=4096, max_surface_points=12000, seed=42):
    rows = []
    for sample_idx, sample in enumerate(samples):
        points, _, _ = load_points_for_sample(
            sample,
            data_root=data_root,
            point_cache_dir=point_cache_dir,
            max_points=max_surface_points,
            seed=seed + sample_idx,
        )
        tree = cKDTree(points)
        for lm in HARD_LANDMARKS:
            center = sample.base[lm]
            expert = sample.expert[lm]
            surface_d, _ = tree.query(expert, k=1)
            base_error = float(np.linalg.norm(center - expert))
            for radius in radius_values:
                idxs, _ = patch_indices(points, center, float(radius), patch_points)
                d = np.linalg.norm(points[idxs] - expert[None, :], axis=1)
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "class": sample.class_name,
                        "gender": sample.gender,
                        "subject_id": sample.subject_id,
                        "landmark": lm,
                        "radius_mm": float(radius),
                        "n_candidates": int(len(idxs)),
                        "base_error": base_error,
                        "surface_nearest_error": float(surface_d),
                        "oracle_error": float(d.min()),
                        "oracle_pck_at_2mm": float(d.min() <= 2.0),
                        "oracle_pck_at_3mm": float(d.min() <= 3.0),
                    }
                )
    return rows
