import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
from trimesh.transformations import transform_points


LANDMARK_RE = re.compile(
    r"^Point\s*#(?P<idx>\d+)\s*,\s*(?P<x>[-+0-9.eE]+)\s*,\s*"
    r"(?P<y>[-+0-9.eE]+)\s*,\s*(?P<z>[-+0-9.eE]+)\s*$"
)
NAMED_LANDMARK_RE = re.compile(
    r"^(?P<name>[^,\s]+)\s*,\s*(?P<x>[-+0-9.eE]+)\s*,\s*"
    r"(?P<y>[-+0-9.eE]+)\s*,\s*(?P<z>[-+0-9.eE]+)\s*$"
)


@dataclass(frozen=True)
class OrthodonticSample:
    mesh_path: Path
    landmark_path: Path
    class_name: str
    gender: str
    subject_id: int

    @property
    def sample_id(self) -> str:
        gender_prefix = "M" if self.gender == "men" else "F"
        return f"{self.class_name}_{gender_prefix}{self.subject_id}"


def read_orthodontic_landmarks(path):
    coords = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            match = LANDMARK_RE.match(stripped)
            if match:
                idx = int(match.group("idx"))
                coords.append((idx, [float(match.group("x")), float(match.group("y")), float(match.group("z"))]))
                continue

            match = NAMED_LANDMARK_RE.match(stripped)
            if match:
                coords.append(
                    (
                        len(coords),
                        [float(match.group("x")), float(match.group("y")), float(match.group("z"))],
                    )
                )

    coords = [xyz for _, xyz in sorted(coords, key=lambda item: item[0])]
    if len(coords) != 23:
        raise ValueError(f"{path} has {len(coords)} landmarks; expected 23")
    return np.asarray(coords, dtype=np.float32)


def discover_orthodontic_samples(root_dir):
    root = Path(root_dir)
    samples = []
    missing_landmarks = []

    for class_dir in sorted(root.glob("Class*")):
        if not class_dir.is_dir():
            continue
        landmark_root = class_dir / f"{class_dir.name}-Landmark"
        for gender in ("men", "women"):
            mesh_dir = class_dir / gender
            lm_dir = landmark_root / gender
            if not mesh_dir.exists():
                continue
            for mesh_path in sorted(mesh_dir.glob("*.ply"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
                if not mesh_path.stem.isdigit():
                    continue
                subject_id = int(mesh_path.stem)
                gender_prefix = "M" if gender == "men" else "F"
                landmark_path = lm_dir / f"{class_dir.name}_{gender_prefix}{subject_id}.txt"
                if landmark_path.exists():
                    samples.append(
                        OrthodonticSample(
                            mesh_path=mesh_path,
                            landmark_path=landmark_path,
                            class_name=class_dir.name,
                            gender=gender,
                            subject_id=subject_id,
                        )
                    )
                else:
                    missing_landmarks.append(mesh_path)

    return samples, missing_landmarks


def ensure_point_count(points, count):
    if len(points) == count:
        return points.astype(np.float32)
    if len(points) == 0:
        raise ValueError("Cannot resample an empty point array")
    replace = len(points) < count
    indices = np.random.choice(len(points), count, replace=replace)
    return points[indices].astype(np.float32)


def mesh_normalization(vertices):
    center = vertices[:, :3].mean(axis=0, keepdims=True).astype(np.float32)
    scale = float(np.linalg.norm(vertices[:, :3] - center, axis=1).max())
    return center, scale if scale > 0 else 1.0


class OrthodonticDataset(Dataset):
    """Dataset adapter for data/dataset Class*/{men,women} PLY meshes and 23 TXT landmarks."""

    def __init__(
        self,
        root_dir,
        cache_dir=None,
        num_surface_points=10000,
        transform=None,
        normalize=False,
        transformation_dir=None,
    ):
        self.root_dir = Path(root_dir)
        self.samples, self.missing_landmarks = discover_orthodontic_samples(self.root_dir)
        if not self.samples:
            raise ValueError(f"No paired .ply/.txt samples found below {self.root_dir}")

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_surface_points = int(num_surface_points)
        self.transform = transform
        self.normalize = normalize
        self.transformation_dir = Path(transformation_dir) if transformation_dir else None

    def __len__(self):
        return len(self.samples)

    def _cache_path(self, sample):
        if not self.cache_dir:
            return None
        safe_mesh = str(sample.mesh_path.relative_to(self.root_dir)).replace(os.sep, "__")
        transform_tag = "aligned" if self.transformation_dir else "raw"
        norm_tag = "normalized" if self.normalize else "original"
        return self.cache_dir / f"{safe_mesh}.{self.num_surface_points}.{transform_tag}.{norm_tag}.npz"

    def _transformation_path(self, sample):
        if not self.transformation_dir:
            return None
        rel_parent = sample.mesh_path.relative_to(self.root_dir).parent
        return self.transformation_dir / rel_parent / f"{sample.mesh_path.stem}_transformation_matrix.npy"

    def _load_transformation(self, sample):
        path = self._transformation_path(sample)
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(f"Missing transformation matrix for {sample.sample_id}: {path}")
        matrix = np.load(path)
        if matrix.shape != (4, 4):
            raise ValueError(f"{path} has shape {matrix.shape}; expected (4, 4)")
        return matrix.astype(np.float64)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cache_path = self._cache_path(sample)

        if cache_path and cache_path.exists():
            data = np.load(cache_path)
            vertices_raw = data["vertices"]
            points = ensure_point_count(data["points"], self.num_surface_points)
            landmarks = data["landmarks"]
        else:
            mesh = trimesh.load(sample.mesh_path, force="mesh")
            vertices_raw = np.asarray(mesh.vertices, dtype=np.float32)
            if len(vertices_raw) == 0:
                raise ValueError(f"{sample.mesh_path} has no vertices")
            landmarks = read_orthodontic_landmarks(sample.landmark_path)

            matrix = self._load_transformation(sample)
            if matrix is not None:
                mesh.apply_transform(matrix)
                vertices_raw = np.asarray(mesh.vertices, dtype=np.float32)
                landmarks = transform_points(landmarks, matrix).astype(np.float32)

            if len(getattr(mesh, "faces", [])) > 0:
                points = trimesh.sample.sample_surface_even(mesh, self.num_surface_points)[0].astype(np.float32)
            else:
                replace = len(vertices_raw) < self.num_surface_points
                indices = np.random.choice(len(vertices_raw), self.num_surface_points, replace=replace)
                points = vertices_raw[indices].astype(np.float32)
            points = ensure_point_count(points, self.num_surface_points)

            if self.normalize:
                center, scale = mesh_normalization(vertices_raw)
                vertices_raw = (vertices_raw - center) / scale
                points = (points - center) / scale
                landmarks = (landmarks - center) / scale

            if cache_path:
                np.savez_compressed(cache_path, vertices=vertices_raw, points=points, landmarks=landmarks)

        points = torch.from_numpy(points.astype(np.float32))
        landmarks = torch.from_numpy(landmarks.astype(np.float32))
        vertices_raw = torch.from_numpy(vertices_raw.astype(np.float32))

        if self.transform:
            points, landmarks = self.transform(points, landmarks)

        return points, landmarks, vertices_raw

    def metadata(self, idx):
        return self.samples[idx]

    def normalization_params(self, idx):
        sample = self.samples[idx]
        mesh = trimesh.load(sample.mesh_path, force="mesh")
        vertices_raw = np.asarray(mesh.vertices, dtype=np.float32)
        return mesh_normalization(vertices_raw)
