import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from torch.utils.data import Dataset

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None


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
    def sample_id(self):
        prefix = "M" if self.gender == "men" else "F"
        return f"{self.class_name}_{prefix}{self.subject_id}"

    def patient_key(self, mode):
        if mode == "sample_id":
            return self.sample_id
        if mode == "gender_subject":
            prefix = "M" if self.gender == "men" else "F"
            return f"{prefix}{self.subject_id}"
        raise ValueError(f"Unsupported patient key mode: {mode}")


def read_landmarks(path):
    coords = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            match = LANDMARK_RE.match(text)
            if match:
                coords.append(
                    (
                        int(match.group("idx")),
                        [float(match.group("x")), float(match.group("y")), float(match.group("z"))],
                    )
                )
                continue
            match = NAMED_LANDMARK_RE.match(text)
            if match:
                coords.append((len(coords), [float(match.group("x")), float(match.group("y")), float(match.group("z"))]))
    if len(coords) != 23:
        raise ValueError(f"{path} has {len(coords)} landmarks; expected 23")
    idxs = [idx for idx, _ in coords]
    if sorted(idxs) != list(range(23)):
        raise ValueError(f"{path} landmark indices are not exactly 0..22")
    return np.asarray([xyz for _, xyz in sorted(coords, key=lambda item: item[0])], dtype=np.float32)


def discover_samples(root_dir):
    root = Path(root_dir)
    samples = []
    missing = []
    for class_dir in sorted(root.glob("Class*")):
        if not class_dir.is_dir():
            continue
        landmark_root = class_dir / f"{class_dir.name}-Landmark"
        for gender in ("men", "women"):
            mesh_dir = class_dir / gender
            landmark_dir = landmark_root / gender
            if not mesh_dir.exists():
                continue
            for mesh_path in sorted(mesh_dir.glob("*.ply"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
                if not mesh_path.stem.isdigit():
                    continue
                subject_id = int(mesh_path.stem)
                prefix = "M" if gender == "men" else "F"
                landmark_path = landmark_dir / f"{class_dir.name}_{prefix}{subject_id}.txt"
                if landmark_path.exists():
                    samples.append(OrthodonticSample(mesh_path, landmark_path, class_dir.name, gender, subject_id))
                else:
                    missing.append(str(mesh_path))
    return samples, missing


def normalize_vectors(vectors):
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(denom, 1e-8, None)


def local_geometry_features(points, k=16):
    density = np.zeros((len(points), 1), dtype=np.float32)
    curvature = np.zeros((len(points), 1), dtype=np.float32)
    if NearestNeighbors is None or len(points) < 4:
        return density, curvature
    k = min(int(k), len(points))
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    dists, idx = nbrs.kneighbors(points)
    kth = dists[:, -1:]
    density = (kth / max(float(np.median(kth)), 1e-6)).astype(np.float32)
    for i in range(len(points)):
        neigh = points[idx[i]]
        centered = neigh - neigh.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, len(neigh) - 1)
        eigvals = np.linalg.eigvalsh(cov)
        curvature[i, 0] = float(np.clip(eigvals[0] / max(float(eigvals.sum()), 1e-12), 0.0, 1.0))
    return density.astype(np.float32), curvature.astype(np.float32)


def make_patient_folds(samples, n_folds=5, patient_key_mode="gender_subject", seed=42, val_fraction=0.2):
    indices = np.arange(len(samples))
    groups = np.asarray([sample.patient_key(patient_key_mode) for sample in samples])
    if patient_key_mode == "sample_id":
        splitter = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
        outer = splitter.split(indices)
    else:
        splitter = GroupKFold(n_splits=int(n_folds))
        outer = splitter.split(indices, groups=groups)
    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(outer):
        train_val_idx = np.asarray(train_val_idx)
        train_val_groups = groups[train_val_idx]
        if len(np.unique(train_val_groups)) > 1:
            val_splitter = GroupShuffleSplit(n_splits=1, test_size=float(val_fraction), random_state=int(seed) + fold_idx)
            tr_local, val_local = next(val_splitter.split(train_val_idx, groups=train_val_groups))
            train_idx = train_val_idx[tr_local]
            val_idx = train_val_idx[val_local]
        else:
            cut = max(1, int(len(train_val_idx) * (1.0 - float(val_fraction))))
            train_idx, val_idx = train_val_idx[:cut], train_val_idx[cut:]
        folds.append({"train": train_idx.tolist(), "val": val_idx.tolist(), "test": np.asarray(test_idx).tolist()})
    return folds


def split_leakage_report(samples, split_indices, patient_key_mode):
    report = {}
    for split_name, idxs in split_indices.items():
        report[split_name] = {
            "n_samples": len(idxs),
            "sample_ids": [samples[i].sample_id for i in idxs],
            "patient_keys": [samples[i].patient_key(patient_key_mode) for i in idxs],
        }
    checks = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_ids = set(report[left]["sample_ids"])
        right_ids = set(report[right]["sample_ids"])
        left_keys = set(report[left]["patient_keys"])
        right_keys = set(report[right]["patient_keys"])
        checks[f"{left}_{right}"] = {
            "sample_id_overlap": sorted(left_ids & right_ids),
            "patient_key_overlap": sorted(left_keys & right_keys),
        }
    report["checks"] = checks
    return report


def apply_matrix(points, matrix):
    points = np.asarray(points, dtype=np.float32)
    homo = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (homo @ np.asarray(matrix, dtype=np.float32).T)[:, :3].astype(np.float32)


def rotate_vectors(vectors, matrix):
    rotation = np.asarray(matrix, dtype=np.float32)[:3, :3]
    return normalize_vectors(np.asarray(vectors, dtype=np.float32) @ rotation.T).astype(np.float32)


def load_mesh_sample(sample, num_points, seed):
    import trimesh

    mesh = trimesh.load(sample.mesh_path, force="mesh")
    rng = np.random.default_rng(int(seed))
    if len(getattr(mesh, "faces", [])) > 0:
        state = np.random.get_state()
        np.random.seed(int(seed))
        points, face_indices = trimesh.sample.sample_surface_even(mesh, int(num_points))
        if len(points) < int(num_points):
            extra, extra_faces = trimesh.sample.sample_surface(mesh, int(num_points) - len(points))
            points = np.concatenate([points, extra], axis=0)
            face_indices = np.concatenate([face_indices, extra_faces], axis=0)
        points = points[: int(num_points)].astype(np.float32)
        face_indices = face_indices[: int(num_points)]
        np.random.set_state(state)
        normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float32)
    else:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        replace = len(vertices) < int(num_points)
        idx = rng.choice(len(vertices), int(num_points), replace=replace)
        points = vertices[idx].astype(np.float32)
        normals = np.asarray(mesh.vertex_normals[idx], dtype=np.float32) if hasattr(mesh, "vertex_normals") else np.zeros_like(points)
    return points, normalize_vectors(normals).astype(np.float32)


def fit_normalizer(samples, train_idx, transforms, num_points, seed):
    chunks = []
    for idx in train_idx:
        sample = samples[idx]
        points, _ = load_mesh_sample(sample, min(int(num_points), 4096), int(seed) + idx)
        landmarks = read_landmarks(sample.landmark_path)
        matrix = transforms[sample.sample_id]
        chunks.append(apply_matrix(points, matrix))
        chunks.append(apply_matrix(landmarks, matrix))
    arr = np.concatenate(chunks, axis=0)
    center = arr.mean(axis=0).astype(np.float32)
    scale = float(np.linalg.norm(arr - center[None, :], axis=1).max())
    return {"center": center.astype(float).tolist(), "scale": scale if scale > 0 else 1.0}


class AtlasDataset(Dataset):
    def __init__(
        self,
        samples,
        indices,
        transforms,
        normalizer,
        cache_dir,
        num_points=4096,
        local_geometry_k=16,
        use_normals=True,
        use_curvature=True,
        seed=42,
    ):
        self.samples = list(samples)
        self.indices = list(indices)
        self.transforms = transforms
        self.normalizer = normalizer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_points = int(num_points)
        self.local_geometry_k = int(local_geometry_k)
        self.use_normals = bool(use_normals)
        self.use_curvature = bool(use_curvature)
        self.seed = int(seed)

    def __len__(self):
        return len(self.indices)

    def metadata(self, local_idx):
        return self.samples[self.indices[local_idx]]

    def _cache_path(self, sample):
        return self.cache_dir / f"{sample.sample_id}.{self.num_points}.npz"

    def __getitem__(self, local_idx):
        sample_idx = self.indices[local_idx]
        sample = self.samples[sample_idx]
        cache_path = self._cache_path(sample)
        center = np.asarray(self.normalizer["center"], dtype=np.float32)
        scale = float(self.normalizer["scale"])
        if cache_path.exists():
            data = np.load(cache_path)
            points_world = data["points_world"]
            normals = data["normals"]
            landmarks_world = data["landmarks_world"]
        else:
            points, normals = load_mesh_sample(sample, self.num_points, self.seed + sample_idx)
            landmarks = read_landmarks(sample.landmark_path)
            matrix = self.transforms[sample.sample_id]
            points_world = apply_matrix(points, matrix)
            landmarks_world = apply_matrix(landmarks, matrix)
            normals = rotate_vectors(normals, matrix)
            np.savez_compressed(cache_path, points_world=points_world, normals=normals, landmarks_world=landmarks_world)
        points_norm = ((points_world - center[None, :]) / scale).astype(np.float32)
        landmarks_norm = ((landmarks_world - center[None, :]) / scale).astype(np.float32)
        parts = [points_norm]
        if self.use_normals:
            parts.append(normals.astype(np.float32))
        if self.use_curvature:
            density, curvature = local_geometry_features(points_norm, self.local_geometry_k)
            parts.extend([density, curvature])
        features = np.concatenate(parts, axis=1).astype(np.float32)
        return {
            "points_norm": torch.tensor(points_norm, dtype=torch.float32),
            "features": torch.tensor(features, dtype=torch.float32),
            "landmarks_norm": torch.tensor(landmarks_norm, dtype=torch.float32),
            "points_world": torch.tensor(points_world, dtype=torch.float32),
            "landmarks_world": torch.tensor(landmarks_world, dtype=torch.float32),
            "sample_index": torch.tensor(sample_idx, dtype=torch.long),
            "scale": torch.tensor([scale], dtype=torch.float32),
        }
