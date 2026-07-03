import argparse
import csv
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import trimesh
from trimesh.transformations import transform_points

for parent in Path(__file__).resolve().parents:
    if (parent / "shared_metrics" / "orthodontic_analysis.py").exists():
        sys.path.append(str(parent))
        break

from shared_metrics.orthodontic_analysis import build_error_analysis, write_analysis_csvs


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
            lm_dir = landmark_root / gender
            if not mesh_dir.exists():
                continue
            for mesh_path in sorted(mesh_dir.glob("*.ply"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
                if not mesh_path.stem.isdigit():
                    continue
                subject_id = int(mesh_path.stem)
                prefix = "M" if gender == "men" else "F"
                landmark_path = lm_dir / f"{class_dir.name}_{prefix}{subject_id}.txt"
                if landmark_path.exists():
                    samples.append(OrthodonticSample(mesh_path, landmark_path, class_dir.name, gender, subject_id))
                else:
                    missing.append(str(mesh_path))
    return samples, missing


def mesh_normalization(points):
    center = points.mean(axis=0, keepdims=True).astype(np.float32)
    scale = float(np.linalg.norm(points - center, axis=1).max())
    return center, scale if scale > 0 else 1.0


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def summarize_errors(errors):
    errors = np.asarray(errors, dtype=np.float64)
    summary = {
        "ale": float(errors.mean()),
        "std": float(errors.std()),
        "median": float(np.median(errors)),
        "max": float(errors.max()),
        "per_landmark_ale": errors.reshape((-1, 23)).mean(axis=0).astype(float).tolist(),
        "per_sample_ale": errors.reshape((-1, 23)).mean(axis=1).astype(float).tolist(),
    }
    for threshold in (2.0, 2.5, 3.0):
        key = ("%g" % threshold).replace(".", "_")
        summary[f"pck_at_{key}mm"] = float((errors <= threshold).mean())
    return summary


def analysis_from_eval_rows(dataset, eval_rows):
    samples = []
    error_rows = []
    for sample_idx, _, _, errors in eval_rows:
        samples.append(dataset.metadata(sample_idx))
        error_rows.append(np.asarray(errors, dtype=np.float64))
    return build_error_analysis(samples, np.stack(error_rows, axis=0))


class OrthodonticPointCloudDataset(Dataset):
    def __init__(
        self,
        root_dir,
        cache_dir,
        num_points=1024,
        mask_radius=3.5,
        target_mode="gaussian",
        heatmap_sigma=3.5,
        use_normals=True,
        transformation_dir=None,
        seed=42,
    ):
        self.root_dir = Path(root_dir)
        self.samples, self.missing_landmarks = discover_samples(self.root_dir)
        if not self.samples:
            raise ValueError(f"No paired samples found below {self.root_dir}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_points = int(num_points)
        self.mask_radius = float(mask_radius)
        self.target_mode = target_mode
        self.heatmap_sigma = float(heatmap_sigma)
        self.use_normals = bool(use_normals)
        self.transformation_dir = Path(transformation_dir) if transformation_dir else None
        self.seed = int(seed)

    def __len__(self):
        return len(self.samples)

    def _transform_path(self, sample):
        if self.transformation_dir is None:
            return None
        rel_parent = sample.mesh_path.relative_to(self.root_dir).parent
        return self.transformation_dir / rel_parent / f"{sample.mesh_path.stem}_transformation_matrix.npy"

    def _cache_path(self, sample):
        transform_tag = "aligned" if self.transformation_dir else "raw"
        feature_tag = "xyz_normal" if self.use_normals else "xyz"
        if self.target_mode == "gaussian":
            target_tag = f"gauss{self.heatmap_sigma:g}"
        else:
            target_tag = f"mask{self.mask_radius:g}"
        safe_name = str(sample.mesh_path.relative_to(self.root_dir)).replace(os.sep, "__")
        return self.cache_dir / f"{safe_name}.{self.num_points}.{transform_tag}.{feature_tag}.{target_tag}.npz"

    def _load_arrays(self, idx):
        sample = self.samples[idx]
        cache_path = self._cache_path(sample)
        if cache_path.exists():
            data = np.load(cache_path)
            return (
                data["points_norm"],
                data["features"],
                data["points_world"],
                data["landmarks_norm"],
                data["landmarks_world"],
                data["targets"],
            )

        mesh = trimesh.load(sample.mesh_path, force="mesh")
        landmarks = read_landmarks(sample.landmark_path)
        transform_path = self._transform_path(sample)
        if transform_path is not None:
            if not transform_path.exists():
                raise FileNotFoundError(f"Missing transform for {sample.sample_id}: {transform_path}")
            matrix = np.load(transform_path)
            mesh.apply_transform(matrix)
            landmarks = transform_points(landmarks, matrix).astype(np.float32)

        rng = np.random.default_rng(self.seed + idx)
        if len(getattr(mesh, "faces", [])) > 0:
            state = np.random.get_state()
            np.random.seed(self.seed + idx)
            points_world, face_indices = trimesh.sample.sample_surface_even(mesh, self.num_points)
            if len(points_world) < self.num_points:
                extra_points, extra_faces = trimesh.sample.sample_surface(mesh, self.num_points - len(points_world))
                points_world = np.concatenate([points_world, extra_points], axis=0)
                face_indices = np.concatenate([face_indices, extra_faces], axis=0)
            elif len(points_world) > self.num_points:
                points_world = points_world[: self.num_points]
                face_indices = face_indices[: self.num_points]
            np.random.set_state(state)
            points_world = points_world.astype(np.float32)
            normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float32)
        else:
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            replace = len(vertices) < self.num_points
            vertex_indices = rng.choice(len(vertices), self.num_points, replace=replace)
            points_world = vertices[vertex_indices].astype(np.float32)
            if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) == len(vertices):
                normals = np.asarray(mesh.vertex_normals[vertex_indices], dtype=np.float32)
            else:
                normals = np.zeros_like(points_world, dtype=np.float32)

        center, scale = mesh_normalization(points_world)
        points_norm = ((points_world - center) / scale).astype(np.float32)
        landmarks_norm = ((landmarks - center) / scale).astype(np.float32)
        normals = normalize_vectors(normals).astype(np.float32)
        features = np.concatenate([points_norm, normals], axis=1).astype(np.float32) if self.use_normals else points_norm
        world_dists = np.linalg.norm(points_world[:, None, :] - landmarks[None, :, :], axis=-1)
        if self.target_mode == "gaussian":
            sigma = max(self.heatmap_sigma, 1e-6)
            targets = np.exp(-(world_dists**2) / (2.0 * sigma**2)).astype(np.float32)
        else:
            targets = (world_dists <= self.mask_radius).astype(np.float32)
        np.savez_compressed(
            cache_path,
            points_norm=points_norm,
            features=features,
            points_world=points_world.astype(np.float32),
            landmarks_norm=landmarks_norm.astype(np.float32),
            landmarks_world=landmarks.astype(np.float32),
            targets=targets,
        )
        return points_norm, features, points_world, landmarks_norm, landmarks, targets

    def __getitem__(self, idx):
        points_norm, features, points_world, landmarks_norm, landmarks_world, targets = self._load_arrays(idx)
        return {
            "points_norm": torch.tensor(points_norm, dtype=torch.float32),
            "features": torch.tensor(features, dtype=torch.float32),
            "points_world": torch.tensor(points_world, dtype=torch.float32),
            "landmarks_norm": torch.tensor(landmarks_norm, dtype=torch.float32),
            "landmarks_world": torch.tensor(landmarks_world, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "sample_index": torch.tensor(idx, dtype=torch.long),
        }

    def metadata(self, idx):
        return self.samples[idx]


def square_distance(src, dst):
    return torch.cdist(src, dst, p=2) ** 2


def index_points(points, idx):
    device = points.device
    batch = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    batch, n_points, _ = xyz.shape
    npoint = min(int(npoint), n_points)
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=device)
    distance = torch.ones(batch, n_points, device=device) * 1e10
    farthest = torch.randint(0, n_points, (batch,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_point(nsample, xyz, new_xyz):
    nsample = min(int(nsample), xyz.shape[1])
    dist = square_distance(new_xyz, xyz)
    return dist.topk(nsample, dim=-1, largest=False, sorted=False)[1]


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = int(npoint)
        self.nsample = int(nsample)
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.extend(
                [
                    nn.Conv2d(last_channel, out_channel, 1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True),
                ]
            )
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = knn_point(self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=2)[0].transpose(1, 2).contiguous()
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.extend(
                [
                    nn.Conv1d(last_channel, out_channel, 1, bias=False),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU(inplace=True),
                ]
            )
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        if xyz2.shape[1] == 1:
            interpolated = points2.repeat(1, xyz1.shape[1], 1)
        else:
            dist = square_distance(xyz1, xyz2)
            k = min(3, xyz2.shape[1])
            dist, idx = dist.topk(k, dim=-1, largest=False, sorted=False)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            grouped_points = index_points(points2, idx)
            interpolated = torch.sum(grouped_points * weight[:, :, :, None], dim=2)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated
        new_points = new_points.transpose(1, 2).contiguous()
        new_points = self.mlp(new_points)
        return new_points.transpose(1, 2).contiguous()


class PointNet2LandmarkSeg(nn.Module):
    def __init__(self, num_landmarks=23, input_dim=6, sa1_points=256, sa2_points=64, sa3_points=16, nsample=32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.sa1 = PointNetSetAbstraction(sa1_points, nsample, 3 + self.input_dim, [32, 32, 64])
        self.sa2 = PointNetSetAbstraction(sa2_points, nsample, 64 + 3, [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(sa3_points, nsample, 128 + 3, [128, 256, 512])
        self.fp3 = PointNetFeaturePropagation(512 + 128, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 64, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + self.input_dim, [128, 128, 128])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(128, num_landmarks, 1),
        )

    def forward(self, xyz, features=None):
        if features is None:
            features = xyz
        l0_xyz = xyz
        l0_points = features
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        logits = self.head(l0_points.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        return logits


def differentiable_landmark_coordinates(logits, points_norm, temperature=1.0):
    weights = torch.softmax(logits / max(float(temperature), 1e-6), dim=1)
    return torch.einsum("bnl,bnd->bld", weights, points_norm)


def landmark_loss(
    logits,
    targets,
    points_norm=None,
    landmarks_norm=None,
    target_mode="gaussian",
    positive_weight=0.0,
    dice_weight=0.0,
    coord_weight=0.25,
    coord_temperature=1.0,
):
    if target_mode == "gaussian":
        heatmap_loss = F.mse_loss(torch.sigmoid(logits), targets)
    else:
        if positive_weight > 0:
            pos_weight = torch.full((logits.shape[-1],), positive_weight, dtype=logits.dtype, device=logits.device)
        else:
            positives = targets.sum(dim=(0, 1)).clamp_min(1.0)
            negatives = (targets.shape[0] * targets.shape[1] - targets.sum(dim=(0, 1))).clamp_min(1.0)
            pos_weight = (negatives / positives).clamp(max=200.0)
        heatmap_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    if dice_weight > 0:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(0, 1))
        union = probs.sum(dim=(0, 1)) + targets.sum(dim=(0, 1))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
        heatmap_loss = heatmap_loss + float(dice_weight) * dice

    if coord_weight <= 0 or points_norm is None or landmarks_norm is None:
        return heatmap_loss

    pred_norm = differentiable_landmark_coordinates(logits, points_norm, temperature=coord_temperature)
    coord_loss = F.smooth_l1_loss(pred_norm, landmarks_norm)
    return heatmap_loss + float(coord_weight) * coord_loss


def predict_landmarks(logits, points_world, postprocess="softmax", temperature=1.0, topk=10):
    logits = logits.detach().cpu().numpy()
    points_world = points_world.detach().cpu().numpy()
    batch, n_points, n_landmarks = logits.shape
    preds = np.zeros((batch, n_landmarks, 3), dtype=np.float32)
    for b in range(batch):
        for lm_idx in range(n_landmarks):
            scores = logits[b, :, lm_idx]
            if postprocess == "argmax":
                preds[b, lm_idx] = points_world[b, scores.argmax()]
                continue
            if postprocess == "topk_softmax":
                k = min(int(topk), n_points)
                idx = np.argpartition(scores, -k)[-k:]
                local_scores = scores[idx]
                local_points = points_world[b, idx]
            else:
                local_scores = scores
                local_points = points_world[b]
            local_scores = local_scores / max(float(temperature), 1e-6)
            local_scores = local_scores - local_scores.max()
            weights = np.exp(local_scores)
            weights = weights / max(weights.sum(), 1e-12)
            preds[b, lm_idx] = np.sum(local_points * weights[:, None], axis=0)
    return preds


def make_splits(dataset, test_size, val_size, seed):
    indices = np.arange(len(dataset))
    strata = np.array([f"{s.class_name}_{s.gender}" for s in dataset.samples])
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=strata)
    val_fraction = val_size / (1.0 - test_size)
    train_val_strata = strata[train_val_idx]
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_fraction, random_state=seed, stratify=train_val_strata)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def ids_to_indices(dataset, sample_ids):
    index_by_id = {sample.sample_id: idx for idx, sample in enumerate(dataset.samples)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"Split file references samples not found in dataset: {missing[:10]}")
    return [index_by_id[sample_id] for sample_id in sample_ids]


def limit_samples_balanced(samples, max_samples, seed):
    if max_samples is None or max_samples >= len(samples):
        return samples
    rng = random.Random(seed)
    groups = {}
    for sample in samples:
        groups.setdefault((sample.class_name, sample.gender), []).append(sample)
    selected = []
    group_keys = sorted(groups)
    per_group = max_samples // len(group_keys)
    remainder = max_samples % len(group_keys)
    for idx, key in enumerate(group_keys):
        group = groups[key][:]
        rng.shuffle(group)
        take = per_group + (1 if idx < remainder else 0)
        selected.extend(group[:take])
    return sorted(selected, key=lambda s: (s.class_name, s.gender, s.subject_id))


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    target_mode,
    positive_weight,
    dice_weight,
    coord_weight,
    coord_temperature,
):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        points = batch["points_norm"].to(device)
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        landmarks_norm = batch["landmarks_norm"].to(device)
        logits = model(points, features)
        loss = landmark_loss(
            logits,
            targets,
            points_norm=points,
            landmarks_norm=landmarks_norm,
            target_mode=target_mode,
            positive_weight=positive_weight,
            dice_weight=dice_weight,
            coord_weight=coord_weight,
            coord_temperature=coord_temperature,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * points.shape[0]
    return total / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    target_mode,
    positive_weight,
    dice_weight,
    coord_weight,
    coord_temperature,
    postprocess,
    temperature,
    topk,
):
    model.eval()
    total = 0.0
    rows = []
    all_errors = []
    for batch in tqdm(loader, desc="eval", leave=False):
        points = batch["points_norm"].to(device)
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        landmarks_norm = batch["landmarks_norm"].to(device)
        logits = model(points, features)
        loss = landmark_loss(
            logits,
            targets,
            points_norm=points,
            landmarks_norm=landmarks_norm,
            target_mode=target_mode,
            positive_weight=positive_weight,
            dice_weight=dice_weight,
            coord_weight=coord_weight,
            coord_temperature=coord_temperature,
        )
        total += float(loss.detach().cpu()) * points.shape[0]
        preds = predict_landmarks(logits, batch["points_world"], postprocess=postprocess, temperature=temperature, topk=topk)
        landmarks = batch["landmarks_world"].cpu().numpy()
        errors = np.linalg.norm(preds - landmarks, axis=-1)
        all_errors.extend(errors.reshape(-1).tolist())
        for row_idx in range(points.shape[0]):
            rows.append((int(batch["sample_index"][row_idx]), preds[row_idx], landmarks[row_idx], errors[row_idx]))
    return total / max(1, len(loader.dataset)), rows, np.asarray(all_errors, dtype=np.float32)


def write_predictions(path, dataset, eval_rows):
    fieldnames = [
        "sample_id",
        "class",
        "gender",
        "subject_id",
        "landmark",
        "expert_x",
        "expert_y",
        "expert_z",
        "pointnet2_x",
        "pointnet2_y",
        "pointnet2_z",
        "localization_error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx, pred, expert, errors in eval_rows:
            meta = dataset.metadata(sample_idx)
            for lm_idx in range(23):
                writer.writerow(
                    {
                        "sample_id": meta.sample_id,
                        "class": meta.class_name,
                        "gender": meta.gender,
                        "subject_id": meta.subject_id,
                        "landmark": lm_idx,
                        "expert_x": expert[lm_idx, 0],
                        "expert_y": expert[lm_idx, 1],
                        "expert_z": expert[lm_idx, 2],
                        "pointnet2_x": pred[lm_idx, 0],
                        "pointnet2_y": pred[lm_idx, 1],
                        "pointnet2_z": pred[lm_idx, 2],
                        "localization_error": errors[lm_idx],
                    }
                )


def write_group_metrics(path, dataset, eval_rows):
    groups = {}
    for sample_idx, _, _, errors in eval_rows:
        meta = dataset.metadata(sample_idx)
        groups.setdefault((meta.class_name, meta.gender), []).extend(errors.tolist())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "gender", "n_samples", "ale", "std", "median"])
        writer.writeheader()
        for (class_name, gender), errors in sorted(groups.items()):
            arr = np.asarray(errors, dtype=np.float64)
            writer.writerow(
                {
                    "class": class_name,
                    "gender": gender,
                    "n_samples": int(len(arr) / 23),
                    "ale": float(arr.mean()),
                    "std": float(arr.std()),
                    "median": float(np.median(arr)),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Train PointNet++ for orthodontic 3D landmark localization.")
    parser.add_argument("--data-root", default="../data/dataset")
    parser.add_argument("--output-dir", default="runs/pointnet2_orthodontic_maskseg")
    parser.add_argument("--transformation-dir", default=None)
    parser.add_argument("--surface-points", type=int, default=1024)
    parser.add_argument("--eval-surface-points", type=int, default=None)
    parser.add_argument("--mask-radius", type=float, default=3.5)
    parser.add_argument("--target-mode", choices=["gaussian", "mask"], default="gaussian")
    parser.add_argument("--heatmap-sigma", type=float, default=3.5)
    parser.add_argument("--use-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--positive-weight", type=float, default=0.0)
    parser.add_argument("--dice-weight", type=float, default=0.0)
    parser.add_argument("--coord-weight", type=float, default=0.25)
    parser.add_argument("--coord-temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="cosine")
    parser.add_argument("--sa1-points", type=int, default=256)
    parser.add_argument("--sa2-points", type=int, default=64)
    parser.add_argument("--sa3-points", type=int, default=16)
    parser.add_argument("--nsample", type=int, default=32)
    parser.add_argument("--postprocess", choices=["softmax", "topk_softmax", "argmax"], default="softmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--splits-json", default=None, help="Shared split JSON with train/val/test sample_id lists.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = OrthodonticPointCloudDataset(
        root_dir=args.data_root,
        cache_dir=output_dir / "point_cache",
        num_points=args.surface_points,
        mask_radius=args.mask_radius,
        target_mode=args.target_mode,
        heatmap_sigma=args.heatmap_sigma,
        use_normals=args.use_normals,
        transformation_dir=args.transformation_dir,
        seed=args.seed,
    )
    if args.max_samples is not None:
        dataset.samples = limit_samples_balanced(dataset.samples, args.max_samples, args.seed)
    print(f"Paired samples: {len(dataset)}")
    print(f"Meshes without matching landmark file: {len(dataset.missing_landmarks)}")
    print(f"Device: {device}")

    source_splits_json = None
    if args.splits_json:
        source_splits_json = str(Path(args.splits_json))
        split_source = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))
        train_idx = ids_to_indices(dataset, split_source["train"])
        val_idx = ids_to_indices(dataset, split_source["val"])
        test_idx = ids_to_indices(dataset, split_source["test"])
    else:
        train_idx, val_idx, test_idx = make_splits(dataset, args.test_size, args.val_size, args.seed)
    split_payload = {
        "train": [dataset.samples[i].sample_id for i in train_idx],
        "val": [dataset.samples[i].sample_id for i in val_idx],
        "test": [dataset.samples[i].sample_id for i in test_idx],
        "source_splits_json": source_splits_json,
        "missing_landmark_meshes": dataset.missing_landmarks,
    }
    (output_dir / "splits.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    eval_dataset = dataset
    if args.eval_surface_points is not None and args.eval_surface_points != args.surface_points:
        eval_dataset = OrthodonticPointCloudDataset(
            root_dir=args.data_root,
            cache_dir=output_dir / "point_cache",
            num_points=args.eval_surface_points,
            mask_radius=args.mask_radius,
            target_mode=args.target_mode,
            heatmap_sigma=args.heatmap_sigma,
            use_normals=args.use_normals,
            transformation_dir=args.transformation_dir,
            seed=args.seed,
        )
        if args.max_samples is not None:
            eval_dataset.samples = limit_samples_balanced(eval_dataset.samples, args.max_samples, args.seed)
    test_loader = DataLoader(Subset(eval_dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PointNet2LandmarkSeg(
        num_landmarks=23,
        input_dim=6 if args.use_normals else 3,
        sa1_points=args.sa1_points,
        sa2_points=args.sa2_points,
        sa3_points=args.sa3_points,
        nsample=args.nsample,
    ).to(device)

    if args.evaluate_only:
        model_path = Path(args.model_path) if args.model_path else output_dir / "best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_loss, test_rows, test_errors = evaluate(
            model,
            test_loader,
            device,
            args.target_mode,
            args.positive_weight,
            args.dice_weight,
            args.coord_weight,
            args.coord_temperature,
            args.postprocess,
            args.temperature,
            args.topk,
        )
        metrics = {
            "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
            "unit": "dataset coordinate unit after optional transformation",
            "clinical_threshold_unit": "mm",
            "model": "PointNet++ set abstraction / feature propagation landmark mask segmentation",
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "surface_points": args.surface_points,
            "eval_surface_points": args.eval_surface_points or args.surface_points,
            "mask_radius": args.mask_radius,
            "target_mode": args.target_mode,
            "heatmap_sigma": args.heatmap_sigma,
            "use_normals": args.use_normals,
            "positive_weight": args.positive_weight,
            "dice_weight": args.dice_weight,
            "coord_weight": args.coord_weight,
            "coord_temperature": args.coord_temperature,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "sa1_points": args.sa1_points,
            "sa2_points": args.sa2_points,
            "sa3_points": args.sa3_points,
            "nsample": args.nsample,
            "postprocess": args.postprocess,
            "temperature": args.temperature,
            "topk": args.topk,
            "test_loss": test_loss,
            "pointnet2": summarize_errors(test_errors),
        }
        advanced_analysis = analysis_from_eval_rows(eval_dataset, test_rows)
        metrics.update(advanced_analysis)
        (output_dir / "metrics_eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_predictions(output_dir / "predictions_eval_test.csv", eval_dataset, test_rows)
        write_group_metrics(output_dir / "group_metrics_eval_test.csv", eval_dataset, test_rows)
        write_analysis_csvs(output_dir, advanced_analysis, suffix="eval_test")
        print("\nEvaluation against expert orthodontist landmarks")
        print(f"PointNet++ ALE: {metrics['pointnet2']['ale']:.4f}")
        print(f"PointNet++ median: {metrics['pointnet2']['median']:.4f}")
        print(f"Evaluation saved to: {output_dir}")
        return

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    else:
        scheduler = None

    history = []
    best_val_ale = math.inf
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.target_mode,
            args.positive_weight,
            args.dice_weight,
            args.coord_weight,
            args.coord_temperature,
        )
        val_loss, _, val_errors = evaluate(
            model,
            val_loader,
            device,
            args.target_mode,
            args.positive_weight,
            args.dice_weight,
            args.coord_weight,
            args.coord_temperature,
            args.postprocess,
            args.temperature,
            args.topk,
        )
        if args.scheduler == "plateau" and scheduler is not None:
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        val_ale = float(val_errors.mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ale": val_ale,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f} val_ALE={val_ale:.4f}")
        if val_ale < best_val_ale:
            best_val_ale = val_ale
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    test_loss, test_rows, test_errors = evaluate(
        model,
        test_loader,
        device,
        args.target_mode,
        args.positive_weight,
        args.dice_weight,
        args.coord_weight,
        args.coord_temperature,
        args.postprocess,
        args.temperature,
        args.topk,
    )

    metrics = {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "unit": "dataset coordinate unit after optional transformation",
        "clinical_threshold_unit": "mm",
        "model": "PointNet++ set abstraction / feature propagation landmark mask segmentation",
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "surface_points": args.surface_points,
        "eval_surface_points": args.eval_surface_points or args.surface_points,
        "mask_radius": args.mask_radius,
        "target_mode": args.target_mode,
        "heatmap_sigma": args.heatmap_sigma,
        "use_normals": args.use_normals,
        "positive_weight": args.positive_weight,
        "dice_weight": args.dice_weight,
        "coord_weight": args.coord_weight,
        "coord_temperature": args.coord_temperature,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "sa1_points": args.sa1_points,
        "sa2_points": args.sa2_points,
        "sa3_points": args.sa3_points,
        "nsample": args.nsample,
        "postprocess": args.postprocess,
        "temperature": args.temperature,
        "topk": args.topk,
        "best_val_ale": best_val_ale,
        "test_loss": test_loss,
        "pointnet2": summarize_errors(test_errors),
    }
    advanced_analysis = analysis_from_eval_rows(eval_dataset, test_rows)
    metrics.update(advanced_analysis)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "predictions_test.csv", eval_dataset, test_rows)
    write_group_metrics(output_dir / "group_metrics_test.csv", eval_dataset, test_rows)
    write_analysis_csvs(output_dir, advanced_analysis, suffix="test")

    print("\nEvaluation against expert orthodontist landmarks")
    print(f"PointNet++ ALE: {metrics['pointnet2']['ale']:.4f}")
    print(f"PointNet++ median: {metrics['pointnet2']['median']:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
