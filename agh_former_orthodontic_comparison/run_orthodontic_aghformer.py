import argparse
import csv
import json
import math
import os
import random
import re
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

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None


for parent in Path(__file__).resolve().parents:
    if (parent / "shared_metrics" / "orthodontic_analysis.py").exists():
        import sys

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
                coords.append((len(coords), [float(match.group("x")), float(match.group("y")), float(match.group("z"))]))
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


def normalize_vectors(vectors):
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(denom, 1e-8, None)


def mesh_normalization(points):
    center = points.mean(axis=0, keepdims=True).astype(np.float32)
    scale = float(np.linalg.norm(points - center, axis=1).max())
    return center, scale if scale > 0 else 1.0


def local_geometry_features(points, k=16):
    n_points = len(points)
    density = np.zeros((n_points, 1), dtype=np.float32)
    curvature = np.zeros((n_points, 1), dtype=np.float32)
    if NearestNeighbors is None or n_points < 4:
        return density, curvature
    k = min(int(k), n_points)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(points)
    dists, idx = nbrs.kneighbors(points)
    kth = dists[:, -1:]
    density = (kth / max(float(np.median(kth)), 1e-6)).astype(np.float32)
    for i in range(n_points):
        neigh = points[idx[i]]
        centered = neigh - neigh.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, len(neigh) - 1)
        eigvals = np.linalg.eigvalsh(cov)
        denom = float(np.clip(eigvals.sum(), 1e-12, None))
        curvature[i, 0] = float(np.clip(eigvals[0] / denom, 0.0, 1.0))
    return density.astype(np.float32), curvature.astype(np.float32)


class AGHFormerDataset(Dataset):
    def __init__(
        self,
        root_dir,
        cache_dir,
        num_points=4096,
        heatmap_sigma=5.0,
        use_normals=True,
        use_local_geometry=True,
        local_geometry_k=16,
        transformation_dir=None,
        seed=42,
    ):
        self.root_dir = Path(root_dir)
        self.samples, self.missing_landmarks = discover_samples(self.root_dir)
        if not self.samples:
            raise ValueError(f"No paired .ply/.txt samples found below {self.root_dir}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_points = int(num_points)
        self.heatmap_sigma = float(heatmap_sigma)
        self.use_normals = bool(use_normals)
        self.use_local_geometry = bool(use_local_geometry)
        self.local_geometry_k = int(local_geometry_k)
        self.transformation_dir = Path(transformation_dir) if transformation_dir else None
        self.seed = int(seed)

    def __len__(self):
        return len(self.samples)

    def metadata(self, idx):
        return self.samples[idx]

    def _transform_path(self, sample):
        if self.transformation_dir is None:
            return None
        rel_parent = sample.mesh_path.relative_to(self.root_dir).parent
        return self.transformation_dir / rel_parent / f"{sample.mesh_path.stem}_transformation_matrix.npy"

    def _cache_path(self, sample):
        transform_tag = "aligned" if self.transformation_dir else "raw"
        normal_tag = "normals" if self.use_normals else "nonormals"
        geom_tag = f"geom{self.local_geometry_k}" if self.use_local_geometry else "nogeom"
        safe_name = str(sample.mesh_path.relative_to(self.root_dir)).replace(os.sep, "__")
        return self.cache_dir / f"{safe_name}.{self.num_points}.{transform_tag}.{normal_tag}.{geom_tag}.sigma{self.heatmap_sigma:g}.npz"

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
                data["center"],
                data["scale"],
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
        feature_parts = [points_norm]
        if self.use_normals:
            feature_parts.append(normals)
        if self.use_local_geometry:
            density, curvature = local_geometry_features(points_norm, self.local_geometry_k)
            feature_parts.extend([density, curvature])
        features = np.concatenate(feature_parts, axis=1).astype(np.float32)
        world_dists = np.linalg.norm(points_world[:, None, :] - landmarks[None, :, :], axis=-1)
        sigma = max(float(self.heatmap_sigma), 1e-6)
        targets = np.exp(-(world_dists**2) / (2.0 * sigma**2)).astype(np.float32)

        np.savez_compressed(
            cache_path,
            points_norm=points_norm,
            features=features,
            points_world=points_world,
            landmarks_norm=landmarks_norm,
            landmarks_world=landmarks.astype(np.float32),
            targets=targets,
            center=center.astype(np.float32),
            scale=np.asarray([scale], dtype=np.float32),
        )
        return points_norm, features, points_world, landmarks_norm, landmarks, targets, center.astype(np.float32), np.asarray([scale], dtype=np.float32)

    def __getitem__(self, idx):
        points_norm, features, points_world, landmarks_norm, landmarks_world, targets, center, scale = self._load_arrays(idx)
        return {
            "points_norm": torch.tensor(points_norm, dtype=torch.float32),
            "features": torch.tensor(features, dtype=torch.float32),
            "points_world": torch.tensor(points_world, dtype=torch.float32),
            "landmarks_norm": torch.tensor(landmarks_norm, dtype=torch.float32),
            "landmarks_world": torch.tensor(landmarks_world, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "center": torch.tensor(center.reshape(3), dtype=torch.float32),
            "scale": torch.tensor(scale.reshape(1), dtype=torch.float32),
            "sample_index": torch.tensor(idx, dtype=torch.long),
        }


def parse_pairs(text):
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split("-")
        pairs.append((int(left), int(right)))
    return pairs


def parse_indices(text):
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_anatomical_adjacency(num_landmarks, symmetry_pairs, midline_indices):
    adjacency = torch.eye(num_landmarks, dtype=torch.float32)
    groups = [
        list(range(0, 5)),
        list(range(5, 10)),
        list(range(10, 18)),
        list(range(18, 23)),
    ]
    for group in groups:
        for i in group:
            for j in group:
                if i < num_landmarks and j < num_landmarks:
                    adjacency[i, j] = 1.0
    for i, j in symmetry_pairs:
        if i < num_landmarks and j < num_landmarks:
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
    for i in midline_indices:
        for j in midline_indices:
            if i < num_landmarks and j < num_landmarks:
                adjacency[i, j] = 1.0
    return adjacency


class SurfaceEncoderBlock(nn.Module):
    def __init__(self, width, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width * 3, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, x):
        global_mean = x.mean(dim=1, keepdim=True).expand_as(x)
        global_max = x.max(dim=1, keepdim=True).values.expand_as(x)
        update = self.mlp(torch.cat([self.norm(x), global_mean, global_max], dim=-1))
        return x + update


class GraphTokenBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))

    def forward(self, tokens, attn_mask=None):
        x = self.norm1(tokens)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class CrossTokenBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.surface_norm = nn.LayerNorm(width)
        self.cross_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.graph = GraphTokenBlock(width, heads, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, tokens, surface, graph_mask=None):
        q = self.token_norm(tokens)
        kv = self.surface_norm(surface)
        cross, _ = self.cross_attn(q, kv, kv, need_weights=False)
        tokens = tokens + cross
        return self.graph(tokens, attn_mask=graph_mask)


class AGHFormer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_landmarks=23,
        width=128,
        blocks=3,
        heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        graph_adjacency=None,
        residual_scale=0.08,
    ):
        super().__init__()
        self.num_landmarks = int(num_landmarks)
        self.width = int(width)
        self.residual_scale = float(residual_scale)
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.surface_blocks = nn.ModuleList([SurfaceEncoderBlock(width, dropout=dropout) for _ in range(max(1, blocks // 2))])
        self.landmark_tokens = nn.Parameter(torch.randn(num_landmarks, width) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [CrossTokenBlock(width, heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(blocks)]
        )
        self.surface_heat = nn.Linear(width, width, bias=False)
        self.token_heat = nn.Linear(width, width, bias=False)
        self.coord_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.log_var_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_landmarks))
        if graph_adjacency is None:
            graph_adjacency = torch.ones(num_landmarks, num_landmarks)
        disallowed = graph_adjacency <= 0
        graph_mask = torch.zeros_like(graph_adjacency, dtype=torch.float32)
        graph_mask[disallowed] = -10000.0
        self.register_buffer("graph_mask", graph_mask)

    def forward(self, points_norm, features):
        surface = self.feature_proj(features)
        for block in self.surface_blocks:
            surface = block(surface)
        batch = features.shape[0]
        tokens = self.landmark_tokens.unsqueeze(0).expand(batch, -1, -1)
        for block in self.cross_blocks:
            tokens = block(tokens, surface, self.graph_mask)
        surface_key = F.normalize(self.surface_heat(surface), dim=-1)
        token_query = F.normalize(self.token_heat(tokens), dim=-1)
        logits = torch.einsum("bnc,blc->bnl", surface_key, token_query) * math.sqrt(self.width)
        weights = torch.softmax(logits, dim=1)
        heatmap_coords = torch.einsum("bnl,bnd->bld", weights, points_norm)
        residual = torch.tanh(self.coord_head(tokens)) * self.residual_scale
        pred_norm = heatmap_coords + residual
        log_vars = self.log_var_head(tokens).diagonal(dim1=1, dim2=2).contiguous()
        return {"logits": logits, "pred_norm": pred_norm, "residual_norm": residual, "log_vars": log_vars}


def sharpen_targets(targets, sigma_start, sigma_current):
    sigma_start = max(float(sigma_start), 1e-6)
    sigma_current = max(float(sigma_current), 1e-6)
    exponent = (sigma_start / sigma_current) ** 2
    return torch.clamp(targets, 0.0, 1.0) ** exponent


def pairwise_structure_loss(pred_norm, target_norm):
    pred_dist = torch.cdist(pred_norm, pred_norm, p=2)
    target_dist = torch.cdist(target_norm, target_norm, p=2)
    n = pred_dist.shape[1]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=pred_dist.device), diagonal=1)
    return F.smooth_l1_loss(pred_dist[:, mask], target_dist[:, mask])


def symmetry_loss(pred_norm, symmetry_pairs, midline_indices):
    if not symmetry_pairs:
        return pred_norm.new_tensor(0.0)
    midline = pred_norm[:, midline_indices, 0].mean(dim=1, keepdim=True) if midline_indices else pred_norm[:, :, 0].mean(dim=1, keepdim=True)
    losses = []
    for left, right in symmetry_pairs:
        left_p = pred_norm[:, left]
        right_p = pred_norm[:, right]
        x_loss = F.smooth_l1_loss((left_p[:, 0] + right_p[:, 0]) * 0.5, midline.squeeze(1))
        yz_loss = F.smooth_l1_loss(left_p[:, 1:], right_p[:, 1:])
        losses.append(x_loss + 0.25 * yz_loss)
    return torch.stack(losses).mean()


def clinical_threshold_loss(pred_norm, target_norm, scale, threshold_mm=2.0, margin_mm=0.5):
    err_mm = torch.linalg.norm(pred_norm - target_norm, dim=-1) * scale.view(-1, 1)
    return F.softplus((err_mm - float(threshold_mm)) / max(float(margin_mm), 1e-6)).mean()


def uncertainty_loss(pred_norm, target_norm, log_vars):
    err = torch.linalg.norm(pred_norm - target_norm, dim=-1).detach()
    log_vars = torch.clamp(log_vars, min=-6.0, max=6.0)
    return (torch.exp(-log_vars) * err + log_vars).mean()


def aghformer_loss(
    outputs,
    targets,
    landmarks_norm,
    scale,
    sigma_start,
    sigma_current,
    coord_weight,
    structure_weight,
    symmetry_weight,
    clinical_weight,
    uncertainty_weight,
    symmetry_pairs,
    midline_indices,
    clinical_threshold_mm,
):
    target_heatmaps = sharpen_targets(targets, sigma_start, sigma_current)
    heatmap_loss = F.mse_loss(torch.sigmoid(outputs["logits"]), target_heatmaps)
    coord_loss = F.smooth_l1_loss(outputs["pred_norm"], landmarks_norm)
    loss = heatmap_loss + float(coord_weight) * coord_loss
    structure = pairwise_structure_loss(outputs["pred_norm"], landmarks_norm)
    sym = symmetry_loss(outputs["pred_norm"], symmetry_pairs, midline_indices)
    clinical = clinical_threshold_loss(outputs["pred_norm"], landmarks_norm, scale, clinical_threshold_mm)
    uncertain = uncertainty_loss(outputs["pred_norm"], landmarks_norm, outputs["log_vars"])
    loss = (
        loss
        + float(structure_weight) * structure
        + float(symmetry_weight) * sym
        + float(clinical_weight) * clinical
        + float(uncertainty_weight) * uncertain
    )
    return loss, {
        "heatmap_loss": float(heatmap_loss.detach().cpu()),
        "coord_loss": float(coord_loss.detach().cpu()),
        "structure_loss": float(structure.detach().cpu()),
        "symmetry_loss": float(sym.detach().cpu()),
        "clinical_loss": float(clinical.detach().cpu()),
        "uncertainty_loss": float(uncertain.detach().cpu()),
    }


def random_rotation_matrix(batch, max_deg, device):
    max_rad = math.radians(float(max_deg))
    angles = (torch.rand(batch, 3, device=device) * 2.0 - 1.0) * max_rad
    cx, cy, cz = torch.cos(angles[:, 0]), torch.cos(angles[:, 1]), torch.cos(angles[:, 2])
    sx, sy, sz = torch.sin(angles[:, 0]), torch.sin(angles[:, 1]), torch.sin(angles[:, 2])
    zeros = torch.zeros(batch, device=device)
    ones = torch.ones(batch, device=device)
    rx = torch.stack([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=1).view(batch, 3, 3)
    ry = torch.stack([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=1).view(batch, 3, 3)
    rz = torch.stack([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=1).view(batch, 3, 3)
    return rz @ ry @ rx


def augment_batch(points_norm, features, landmarks_norm, rotation_aug_deg, point_jitter_std, feature_dropout, use_normals):
    if rotation_aug_deg > 0:
        rot = random_rotation_matrix(points_norm.shape[0], rotation_aug_deg, points_norm.device)
        points_norm = torch.einsum("bij,bnj->bni", rot, points_norm)
        landmarks_norm = torch.einsum("bij,blj->bli", rot, landmarks_norm)
        features = features.clone()
        features[:, :, :3] = points_norm
        if use_normals and features.shape[-1] >= 6:
            features[:, :, 3:6] = torch.einsum("bij,bnj->bni", rot, features[:, :, 3:6])
    if point_jitter_std > 0:
        jitter = torch.randn_like(points_norm) * float(point_jitter_std)
        points_norm = points_norm + jitter
        features = features.clone()
        features[:, :, :3] = points_norm
    if feature_dropout > 0:
        keep = (torch.rand(features.shape[:2], device=features.device) > float(feature_dropout)).float().unsqueeze(-1)
        features = features * keep
    return points_norm, features, landmarks_norm


def predict_landmarks(outputs, points_world, scale, postprocess="topk_softmax", temperature=1.0, topk=30, snap=True):
    logits = outputs["logits"].detach().cpu().numpy()
    residual_norm = outputs["residual_norm"].detach().cpu().numpy()
    points_world_np = points_world.detach().cpu().numpy()
    scale_np = scale.detach().cpu().numpy().reshape(-1)
    batch, n_points, n_landmarks = logits.shape
    preds = np.zeros((batch, n_landmarks, 3), dtype=np.float32)
    raw_preds = np.zeros((batch, n_landmarks, 3), dtype=np.float32)
    for b in range(batch):
        for lm_idx in range(n_landmarks):
            scores = logits[b, :, lm_idx]
            if postprocess == "argmax":
                base = points_world_np[b, scores.argmax()]
            else:
                if postprocess == "topk_softmax":
                    k = min(int(topk), n_points)
                    idx = np.argpartition(scores, -k)[-k:]
                    local_scores = scores[idx]
                    local_points = points_world_np[b, idx]
                else:
                    local_scores = scores
                    local_points = points_world_np[b]
                local_scores = local_scores / max(float(temperature), 1e-6)
                local_scores = local_scores - local_scores.max()
                weights = np.exp(local_scores)
                weights = weights / max(float(weights.sum()), 1e-12)
                base = np.sum(local_points * weights[:, None], axis=0)
            raw = base + residual_norm[b, lm_idx] * scale_np[b]
            raw_preds[b, lm_idx] = raw
            if snap:
                nearest = np.linalg.norm(points_world_np[b] - raw[None, :], axis=1).argmin()
                preds[b, lm_idx] = points_world_np[b, nearest]
            else:
                preds[b, lm_idx] = raw
    return raw_preds, preds


def summarize_errors(errors):
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
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
    for sample_idx, _, pred_snapped, expert, _, snapped_errors, _ in eval_rows:
        samples.append(dataset.metadata(sample_idx))
        error_rows.append(np.asarray(snapped_errors, dtype=np.float64))
    return build_error_analysis(samples, np.stack(error_rows, axis=0))


def structure_metrics(eval_rows):
    rows = []
    all_errors = []
    for sample_idx, _, pred, expert, _, _, _ in eval_rows:
        pred_dist = np.linalg.norm(pred[:, None, :] - pred[None, :, :], axis=-1)
        expert_dist = np.linalg.norm(expert[:, None, :] - expert[None, :, :], axis=-1)
        mask = np.triu(np.ones_like(pred_dist, dtype=bool), k=1)
        err = np.abs(pred_dist[mask] - expert_dist[mask])
        all_errors.extend(err.tolist())
        rows.append({"sample_index": sample_idx, "mean_pair_distance_error": float(err.mean()), "median_pair_distance_error": float(np.median(err)), "max_pair_distance_error": float(err.max())})
    arr = np.asarray(all_errors, dtype=np.float64)
    return rows, {
        "mean_pair_distance_error": float(arr.mean()),
        "median_pair_distance_error": float(np.median(arr)),
        "max_pair_distance_error": float(arr.max()),
    }


def uncertainty_metrics(eval_rows):
    uncertainties = []
    errors = []
    for _, _, _, _, _, snapped_errors, log_vars in eval_rows:
        uncertainties.extend(np.exp(log_vars).reshape(-1).tolist())
        errors.extend(np.asarray(snapped_errors).reshape(-1).tolist())
    if len(errors) < 2 or np.std(uncertainties) == 0 or np.std(errors) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(uncertainties, errors)[0, 1])
    return {"uncertainty_error_pearson": corr, "mean_uncertainty": float(np.mean(uncertainties)), "median_uncertainty": float(np.median(uncertainties))}


def train_epoch(model, loader, optimizer, device, args, symmetry_pairs, midline_indices, epoch):
    model.train()
    total = 0.0
    parts_total = {}
    progress = tqdm(loader, desc="train", leave=False, disable=args.no_tqdm)
    denom = max(1, args.epochs - 1)
    sigma_current = args.heatmap_sigma_start + (args.heatmap_sigma_end - args.heatmap_sigma_start) * ((epoch - 1) / denom)
    for batch in progress:
        points = batch["points_norm"].to(device)
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        landmarks_norm = batch["landmarks_norm"].to(device)
        scale = batch["scale"].to(device)
        points, features, landmarks_norm = augment_batch(
            points,
            features,
            landmarks_norm,
            args.rotation_aug_deg,
            args.point_jitter_std,
            args.feature_dropout,
            args.use_normals,
        )
        outputs = model(points, features)
        loss, parts = aghformer_loss(
            outputs,
            targets,
            landmarks_norm,
            scale,
            args.heatmap_sigma_start,
            sigma_current,
            args.coord_weight,
            args.structure_weight,
            args.symmetry_weight,
            args.clinical_weight,
            args.uncertainty_weight,
            symmetry_pairs,
            midline_indices,
            args.clinical_threshold_mm,
        )
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(loss.detach().cpu()) * points.shape[0]
        for key, value in parts.items():
            parts_total[key] = parts_total.get(key, 0.0) + value * points.shape[0]
    n = max(1, len(loader.dataset))
    return total / n, {key: value / n for key, value in parts_total.items()}, sigma_current


@torch.no_grad()
def evaluate(model, loader, device, args, symmetry_pairs, midline_indices, sigma_current=None):
    model.eval()
    total = 0.0
    parts_total = {}
    rows = []
    raw_errors_all = []
    snapped_errors_all = []
    sigma_current = args.heatmap_sigma_end if sigma_current is None else sigma_current
    progress = tqdm(loader, desc="eval", leave=False, disable=args.no_tqdm)
    for batch in progress:
        points = batch["points_norm"].to(device)
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        landmarks_norm = batch["landmarks_norm"].to(device)
        scale = batch["scale"].to(device)
        outputs = model(points, features)
        loss, parts = aghformer_loss(
            outputs,
            targets,
            landmarks_norm,
            scale,
            args.heatmap_sigma_start,
            sigma_current,
            args.coord_weight,
            args.structure_weight,
            args.symmetry_weight,
            args.clinical_weight,
            args.uncertainty_weight,
            symmetry_pairs,
            midline_indices,
            args.clinical_threshold_mm,
        )
        total += float(loss.detach().cpu()) * points.shape[0]
        for key, value in parts.items():
            parts_total[key] = parts_total.get(key, 0.0) + value * points.shape[0]
        raw_preds, snapped_preds = predict_landmarks(
            outputs,
            batch["points_world"],
            batch["scale"],
            postprocess=args.postprocess,
            temperature=args.temperature,
            topk=args.topk,
            snap=True,
        )
        landmarks = batch["landmarks_world"].cpu().numpy()
        raw_errors = np.linalg.norm(raw_preds - landmarks, axis=-1)
        snapped_errors = np.linalg.norm(snapped_preds - landmarks, axis=-1)
        log_vars = outputs["log_vars"].detach().cpu().numpy()
        raw_errors_all.extend(raw_errors.reshape(-1).tolist())
        snapped_errors_all.extend(snapped_errors.reshape(-1).tolist())
        for row_idx in range(points.shape[0]):
            rows.append(
                (
                    int(batch["sample_index"][row_idx]),
                    raw_preds[row_idx],
                    snapped_preds[row_idx],
                    landmarks[row_idx],
                    raw_errors[row_idx],
                    snapped_errors[row_idx],
                    log_vars[row_idx],
                )
            )
    n = max(1, len(loader.dataset))
    return (
        total / n,
        {key: value / n for key, value in parts_total.items()},
        rows,
        np.asarray(raw_errors_all, dtype=np.float32),
        np.asarray(snapped_errors_all, dtype=np.float32),
    )


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
        "aghformer_raw_x",
        "aghformer_raw_y",
        "aghformer_raw_z",
        "aghformer_snapped_x",
        "aghformer_snapped_y",
        "aghformer_snapped_z",
        "raw_localization_error",
        "snapped_localization_error",
        "uncertainty",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx, raw_pred, snapped_pred, expert, raw_errors, snapped_errors, log_vars in eval_rows:
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
                        "aghformer_raw_x": raw_pred[lm_idx, 0],
                        "aghformer_raw_y": raw_pred[lm_idx, 1],
                        "aghformer_raw_z": raw_pred[lm_idx, 2],
                        "aghformer_snapped_x": snapped_pred[lm_idx, 0],
                        "aghformer_snapped_y": snapped_pred[lm_idx, 1],
                        "aghformer_snapped_z": snapped_pred[lm_idx, 2],
                        "raw_localization_error": raw_errors[lm_idx],
                        "snapped_localization_error": snapped_errors[lm_idx],
                        "uncertainty": float(np.exp(log_vars[lm_idx])),
                    }
                )


def write_dict_rows(path, rows):
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_group_metrics(path, dataset, eval_rows):
    groups = {}
    for sample_idx, _, _, _, _, snapped_errors, _ in eval_rows:
        meta = dataset.metadata(sample_idx)
        groups.setdefault((meta.class_name, meta.gender), []).extend(snapped_errors.tolist())
    rows = []
    for (class_name, gender), errors in sorted(groups.items()):
        arr = np.asarray(errors, dtype=np.float64)
        rows.append(
            {
                "class": class_name,
                "gender": gender,
                "n_samples": int(len(arr) / 23),
                "ale": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
            }
        )
    write_dict_rows(path, rows)


def make_splits(dataset, test_size, val_size, seed):
    indices = np.arange(len(dataset))
    strata = np.array([f"{s.class_name}_{s.gender}" for s in dataset.samples])
    n_groups = len(set(strata.tolist()))
    effective_test_size = max(float(test_size), n_groups / max(1, len(indices)))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=effective_test_size,
        random_state=seed,
        stratify=strata,
    )
    val_fraction = val_size / (1.0 - effective_test_size)
    train_val_strata = strata[train_val_idx]
    effective_val_size = max(float(val_fraction), n_groups / max(1, len(train_val_idx)))
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=effective_val_size,
        random_state=seed,
        stratify=train_val_strata,
    )
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


def build_loaders(args):
    output_dir = Path(args.output_dir)
    dataset = AGHFormerDataset(
        root_dir=args.data_root,
        cache_dir=output_dir / "point_cache",
        num_points=args.surface_points,
        heatmap_sigma=args.heatmap_sigma_start,
        use_normals=args.use_normals,
        use_local_geometry=args.use_local_geometry,
        local_geometry_k=args.local_geometry_k,
        transformation_dir=args.transformation_dir,
        seed=args.seed,
    )
    if args.max_samples is not None:
        dataset.samples = limit_samples_balanced(dataset.samples, args.max_samples, args.seed)
    if args.splits_json and args.max_samples is None:
        split_source = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))
        train_idx = ids_to_indices(dataset, split_source["train"])
        val_idx = ids_to_indices(dataset, split_source["val"])
        test_idx = ids_to_indices(dataset, split_source["test"])
        source_splits_json = str(Path(args.splits_json))
    else:
        train_idx, val_idx, test_idx = make_splits(dataset, args.test_size, args.val_size, args.seed)
        source_splits_json = None if not args.splits_json else f"{Path(args.splits_json)} ignored because --max-samples was used"

    eval_dataset = dataset
    if args.eval_surface_points is not None and args.eval_surface_points != args.surface_points:
        eval_dataset = AGHFormerDataset(
            root_dir=args.data_root,
            cache_dir=output_dir / "point_cache",
            num_points=args.eval_surface_points,
            heatmap_sigma=args.heatmap_sigma_start,
            use_normals=args.use_normals,
            use_local_geometry=args.use_local_geometry,
            local_geometry_k=args.local_geometry_k,
            transformation_dir=args.transformation_dir,
            seed=args.seed,
        )
        if args.max_samples is not None:
            eval_dataset.samples = limit_samples_balanced(eval_dataset.samples, args.max_samples, args.seed)

    split_payload = {
        "train": [dataset.samples[i].sample_id for i in train_idx],
        "val": [dataset.samples[i].sample_id for i in val_idx],
        "test": [dataset.samples[i].sample_id for i in test_idx],
        "source_splits_json": source_splits_json,
        "missing_landmark_meshes": dataset.missing_landmarks,
    }
    (output_dir / "splits.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(Subset(eval_dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return dataset, eval_dataset, train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


def save_outputs(output_dir, args, eval_dataset, test_rows, raw_errors, snapped_errors, test_loss, test_parts, best_val_ale=None, eval_only=False):
    advanced_analysis = analysis_from_eval_rows(eval_dataset, test_rows)
    structure_rows, structure_summary = structure_metrics(test_rows)
    uncertainty_summary = uncertainty_metrics(test_rows)
    metrics = {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "unit": "dataset coordinate unit after optional transformation",
        "clinical_threshold_unit": "mm",
        "model": "AGH-Former anatomy-aware geodesic heatmap transformer",
        "surface_points": args.surface_points,
        "eval_surface_points": args.eval_surface_points or args.surface_points,
        "use_normals": args.use_normals,
        "use_local_geometry": args.use_local_geometry,
        "local_geometry_k": args.local_geometry_k,
        "width": args.width,
        "blocks": args.blocks,
        "heads": args.heads,
        "mlp_ratio": args.mlp_ratio,
        "heatmap_sigma_start": args.heatmap_sigma_start,
        "heatmap_sigma_end": args.heatmap_sigma_end,
        "coord_weight": args.coord_weight,
        "structure_weight": args.structure_weight,
        "symmetry_weight": args.symmetry_weight,
        "clinical_weight": args.clinical_weight,
        "uncertainty_weight": args.uncertainty_weight,
        "postprocess": args.postprocess,
        "temperature": args.temperature,
        "topk": args.topk,
        "test_loss": test_loss,
        "test_loss_parts": test_parts,
        "best_val_ale": best_val_ale,
        "aghformer_raw": summarize_errors(raw_errors),
        "aghformer_snapped": summarize_errors(snapped_errors),
        "structure_consistency": structure_summary,
        "uncertainty_analysis": uncertainty_summary,
    }
    metrics.update(advanced_analysis)
    metrics_path = output_dir / ("metrics_eval.json" if eval_only else "metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / ("predictions_eval_test.csv" if eval_only else "predictions_test.csv"), eval_dataset, test_rows)
    write_group_metrics(output_dir / ("group_metrics_eval_test.csv" if eval_only else "group_metrics_test.csv"), eval_dataset, test_rows)
    write_analysis_csvs(output_dir, advanced_analysis, suffix="eval_test" if eval_only else "test")
    write_dict_rows(output_dir / ("structure_metrics_eval_test.csv" if eval_only else "structure_metrics_test.csv"), structure_rows)
    write_dict_rows(output_dir / ("uncertainty_metrics_eval_test.csv" if eval_only else "uncertainty_metrics_test.csv"), [uncertainty_summary])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train AGH-Former for orthodontic 3D landmark localization.")
    parser.add_argument("--data-root", default="../data/dataset")
    parser.add_argument("--output-dir", default="runs/aghformer_orthodontic")
    parser.add_argument("--transformation-dir", default=None)
    parser.add_argument("--surface-points", type=int, default=4096)
    parser.add_argument("--eval-surface-points", type=int, default=None)
    parser.add_argument("--use-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-local-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-geometry-k", type=int, default=16)
    parser.add_argument("--heatmap-sigma-start", type=float, default=5.0)
    parser.add_argument("--heatmap-sigma-end", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-scale", type=float, default=0.08)
    parser.add_argument("--coord-weight", type=float, default=0.45)
    parser.add_argument("--structure-weight", type=float, default=0.08)
    parser.add_argument("--symmetry-weight", type=float, default=0.02)
    parser.add_argument("--clinical-weight", type=float, default=0.05)
    parser.add_argument("--uncertainty-weight", type=float, default=0.02)
    parser.add_argument("--clinical-threshold-mm", type=float, default=2.0)
    parser.add_argument("--symmetry-pairs", default="1-2,3-4,7-8,10-11,12-13,14-15,16-17,19-20,21-22")
    parser.add_argument("--midline-indices", default="0,5,6,9,18")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default="cosine")
    parser.add_argument("--rotation-aug-deg", type=float, default=0.0)
    parser.add_argument("--point-jitter-std", type=float, default=0.0)
    parser.add_argument("--feature-dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--postprocess", choices=["softmax", "topk_softmax", "argmax"], default="topk_softmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--splits-json", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-tqdm", action="store_true")
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
    symmetry_pairs = parse_pairs(args.symmetry_pairs)
    midline_indices = parse_indices(args.midline_indices)
    graph_adjacency = build_anatomical_adjacency(23, symmetry_pairs, midline_indices)
    dataset, eval_dataset, train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = build_loaders(args)
    input_dim = 3 + (3 if args.use_normals else 0) + (2 if args.use_local_geometry else 0)
    model = AGHFormer(
        input_dim=input_dim,
        num_landmarks=23,
        width=args.width,
        blocks=args.blocks,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        graph_adjacency=graph_adjacency,
        residual_scale=args.residual_scale,
    ).to(device)
    config = vars(args).copy()
    config.update(
        {
            "n_total": len(dataset),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "input_dim": input_dim,
            "symmetry_pairs_parsed": symmetry_pairs,
            "midline_indices_parsed": midline_indices,
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Paired samples: {len(dataset)}", flush=True)
    print(f"Train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}", flush=True)
    print(f"Meshes without matching landmark file: {len(dataset.missing_landmarks)}", flush=True)
    print(f"Device: {device}", flush=True)

    if args.evaluate_only:
        model_path = Path(args.model_path) if args.model_path else output_dir / "best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_loss, test_parts, test_rows, raw_errors, snapped_errors = evaluate(
            model, test_loader, device, args, symmetry_pairs, midline_indices
        )
        metrics = save_outputs(output_dir, args, eval_dataset, test_rows, raw_errors, snapped_errors, test_loss, test_parts, eval_only=True)
        print("\nEvaluation against expert orthodontist landmarks", flush=True)
        print(f"AGH-Former snapped ALE: {metrics['aghformer_snapped']['ale']:.4f}", flush=True)
        print(f"AGH-Former snapped median: {metrics['aghformer_snapped']['median']:.4f}", flush=True)
        print(f"Evaluation saved to: {output_dir}", flush=True)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        train_loss, train_parts, sigma_current = train_epoch(model, train_loader, optimizer, device, args, symmetry_pairs, midline_indices, epoch)
        val_loss, val_parts, _, _, val_snapped_errors = evaluate(model, val_loader, device, args, symmetry_pairs, midline_indices, sigma_current)
        val_ale = float(val_snapped_errors.mean())
        if args.scheduler == "plateau" and scheduler is not None:
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ale": val_ale,
                "sigma_current": sigma_current,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_parts": train_parts,
                "val_parts": val_parts,
            }
        )
        print(
            f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f} "
            f"val_ALE={val_ale:.4f} sigma={sigma_current:.3f}",
            flush=True,
        )
        if val_ale < best_val_ale:
            best_val_ale = val_ale
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    test_loss, test_parts, test_rows, raw_errors, snapped_errors = evaluate(model, test_loader, device, args, symmetry_pairs, midline_indices)
    metrics = save_outputs(
        output_dir,
        args,
        eval_dataset,
        test_rows,
        raw_errors,
        snapped_errors,
        test_loss,
        test_parts,
        best_val_ale=best_val_ale,
        eval_only=False,
    )
    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"AGH-Former raw ALE: {metrics['aghformer_raw']['ale']:.4f}", flush=True)
    print(f"AGH-Former snapped ALE: {metrics['aghformer_snapped']['ale']:.4f}", flush=True)
    print(f"AGH-Former snapped median: {metrics['aghformer_snapped']['median']:.4f}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
