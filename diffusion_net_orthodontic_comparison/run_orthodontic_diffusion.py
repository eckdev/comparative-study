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
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import trimesh
from trimesh.transformations import transform_points


ROOT = Path(__file__).resolve().parent
DIFFUSION_SRC = ROOT / "upstream" / "src"
sys.path.append(str(DIFFUSION_SRC))
import diffusion_net  # noqa: E402

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


class DiffusionOrthodonticDataset(Dataset):
    def __init__(
        self,
        root_dir,
        cache_dir,
        op_cache_dir,
        num_points=2048,
        k_eig=64,
        sigma=0.04,
        mask_radius=3.5,
        transformation_dir=None,
        use_mesh_vertices=False,
        seed=42,
    ):
        self.root_dir = Path(root_dir)
        self.samples, self.missing_landmarks = discover_samples(self.root_dir)
        if not self.samples:
            raise ValueError(f"No paired samples found below {self.root_dir}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.op_cache_dir = Path(op_cache_dir)
        self.op_cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_points = int(num_points)
        self.k_eig = int(k_eig)
        self.sigma = float(sigma)
        self.mask_radius = float(mask_radius)
        self.transformation_dir = Path(transformation_dir) if transformation_dir else None
        self.use_mesh_vertices = bool(use_mesh_vertices)
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
        representation_tag = "meshverts" if self.use_mesh_vertices else str(self.num_points)
        safe_name = str(sample.mesh_path.relative_to(self.root_dir)).replace(os.sep, "__")
        return self.cache_dir / f"{safe_name}.{representation_tag}.{transform_tag}.r{self.mask_radius:g}.npz"

    def _load_arrays(self, idx):
        sample = self.samples[idx]
        cache_path = self._cache_path(sample)
        if cache_path.exists():
            data = np.load(cache_path)
            points_world = data["points_world"]
            landmarks_world = data["landmarks_world"]
            points_norm = data["points_norm"]
            targets = data["targets"]
            faces = data["faces"] if "faces" in data else np.empty((0, 3), dtype=np.int64)
            masks = data["masks"] if "masks" in data else None
            if "target_indices" in data:
                target_indices = data["target_indices"]
            else:
                center, scale = mesh_normalization(points_world)
                landmarks_norm = ((landmarks_world - center) / scale).astype(np.float32)
                dists = np.linalg.norm(points_norm[:, None, :] - landmarks_norm[None, :, :], axis=-1)
                target_indices = dists.argmin(axis=0).astype(np.int64)
            if masks is None:
                world_dists = np.linalg.norm(points_world[:, None, :] - landmarks_world[None, :, :], axis=-1)
                masks = (world_dists <= self.mask_radius).astype(np.float32)
            return points_world, landmarks_world, points_norm, faces.astype(np.int64), targets, masks, target_indices

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
        if self.use_mesh_vertices:
            points_world = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int64)
        elif len(getattr(mesh, "faces", [])) > 0:
            state = np.random.get_state()
            np.random.seed(self.seed + idx)
            points_world = trimesh.sample.sample_surface_even(mesh, self.num_points)[0].astype(np.float32)
            np.random.set_state(state)
            faces = np.empty((0, 3), dtype=np.int64)
        else:
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            replace = len(vertices) < self.num_points
            points_world = vertices[rng.choice(len(vertices), self.num_points, replace=replace)].astype(np.float32)
            faces = np.empty((0, 3), dtype=np.int64)

        center, scale = mesh_normalization(points_world)
        points_norm = ((points_world - center) / scale).astype(np.float32)
        landmarks_norm = ((landmarks - center) / scale).astype(np.float32)
        dists = np.linalg.norm(points_norm[:, None, :] - landmarks_norm[None, :, :], axis=-1)
        targets = np.exp(-(dists ** 2) / (2.0 * self.sigma ** 2)).astype(np.float32)
        target_indices = dists.argmin(axis=0).astype(np.int64)
        world_dists = np.linalg.norm(points_world[:, None, :] - landmarks[None, :, :], axis=-1)
        masks = (world_dists <= self.mask_radius).astype(np.float32)

        np.savez_compressed(
            cache_path,
            points_world=points_world.astype(np.float32),
            landmarks_world=landmarks.astype(np.float32),
            points_norm=points_norm,
            faces=faces.astype(np.int64),
            targets=targets,
            masks=masks,
            target_indices=target_indices,
        )
        return points_world, landmarks, points_norm, faces, targets, masks, target_indices

    def __getitem__(self, idx):
        points_world, landmarks_world, points_norm, faces_np, targets, masks, target_indices = self._load_arrays(idx)
        verts = torch.tensor(points_norm, dtype=torch.float32)
        faces = torch.tensor(faces_np, dtype=torch.long)
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
            verts, faces, k_eig=self.k_eig, op_cache_dir=str(self.op_cache_dir)
        )
        return {
            "verts": verts,
            "faces": faces,
            "mass": mass,
            "L": L,
            "evals": evals,
            "evecs": evecs,
            "gradX": gradX,
            "gradY": gradY,
            "targets": torch.tensor(targets, dtype=torch.float32),
            "masks": torch.tensor(masks, dtype=torch.float32),
            "target_indices": torch.tensor(target_indices, dtype=torch.long),
            "points_world": torch.tensor(points_world, dtype=torch.float32),
            "landmarks_world": torch.tensor(landmarks_world, dtype=torch.float32),
            "sample_index": idx,
        }

    def metadata(self, idx):
        return self.samples[idx]


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


def move_data(data, device):
    moved = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def landmark_cross_entropy(outputs, target_indices):
    return F.cross_entropy(outputs.transpose(0, 1), target_indices)


def landmark_mask_bce(outputs, masks, positive_weight=0.0):
    if positive_weight > 0:
        pos_weight = torch.full((outputs.shape[-1],), positive_weight, dtype=outputs.dtype, device=outputs.device)
    else:
        positives = masks.sum(dim=0).clamp_min(1.0)
        negatives = (masks.shape[0] - masks.sum(dim=0)).clamp_min(1.0)
        pos_weight = (negatives / positives).clamp(max=200.0)
    return F.binary_cross_entropy_with_logits(outputs, masks, pos_weight=pos_weight)


def compute_loss(outputs, data, loss_mode, positive_weight):
    if loss_mode == "ce":
        return landmark_cross_entropy(outputs, data["target_indices"])
    if loss_mode == "mask_bce":
        return landmark_mask_bce(outputs, data["masks"], positive_weight=positive_weight)
    raise ValueError(f"Unsupported loss mode: {loss_mode}")


def build_features(data, input_features, hks_dim):
    if input_features == "xyz":
        return data["verts"]
    hks = diffusion_net.geometry.compute_hks_autoscale(data["evals"], data["evecs"], hks_dim)
    if input_features == "hks":
        return hks
    if input_features == "xyz_hks":
        return torch.cat((data["verts"], hks), dim=-1)
    raise ValueError(f"Unsupported input feature set: {input_features}")


def predict_landmarks(scores, points_world, refine_topk=1, refine_temperature=1.0, postprocess="topk_softmax"):
    if postprocess == "argmax":
        pred_indices = scores.argmax(axis=0)
        return points_world[pred_indices]

    if postprocess == "softmax":
        logits = scores / max(float(refine_temperature), 1e-6)
        logits = logits - logits.max(axis=0, keepdims=True)
        weights = np.exp(logits)
        weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
        return weights.T @ points_world

    if postprocess == "sigmoid_weighted":
        weights = 1.0 / (1.0 + np.exp(-scores))
        weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
        return weights.T @ points_world

    refine_topk = int(refine_topk)
    if refine_topk <= 1:
        pred_indices = scores.argmax(axis=0)
        return points_world[pred_indices]

    n_points, n_landmarks = scores.shape
    k = min(refine_topk, n_points)
    temperature = max(float(refine_temperature), 1e-6)
    pred = np.zeros((n_landmarks, 3), dtype=np.float32)
    for lm_idx in range(n_landmarks):
        landmark_scores = scores[:, lm_idx]
        top_idx = np.argpartition(landmark_scores, -k)[-k:]
        logits = landmark_scores[top_idx] / temperature
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / max(weights.sum(), 1e-12)
        pred[lm_idx] = np.sum(points_world[top_idx] * weights[:, None], axis=0)
    return pred


def train_epoch(model, subset, optimizer, device, input_features, hks_dim, loss_mode, positive_weight):
    model.train()
    total = 0.0
    order = list(range(len(subset)))
    random.shuffle(order)
    for rel_idx in tqdm(order, desc="train", leave=False):
        data = move_data(subset[rel_idx], device)
        features = build_features(data, input_features, hks_dim)
        outputs = model(
            features,
            data["mass"],
            L=data["L"],
            evals=data["evals"],
            evecs=data["evecs"],
            gradX=data["gradX"],
            gradY=data["gradY"],
            faces=data["faces"],
        )
        loss = compute_loss(outputs, data, loss_mode, positive_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
    return total / max(1, len(subset))


@torch.no_grad()
def evaluate(
    model,
    subset,
    device,
    input_features,
    hks_dim,
    loss_mode="ce",
    positive_weight=0.0,
    refine_topk=1,
    refine_temperature=1.0,
    postprocess="topk_softmax",
):
    model.eval()
    total_loss = 0.0
    rows = []
    all_errors = []
    for data in tqdm(subset, desc="eval", leave=False):
        data = move_data(data, device)
        features = build_features(data, input_features, hks_dim)
        outputs = model(
            features,
            data["mass"],
            L=data["L"],
            evals=data["evals"],
            evecs=data["evecs"],
            gradX=data["gradX"],
            gradY=data["gradY"],
            faces=data["faces"],
        )
        loss = compute_loss(outputs, data, loss_mode, positive_weight)
        total_loss += float(loss.detach().cpu())

        points_world = data["points_world"].detach().cpu().numpy()
        landmarks_world = data["landmarks_world"].detach().cpu().numpy()
        scores = outputs.detach().cpu().numpy()
        pred = predict_landmarks(scores, points_world, refine_topk, refine_temperature, postprocess)
        errors = np.linalg.norm(pred - landmarks_world, axis=1)
        all_errors.extend(errors.tolist())
        rows.append((int(data["sample_index"]), pred, landmarks_world, errors))
    return total_loss / max(1, len(subset)), rows, np.asarray(all_errors, dtype=np.float32)


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
        "diffusionnet_x",
        "diffusionnet_y",
        "diffusionnet_z",
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
                        "diffusionnet_x": pred[lm_idx, 0],
                        "diffusionnet_y": pred[lm_idx, 1],
                        "diffusionnet_z": pred[lm_idx, 2],
                        "localization_error": errors[lm_idx],
                    }
                )


def write_group_metrics(path, dataset, eval_rows):
    groups = {}
    for sample_idx, _, _, errors in eval_rows:
        meta = dataset.metadata(sample_idx)
        key = (meta.class_name, meta.gender)
        groups.setdefault(key, []).extend(errors.tolist())
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
    parser = argparse.ArgumentParser(description="Train DiffusionNet heatmaps for orthodontic 3D landmarks.")
    parser.add_argument("--data-root", default="../data/dataset")
    parser.add_argument("--output-dir", default="runs/diffusionnet_orthodontic_heatmap")
    parser.add_argument("--transformation-dir", default=None)
    parser.add_argument("--surface-points", type=int, default=2048)
    parser.add_argument("--k-eig", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--mask-radius", type=float, default=3.5)
    parser.add_argument("--positive-weight", type=float, default=0.0, help="Mask BCE positive weight. Use 0 for automatic per-sample weighting.")
    parser.add_argument("--use-mesh-vertices", action="store_true", help="Use all mesh vertices and faces instead of sampled point clouds.")
    parser.add_argument("--loss-mode", choices=["ce", "mask_bce"], default="ce")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--input-features", choices=["xyz", "hks", "xyz_hks"], default="xyz")
    parser.add_argument("--hks-dim", type=int, default=16)
    parser.add_argument("--refine-topk", type=int, default=1)
    parser.add_argument("--refine-temperature", type=float, default=1.0)
    parser.add_argument("--postprocess", choices=["argmax", "topk_softmax", "softmax", "sigmoid_weighted"], default="topk_softmax")
    parser.add_argument("--mlp-hidden-dims", default=None, help="Comma-separated MLP hidden dims, e.g. 768 or 384,384.")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--splits-json", default=None, help="Shared split JSON with train/val/test sample_id lists.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="Use 'auto' to prefer CUDA, then MPS, then CPU.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional debug limit after sample discovery.")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DiffusionOrthodonticDataset(
        root_dir=args.data_root,
        cache_dir=output_dir / "point_cache",
        op_cache_dir=output_dir / "op_cache",
        num_points=args.surface_points,
        k_eig=args.k_eig,
        sigma=args.sigma,
        mask_radius=args.mask_radius,
        transformation_dir=args.transformation_dir,
        use_mesh_vertices=args.use_mesh_vertices,
        seed=args.seed,
    )
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: args.max_samples]
    print(f"Paired samples: {len(dataset)}")
    print(f"Meshes without matching landmark file: {len(dataset.missing_landmarks)}")

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

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    print("Pre-caching point clouds and DiffusionNet operators...")
    for subset in (train_ds, val_ds, test_ds):
        for idx in tqdm(range(len(subset)), leave=False):
            _ = subset[idx]

    device = torch.device(args.device)
    print(f"Device: {device}")
    feature_dims = {"xyz": 3, "hks": args.hks_dim, "xyz_hks": 3 + args.hks_dim}
    mlp_hidden_dims = None
    if args.mlp_hidden_dims:
        mlp_hidden_dims = [int(part.strip()) for part in args.mlp_hidden_dims.split(",") if part.strip()]
    model = diffusion_net.layers.DiffusionNet(
        C_in=feature_dims[args.input_features],
        C_out=23,
        C_width=args.width,
        N_block=args.blocks,
        outputs_at="vertices",
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=False,
    ).to(device)

    if args.evaluate_only:
        model_path = Path(args.model_path) if args.model_path else output_dir / "best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_loss, test_rows, test_errors = evaluate(
            model,
            test_ds,
            device,
            args.input_features,
            args.hks_dim,
            loss_mode=args.loss_mode,
            positive_weight=args.positive_weight,
            refine_topk=args.refine_topk,
            refine_temperature=args.refine_temperature,
            postprocess=args.postprocess,
        )
        metrics = {
            "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
            "unit": "dataset coordinate unit after optional transformation",
            "clinical_threshold_unit": "mm",
            "model": "DiffusionNet point-cloud landmark classification adaptation",
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "surface_points": args.surface_points,
            "use_mesh_vertices": args.use_mesh_vertices,
            "k_eig": args.k_eig,
            "sigma": args.sigma,
            "mask_radius": args.mask_radius,
            "loss_mode": args.loss_mode,
            "positive_weight": args.positive_weight,
            "input_features": args.input_features,
            "hks_dim": args.hks_dim,
            "refine_topk": args.refine_topk,
            "refine_temperature": args.refine_temperature,
            "postprocess": args.postprocess,
            "width": args.width,
            "blocks": args.blocks,
            "mlp_hidden_dims": mlp_hidden_dims,
            "test_loss": test_loss,
            "diffusionnet_heatmap": summarize_errors(test_errors),
        }
        advanced_analysis = analysis_from_eval_rows(dataset, test_rows)
        metrics.update(advanced_analysis)
        (output_dir / "metrics_eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_predictions(output_dir / "predictions_eval_test.csv", dataset, test_rows)
        write_group_metrics(output_dir / "group_metrics_eval_test.csv", dataset, test_rows)
        write_analysis_csvs(output_dir, advanced_analysis, suffix="eval_test")
        print("\nEvaluation against expert orthodontist landmarks")
        print(f"DiffusionNet ALE: {metrics['diffusionnet_heatmap']['ale']:.4f}")
        print(f"DiffusionNet median: {metrics['diffusionnet_heatmap']['median']:.4f}")
        print(f"Evaluation saved to: {output_dir}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = []
    best_val = math.inf
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_ds,
            optimizer,
            device,
            args.input_features,
            args.hks_dim,
            args.loss_mode,
            args.positive_weight,
        )
        val_loss, _, val_errors = evaluate(
            model,
            val_ds,
            device,
            args.input_features,
            args.hks_dim,
            loss_mode=args.loss_mode,
            positive_weight=args.positive_weight,
            refine_topk=args.refine_topk,
            refine_temperature=args.refine_temperature,
            postprocess=args.postprocess,
        )
        scheduler.step(val_loss)
        val_ale = float(val_errors.mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ale": val_ale})
        print(f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f} val_ALE={val_ale:.3f}")
        if val_loss < best_val:
            best_val = val_loss
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
        test_ds,
        device,
        args.input_features,
        args.hks_dim,
        loss_mode=args.loss_mode,
        positive_weight=args.positive_weight,
        refine_topk=args.refine_topk,
        refine_temperature=args.refine_temperature,
        postprocess=args.postprocess,
    )

    metrics = {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "unit": "dataset coordinate unit after optional transformation",
        "clinical_threshold_unit": "mm",
        "model": "DiffusionNet point-cloud landmark classification adaptation",
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "surface_points": args.surface_points,
        "use_mesh_vertices": args.use_mesh_vertices,
        "k_eig": args.k_eig,
        "sigma": args.sigma,
        "mask_radius": args.mask_radius,
        "loss_mode": args.loss_mode,
        "positive_weight": args.positive_weight,
        "input_features": args.input_features,
        "hks_dim": args.hks_dim,
        "refine_topk": args.refine_topk,
        "refine_temperature": args.refine_temperature,
        "postprocess": args.postprocess,
        "width": args.width,
        "blocks": args.blocks,
        "mlp_hidden_dims": mlp_hidden_dims,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "diffusionnet_heatmap": summarize_errors(test_errors),
    }
    advanced_analysis = analysis_from_eval_rows(dataset, test_rows)
    metrics.update(advanced_analysis)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "predictions_test.csv", dataset, test_rows)
    write_group_metrics(output_dir / "group_metrics_test.csv", dataset, test_rows)
    write_analysis_csvs(output_dir, advanced_analysis, suffix="test")

    print("\nEvaluation against expert orthodontist landmarks")
    print(f"DiffusionNet heatmap ALE: {metrics['diffusionnet_heatmap']['ale']:.4f}")
    print(f"DiffusionNet heatmap median: {metrics['diffusionnet_heatmap']['median']:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
