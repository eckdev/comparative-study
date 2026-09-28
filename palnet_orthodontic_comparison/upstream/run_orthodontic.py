import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.datasets.orthodontic_dataset import OrthodonticDataset, read_orthodontic_landmarks
from src.datasets.patch_dataset import PatchDataset
from src.models.loss import CombinedLoss, localizationLoss
from src.models.model import PALNET, PLNET_noatt

for parent in Path(__file__).resolve().parents:
    if (parent / "shared_metrics" / "orthodontic_analysis.py").exists():
        sys.path.append(str(parent))
        break

from shared_metrics.orthodontic_analysis import build_error_analysis, write_analysis_csvs


DISABLE_TQDM = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value):
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_torch(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_splits(dataset, test_size, val_size, seed):
    indices = np.arange(len(dataset))
    strata = np.array([f"{s.class_name}_{s.gender}" for s in dataset.samples])

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=strata,
    )
    train_val_strata = strata[train_val_idx]
    val_fraction = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
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


def validate_split_indices(dataset, train_idx, val_idx, test_idx):
    split_sets = {
        "train": set(train_idx),
        "val": set(val_idx),
        "test": set(test_idx),
    }
    for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(indices) != len(split_sets[name]):
            raise ValueError(f"Duplicate sample detected inside {name} split")
    if split_sets["train"] & split_sets["val"]:
        raise ValueError("Train/validation split overlap detected")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("Train/test split overlap detected")
    if split_sets["val"] & split_sets["test"]:
        raise ValueError("Validation/test split overlap detected")
    unknown = set.union(*split_sets.values()) - set(range(len(dataset)))
    if unknown:
        raise ValueError(f"Split contains invalid dataset indices: {sorted(unknown)[:10]}")


def limit_split_indices(dataset, indices, max_count, seed):
    if max_count is None or max_count <= 0 or max_count >= len(indices):
        return list(indices)

    grouped = {}
    for idx in indices:
        sample = dataset.samples[idx]
        grouped.setdefault((sample.class_name, sample.gender), []).append(idx)

    rng = random.Random(seed)
    for group_indices in grouped.values():
        rng.shuffle(group_indices)

    selected = []
    group_keys = sorted(grouped)
    cursor = 0
    while len(selected) < max_count and group_keys:
        key = group_keys[cursor % len(group_keys)]
        if grouped[key]:
            selected.append(grouped[key].pop())
        group_keys = [group_key for group_key in group_keys if grouped[group_key]]
        cursor += 1

    return sorted(selected)


def mean_landmarks(base_dataset, subset_indices):
    total = torch.zeros(23, 3)
    for idx in subset_indices:
        _, landmarks, _ = base_dataset[idx]
        total += landmarks
    return total / len(subset_indices)


def nearest_surface_predictions(point_clouds, predictions, k=1):
    fixed = []
    for pc, pred in zip(point_clouds, predictions):
        tree = cKDTree(pc[:, :3])
        _, idx = tree.query(pred, k=k)
        if k == 1:
            fixed.append(pc[idx, :3])
        else:
            fixed.append(pc[idx, :3].mean(axis=1))
    return np.asarray(fixed, dtype=np.float32)


def localization_errors(y_true, y_pred):
    return np.linalg.norm(y_pred - y_true, axis=-1)


def summarize_error_values(errors):
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    summary = {
        "ale": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }
    for threshold in (2.0, 2.5, 3.0, 4.0):
        key = ("%g" % threshold).replace(".", "_")
        summary[f"pck_at_{key}mm"] = float((values <= threshold).mean())
        summary[f"sdr_at_{key}mm"] = summary[f"pck_at_{key}mm"]
    return summary


def ale_summary(y_true, y_pred):
    errors = localization_errors(y_true, y_pred)
    summary = {
        **summarize_error_values(errors),
        "per_landmark_ale": errors.mean(axis=0).tolist(),
        "per_sample_ale": errors.mean(axis=1).tolist(),
    }
    return summary


def template_baseline(train_mean, y_true, point_clouds=None):
    pred = np.repeat(train_mean[None, :, :], repeats=len(y_true), axis=0).astype(np.float32)
    if point_clouds is not None:
        pred = nearest_surface_predictions(point_clouds, pred, k=1)
    return pred


def inverse_normalize_arrays(dataset, indices, *arrays):
    restored = [np.empty_like(array, dtype=np.float32) for array in arrays]
    for row, dataset_idx in enumerate(indices):
        center, scale = dataset.normalization_params(dataset_idx)
        center = center.reshape(1, 3)
        for out, array in zip(restored, arrays):
            out[row] = array[row] * scale + center
    return restored


def collect_predictions(model, loader, device, snap_k):
    model.eval()
    preds = []
    truths = []
    point_clouds = []
    with torch.no_grad():
        for patches, landmarks, sampled_points in loader:
            patches = patches.to(device, non_blocking=True)
            outputs = model(patches).cpu().numpy()
            preds.append(outputs)
            truths.append(landmarks.numpy())
            point_clouds.append(sampled_points.numpy())

    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)
    point_clouds = np.concatenate(point_clouds, axis=0)
    snapped = nearest_surface_predictions(point_clouds, preds.copy(), k=snap_k)
    return preds, snapped, truths, point_clouds


def evaluate_model(model, loader, criterion, device, snap_k):
    model.eval()
    preds = []
    truths = []
    point_clouds = []
    total_loss = 0.0
    with torch.no_grad():
        for patches, landmarks, sampled_points in loader:
            patches = patches.to(device, non_blocking=True)
            landmarks_device = landmarks.to(device, non_blocking=True)
            outputs = model(patches)
            total_loss += criterion(landmarks_device, outputs).item() * patches.size(0)
            preds.append(outputs.cpu().numpy())
            truths.append(landmarks.numpy())
            point_clouds.append(sampled_points.numpy())

    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)
    point_clouds = np.concatenate(point_clouds, axis=0)
    snapped = nearest_surface_predictions(point_clouds, preds.copy(), k=snap_k)
    return total_loss / len(loader.dataset), preds, snapped, truths, point_clouds


def write_prediction_csv(path, samples, y_true, y_pred):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "class",
                "gender",
                "subject_id",
                "landmark",
                "expert_x",
                "expert_y",
                "expert_z",
                "palnet_x",
                "palnet_y",
                "palnet_z",
                "localization_error",
            ]
        )
        errors = localization_errors(y_true, y_pred)
        for sample, truth, pred, sample_errors in zip(samples, y_true, y_pred, errors):
            for lm_idx in range(truth.shape[0]):
                writer.writerow(
                    [
                        sample.sample_id,
                        sample.class_name,
                        sample.gender,
                        sample.subject_id,
                        lm_idx,
                        *truth[lm_idx].tolist(),
                        *pred[lm_idx].tolist(),
                        float(sample_errors[lm_idx]),
                    ]
                )


def write_group_metrics(path, samples, y_true, y_pred):
    groups = {}
    for i, sample in enumerate(samples):
        groups.setdefault((sample.class_name, sample.gender), []).append(i)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "gender", "n_samples", "ale", "std", "median"])
        for (class_name, gender), idxs in sorted(groups.items()):
            summary = ale_summary(y_true[idxs], y_pred[idxs])
            writer.writerow([class_name, gender, len(idxs), summary["ale"], summary["std"], summary["median"]])


def subset_samples(dataset, indices):
    return [dataset.samples[i] for i in indices]


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def compute_template_bank(dataset, train_idx):
    landmarks = []
    samples = []
    for idx in train_idx:
        _, lm, _ = dataset[idx]
        landmarks.append(lm.numpy())
        samples.append(dataset.samples[idx])
    landmarks = np.asarray(landmarks, dtype=np.float32)
    bank = {
        "global": landmarks.mean(axis=0),
        "class": {},
        "gender": {},
        "class_gender": {},
    }
    for class_name in sorted({sample.class_name for sample in samples}):
        selected = [i for i, sample in enumerate(samples) if sample.class_name == class_name]
        bank["class"][class_name] = landmarks[selected].mean(axis=0)
    for gender in sorted({sample.gender for sample in samples}):
        selected = [i for i, sample in enumerate(samples) if sample.gender == gender]
        bank["gender"][gender] = landmarks[selected].mean(axis=0)
    for key in sorted({(sample.class_name, sample.gender) for sample in samples}):
        selected = [i for i, sample in enumerate(samples) if (sample.class_name, sample.gender) == key]
        bank["class_gender"][f"{key[0]}__{key[1]}"] = landmarks[selected].mean(axis=0)
    return bank


def template_for_sample(bank, sample, mode):
    if mode == "class_gender":
        key = f"{sample.class_name}__{sample.gender}"
        if key in bank["class_gender"]:
            return bank["class_gender"][key]
    if mode in ("class_gender", "class") and sample.class_name in bank["class"]:
        return bank["class"][sample.class_name]
    if mode in ("class_gender", "gender") and sample.gender in bank["gender"]:
        return bank["gender"][sample.gender]
    return bank["global"]


def template_centers_for_indices(dataset, indices, bank, mode):
    return np.asarray([template_for_sample(bank, dataset.samples[i], mode) for i in indices], dtype=np.float32)


def snap_centers_to_surface(base_dataset, centers):
    snapped = []
    for idx in range(len(base_dataset)):
        sampled_points, _, raw_vertices = base_dataset[idx]
        point_cloud = raw_vertices.numpy() if raw_vertices is not None else sampled_points.numpy()
        tree = cKDTree(point_cloud[:, :3])
        _, nn_idx = tree.query(centers[idx], k=1)
        snapped.append(point_cloud[nn_idx, :3])
    return np.asarray(snapped, dtype=np.float32)


def extract_centered_patches(raw_vertices, centers, patch_size, reference_point=(0, 0, 0)):
    point_cloud = np.asarray(raw_vertices, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    tree = cKDTree(point_cloud[:, :3])
    _, indices = tree.query(centers, k=patch_size)
    if patch_size == 1:
        indices = indices[:, None]
    patches = point_cloud[indices].astype(np.float32)
    reference = np.asarray(reference_point, dtype=np.float32).reshape(1, 1, 3)
    distances = np.linalg.norm(patches[:, :, :3] - reference, axis=2)
    sorted_idx = np.argsort(distances, axis=1)
    return np.take_along_axis(patches, sorted_idx[..., None], axis=1).astype(np.float32)


class RefinerPatchDataset(Dataset):
    def __init__(
        self,
        base_ds,
        centers,
        patch_size,
        cache_dir,
        center_jitter_mm=0.0,
        point_noise_mm=0.0,
        point_dropout=0.0,
        augment=False,
        return_centers=True,
    ):
        self.base_ds = base_ds
        self.centers = np.asarray(centers, dtype=np.float32)
        self.patch_size = int(patch_size)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.center_jitter_mm = float(center_jitter_mm)
        self.point_noise_mm = float(point_noise_mm)
        self.point_dropout = float(point_dropout)
        self.augment = bool(augment)
        self.return_centers = bool(return_centers)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_sampled, landmarks, raw_vertices = self.base_ds[idx]
        centers = self.centers[idx].copy()
        if self.augment and self.center_jitter_mm > 0:
            centers += np.random.normal(0.0, self.center_jitter_mm, size=centers.shape).astype(np.float32)

        cache_fp = self.cache_dir / f"{idx:06d}_patch.npy"
        if cache_fp.exists() and not self.augment:
            patch = np.load(cache_fp)
        else:
            patch = extract_centered_patches(raw_vertices.numpy(), centers, self.patch_size)
            if not self.augment:
                np.save(cache_fp, patch)

        if self.augment and self.point_dropout > 0:
            mask = np.random.random(size=patch.shape[:2]) < self.point_dropout
            if mask.any():
                replacement = patch[:, :1, :]
                patch = patch.copy()
                patch[mask] = np.repeat(replacement, patch.shape[1], axis=1)[mask]
        if self.augment and self.point_noise_mm > 0:
            patch = patch + np.random.normal(0.0, self.point_noise_mm, size=patch.shape).astype(np.float32)

        patch_tensor = torch.from_numpy(patch.astype(np.float32))
        center_tensor = torch.from_numpy(centers.astype(np.float32))
        if self.return_centers:
            return patch_tensor, landmarks, x_sampled, center_tensor
        return patch_tensor, landmarks, x_sampled


class WeightedCombinedLoss(torch.nn.Module):
    def __init__(self, landmark_weights=None, alpha=0.6, beta=0.4):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        if landmark_weights is None:
            self.register_buffer("landmark_weights", torch.ones(23, dtype=torch.float32))
        else:
            self.register_buffer("landmark_weights", torch.as_tensor(landmark_weights, dtype=torch.float32))

    def forward(self, y_true, y_pred):
        errors = torch.norm(y_pred - y_true, dim=-1)
        weights = self.landmark_weights.to(errors.device).view(1, -1)
        loc = (errors * weights).sum() / (weights.sum() * errors.shape[0])
        dist = torch.abs(torch.cdist(y_true, y_true) - torch.cdist(y_pred, y_pred)).mean()
        return self.alpha * loc + self.beta * dist


def compute_landmark_weights(y_true, y_pred, mode):
    if mode == "none":
        weights = np.ones(23, dtype=np.float32)
    else:
        per_landmark = localization_errors(y_true, y_pred).mean(axis=0)
        weights = per_landmark / max(float(per_landmark.mean()), 1e-6)
        weights = np.clip(weights, 0.75, 2.5).astype(np.float32)
    return weights


def write_landmark_weights(path, weights, mode):
    payload = {
        "weighting": mode,
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "weights": [float(w) for w in weights],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_refiner_predictions(model, loader, device, residual_target=True, snap_k=1):
    model.eval()
    preds = []
    truths = []
    point_clouds = []
    centers_all = []
    with torch.no_grad():
        for patches, landmarks, sampled_points, centers in loader:
            patches = patches.to(device, non_blocking=True)
            outputs = model(patches).cpu()
            if residual_target:
                outputs = outputs + centers
            preds.append(outputs.numpy())
            truths.append(landmarks.numpy())
            point_clouds.append(sampled_points.numpy())
            centers_all.append(centers.numpy())
    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)
    point_clouds = np.concatenate(point_clouds, axis=0)
    centers_all = np.concatenate(centers_all, axis=0)
    snapped = nearest_surface_predictions(point_clouds, preds.copy(), k=snap_k)
    return preds, snapped, truths, point_clouds, centers_all


def maybe_inverse(dataset, indices, normalize, *arrays):
    if not normalize:
        return arrays
    return inverse_normalize_arrays(dataset, indices, *arrays)


def train_refiner(
    args,
    output_dir,
    dataset,
    train_idx,
    val_idx,
    test_idx,
    stage1_centers_train,
    stage1_centers_val,
    stage1_centers_test,
    y_val_internal,
    stage1_val_internal,
    device,
    model_cls,
    output_shape,
):
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    if args.refine_center == "template":
        bank = compute_template_bank(dataset, train_idx)
        stage1_centers_train = snap_centers_to_surface(train_ds, template_centers_for_indices(dataset, train_idx, bank, args.template_mode))
        stage1_centers_val = snap_centers_to_surface(val_ds, template_centers_for_indices(dataset, val_idx, bank, args.template_mode))
        stage1_centers_test = snap_centers_to_surface(test_ds, template_centers_for_indices(dataset, test_idx, bank, args.template_mode))

    landmark_weights = compute_landmark_weights(y_val_internal, stage1_val_internal, args.landmark_weighting)
    write_landmark_weights(output_dir / "landmark_weights.json", landmark_weights, args.landmark_weighting)
    refiner_patch_size = args.refiner_patch_size or args.patch_size
    print(f"Refiner patch size: {refiner_patch_size}", flush=True)

    train_refiner_ds = RefinerPatchDataset(
        train_ds,
        stage1_centers_train,
        refiner_patch_size,
        output_dir / "refiner_patch_cache_train",
        center_jitter_mm=args.center_jitter_mm,
        point_noise_mm=args.point_noise_mm,
        point_dropout=args.point_dropout,
        augment=True,
    )
    val_refiner_ds = RefinerPatchDataset(
        val_ds,
        stage1_centers_val,
        refiner_patch_size,
        output_dir / "refiner_patch_cache_val",
        augment=False,
    )
    test_refiner_ds = RefinerPatchDataset(
        test_ds,
        stage1_centers_test,
        refiner_patch_size,
        output_dir / "refiner_patch_cache_test",
        augment=False,
    )

    train_loader = DataLoader(train_refiner_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_refiner_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_refiner_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    first_patch, _, _, _ = train_refiner_ds[0]
    refiner = model_cls(first_patch.shape, output_shape, seed=args.seed + 1000).to(device)
    criterion = WeightedCombinedLoss(landmark_weights, alpha=0.6, beta=0.4)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_val_ale = float("inf")
    epochs_no_improve = 0
    history = []
    best_path = output_dir / "best_refiner_model.pth"

    for epoch in range(args.epochs):
        refiner.train()
        train_loss = 0.0
        for patches, landmarks, _, centers in train_loader:
            patches = patches.to(device, non_blocking=True)
            landmarks = landmarks.to(device, non_blocking=True)
            centers = centers.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = refiner(patches)
            pred_abs = outputs + centers if args.residual_target else outputs
            loss = criterion(landmarks, pred_abs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * patches.size(0)
        train_loss /= len(train_refiner_ds)

        refiner.eval()
        val_loss = 0.0
        with torch.no_grad():
            for patches, landmarks, _, centers in val_loader:
                patches = patches.to(device, non_blocking=True)
                landmarks = landmarks.to(device, non_blocking=True)
                centers = centers.to(device, non_blocking=True)
                outputs = refiner(patches)
                pred_abs = outputs + centers if args.residual_target else outputs
                val_loss += criterion(landmarks, pred_abs).item() * patches.size(0)
        val_loss /= len(val_refiner_ds)
        scheduler.step(val_loss)

        _, val_snapped, y_val, _, _ = collect_refiner_predictions(
            refiner,
            val_loader,
            device,
            residual_target=args.residual_target,
            snap_k=1,
        )
        val_ale = ale_summary(y_val, val_snapped)["ale"]
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_ale_snap1": val_ale})
        print(
            f"Refiner epoch {epoch + 1:04d}/{args.epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f} val_ALE={val_ale:.4f}",
            flush=True,
        )

        if val_ale < best_val_ale:
            best_val_ale = val_ale
            epochs_no_improve = 0
            torch.save(refiner.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Refiner early stopping at epoch {epoch + 1}", flush=True)
                break

    (output_dir / "refiner_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    refiner.load_state_dict(torch.load(best_path, map_location=device))

    snap_candidates = parse_int_list(args.refiner_snap_k_candidates)
    val_raw, _, y_val, val_point_clouds, _ = collect_refiner_predictions(
        refiner,
        val_loader,
        device,
        residual_target=args.residual_target,
        snap_k=1,
    )
    snap_scores = {}
    for snap_k in snap_candidates:
        val_snapped = nearest_surface_predictions(val_point_clouds, val_raw.copy(), k=snap_k)
        snap_scores[str(snap_k)] = ale_summary(y_val, val_snapped)
    best_snap_k = min(snap_scores, key=lambda key: snap_scores[key]["ale"])
    best_snap_k = int(best_snap_k)

    test_raw, test_snapped, y_test, test_point_clouds, _ = collect_refiner_predictions(
        refiner,
        test_loader,
        device,
        residual_target=args.residual_target,
        snap_k=best_snap_k,
    )
    test_samples = subset_samples(dataset, test_idx)
    test_raw_out, test_snapped_out, y_test_out, test_point_clouds_out = maybe_inverse(
        dataset,
        test_idx,
        args.normalize,
        test_raw,
        test_snapped,
        y_test,
        test_point_clouds,
    )
    refined_raw = ale_summary(y_test_out, test_raw_out)
    refined_snapped = ale_summary(y_test_out, test_snapped_out)
    advanced_analysis = build_error_analysis(test_samples, localization_errors(y_test_out, test_snapped_out))
    metrics = {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "unit": "dataset coordinate unit",
        "clinical_threshold_unit": "mm",
        "model": args.model,
        "stage": "palnet_residual_refiner",
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "template_mode": args.template_mode,
        "refine_center": args.refine_center,
        "residual_target": args.residual_target,
        "landmark_weighting": args.landmark_weighting,
        "center_jitter_mm": args.center_jitter_mm,
        "point_noise_mm": args.point_noise_mm,
        "point_dropout": args.point_dropout,
        "stage1_patch_size": args.patch_size,
        "refiner_patch_size": refiner_patch_size,
        "landmark_weights": [float(w) for w in landmark_weights],
        "snap_candidates": snap_scores,
        "best_snap_k": best_snap_k,
        "best_val_ale": best_val_ale,
        "palnet_refined_raw": refined_raw,
        "palnet_refined_snapped": refined_snapped,
        **advanced_analysis,
    }
    (output_dir / "metrics_refined.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_prediction_csv(output_dir / "refined_predictions_test.csv", test_samples, y_test_out, test_snapped_out)
    write_group_metrics(output_dir / "group_metrics_refined_test.csv", test_samples, y_test_out, test_snapped_out)
    write_analysis_csvs(output_dir, advanced_analysis, suffix="refined_test")

    return metrics


def build_patch_dataset(dataset, indices, train_mean, template_bank, args, cache_dir):
    subset = Subset(dataset, indices)
    if args.template_mode == "global":
        patches = PatchDataset(subset, train_mean, args.patch_size, cache_dir)
    else:
        centers = snap_centers_to_surface(
            subset,
            template_centers_for_indices(dataset, indices, template_bank, args.template_mode),
        )
        patches = RefinerPatchDataset(
            subset,
            centers,
            args.patch_size,
            cache_dir,
            augment=False,
            return_centers=False,
        )
    return subset, patches


def precache_patches(name, patch_dataset):
    print(f"  cache {name}: {len(patch_dataset)} samples", flush=True)
    iterator = tqdm(
        range(len(patch_dataset)),
        desc=f"cache {name}",
        leave=True,
        mininterval=1.0,
        file=sys.stdout,
        disable=DISABLE_TQDM,
    )
    for position, idx in enumerate(iterator, start=1):
        _ = patch_dataset[idx]
        if DISABLE_TQDM and (position % 10 == 0 or position == len(patch_dataset)):
            print(f"  cache {name}: {position}/{len(patch_dataset)}", flush=True)
    print(f"  cache {name}: done", flush=True)


def common_metrics(args, train_idx, val_idx, test_idx, parameter_count, run_signature):
    return {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "unit": "millimetres in the rigidly aligned source coordinate system",
        "clinical_threshold_unit": "mm",
        "model": args.model,
        "adaptation": "PAL-Net orthodontic 23-landmark patch regression",
        "seed": args.seed,
        "fold": args.fold_number,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "patch_size": args.patch_size,
        "surface_points": args.surface_points,
        "snap_k": args.snap_k,
        "template_mode": args.template_mode,
        "loss": args.loss,
        "checkpoint_metric": args.checkpoint_metric,
        "normalize": args.normalize,
        "parameter_count": int(parameter_count),
        "run_signature": run_signature,
    }


def prepare_evaluation_arrays(dataset, indices, train_mean, normalize, raw, snapped, truth, point_clouds):
    baseline_raw = template_baseline(train_mean.numpy(), truth)
    baseline_snapped = template_baseline(train_mean.numpy(), truth, point_clouds)
    if normalize:
        raw, snapped, truth, point_clouds, baseline_raw, baseline_snapped = inverse_normalize_arrays(
            dataset,
            indices,
            raw,
            snapped,
            truth,
            point_clouds,
            baseline_raw,
            baseline_snapped,
        )
    return raw, snapped, truth, point_clouds, baseline_raw, baseline_snapped


def evaluation_payload(common, split, loss, samples, truth, raw, snapped, baseline_raw, baseline_snapped):
    errors = localization_errors(truth, snapped)
    analysis = build_error_analysis(samples, errors)
    return {
        **common,
        "split": split,
        "loss_value": float(loss),
        "palnet_raw": ale_summary(truth, raw),
        "palnet_snapped": ale_summary(truth, snapped),
        "core20": summarize_error_values(errors[:, 1:21]),
        "hard3": summarize_error_values(errors[:, [0, 21, 22]]),
        "mean_shape_baseline_raw": ale_summary(truth, baseline_raw),
        "mean_shape_baseline_snapped": ale_summary(truth, baseline_snapped),
        **analysis,
    }


def write_evaluation(output_dir, split, payload, samples, truth, snapped):
    metrics_name = "metrics.json" if split == "test" else f"metrics_{split}.json"
    (output_dir / metrics_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if split == "test":
        (output_dir / "metrics_test.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    write_prediction_csv(output_dir / f"predictions_{split}.csv", samples, truth, snapped)
    write_prediction_csv(output_dir / f"stage1_predictions_{split}.csv", samples, truth, snapped)
    write_group_metrics(output_dir / f"group_metrics_{split}.csv", samples, truth, snapped)
    write_analysis_csvs(output_dir, payload, suffix=split)


def main():
    parser = argparse.ArgumentParser(description="Train PAL-Net on the 23-point orthodontic dataset and report ALE.")
    parser.add_argument("--data-root", default="../../data/dataset", help="Path to Class*/ mesh and landmark folders.")
    parser.add_argument("--output-dir", default="../runs/orthodontic_palnet")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=250)
    parser.add_argument(
        "--refiner-patch-size",
        type=int,
        default=None,
        help="Patch size for Stage 2 residual refiner. Defaults to --patch-size.",
    )
    parser.add_argument("--surface-points", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--splits-json", default=None, help="Shared split JSON with train/val/test sample_id lists.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples for smoke/debug runs.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples for smoke/debug runs.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples for smoke/debug runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--snap-k", type=int, default=1, help="Nearest sampled surface points used to snap PAL-Net output.")
    parser.add_argument("--model", choices=["PALNET", "PLNET_noatt"], default="PALNET")
    parser.add_argument("--loss", choices=["combined", "localization"], default="combined")
    parser.add_argument("--checkpoint-metric", choices=["val_ale", "val_loss"], default="val_loss")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fold-number", type=int, default=1)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--normalize", action="store_true", help="Normalize each face to unit scale before training.")
    parser.add_argument("--template-mode", choices=["global", "class", "gender", "class_gender"], default="global")
    parser.add_argument("--stage1-model-path", default=None, help="Optional existing PAL-Net checkpoint for stage 1.")
    parser.add_argument("--train-refiner", action="store_true", help="Train a residual PAL-Net refiner after stage 1.")
    parser.add_argument("--refine-center", choices=["stage1", "template"], default="stage1")
    parser.add_argument("--residual-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--landmark-weighting", choices=["none", "val_error"], default="val_error")
    parser.add_argument("--center-jitter-mm", type=float, default=0.0)
    parser.add_argument("--point-noise-mm", type=float, default=0.0)
    parser.add_argument("--point-dropout", type=float, default=0.0)
    parser.add_argument("--refiner-snap-k-candidates", default="1,3,5")
    parser.add_argument(
        "--transformation-dir",
        default=None,
        help="Directory containing PAL-Net-style *_transformation_matrix.npy files.",
    )
    parser.add_argument(
        "--transformation-npz",
        default=None,
        help="Fold-specific archive containing sample_id -> rigid 4x4 matrices.",
    )
    parser.add_argument("--alignment-report", default=None)
    parser.add_argument("--require-label-free-alignment", action="store_true")
    args = parser.parse_args()

    global DISABLE_TQDM
    DISABLE_TQDM = bool(args.no_tqdm)
    if args.require_label_free_alignment and args.train_refiner:
        raise ValueError(
            "The legacy residual refiner is not part of the frozen publication CV protocol. "
            "Run Stage 1 PAL-Net alone or use a separately frozen refiner protocol."
        )
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    dataset = OrthodonticDataset(
        args.data_root,
        cache_dir=output_dir / "mesh_cache",
        num_surface_points=args.surface_points,
        normalize=args.normalize,
        transformation_dir=args.transformation_dir,
        transformation_npz=args.transformation_npz,
        seed=args.seed,
    )
    print(f"Paired samples: {len(dataset)}", flush=True)
    print(f"Meshes without matching landmark file: {len(dataset.missing_landmarks)}", flush=True)
    print(f"Device: {device}", flush=True)

    source_splits_json = None
    split_sha256 = None
    if args.splits_json:
        split_path = Path(args.splits_json)
        source_splits_json = str(split_path)
        split_sha256 = file_sha256(split_path)
        split_source = json.loads(split_path.read_text(encoding="utf-8"))
        train_idx = ids_to_indices(dataset, split_source["train"])
        val_idx = ids_to_indices(dataset, split_source["val"])
        test_idx = ids_to_indices(dataset, split_source["test"])
    else:
        train_idx, val_idx, test_idx = make_splits(dataset, args.test_size, args.val_size, args.seed)

    full_counts = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    alignment_train_ids = {dataset.samples[index].sample_id for index in train_idx}
    train_idx = limit_split_indices(dataset, train_idx, args.max_train_samples, args.seed + 101)
    val_idx = limit_split_indices(dataset, val_idx, args.max_val_samples, args.seed + 202)
    test_idx = limit_split_indices(dataset, test_idx, args.max_test_samples, args.seed + 303)
    validate_split_indices(dataset, train_idx, val_idx, test_idx)
    print(
        "Using samples: "
        f"train={len(train_idx)}/{full_counts['train']} "
        f"val={len(val_idx)}/{full_counts['val']} "
        f"test={len(test_idx)}/{full_counts['test']}",
        flush=True,
    )

    split_payload = {
        "fold": args.fold_number,
        "train": [dataset.samples[i].sample_id for i in train_idx],
        "val": [dataset.samples[i].sample_id for i in val_idx],
        "test": [dataset.samples[i].sample_id for i in test_idx],
        "source_splits_json": source_splits_json,
        "source_splits_sha256": split_sha256,
        "source_split_counts": full_counts,
        "sample_limits": {
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
        },
        "missing_landmarks": [str(p) for p in dataset.missing_landmarks],
    }
    (output_dir / "splits.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    alignment_report = None
    if args.alignment_report:
        alignment_report_path = Path(args.alignment_report)
        if not alignment_report_path.exists():
            raise FileNotFoundError(f"Alignment report not found: {alignment_report_path}")
        alignment_report = json.loads(alignment_report_path.read_text(encoding="utf-8"))
        if alignment_report.get("uses_expert_landmarks") is not False:
            raise ValueError("Alignment report does not certify label-free registration")
        if alignment_report.get("scale") is not False:
            raise ValueError("Alignment report indicates scale-changing registration")
        fitted_ids = set(alignment_report.get("atlas_sample_ids", []))
        fitted_ids.update(alignment_report.get("train_template_sample_ids", []))
        medoid_id = alignment_report.get("train_medoid_sample_id")
        if medoid_id:
            fitted_ids.add(medoid_id)
        if fitted_ids and not fitted_ids <= alignment_train_ids:
            raise ValueError("Alignment atlas contains validation/test samples")
    if args.require_label_free_alignment:
        if alignment_report is None:
            raise ValueError("--require-label-free-alignment requires --alignment-report")
        if not args.transformation_npz:
            raise ValueError("Publication CV requires --transformation-npz")

    transform_info = {
        "transformation_dir": args.transformation_dir,
        "transformation_npz": args.transformation_npz,
        "transformation_npz_sha256": (
            file_sha256(args.transformation_npz) if args.transformation_npz else None
        ),
        "alignment_report": args.alignment_report,
        "alignment_report_sha256": (
            file_sha256(args.alignment_report) if args.alignment_report else None
        ),
        "method": alignment_report.get("method") if alignment_report else None,
        "uses_expert_landmarks": (
            alignment_report.get("uses_expert_landmarks") if alignment_report else None
        ),
        "scale": alignment_report.get("scale") if alignment_report else None,
        "atlas_train_only": True if alignment_report else None,
    }
    audit = {
        "fold": args.fold_number,
        "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "overlap": {"train_val": [], "train_test": [], "val_test": []},
        "split_source": source_splits_json,
        "split_sha256": split_sha256,
        "alignment": transform_info,
        "test_label_protocol": {
            "consumed_only_after_validation_checkpoint_lock": True,
            "test_evaluated": False,
        },
    }
    audit_path = output_dir / "split_and_leakage_report.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if args.preflight_only:
        for sample in dataset.samples:
            read_orthodontic_landmarks(sample.landmark_path)
            matrix = dataset._load_transformation(sample)
            if args.require_label_free_alignment and matrix is None:
                raise ValueError(f"No alignment matrix for {sample.sample_id}")
        report = {
            **audit,
            "passed": True,
            "dataset_samples": len(dataset),
            "landmarks_per_sample": 23,
            "physical_scale_preserved": not args.normalize and transform_info["scale"] is False,
        }
        (output_dir / "preflight_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"Preflight passed for fold {args.fold_number}: "
            f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
            flush=True,
        )
        return

    train_mean = mean_landmarks(dataset, train_idx)
    np.save(output_dir / "train_mean_landmarks.npy", train_mean.numpy())
    template_bank = compute_template_bank(dataset, train_idx)
    np.savez_compressed(
        output_dir / "template_bank.npz",
        global_template=template_bank["global"],
        class_templates=np.asarray(list(template_bank["class"].values()), dtype=np.float32),
        class_template_keys=np.asarray(list(template_bank["class"].keys())),
        gender_templates=np.asarray(list(template_bank["gender"].values()), dtype=np.float32),
        gender_template_keys=np.asarray(list(template_bank["gender"].keys())),
        class_gender_templates=np.asarray(list(template_bank["class_gender"].values()), dtype=np.float32),
        class_gender_template_keys=np.asarray(list(template_bank["class_gender"].keys())),
    )

    train_ds, train_patches = build_patch_dataset(
        dataset, train_idx, train_mean, template_bank, args, output_dir / "patch_cache_train"
    )
    val_ds, val_patches = build_patch_dataset(
        dataset, val_idx, train_mean, template_bank, args, output_dir / "patch_cache_val"
    )

    print("Pre-caching train/validation patches; outer test remains sealed...", flush=True)
    precache_patches("train", train_patches)
    precache_patches("val", val_patches)

    train_loader = DataLoader(train_patches, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_patches, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_patches, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    first_patch, first_landmark, _ = train_patches[0]
    input_shape = first_patch.shape
    output_shape = first_landmark.shape

    model_cls = PALNET if args.model == "PALNET" else PLNET_noatt
    model = model_cls(input_shape, output_shape, seed=args.seed).to(device)
    criterion = CombinedLoss(alpha=0.6, beta=0.4) if args.loss == "combined" else localizationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameter_count:,}", flush=True)

    signature_payload = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output_dir", "resume", "preflight_only"}
    }
    signature_payload.update(
        {
            "split_sha256": split_sha256,
            "transformation_npz_sha256": transform_info["transformation_npz_sha256"],
            "alignment_report_sha256": transform_info["alignment_report_sha256"],
        }
    )
    run_signature = config_sha256(signature_payload)

    best_selection = float("inf")
    best_val_loss = float("inf")
    best_val_ale = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_path = output_dir / "best_model.pth"
    last_path = output_dir / "last_model.pth"
    history = []
    start_epoch = 1
    previous_training_seconds = 0.0

    if args.stage1_model_path:
        print(f"Loading stage 1 model: {args.stage1_model_path}", flush=True)
        model.load_state_dict(load_torch(args.stage1_model_path, device))
        torch.save(model.state_dict(), best_path)
        history.append({"stage": "loaded_stage1", "model_path": str(args.stage1_model_path)})
    else:
        if args.resume and last_path.exists():
            state = load_torch(last_path, device)
            if state.get("run_signature") != run_signature:
                raise RuntimeError(
                    "Resume checkpoint configuration does not match the current fold command. "
                    "Use a new output directory or rerun without --resume."
                )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history = state["history"]
            best_selection = float(state["best_selection"])
            best_val_loss = float(state["best_val_loss"])
            best_val_ale = float(state["best_val_ale"])
            best_epoch = int(state["best_epoch"])
            epochs_no_improve = int(state["epochs_no_improve"])
            previous_training_seconds = float(state.get("training_seconds", 0.0))
            start_epoch = int(state["epoch"]) + 1
            restore_rng_state(state.get("rng_state"))
            print(f"Resuming after epoch {start_epoch - 1}; best epoch={best_epoch}", flush=True)

        training_started = time.time()
        training_already_stopped = (
            start_epoch > args.min_epochs and epochs_no_improve >= args.patience
        )
        epoch_range = range(start_epoch, args.epochs + 1) if not training_already_stopped else ()
        if training_already_stopped:
            print(f"Training was already complete; best epoch={best_epoch}", flush=True)
        for epoch in epoch_range:
            model.train()
            train_loss = 0.0
            for patches, landmarks, _ in train_loader:
                patches = patches.to(device, non_blocking=True)
                landmarks = landmarks.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(patches)
                loss = criterion(landmarks, outputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * patches.size(0)
            train_loss /= len(train_patches)

            val_loss, val_raw_epoch, val_snapped_epoch, y_val_epoch, _ = evaluate_model(
                model, val_loader, criterion, device, args.snap_k
            )
            scheduler.step(val_loss)
            val_ale = ale_summary(y_val_epoch, val_snapped_epoch)["ale"]
            val_raw_ale = ale_summary(y_val_epoch, val_raw_epoch)["ale"]
            selection = val_ale if args.checkpoint_metric == "val_ale" else val_loss
            if selection < best_selection:
                best_selection = selection
                best_val_loss = val_loss
                best_val_ale = val_ale
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                epochs_no_improve += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ale_raw": val_raw_ale,
                    "val_ale_snapped": val_ale,
                    "selection": selection,
                    "best_epoch": best_epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            print(
                f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.4f} "
                f"val={val_loss:.4f} val_ALE={val_ale:.4f} best={best_epoch}",
                flush=True,
            )
            elapsed = previous_training_seconds + (time.time() - training_started)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "best_selection": best_selection,
                    "best_val_loss": best_val_loss,
                    "best_val_ale": best_val_ale,
                    "best_epoch": best_epoch,
                    "epochs_no_improve": epochs_no_improve,
                    "training_seconds": elapsed,
                    "run_signature": run_signature,
                    "rng_state": capture_rng_state(),
                },
                last_path,
            )
            (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if epoch >= args.min_epochs and epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
                break

        training_seconds = previous_training_seconds + (time.time() - training_started)
        if not best_path.exists():
            raise RuntimeError("Training did not produce best_model.pth")
        model.load_state_dict(load_torch(best_path, device))

    if args.stage1_model_path:
        training_seconds = 0.0
        val_loss_loaded, _, val_snapped_loaded, y_val_loaded, _ = evaluate_model(
            model, val_loader, criterion, device, args.snap_k
        )
        best_val_loss = val_loss_loaded
        best_val_ale = ale_summary(y_val_loaded, val_snapped_loaded)["ale"]

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    common = common_metrics(
        args, train_idx, val_idx, test_idx, parameter_count, run_signature
    )
    common.update(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_ale": best_val_ale,
            "training_seconds": training_seconds,
            "stage1_model_path": args.stage1_model_path,
            "alignment": transform_info,
        }
    )

    print("Evaluating locked checkpoint on validation split...", flush=True)
    val_loss, stage1_val_raw, stage1_val_snapped, y_val, val_point_clouds = evaluate_model(
        model, val_loader, criterion, device, args.snap_k
    )
    stage1_val_snapped_internal = stage1_val_snapped.copy()
    y_val_internal = y_val.copy()
    (
        val_raw_out,
        val_snapped_out,
        y_val_out,
        _,
        val_baseline_raw,
        val_baseline_snapped,
    ) = prepare_evaluation_arrays(
        dataset,
        val_idx,
        train_mean,
        args.normalize,
        stage1_val_raw,
        stage1_val_snapped,
        y_val,
        val_point_clouds,
    )
    val_samples = subset_samples(dataset, val_idx)
    val_metrics = evaluation_payload(
        common,
        "val",
        val_loss,
        val_samples,
        y_val_out,
        val_raw_out,
        val_snapped_out,
        val_baseline_raw,
        val_baseline_snapped,
    )
    write_evaluation(output_dir, "val", val_metrics, val_samples, y_val_out, val_snapped_out)

    print("Checkpoint locked. Preparing outer-test patches and consuming test labels...", flush=True)
    test_ds, test_patches = build_patch_dataset(
        dataset, test_idx, train_mean, template_bank, args, output_dir / "patch_cache_test"
    )
    precache_patches("test", test_patches)
    test_loader = DataLoader(
        test_patches, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loss, raw_pred_internal, snapped_pred_internal, y_test_internal, point_clouds_internal = evaluate_model(
        model, test_loader, criterion, device, args.snap_k
    )
    stage1_test_snapped_internal = snapped_pred_internal.copy()
    (
        raw_pred,
        snapped_pred,
        y_test,
        point_clouds,
        baseline_raw_pred,
        baseline_snapped_pred,
    ) = prepare_evaluation_arrays(
        dataset,
        test_idx,
        train_mean,
        args.normalize,
        raw_pred_internal,
        snapped_pred_internal,
        y_test_internal,
        point_clouds_internal,
    )
    test_samples = subset_samples(dataset, test_idx)
    metrics = evaluation_payload(
        common,
        "test",
        test_loss,
        test_samples,
        y_test,
        raw_pred,
        snapped_pred,
        baseline_raw_pred,
        baseline_snapped_pred,
    )
    write_evaluation(output_dir, "test", metrics, test_samples, y_test, snapped_pred)
    audit["test_label_protocol"]["test_evaluated"] = True
    audit["test_label_protocol"]["best_epoch"] = best_epoch
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    refined_metrics = None
    if args.train_refiner:
        stage1_train_raw, stage1_train_snapped, _, _ = collect_predictions(
            model, train_eval_loader, device, args.snap_k
        )
        refined_metrics = train_refiner(
            args,
            output_dir,
            dataset,
            train_idx,
            val_idx,
            test_idx,
            stage1_train_snapped,
            stage1_val_snapped_internal,
            stage1_test_snapped_internal,
            y_val_internal,
            stage1_val_snapped_internal,
            device,
            model_cls,
            output_shape,
        )

    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"PAL-Net raw ALE:      {metrics['palnet_raw']['ale']:.4f}", flush=True)
    print(f"PAL-Net snapped ALE:  {metrics['palnet_snapped']['ale']:.4f}", flush=True)
    print(f"PAL-Net Core20 ALE:   {metrics['core20']['ale']:.4f}", flush=True)
    print(f"PAL-Net Hard3 ALE:    {metrics['hard3']['ale']:.4f}", flush=True)
    if refined_metrics:
        print(f"PAL-Net refined ALE:  {refined_metrics['palnet_refined_snapped']['ale']:.4f}", flush=True)
    print(f"Mean-template ALE:    {metrics['mean_shape_baseline_snapped']['ale']:.4f}", flush=True)
    print(f"Results saved to:     {output_dir}", flush=True)


if __name__ == "__main__":
    main()
