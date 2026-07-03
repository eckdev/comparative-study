import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.datasets.orthodontic_dataset import OrthodonticDataset
from src.datasets.patch_dataset import PatchDataset
from src.models.loss import CombinedLoss, localizationLoss
from src.models.model import PALNET, PLNET_noatt

for parent in Path(__file__).resolve().parents:
    if (parent / "shared_metrics" / "orthodontic_analysis.py").exists():
        sys.path.append(str(parent))
        break

from shared_metrics.orthodontic_analysis import build_error_analysis, write_analysis_csvs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def ale_summary(y_true, y_pred):
    errors = localization_errors(y_true, y_pred)
    summary = {
        "ale": float(errors.mean()),
        "std": float(errors.std()),
        "median": float(np.median(errors)),
        "max": float(errors.max()),
        "per_landmark_ale": errors.mean(axis=0).tolist(),
        "per_sample_ale": errors.mean(axis=1).tolist(),
    }
    for threshold in (2.0, 2.5, 3.0):
        key = ("%g" % threshold).replace(".", "_")
        summary[f"pck_at_{key}mm"] = float((errors <= threshold).mean())
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


def main():
    parser = argparse.ArgumentParser(description="Train PAL-Net on the 23-point orthodontic dataset and report ALE.")
    parser.add_argument("--data-root", default="../../data/dataset", help="Path to Class*/ mesh and landmark folders.")
    parser.add_argument("--output-dir", default="../runs/orthodontic_palnet")
    parser.add_argument("--epochs", type=int, default=200)
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
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OrthodonticDataset(
        args.data_root,
        cache_dir=output_dir / "mesh_cache",
        num_surface_points=args.surface_points,
        normalize=args.normalize,
        transformation_dir=args.transformation_dir,
    )
    print(f"Paired samples: {len(dataset)}", flush=True)
    print(f"Meshes without matching landmark file: {len(dataset.missing_landmarks)}", flush=True)

    source_splits_json = None
    if args.splits_json:
        source_splits_json = str(Path(args.splits_json))
        split_source = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))
        train_idx = ids_to_indices(dataset, split_source["train"])
        val_idx = ids_to_indices(dataset, split_source["val"])
        test_idx = ids_to_indices(dataset, split_source["test"])
    else:
        train_idx, val_idx, test_idx = make_splits(dataset, args.test_size, args.val_size, args.seed)

    full_counts = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    train_idx = limit_split_indices(dataset, train_idx, args.max_train_samples, args.seed + 101)
    val_idx = limit_split_indices(dataset, val_idx, args.max_val_samples, args.seed + 202)
    test_idx = limit_split_indices(dataset, test_idx, args.max_test_samples, args.seed + 303)
    print(
        "Using samples: "
        f"train={len(train_idx)}/{full_counts['train']} "
        f"val={len(val_idx)}/{full_counts['val']} "
        f"test={len(test_idx)}/{full_counts['test']}",
        flush=True,
    )

    split_payload = {
        "train": [dataset.samples[i].sample_id for i in train_idx],
        "val": [dataset.samples[i].sample_id for i in val_idx],
        "test": [dataset.samples[i].sample_id for i in test_idx],
        "source_splits_json": source_splits_json,
        "source_split_counts": full_counts,
        "sample_limits": {
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
        },
        "missing_landmarks": [str(p) for p in dataset.missing_landmarks],
    }
    (output_dir / "splits.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

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

    if args.template_mode == "global":
        train_patches = PatchDataset(train_ds, train_mean, args.patch_size, output_dir / "patch_cache_train")
        val_patches = PatchDataset(val_ds, train_mean, args.patch_size, output_dir / "patch_cache_val")
        test_patches = PatchDataset(test_ds, train_mean, args.patch_size, output_dir / "patch_cache_test")
    else:
        train_template_centers = snap_centers_to_surface(
            train_ds,
            template_centers_for_indices(dataset, train_idx, template_bank, args.template_mode),
        )
        val_template_centers = snap_centers_to_surface(
            val_ds,
            template_centers_for_indices(dataset, val_idx, template_bank, args.template_mode),
        )
        test_template_centers = snap_centers_to_surface(
            test_ds,
            template_centers_for_indices(dataset, test_idx, template_bank, args.template_mode),
        )
        train_patches = RefinerPatchDataset(
            train_ds,
            train_template_centers,
            args.patch_size,
            output_dir / "patch_cache_train",
            augment=False,
            return_centers=False,
        )
        val_patches = RefinerPatchDataset(
            val_ds,
            val_template_centers,
            args.patch_size,
            output_dir / "patch_cache_val",
            augment=False,
            return_centers=False,
        )
        test_patches = RefinerPatchDataset(
            test_ds,
            test_template_centers,
            args.patch_size,
            output_dir / "patch_cache_test",
            augment=False,
            return_centers=False,
        )

    print("Pre-caching train/val/test patches...", flush=True)
    for split_name, patch_ds in (("train", train_patches), ("val", val_patches), ("test", test_patches)):
        print(f"  cache {split_name}: {len(patch_ds)} samples", flush=True)
        for i in tqdm(range(len(patch_ds)), desc=f"cache {split_name}", leave=True, mininterval=1.0, file=sys.stdout):
            _ = patch_ds[i]
        print(f"  cache {split_name}: done", flush=True)

    train_loader = DataLoader(train_patches, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_patches, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_patches, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_patches, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    first_patch, first_landmark, _ = train_patches[0]
    input_shape = first_patch.shape
    output_shape = first_landmark.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    model_cls = PALNET if args.model == "PALNET" else PLNET_noatt
    model = model_cls(input_shape, output_shape, seed=args.seed).to(device)
    criterion = CombinedLoss(alpha=0.6, beta=0.4) if args.loss == "combined" else localizationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_val = float("inf")
    epochs_no_improve = 0
    best_path = output_dir / "best_model.pth"
    history = []

    if args.stage1_model_path:
        print(f"Loading stage 1 model: {args.stage1_model_path}", flush=True)
        model.load_state_dict(torch.load(args.stage1_model_path, map_location=device))
        history.append({"stage": "loaded_stage1", "model_path": str(args.stage1_model_path)})
        best_val = None
    else:
        for epoch in range(args.epochs):
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

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for patches, landmarks, _ in val_loader:
                    patches = patches.to(device, non_blocking=True)
                    landmarks = landmarks.to(device, non_blocking=True)
                    outputs = model(patches)
                    val_loss += criterion(landmarks, outputs).item() * patches.size(0)
            val_loss /= len(val_patches)
            scheduler.step(val_loss)

            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            print(f"Epoch {epoch + 1:04d}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f}", flush=True)

            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1}", flush=True)
                    break
        model.load_state_dict(torch.load(best_path, map_location=device))

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("Collecting stage 1 predictions...", flush=True)
    stage1_train_raw, stage1_train_snapped, y_train, train_point_clouds = collect_predictions(
        model, train_eval_loader, device, args.snap_k
    )
    stage1_val_raw, stage1_val_snapped, y_val, val_point_clouds = collect_predictions(
        model, val_loader, device, args.snap_k
    )
    raw_pred, snapped_pred, y_test, point_clouds = collect_predictions(model, test_loader, device, args.snap_k)
    stage1_test_snapped_internal = snapped_pred.copy()
    test_samples = [dataset.samples[i] for i in test_idx]
    baseline_raw_pred = template_baseline(train_mean.numpy(), y_test)
    baseline_snapped_pred = template_baseline(train_mean.numpy(), y_test, point_clouds)

    val_samples = [dataset.samples[i] for i in val_idx]
    val_raw_out, val_snapped_out, y_val_out = maybe_inverse(
        dataset,
        val_idx,
        args.normalize,
        stage1_val_raw,
        stage1_val_snapped,
        y_val,
    )
    write_prediction_csv(output_dir / "stage1_predictions_val.csv", val_samples, y_val_out, val_snapped_out)

    if args.normalize:
        raw_pred, snapped_pred, y_test, point_clouds, baseline_raw_pred, baseline_snapped_pred = inverse_normalize_arrays(
            dataset,
            test_idx,
            raw_pred,
            snapped_pred,
            y_test,
            point_clouds,
            baseline_raw_pred,
            baseline_snapped_pred,
        )

    write_prediction_csv(output_dir / "stage1_predictions_test.csv", test_samples, y_test, snapped_pred)

    palnet_raw = ale_summary(y_test, raw_pred)
    palnet_snapped = ale_summary(y_test, snapped_pred)
    baseline_raw = ale_summary(y_test, baseline_raw_pred)
    baseline_snapped = ale_summary(y_test, baseline_snapped_pred)
    advanced_analysis = build_error_analysis(test_samples, localization_errors(y_test, snapped_pred))

    metrics = {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "unit": "dataset coordinate unit",
        "clinical_threshold_unit": "mm",
        "model": args.model,
        "stage1_model_path": args.stage1_model_path,
        "template_mode": args.template_mode,
        "train_refiner": args.train_refiner,
        "refine_center": args.refine_center,
        "residual_target": args.residual_target,
        "landmark_weighting": args.landmark_weighting,
        "center_jitter_mm": args.center_jitter_mm,
        "point_noise_mm": args.point_noise_mm,
        "point_dropout": args.point_dropout,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "palnet_raw": palnet_raw,
        "palnet_snapped": palnet_snapped,
        "mean_shape_baseline_raw": baseline_raw,
        "mean_shape_baseline_snapped": baseline_snapped,
        **advanced_analysis,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_prediction_csv(output_dir / "predictions_test.csv", test_samples, y_test, snapped_pred)
    write_group_metrics(output_dir / "group_metrics_test.csv", test_samples, y_test, snapped_pred)
    write_analysis_csvs(output_dir, advanced_analysis, suffix="test")

    refined_metrics = None
    if args.train_refiner:
        refined_metrics = train_refiner(
            args,
            output_dir,
            dataset,
            train_idx,
            val_idx,
            test_idx,
            stage1_train_snapped,
            stage1_val_snapped,
            stage1_test_snapped_internal,
            y_val,
            stage1_val_snapped,
            device,
            model_cls,
            output_shape,
        )

    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"PAL-Net raw ALE:      {palnet_raw['ale']:.4f}", flush=True)
    print(f"PAL-Net snapped ALE:  {palnet_snapped['ale']:.4f}", flush=True)
    if refined_metrics:
        print(f"PAL-Net refined ALE:  {refined_metrics['palnet_refined_snapped']['ale']:.4f}", flush=True)
    print(f"Mean-template ALE:    {baseline_snapped['ale']:.4f}", flush=True)
    print(f"Results saved to:     {output_dir}", flush=True)


if __name__ == "__main__":
    main()
