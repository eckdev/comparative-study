import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CORE20 = [idx for idx in range(23) if idx not in {0, 21, 22}]


def import_aghformer_dataset():
    for parent in Path(__file__).resolve().parents:
        if (parent / "agh_former_orthodontic_comparison" / "run_orthodontic_aghformer.py").exists():
            sys.path.append(str(parent / "agh_former_orthodontic_comparison"))
            break
    from run_orthodontic_aghformer import AGHFormerDataset, ids_to_indices

    return AGHFormerDataset, ids_to_indices


def parse_ints(value):
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def coord(row, prefix):
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")], dtype=np.float32)


def summarize(errors):
    arr = np.asarray(errors, dtype=np.float64)
    flat = arr.reshape(-1)
    out = {
        "ale": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "max": float(flat.max()),
    }
    if arr.ndim == 2:
        out["per_landmark_ale"] = arr.mean(axis=0).astype(float).tolist()
        out["per_landmark_median"] = np.median(arr, axis=0).astype(float).tolist()
        out["per_sample_ale"] = arr.mean(axis=1).astype(float).tolist()
    for threshold in (2.0, 2.5, 3.0):
        key = ("%g" % threshold).replace(".", "_")
        out[f"pck_at_{key}mm"] = float((flat <= threshold).mean())
    return out


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError("No base prediction CSV found. Tried: " + ", ".join(str(path) for path in paths))


def infer_prediction_prefix(row, requested_prefix):
    if f"{requested_prefix}_x" in row:
        return requested_prefix
    for prefix in ("final", "stage3", "stage2_raw", "stage2_snapped", "base"):
        if f"{prefix}_x" in row:
            return prefix
    raise KeyError(f"Could not infer prediction coordinate prefix from columns: {sorted(row.keys())}")


def infer_error_column(row, prefix):
    candidates = [
        f"{prefix}_error",
        "raw_localization_error",
        "snapped_localization_error",
        "final_error",
        "stage3_error",
        "base_error",
    ]
    for key in candidates:
        if key in row:
            return key
    return None


def load_base_predictions(base_run_dir, split, dataset, source_prefix="stage2_raw", base_prediction_dir=None):
    base_run_dir = Path(base_run_dir)
    prediction_dir = Path(base_prediction_dir) if base_prediction_dir else base_run_dir
    csv_path = first_existing(
        [
            prediction_dir / f"base_stage2_predictions_{split}.csv",
            prediction_dir / f"refined_predictions_{split}.csv",
            base_run_dir / f"base_stage2_predictions_{split}.csv",
            base_run_dir / f"refined_predictions_{split}.csv",
        ]
    )
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f"Base prediction CSV is empty: {csv_path}")
    prefix = infer_prediction_prefix(rows[0], source_prefix)
    error_column = infer_error_column(rows[0], prefix)
    sample_to_idx = {dataset.metadata(i).sample_id: i for i in range(len(dataset))}
    grouped = {}
    for row in rows:
        if row["sample_id"] not in sample_to_idx:
            continue
        sample_idx = sample_to_idx[row["sample_id"]]
        lm_idx = int(row["landmark"])
        item = grouped.setdefault(
            sample_idx,
            {
                "base": np.zeros((23, 3), dtype=np.float32),
                "expert": np.zeros((23, 3), dtype=np.float32),
                "error": np.zeros(23, dtype=np.float32),
            },
        )
        item["base"][lm_idx] = coord(row, prefix)
        item["expert"][lm_idx] = coord(row, "expert")
        if error_column:
            item["error"][lm_idx] = float(row[error_column])
        else:
            item["error"][lm_idx] = float(np.linalg.norm(item["base"][lm_idx] - item["expert"][lm_idx]))
    print(f"Loaded {split} base predictions from {csv_path} using prefix '{prefix}'", flush=True)
    return grouped


def tangent_frame(normal_hint):
    n = normal_hint.astype(np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-8)
    ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(ref, n)
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    return u.astype(np.float32), v.astype(np.float32), n.astype(np.float32)


def rasterize_patch(points, features, center, expert, grid_size, radius, heatmap_sigma):
    rel = points - center[None, :]
    dists = np.linalg.norm(rel, axis=1)
    near = np.argpartition(dists, min(len(dists), 64) - 1)[: min(len(dists), 64)]
    normal_hint = features[near, 3:6].mean(axis=0) if features.shape[1] >= 6 else np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    u, v, n = tangent_frame(normal_hint)
    x = rel @ u
    y = rel @ v
    z = rel @ n
    mask = (np.abs(x) <= radius) & (np.abs(y) <= radius)
    if not np.any(mask):
        mask = dists <= np.partition(dists, min(len(dists) - 1, 255))[min(len(dists) - 1, 255)]
    x = x[mask]
    y = y[mask]
    z = z[mask]
    local_features = features[mask]
    points_local = points[mask]
    g = int(grid_size)
    channels = np.zeros((6, g, g), dtype=np.float32)
    counts = np.zeros((g, g), dtype=np.float32)
    best_abs_z = np.full((g, g), np.inf, dtype=np.float32)
    ix = np.clip(((x + radius) / (2.0 * radius) * (g - 1)).round().astype(int), 0, g - 1)
    iy = np.clip(((y + radius) / (2.0 * radius) * (g - 1)).round().astype(int), 0, g - 1)
    normals = local_features[:, 3:6] if local_features.shape[1] >= 6 else np.zeros((len(x), 3), dtype=np.float32)
    curvature = local_features[:, 7] if local_features.shape[1] >= 8 else np.zeros(len(x), dtype=np.float32)
    for row_i in range(len(x)):
        gx = ix[row_i]
        gy = iy[row_i]
        counts[gy, gx] += 1.0
        if abs(float(z[row_i])) < best_abs_z[gy, gx]:
            best_abs_z[gy, gx] = abs(float(z[row_i]))
            channels[0, gy, gx] = float(z[row_i]) / max(radius, 1e-6)
            channels[1:4, gy, gx] = normals[row_i]
            channels[4, gy, gx] = float(curvature[row_i])
            channels[5, gy, gx] = 1.0
    if counts.max() > 0:
        channels[5] = counts / max(float(counts.max()), 1.0)
    expert_rel = expert - center
    expert_x = float(expert_rel @ u)
    expert_y = float(expert_rel @ v)
    target_xy = np.asarray([expert_x / max(radius, 1e-6), expert_y / max(radius, 1e-6)], dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    tx = (expert_x + radius) / (2.0 * radius) * (g - 1)
    ty = (expert_y + radius) / (2.0 * radius) * (g - 1)
    sigma_px = max(float(heatmap_sigma) / (2.0 * float(radius)) * (g - 1), 1e-6)
    heatmap = np.exp(-((xx - tx) ** 2 + (yy - ty) ** 2) / (2.0 * sigma_px**2)).astype(np.float32)
    target_world = center + target_xy[0] * radius * u + target_xy[1] * radius * v
    return channels, heatmap, target_xy, u, v, n, target_world.astype(np.float32)


class Local2p5DDataset(Dataset):
    def __init__(
        self,
        dataset,
        sample_indices,
        base_predictions,
        target_landmarks,
        grid_size=64,
        patch_radius_mm=8.0,
        heatmap_sigma_mm=1.5,
        center_jitter_mm=0.0,
        focus_min_mm=1.5,
        focus_max_mm=3.2,
        focus_weight=2.0,
        seed=42,
    ):
        self.dataset = dataset
        self.sample_indices = list(sample_indices)
        self.base_predictions = base_predictions
        self.target_landmarks = list(target_landmarks)
        self.grid_size = int(grid_size)
        self.patch_radius_mm = float(patch_radius_mm)
        self.heatmap_sigma_mm = float(heatmap_sigma_mm)
        self.center_jitter_mm = float(center_jitter_mm)
        self.focus_min_mm = float(focus_min_mm)
        self.focus_max_mm = float(focus_max_mm)
        self.focus_weight = float(focus_weight)
        self.seed = int(seed)

    def __len__(self):
        return len(self.sample_indices) * len(self.target_landmarks)

    def __getitem__(self, item_idx):
        sample_pos = item_idx // len(self.target_landmarks)
        lm_idx = self.target_landmarks[item_idx % len(self.target_landmarks)]
        sample_idx = self.sample_indices[sample_pos]
        data = self.dataset[sample_idx]
        points = data["points_world"].numpy().astype(np.float32)
        features = data["features"].numpy().astype(np.float32)
        base = self.base_predictions[sample_idx]["base"][lm_idx].astype(np.float32)
        expert = self.base_predictions[sample_idx]["expert"][lm_idx].astype(np.float32)
        base_error = float(np.linalg.norm(base - expert))
        center = base.copy()
        if self.center_jitter_mm > 0:
            center = center + np.random.normal(0.0, self.center_jitter_mm, size=3).astype(np.float32)
        image, heatmap, target_xy, u, v, n, target_world = rasterize_patch(
            points,
            features,
            center,
            expert,
            self.grid_size,
            self.patch_radius_mm,
            self.heatmap_sigma_mm,
        )
        g = self.grid_size
        context = np.zeros((4, g, g), dtype=np.float32)
        context[0, :, :] = float(lm_idx) / 22.0
        context[1, :, :] = float(center[0]) / 200.0
        context[2, :, :] = float(center[1]) / 200.0
        context[3, :, :] = float(center[2]) / 200.0
        image = np.concatenate([image, context], axis=0)
        focus = 1.0
        if self.focus_min_mm <= base_error <= self.focus_max_mm:
            focus += self.focus_weight
        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "heatmap": torch.tensor(heatmap, dtype=torch.float32),
            "target_xy": torch.tensor(target_xy, dtype=torch.float32),
            "base": torch.tensor(base, dtype=torch.float32),
            "expert": torch.tensor(expert, dtype=torch.float32),
            "u": torch.tensor(u, dtype=torch.float32),
            "v": torch.tensor(v, dtype=torch.float32),
            "base_error": torch.tensor(base_error, dtype=torch.float32),
            "focus_weight": torch.tensor(focus, dtype=torch.float32),
            "landmark": torch.tensor(lm_idx, dtype=torch.long),
            "sample_index": torch.tensor(sample_idx, dtype=torch.long),
        }


class SmallHeatmapCNN(nn.Module):
    def __init__(self, in_channels=10, width=64, dropout=0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, image):
        return self.decoder(self.encoder(image)).squeeze(1)


def soft_argmax_2d(logits):
    b, h, w = logits.shape
    weights = torch.softmax(logits.reshape(b, -1), dim=1).reshape(b, h, w)
    ys = torch.linspace(-1.0, 1.0, h, device=logits.device)
    xs = torch.linspace(-1.0, 1.0, w, device=logits.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = (weights * xx[None]).sum(dim=(1, 2))
    y = (weights * yy[None]).sum(dim=(1, 2))
    return torch.stack([x, y], dim=1)


def loss_fn(logits, batch, args):
    pred_xy = soft_argmax_2d(logits)
    heatmap_loss = F.mse_loss(torch.sigmoid(logits), batch["heatmap"], reduction="none").mean(dim=(1, 2))
    coord_loss = F.smooth_l1_loss(pred_xy, batch["target_xy"], reduction="none").mean(dim=1)
    pred_world = batch["base"] + pred_xy[:, 0:1] * args.patch_radius_mm * batch["u"] + pred_xy[:, 1:2] * args.patch_radius_mm * batch["v"]
    err = torch.linalg.norm(pred_world - batch["expert"], dim=1)
    clinical = F.softplus((err - args.clinical_threshold_mm) / max(args.clinical_margin_mm, 1e-6))
    improve = F.softplus((err - batch["base_error"] + args.improvement_margin_mm) / max(args.improvement_margin_mm, 1e-6))
    reg = torch.linalg.norm(pred_xy, dim=1)
    weights = batch["focus_weight"]
    loss = (
        args.heatmap_weight * (heatmap_loss * weights).mean()
        + args.coord_weight * (coord_loss * weights).mean()
        + args.clinical_weight * clinical.mean()
        + args.improvement_weight * improve.mean()
        + args.delta_reg_weight * reg.mean()
    )
    return loss


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="2.5d train", leave=False, disable=args.no_tqdm):
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch["image"])
        loss = loss_fn(logits, batch, args)
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(loss.detach().cpu()) * batch["image"].shape[0]
    return total / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, base_predictions, split_indices, target_landmarks, device, args):
    model.eval()
    idx_to_pos = {sample_idx: pos for pos, sample_idx in enumerate(split_indices)}
    n = len(split_indices)
    base = np.zeros((n, 23, 3), dtype=np.float32)
    pred = np.zeros_like(base)
    expert = np.zeros_like(base)
    for sample_idx, pos in idx_to_pos.items():
        base[pos] = base_predictions[sample_idx]["base"]
        pred[pos] = base_predictions[sample_idx]["base"]
        expert[pos] = base_predictions[sample_idx]["expert"]
    for batch in tqdm(loader, desc="2.5d eval", leave=False, disable=args.no_tqdm):
        device_batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(device_batch["image"])
        pred_xy = soft_argmax_2d(logits).cpu().numpy().astype(np.float32)
        sample_indices = batch["sample_index"].numpy().astype(int)
        landmarks = batch["landmark"].numpy().astype(int)
        bases = batch["base"].numpy().astype(np.float32)
        us = batch["u"].numpy().astype(np.float32)
        vs = batch["v"].numpy().astype(np.float32)
        for row_i, sample_idx in enumerate(sample_indices):
            lm_idx = int(landmarks[row_i])
            point = bases[row_i] + pred_xy[row_i, 0] * args.patch_radius_mm * us[row_i] + pred_xy[row_i, 1] * args.patch_radius_mm * vs[row_i]
            pred[idx_to_pos[int(sample_idx)], lm_idx] = point
    base_errors = np.linalg.norm(base - expert, axis=-1)
    pred_errors = np.linalg.norm(pred - expert, axis=-1)
    return base, pred, expert, base_errors, pred_errors


def gate_by_landmark(base, pred, expert, target_landmarks, val_base_errors, val_pred_errors, min_improvement):
    enabled = []
    for lm_idx in target_landmarks:
        if float(val_pred_errors[:, lm_idx].mean()) + min_improvement < float(val_base_errors[:, lm_idx].mean()):
            enabled.append(int(lm_idx))
    final = base.copy()
    for lm_idx in enabled:
        final[:, lm_idx] = pred[:, lm_idx]
    return final, np.linalg.norm(final - expert, axis=-1), enabled


def write_prediction_csv(path, dataset, split_indices, base, pred, final, expert, base_errors, pred_errors, final_errors, enabled):
    enabled_set = set(enabled)
    rows = []
    for pos, sample_idx in enumerate(split_indices):
        meta = dataset.metadata(sample_idx)
        for lm_idx in range(23):
            rows.append(
                {
                    "sample_id": meta.sample_id,
                    "class": meta.class_name,
                    "gender": meta.gender,
                    "subject_id": meta.subject_id,
                    "landmark": lm_idx,
                    "enabled": lm_idx in enabled_set,
                    "base_error": float(base_errors[pos, lm_idx]),
                    "pred_error": float(pred_errors[pos, lm_idx]),
                    "final_error": float(final_errors[pos, lm_idx]),
                    "expert_x": float(expert[pos, lm_idx, 0]),
                    "expert_y": float(expert[pos, lm_idx, 1]),
                    "expert_z": float(expert[pos, lm_idx, 2]),
                    "base_x": float(base[pos, lm_idx, 0]),
                    "base_y": float(base[pos, lm_idx, 1]),
                    "base_z": float(base[pos, lm_idx, 2]),
                    "pred_x": float(pred[pos, lm_idx, 0]),
                    "pred_y": float(pred[pos, lm_idx, 1]),
                    "pred_z": float(pred[pos, lm_idx, 2]),
                    "final_x": float(final[pos, lm_idx, 0]),
                    "final_y": float(final[pos, lm_idx, 1]),
                    "final_z": float(final[pos, lm_idx, 2]),
                }
            )
    write_rows(path, rows)


def write_landmark_metrics(path, errors):
    rows = []
    for lm_idx in range(23):
        arr = errors[:, lm_idx].astype(np.float64)
        rows.append(
            {
                "landmark": lm_idx,
                "ale": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "max": float(arr.max()),
                "pck_at_2mm": float((arr <= 2.0).mean()),
                "pck_at_2_5mm": float((arr <= 2.5).mean()),
                "pck_at_3mm": float((arr <= 3.0).mean()),
            }
        )
    write_rows(path, rows)


def main():
    parser = argparse.ArgumentParser(description="Train core20 2.5D local heatmap refiner.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits-json", required=True)
    parser.add_argument("--transformation-dir", default=None)
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--base-prediction-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-landmarks", default=",".join(str(i) for i in CORE20))
    parser.add_argument("--source-prefix", default="stage2_raw")
    parser.add_argument("--surface-points", type=int, default=12000)
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--patch-radius-mm", type=float, default=8.0)
    parser.add_argument("--patch-heatmap-sigma-mm", type=float, default=1.5)
    parser.add_argument("--center-jitter-mm", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heatmap-weight", type=float, default=1.0)
    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--clinical-threshold-mm", type=float, default=2.0)
    parser.add_argument("--clinical-margin-mm", type=float, default=0.25)
    parser.add_argument("--clinical-weight", type=float, default=0.35)
    parser.add_argument("--improvement-margin-mm", type=float, default=0.15)
    parser.add_argument("--improvement-weight", type=float, default=0.35)
    parser.add_argument("--delta-reg-weight", type=float, default=0.01)
    parser.add_argument("--focus-min-mm", type=float, default=1.5)
    parser.add_argument("--focus-max-mm", type=float, default=3.2)
    parser.add_argument("--focus-weight", type=float, default=2.0)
    parser.add_argument("--min-val-improvement-mm", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_landmarks = parse_ints(args.target_landmarks)
    split_payload = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))
    AGHFormerDataset, ids_to_indices = import_aghformer_dataset()
    dataset = AGHFormerDataset(
        root_dir=args.data_root,
        cache_dir=output_dir / "point_cache",
        num_points=args.surface_points,
        heatmap_sigma=5.0,
        use_normals=True,
        use_local_geometry=True,
        local_geometry_k=16,
        transformation_dir=args.transformation_dir,
        seed=args.seed,
    )
    train_idx = ids_to_indices(dataset, split_payload["train"])
    val_idx = ids_to_indices(dataset, split_payload["val"])
    test_idx = ids_to_indices(dataset, split_payload["test"])
    if args.max_samples is not None:
        train_count = max(1, int(args.max_samples) // 2)
        val_count = max(1, int(args.max_samples) // 4)
        test_count = max(1, int(args.max_samples) - train_count - val_count)
        train_idx = train_idx[:train_count]
        val_idx = val_idx[:val_count]
        test_idx = test_idx[:test_count]
    base_train = load_base_predictions(args.base_run_dir, "train", dataset, args.source_prefix, args.base_prediction_dir)
    base_val = load_base_predictions(args.base_run_dir, "val", dataset, args.source_prefix, args.base_prediction_dir)
    base_test = load_base_predictions(args.base_run_dir, "test", dataset, args.source_prefix, args.base_prediction_dir)
    print(f"Samples train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}", flush=True)
    print(f"2.5D items train/val/test: {len(train_idx) * len(target_landmarks)}/{len(val_idx) * len(target_landmarks)}/{len(test_idx) * len(target_landmarks)}", flush=True)
    train_ds = Local2p5DDataset(
        dataset,
        train_idx,
        base_train,
        target_landmarks,
        args.grid_size,
        args.patch_radius_mm,
        args.patch_heatmap_sigma_mm,
        args.center_jitter_mm,
        args.focus_min_mm,
        args.focus_max_mm,
        args.focus_weight,
        args.seed,
    )
    val_ds = Local2p5DDataset(dataset, val_idx, base_val, target_landmarks, args.grid_size, args.patch_radius_mm, args.patch_heatmap_sigma_mm, seed=args.seed)
    test_ds = Local2p5DDataset(dataset, test_idx, base_test, target_landmarks, args.grid_size, args.patch_radius_mm, args.patch_heatmap_sigma_mm, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = SmallHeatmapCNN(in_channels=10, width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01)

    history = []
    best_val = math.inf
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args)
        val_base, val_pred, val_expert, val_base_errors, val_pred_errors = evaluate(model, val_loader, base_val, val_idx, target_landmarks, device, args)
        val_ale = float(val_pred_errors.mean())
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_base_ale": float(val_base_errors.mean()), "val_pred_ale": val_ale, "lr": float(optimizer.param_groups[0]["lr"])})
        print(f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.5f} val_base={float(val_base_errors.mean()):.4f} val_2p5d={val_ale:.4f}", flush=True)
        if val_ale < best_val:
            best_val = val_ale
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_2p5d_refiner.pth")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    (output_dir / "history_2p5d.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.load_state_dict(torch.load(output_dir / "best_2p5d_refiner.pth", map_location=device))
    val_base, val_pred, val_expert, val_base_errors, val_pred_errors = evaluate(model, val_loader, base_val, val_idx, target_landmarks, device, args)
    test_base, test_pred, test_expert, test_base_errors, test_pred_errors = evaluate(model, test_loader, base_test, test_idx, target_landmarks, device, args)
    val_final, val_final_errors, enabled = gate_by_landmark(val_base, val_pred, val_expert, target_landmarks, val_base_errors, val_pred_errors, args.min_val_improvement_mm)
    test_final = test_base.copy()
    for lm_idx in enabled:
        test_final[:, lm_idx] = test_pred[:, lm_idx]
    test_final_errors = np.linalg.norm(test_final - test_expert, axis=-1)
    write_prediction_csv(output_dir / "predictions_val.csv", dataset, val_idx, val_base, val_pred, val_final, val_expert, val_base_errors, val_pred_errors, val_final_errors, enabled)
    write_prediction_csv(output_dir / "predictions_test.csv", dataset, test_idx, test_base, test_pred, test_final, test_expert, test_base_errors, test_pred_errors, test_final_errors, enabled)
    write_landmark_metrics(output_dir / "landmark_metrics_test.csv", test_final_errors)
    metrics = {
        "model": "Core20 2.5D local heatmap refiner",
        "target_landmarks": target_landmarks,
        "enabled_landmarks": enabled,
        "base_validation": summarize(val_base_errors),
        "all_target_validation": summarize(val_pred_errors),
        "gated_validation": summarize(val_final_errors),
        "base_test": summarize(test_base_errors),
        "all_target_test": summarize(test_pred_errors),
        "gated_test": summarize(test_final_errors),
        "base_core20_test": summarize(test_base_errors[:, CORE20]),
        "gated_core20_test": summarize(test_final_errors[:, CORE20]),
    }
    (output_dir / "metrics_2p5d.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"Base ALE: {metrics['base_test']['ale']:.4f}", flush=True)
    print(f"2.5D all-target ALE: {metrics['all_target_test']['ale']:.4f}", flush=True)
    print(f"2.5D gated ALE: {metrics['gated_test']['ale']:.4f}", flush=True)
    print(f"2.5D gated median: {metrics['gated_test']['median']:.4f}", flush=True)
    print(f"Enabled landmarks: {enabled}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
