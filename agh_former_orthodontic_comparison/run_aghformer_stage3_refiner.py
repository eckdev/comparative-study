import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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
        "per_landmark_ale": arr.mean(axis=0).astype(float).tolist(),
        "per_landmark_median": np.median(arr, axis=0).astype(float).tolist(),
        "per_sample_ale": arr.mean(axis=1).astype(float).tolist(),
    }
    for threshold in (2.0, 2.5, 3.0):
        key = ("%g" % threshold).replace(".", "_")
        out[f"pck_at_{key}mm"] = float((flat <= threshold).mean())
    return out


def summarize_subset(errors, landmarks):
    arr = np.asarray(errors, dtype=np.float64)[:, list(landmarks)]
    return summarize(arr.reshape((arr.shape[0], len(landmarks))))


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(config, key, fallback):
    return config[key] if key in config and config[key] is not None else fallback


def load_stage1_prediction_dict(path, dataset):
    rows = read_rows(path)
    sample_to_idx = {dataset.metadata(i).sample_id: i for i in range(len(dataset))}
    grouped = {}
    for row in rows:
        sample_id = row["sample_id"]
        lm_idx = int(row["landmark"])
        sample_idx = sample_to_idx[sample_id]
        entry = grouped.setdefault(
            sample_idx,
            {
                "raw": np.zeros((23, 3), dtype=np.float32),
                "snapped": np.zeros((23, 3), dtype=np.float32),
                "center": np.zeros((23, 3), dtype=np.float32),
                "expert": np.zeros((23, 3), dtype=np.float32),
                "raw_errors": np.zeros(23, dtype=np.float32),
                "snapped_errors": np.zeros(23, dtype=np.float32),
            },
        )
        entry["raw"][lm_idx] = coord(row, "stage1_raw")
        entry["snapped"][lm_idx] = coord(row, "stage1_snapped")
        entry["center"][lm_idx] = coord(row, "stage1_snapped")
        entry["expert"][lm_idx] = coord(row, "expert")
        entry["raw_errors"][lm_idx] = float(row["raw_localization_error"])
        entry["snapped_errors"][lm_idx] = float(row["snapped_localization_error"])
    return grouped


def build_dataset(args, split_payload, base_config):
    from run_orthodontic_aghformer import AGHFormerDataset, ids_to_indices

    dataset = AGHFormerDataset(
        root_dir=args.data_root,
        cache_dir=Path(args.base_run_dir) / "stage1_point_cache",
        num_points=args.surface_points,
        heatmap_sigma=float(resolve(base_config, "heatmap_sigma_start", args.heatmap_sigma)),
        use_normals=True,
        use_local_geometry=True,
        local_geometry_k=16,
        transformation_dir=args.transformation_dir,
        seed=args.seed,
    )
    train_idx = ids_to_indices(dataset, split_payload["train"])
    val_idx = ids_to_indices(dataset, split_payload["val"])
    test_idx = ids_to_indices(dataset, split_payload["test"])
    return dataset, train_idx, val_idx, test_idx


class PatchResidualRefiner(nn.Module):
    def __init__(self, input_dim=7, width=128, landmark_dim=32, dropout=0.1, residual_limit_mm=4.0):
        super().__init__()
        self.residual_limit_mm = float(residual_limit_mm)
        self.landmark_embedding = nn.Embedding(23, landmark_dim)
        self.point_mlp = nn.Sequential(
            nn.Conv1d(input_dim, width // 2, 1, bias=False),
            nn.BatchNorm1d(width // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(width // 2, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2 + landmark_dim, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv1d(width, 1, 1)
        self.delta_head = nn.Linear(width // 2, 3)
        self.log_var_head = nn.Linear(width // 2, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, patch_features, landmark):
        x = patch_features.transpose(1, 2).contiguous()
        point_features = self.point_mlp(x)
        point_logits = self.heatmap_head(point_features).squeeze(1)
        pooled_max = point_features.max(dim=2).values
        pooled_mean = point_features.mean(dim=2)
        hidden = self.head(torch.cat([pooled_max, pooled_mean, self.landmark_embedding(landmark)], dim=1))
        delta = torch.tanh(self.delta_head(hidden)) * self.residual_limit_mm
        return delta, point_logits


class Stage3PatchDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        sample_indices,
        base_predictions,
        landmark_subset,
        patch_points=512,
        patch_radius_mm=8.0,
        heatmap_sigma_mm=1.5,
        center_jitter_mm=0.0,
        point_noise_mm=0.0,
        point_dropout=0.0,
        focus_min_mm=1.8,
        focus_max_mm=3.2,
        focus_weight=2.0,
        seed=42,
    ):
        self.base_dataset = base_dataset
        self.sample_indices = list(sample_indices)
        self.base_predictions = base_predictions
        self.landmark_subset = list(landmark_subset)
        self.patch_points = int(patch_points)
        self.patch_radius_mm = float(patch_radius_mm)
        self.heatmap_sigma_mm = float(heatmap_sigma_mm)
        self.center_jitter_mm = float(center_jitter_mm)
        self.point_noise_mm = float(point_noise_mm)
        self.point_dropout = float(point_dropout)
        self.focus_min_mm = float(focus_min_mm)
        self.focus_max_mm = float(focus_max_mm)
        self.focus_weight = float(focus_weight)
        self.seed = int(seed)

    def __len__(self):
        return len(self.sample_indices) * len(self.landmark_subset)

    def __getitem__(self, item_idx):
        sample_pos = item_idx // len(self.landmark_subset)
        lm_idx = self.landmark_subset[item_idx % len(self.landmark_subset)]
        sample_idx = self.sample_indices[sample_pos]
        data = self.base_dataset[sample_idx]
        points_world = data["points_world"].numpy().astype(np.float32)
        features = data["features"].numpy().astype(np.float32)
        center_clean = self.base_predictions[sample_idx]["center"][lm_idx].astype(np.float32)
        expert = self.base_predictions[sample_idx]["expert"][lm_idx].astype(np.float32)
        base_error = float(np.linalg.norm(center_clean - expert))
        rng = np.random.default_rng(self.seed + item_idx)
        center = center_clean.copy()
        if self.center_jitter_mm > 0:
            center = center + rng.normal(0.0, self.center_jitter_mm, size=3).astype(np.float32)

        dists = np.linalg.norm(points_world - center[None, :], axis=1)
        within = np.where(dists <= self.patch_radius_mm)[0]
        if len(within) >= self.patch_points:
            local_order = within[np.argpartition(dists[within], self.patch_points - 1)[: self.patch_points]]
        else:
            local_order = np.argpartition(dists, min(self.patch_points, len(dists)) - 1)[: self.patch_points]
        if len(local_order) < self.patch_points:
            extra = rng.choice(local_order, self.patch_points - len(local_order), replace=True)
            local_order = np.concatenate([local_order, extra], axis=0)

        points_patch = points_world[local_order].astype(np.float32)
        local_xyz = (points_patch - center[None, :]) / max(self.patch_radius_mm, 1e-6)
        if self.point_noise_mm > 0:
            local_xyz = local_xyz + rng.normal(
                0.0,
                self.point_noise_mm / max(self.patch_radius_mm, 1e-6),
                size=local_xyz.shape,
            ).astype(np.float32)
        normals = features[local_order, 3:6].astype(np.float32) if features.shape[1] >= 6 else np.zeros_like(local_xyz)
        local_dist = np.linalg.norm(local_xyz, axis=1, keepdims=True).astype(np.float32)
        patch_features = np.concatenate([local_xyz.astype(np.float32), normals, local_dist], axis=1)
        if self.point_dropout > 0:
            keep = (rng.random(self.patch_points) > self.point_dropout).astype(np.float32)[:, None]
            patch_features = patch_features * keep

        expert_dists = np.linalg.norm(points_patch - expert[None, :], axis=1)
        sigma = max(self.heatmap_sigma_mm, 1e-6)
        patch_heatmap = np.exp(-(expert_dists**2) / (2.0 * sigma**2)).astype(np.float32)
        focus = 1.0
        if self.focus_min_mm <= base_error <= self.focus_max_mm:
            focus += self.focus_weight
        return {
            "patch_features": torch.tensor(patch_features, dtype=torch.float32),
            "patch_points_world": torch.tensor(points_patch, dtype=torch.float32),
            "patch_heatmap": torch.tensor(patch_heatmap, dtype=torch.float32),
            "center": torch.tensor(center_clean, dtype=torch.float32),
            "expert": torch.tensor(expert, dtype=torch.float32),
            "base_error": torch.tensor(base_error, dtype=torch.float32),
            "focus_weight": torch.tensor(focus, dtype=torch.float32),
            "landmark": torch.tensor(lm_idx, dtype=torch.long),
            "sample_index": torch.tensor(sample_idx, dtype=torch.long),
        }


def patch_heatmap_loss(point_logits, patch_heatmap, positive_weight=10.0, ce_weight=0.02):
    weights = 1.0 + float(positive_weight) * patch_heatmap
    mse = ((torch.sigmoid(point_logits) - patch_heatmap).pow(2.0) * weights).sum() / weights.sum().clamp_min(1.0)
    if ce_weight <= 0:
        return mse
    return mse + float(ce_weight) * F.cross_entropy(point_logits, patch_heatmap.argmax(dim=1))


def stage3_loss(delta, point_logits, batch, args):
    final = batch["center"] + delta
    expert = batch["expert"]
    err = torch.linalg.norm(final - expert, dim=1)
    base_err = batch["base_error"]
    coord = F.smooth_l1_loss(final, expert, reduction="none").mean(dim=1)
    coord = (coord * batch["focus_weight"]).mean()
    clinical = F.softplus((err - args.clinical_threshold_mm) / max(args.clinical_margin_mm, 1e-6)).mean()
    improve = F.softplus((err - base_err + args.improvement_margin_mm) / max(args.improvement_margin_mm, 1e-6)).mean()
    heatmap = patch_heatmap_loss(
        point_logits,
        batch["patch_heatmap"],
        positive_weight=args.patch_heatmap_positive_weight,
        ce_weight=args.patch_heatmap_ce_weight,
    )
    delta_reg = torch.linalg.norm(delta, dim=1).mean()
    loss = (
        coord
        + args.clinical_weight * clinical
        + args.improvement_weight * improve
        + args.patch_heatmap_weight * heatmap
        + args.delta_reg_weight * delta_reg
    )
    return loss, {
        "coord_loss": float(coord.detach().cpu()),
        "clinical_loss": float(clinical.detach().cpu()),
        "improvement_loss": float(improve.detach().cpu()),
        "patch_heatmap_loss": float(heatmap.detach().cpu()),
        "delta_reg": float(delta_reg.detach().cpu()),
    }


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    total = 0.0
    parts_total = {}
    for batch in tqdm(loader, desc="stage3 train", leave=False, disable=args.no_tqdm):
        batch = {key: value.to(device) for key, value in batch.items()}
        delta, point_logits = model(batch["patch_features"], batch["landmark"])
        loss, parts = stage3_loss(delta, point_logits, batch, args)
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(loss.detach().cpu()) * batch["patch_features"].shape[0]
        for key, value in parts.items():
            parts_total[key] = parts_total.get(key, 0.0) + value * batch["patch_features"].shape[0]
    n = max(1, len(loader.dataset))
    return total / n, {key: value / n for key, value in parts_total.items()}


@torch.no_grad()
def evaluate_stage3(model, loader, base_predictions, split_indices, target_landmarks, device, args):
    model.eval()
    target_set = set(target_landmarks)
    index_to_pos = {sample_idx: pos for pos, sample_idx in enumerate(split_indices)}
    n_samples = len(split_indices)
    base_pred = np.zeros((n_samples, 23, 3), dtype=np.float32)
    stage3_pred = np.zeros((n_samples, 23, 3), dtype=np.float32)
    expert = np.zeros((n_samples, 23, 3), dtype=np.float32)
    for sample_idx, pos in index_to_pos.items():
        base_pred[pos] = base_predictions[sample_idx]["center"]
        stage3_pred[pos] = base_predictions[sample_idx]["center"]
        expert[pos] = base_predictions[sample_idx]["expert"]

    for batch in tqdm(loader, desc="stage3 eval", leave=False, disable=args.no_tqdm):
        device_batch = {key: value.to(device) for key, value in batch.items()}
        delta, _ = model(device_batch["patch_features"], device_batch["landmark"])
        final = (device_batch["center"] + delta).cpu().numpy().astype(np.float32)
        sample_indices = batch["sample_index"].numpy().astype(int)
        landmarks = batch["landmark"].numpy().astype(int)
        for row_i, sample_idx in enumerate(sample_indices):
            lm_idx = int(landmarks[row_i])
            if lm_idx not in target_set:
                continue
            stage3_pred[index_to_pos[int(sample_idx)], lm_idx] = final[row_i]
    base_errors = np.linalg.norm(base_pred - expert, axis=-1)
    stage3_errors = np.linalg.norm(stage3_pred - expert, axis=-1)
    return base_pred, stage3_pred, expert, base_errors, stage3_errors


def prediction_dict_from_arrays(dataset, split_indices, pred, expert, errors):
    output = {}
    for pos, sample_idx in enumerate(split_indices):
        output[int(sample_idx)] = {
            "center": pred[pos].astype(np.float32),
            "expert": expert[pos].astype(np.float32),
            "errors": errors[pos].astype(np.float32),
        }
    return output


def write_prediction_csv(path, dataset, split_indices, base_pred, stage3_pred, final_pred, expert, base_errors, stage3_errors, final_errors, enabled):
    rows = []
    enabled_set = set(enabled)
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
                    "expert_x": float(expert[pos, lm_idx, 0]),
                    "expert_y": float(expert[pos, lm_idx, 1]),
                    "expert_z": float(expert[pos, lm_idx, 2]),
                    "base_x": float(base_pred[pos, lm_idx, 0]),
                    "base_y": float(base_pred[pos, lm_idx, 1]),
                    "base_z": float(base_pred[pos, lm_idx, 2]),
                    "stage3_x": float(stage3_pred[pos, lm_idx, 0]),
                    "stage3_y": float(stage3_pred[pos, lm_idx, 1]),
                    "stage3_z": float(stage3_pred[pos, lm_idx, 2]),
                    "final_x": float(final_pred[pos, lm_idx, 0]),
                    "final_y": float(final_pred[pos, lm_idx, 1]),
                    "final_z": float(final_pred[pos, lm_idx, 2]),
                    "base_error": float(base_errors[pos, lm_idx]),
                    "stage3_error": float(stage3_errors[pos, lm_idx]),
                    "final_error": float(final_errors[pos, lm_idx]),
                }
            )
    write_rows(path, rows)


def write_analysis_csvs(output_dir, suffix, dataset, split_indices, errors):
    samples = [dataset.metadata(i) for i in split_indices]
    rows = []
    for lm_idx in range(23):
        arr = errors[:, lm_idx].astype(np.float64)
        rows.append(
            {
                "landmark": lm_idx,
                "n_points": len(arr),
                "ale": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "max": float(arr.max()),
                "pck_at_2mm": float((arr <= 2.0).mean()),
                "pck_at_2_5mm": float((arr <= 2.5).mean()),
                "pck_at_3mm": float((arr <= 3.0).mean()),
            }
        )
    write_rows(Path(output_dir) / f"landmark_metrics_{suffix}.csv", rows)
    groups = {}
    for sample_pos, sample in enumerate(samples):
        groups.setdefault(("class", sample.class_name), []).extend(errors[sample_pos].tolist())
        groups.setdefault(("gender", sample.gender), []).extend(errors[sample_pos].tolist())
        groups.setdefault(("class_gender", f"{sample.class_name}|{sample.gender}"), []).extend(errors[sample_pos].tolist())
    group_rows = []
    for (scope, group), values in sorted(groups.items()):
        arr = np.asarray(values, dtype=np.float64)
        group_rows.append(
            {
                "scope": scope,
                "group": group,
                "n_points": len(values),
                "ale": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "pck_at_2mm": float((arr <= 2.0).mean()),
                "pck_at_2_5mm": float((arr <= 2.5).mean()),
                "pck_at_3mm": float((arr <= 3.0).mean()),
            }
        )
    write_rows(Path(output_dir) / f"group_metrics_{suffix}.csv", group_rows)


def load_or_generate_base_predictions(args, dataset, split_indices, base_config, device, output_dir):
    stage1_all = {}
    for split in ("train", "val", "test"):
        stage1_all.update(load_stage1_prediction_dict(Path(args.base_run_dir) / f"stage1_predictions_{split}.csv", dataset))

    model = PatchResidualRefiner(
        input_dim=7,
        width=int(resolve(base_config, "refiner_width", 256)),
        landmark_dim=int(resolve(base_config, "landmark_embedding_dim", 64)),
        dropout=float(resolve(base_config, "refiner_dropout", 0.1)),
        residual_limit_mm=float(resolve(base_config, "residual_limit_mm", 12.0)),
    ).to(device)
    model.load_state_dict(torch.load(Path(args.base_run_dir) / "best_refiner.pth", map_location=device))

    base_by_split = {}
    for split, indices in split_indices.items():
        ds = Stage3PatchDataset(
            dataset,
            indices,
            stage1_all,
            landmark_subset=list(range(23)),
            patch_points=int(resolve(base_config, "patch_points", 1024)),
            patch_radius_mm=float(resolve(base_config, "patch_radius_mm", 12.0)),
            heatmap_sigma_mm=float(resolve(base_config, "patch_heatmap_sigma_mm", 2.0)),
            seed=args.seed,
        )
        loader = DataLoader(ds, batch_size=args.base_eval_batch_size, shuffle=False, num_workers=args.num_workers)
        base_pred, stage2_pred, expert, _, stage2_errors = evaluate_stage3(
            model,
            loader,
            stage1_all,
            indices,
            list(range(23)),
            device,
            args,
        )
        base_by_split[split] = prediction_dict_from_arrays(dataset, indices, stage2_pred, expert, stage2_errors)
        write_prediction_csv(
            Path(output_dir) / f"base_stage2_predictions_{split}.csv",
            dataset,
            indices,
            base_pred,
            stage2_pred,
            stage2_pred,
            expert,
            np.linalg.norm(base_pred - expert, axis=-1),
            stage2_errors,
            stage2_errors,
            enabled=list(range(23)),
        )
    return base_by_split


def final_with_gate(base_pred, stage3_pred, expert, target_landmarks, val_base_errors, val_stage3_errors, min_improvement):
    enabled = []
    for lm_idx in target_landmarks:
        base_mean = float(val_base_errors[:, lm_idx].mean())
        stage3_mean = float(val_stage3_errors[:, lm_idx].mean())
        if stage3_mean + float(min_improvement) < base_mean:
            enabled.append(int(lm_idx))
    final = base_pred.copy()
    for lm_idx in enabled:
        final[:, lm_idx] = stage3_pred[:, lm_idx]
    errors = np.linalg.norm(final - expert, axis=-1)
    return final, errors, enabled


def main():
    parser = argparse.ArgumentParser(description="Train AGH-Former Stage 3 clinical-threshold fine refiner on top of v6 Stage2 outputs.")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits-json", default=None)
    parser.add_argument("--transformation-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-landmarks", default="2,10,11,12,13,16,19,20")
    parser.add_argument("--surface-points", type=int, default=12000)
    parser.add_argument("--heatmap-sigma", type=float, default=5.0)
    parser.add_argument("--patch-points", type=int, default=512)
    parser.add_argument("--patch-radius-mm", type=float, default=8.0)
    parser.add_argument("--patch-heatmap-sigma-mm", type=float, default=1.5)
    parser.add_argument("--center-jitter-mm", type=float, default=0.25)
    parser.add_argument("--point-noise-mm", type=float, default=0.03)
    parser.add_argument("--point-dropout", type=float, default=0.01)
    parser.add_argument("--refiner-width", type=int, default=128)
    parser.add_argument("--landmark-embedding-dim", type=int, default=32)
    parser.add_argument("--refiner-dropout", type=float, default=0.05)
    parser.add_argument("--residual-limit-mm", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clinical-threshold-mm", type=float, default=2.0)
    parser.add_argument("--clinical-margin-mm", type=float, default=0.25)
    parser.add_argument("--clinical-weight", type=float, default=0.4)
    parser.add_argument("--improvement-margin-mm", type=float, default=0.15)
    parser.add_argument("--improvement-weight", type=float, default=0.5)
    parser.add_argument("--patch-heatmap-weight", type=float, default=0.1)
    parser.add_argument("--patch-heatmap-positive-weight", type=float, default=10.0)
    parser.add_argument("--patch-heatmap-ce-weight", type=float, default=0.02)
    parser.add_argument("--delta-reg-weight", type=float, default=0.01)
    parser.add_argument("--focus-min-mm", type=float, default=1.8)
    parser.add_argument("--focus-max-mm", type=float, default=3.2)
    parser.add_argument("--focus-weight", type=float, default=2.0)
    parser.add_argument("--min-val-improvement-mm", type=float, default=0.0)
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
    base_config = load_json(Path(args.base_run_dir) / "config_stage2.json", default={})
    split_payload = load_json(Path(args.splits_json) if args.splits_json else Path(args.base_run_dir) / "splits.json")
    (output_dir / "config_stage3.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    dataset, train_idx, val_idx, test_idx = build_dataset(args, split_payload, base_config)
    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    print(f"Device: {device}", flush=True)
    print(f"Target landmarks: {target_landmarks}", flush=True)
    print(f"Samples train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}", flush=True)

    base_predictions = load_or_generate_base_predictions(args, dataset, split_indices, base_config, device, output_dir)
    train_ds = Stage3PatchDataset(
        dataset,
        train_idx,
        base_predictions["train"],
        target_landmarks,
        patch_points=args.patch_points,
        patch_radius_mm=args.patch_radius_mm,
        heatmap_sigma_mm=args.patch_heatmap_sigma_mm,
        center_jitter_mm=args.center_jitter_mm,
        point_noise_mm=args.point_noise_mm,
        point_dropout=args.point_dropout,
        focus_min_mm=args.focus_min_mm,
        focus_max_mm=args.focus_max_mm,
        focus_weight=args.focus_weight,
        seed=args.seed,
    )
    val_ds = Stage3PatchDataset(
        dataset,
        val_idx,
        base_predictions["val"],
        target_landmarks,
        args.patch_points,
        args.patch_radius_mm,
        args.patch_heatmap_sigma_mm,
        seed=args.seed,
    )
    test_ds = Stage3PatchDataset(
        dataset,
        test_idx,
        base_predictions["test"],
        target_landmarks,
        args.patch_points,
        args.patch_radius_mm,
        args.patch_heatmap_sigma_mm,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PatchResidualRefiner(
        input_dim=7,
        width=args.refiner_width,
        landmark_dim=args.landmark_embedding_dim,
        dropout=args.refiner_dropout,
        residual_limit_mm=args.residual_limit_mm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01)

    val_base_pred, val_stage3_pred, val_expert, val_base_errors, val_stage3_errors = evaluate_stage3(
        model, val_loader, base_predictions["val"], val_idx, target_landmarks, device, args
    )
    best_val_ale = float(val_stage3_errors.mean())
    torch.save(model.state_dict(), output_dir / "best_stage3_refiner.pth")
    history = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_base_ale": float(val_base_errors.mean()),
            "val_stage3_all_target_ale": best_val_ale,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
    ]
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, parts = train_epoch(model, train_loader, optimizer, device, args)
        val_base_pred, val_stage3_pred, val_expert, val_base_errors, val_stage3_errors = evaluate_stage3(
            model, val_loader, base_predictions["val"], val_idx, target_landmarks, device, args
        )
        val_ale = float(val_stage3_errors.mean())
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_parts": parts,
                "val_base_ale": float(val_base_errors.mean()),
                "val_stage3_all_target_ale": val_ale,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.5f} "
            f"val_base={float(val_base_errors.mean()):.4f} val_stage3={val_ale:.4f}",
            flush=True,
        )
        if val_ale < best_val_ale:
            best_val_ale = val_ale
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_stage3_refiner.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    (output_dir / "history_stage3.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.load_state_dict(torch.load(output_dir / "best_stage3_refiner.pth", map_location=device))
    val_base_pred, val_stage3_pred, val_expert, val_base_errors, val_stage3_errors = evaluate_stage3(
        model, val_loader, base_predictions["val"], val_idx, target_landmarks, device, args
    )
    test_base_pred, test_stage3_pred, test_expert, test_base_errors, test_stage3_errors = evaluate_stage3(
        model, test_loader, base_predictions["test"], test_idx, target_landmarks, device, args
    )
    val_final_pred, val_final_errors, enabled = final_with_gate(
        val_base_pred,
        val_stage3_pred,
        val_expert,
        target_landmarks,
        val_base_errors,
        val_stage3_errors,
        args.min_val_improvement_mm,
    )
    test_final_pred = test_base_pred.copy()
    for lm_idx in enabled:
        test_final_pred[:, lm_idx] = test_stage3_pred[:, lm_idx]
    test_final_errors = np.linalg.norm(test_final_pred - test_expert, axis=-1)

    write_prediction_csv(
        output_dir / "stage3_predictions_val.csv",
        dataset,
        val_idx,
        val_base_pred,
        val_stage3_pred,
        val_final_pred,
        val_expert,
        val_base_errors,
        val_stage3_errors,
        val_final_errors,
        enabled,
    )
    write_prediction_csv(
        output_dir / "stage3_predictions_test.csv",
        dataset,
        test_idx,
        test_base_pred,
        test_stage3_pred,
        test_final_pred,
        test_expert,
        test_base_errors,
        test_stage3_errors,
        test_final_errors,
        enabled,
    )
    write_analysis_csvs(output_dir, "val", dataset, val_idx, val_final_errors)
    write_analysis_csvs(output_dir, "test", dataset, test_idx, test_final_errors)
    metrics = {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "model": "AGH-Former Stage 3 clinical-threshold fine refiner",
        "base_run_dir": str(args.base_run_dir),
        "target_landmarks": target_landmarks,
        "enabled_landmarks": enabled,
        "base_validation": summarize(val_base_errors),
        "stage3_all_target_validation": summarize(val_stage3_errors),
        "stage3_gated_validation": summarize(val_final_errors),
        "base_test": summarize(test_base_errors),
        "stage3_all_target_test": summarize(test_stage3_errors),
        "stage3_gated_test": summarize(test_final_errors),
        "base_target_landmarks_test": summarize_subset(test_base_errors, target_landmarks),
        "stage3_gated_target_landmarks_test": summarize_subset(test_final_errors, target_landmarks),
        "base_core20_test": summarize_subset(test_base_errors, [idx for idx in range(23) if idx not in {0, 21, 22}]),
        "stage3_gated_core20_test": summarize_subset(test_final_errors, [idx for idx in range(23) if idx not in {0, 21, 22}]),
    }
    (output_dir / "metrics_stage3.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"Base Stage2 ALE: {metrics['base_test']['ale']:.4f}", flush=True)
    print(f"Stage3 all-target ALE: {metrics['stage3_all_target_test']['ale']:.4f}", flush=True)
    print(f"Stage3 gated ALE: {metrics['stage3_gated_test']['ale']:.4f}", flush=True)
    print(f"Stage3 gated median: {metrics['stage3_gated_test']['median']:.4f}", flush=True)
    print(f"Enabled landmarks: {enabled}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
