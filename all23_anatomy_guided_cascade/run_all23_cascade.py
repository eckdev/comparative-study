import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


NUM_LANDMARKS = 23
HARD_LANDMARKS = (0, 21, 22)
CORE20 = tuple(idx for idx in range(NUM_LANDMARKS) if idx not in HARD_LANDMARKS)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_float_grid(value):
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_int_grid(value):
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def infer_prefix(row, requested):
    if requested and f"{requested}_x" in row:
        return requested
    for prefix in ("final", "stacked", "shape_prior", "stage2_raw", "stage2_snapped", "base", "pred"):
        if f"{prefix}_x" in row:
            return prefix
    raise KeyError(f"Could not infer coordinate prefix from columns: {sorted(row)}")


@dataclass
class PredictionSplit:
    sample_ids: list
    pred: np.ndarray
    expert: np.ndarray
    metadata: dict
    source_path: str
    source_prefix: str


def load_prediction_file(path, requested_prefix=None):
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    prefix = infer_prefix(rows[0], requested_prefix)
    sample_ids = []
    pred = {}
    expert = {}
    metadata = {}
    seen_landmarks = {}
    for row in rows:
        sample_id = row["sample_id"]
        lm_idx = int(row["landmark"])
        if sample_id not in pred:
            sample_ids.append(sample_id)
            pred[sample_id] = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            expert[sample_id] = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            seen_landmarks[sample_id] = set()
            metadata[sample_id] = {
                "sample_id": sample_id,
                "class": row.get("class", ""),
                "gender": row.get("gender", ""),
                "subject_id": row.get("subject_id", ""),
            }
        if lm_idx in seen_landmarks[sample_id]:
            raise ValueError(f"Duplicate landmark {lm_idx} for {sample_id} in {path}")
        seen_landmarks[sample_id].add(lm_idx)
        pred[sample_id][lm_idx] = [float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")]
        expert[sample_id][lm_idx] = [float(row[f"expert_{axis}"]) for axis in ("x", "y", "z")]
    incomplete = {key: sorted(set(range(NUM_LANDMARKS)) - value) for key, value in seen_landmarks.items() if len(value) != NUM_LANDMARKS}
    if incomplete:
        raise ValueError(f"Samples with incomplete landmark sets in {path}: {incomplete}")
    return PredictionSplit(
        sample_ids=sample_ids,
        pred=np.stack([pred[sample_id] for sample_id in sample_ids]),
        expert=np.stack([expert[sample_id] for sample_id in sample_ids]),
        metadata=metadata,
        source_path=str(path),
        source_prefix=prefix,
    )


def load_base_split(base_dir, split, prefix):
    path = Path(base_dir) / f"refined_predictions_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Export the Stage2 train predictions first with "
            "agh_former_orthodontic_comparison/export_stage2_train_predictions.py."
        )
    return load_prediction_file(path, prefix)


def load_initial_split(initial_dir, split, prefix, base):
    if initial_dir is None:
        return base
    path = Path(initial_dir) / f"predictions_{split}.csv"
    initial = load_prediction_file(path, prefix)
    by_id = {sample_id: idx for idx, sample_id in enumerate(initial.sample_ids)}
    missing = [sample_id for sample_id in base.sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Initial prediction directory is missing samples: {missing[:5]}")
    order = [by_id[sample_id] for sample_id in base.sample_ids]
    expert = initial.expert[order]
    if not np.allclose(expert, base.expert, atol=1e-3):
        raise ValueError(f"Expert coordinates disagree between base and initial predictions for split={split}")
    return PredictionSplit(
        sample_ids=list(base.sample_ids),
        pred=initial.pred[order],
        expert=expert,
        metadata=base.metadata,
        source_path=initial.source_path,
        source_prefix=initial.source_prefix,
    )


class PointCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Point cache does not exist: {self.cache_dir}")
        self.paths = {}
        for path in self.cache_dir.glob("*.npz"):
            parts = path.name.split("__")
            if len(parts) < 3:
                continue
            class_name, gender = parts[0], parts[1]
            subject_text = parts[2].split(".ply", 1)[0]
            prefix = "F" if gender == "women" else "M"
            self.paths[f"{class_name}_{prefix}{int(subject_text)}"] = path
        if not self.paths:
            raise ValueError(f"No AGH point-cache .npz files found in {self.cache_dir}")

    def load(self, sample_id):
        if sample_id not in self.paths:
            raise FileNotFoundError(f"No point-cache file for {sample_id} below {self.cache_dir}")
        data = np.load(self.paths[sample_id])
        return data["points_world"].astype(np.float32), data["features"].astype(np.float32)


def metadata_vector(meta):
    return np.asarray(
        [
            float(meta.get("class") == "Class1"),
            float(meta.get("class") == "Class2"),
            float(meta.get("class") == "Class3"),
            float(meta.get("gender") == "women"),
            float(meta.get("gender") == "men"),
        ],
        dtype=np.float32,
    )


def normalized_landmark_context(points):
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    scale = max(float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))), 1e-6)
    return (centered / scale).reshape(-1).astype(np.float32)


class Hard3SurfaceDataset(Dataset):
    """One item is one face and contains candidate sets for all three hard landmarks."""

    def __init__(
        self,
        split,
        point_cache,
        candidate_points=4096,
        local_fraction=0.75,
        local_radius_mm=35.0,
        heatmap_sigmas=(6.0, 4.5, 4.5),
        training=False,
        center_jitter_mm=0.0,
        point_noise_mm=0.0,
        point_dropout=0.0,
        seed=42,
    ):
        self.split = split
        self.point_cache = point_cache
        self.candidate_points = int(candidate_points)
        self.local_fraction = float(local_fraction)
        self.local_radius_mm = float(local_radius_mm)
        self.heatmap_sigmas = tuple(float(value) for value in heatmap_sigmas)
        self.training = bool(training)
        self.center_jitter_mm = float(center_jitter_mm)
        self.point_noise_mm = float(point_noise_mm)
        self.point_dropout = float(point_dropout)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.split.sample_ids)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _candidate_indices(self, points, center, rng):
        n_points = len(points)
        count = min(self.candidate_points, n_points)
        local_count = min(int(round(count * self.local_fraction)), count)
        dists = np.linalg.norm(points - center[None, :], axis=1)
        within = np.flatnonzero(dists <= self.local_radius_mm)
        if len(within) >= local_count:
            local = within[np.argpartition(dists[within], local_count - 1)[:local_count]]
        else:
            local = np.argpartition(dists, max(local_count - 1, 0))[:local_count]
        remaining_count = count - len(local)
        if remaining_count > 0:
            mask = np.ones(n_points, dtype=bool)
            mask[local] = False
            remaining = np.flatnonzero(mask)
            if self.training:
                global_idx = rng.choice(remaining, remaining_count, replace=len(remaining) < remaining_count)
            else:
                positions = np.linspace(0, max(len(remaining) - 1, 0), remaining_count, dtype=int)
                global_idx = remaining[positions]
            selected = np.concatenate([local, global_idx])
        else:
            selected = local
        if len(selected) < self.candidate_points:
            extra = rng.choice(selected, self.candidate_points - len(selected), replace=True)
            selected = np.concatenate([selected, extra])
        return selected.astype(np.int64)

    def __getitem__(self, item_idx):
        sample_id = self.split.sample_ids[item_idx]
        points_world, surface_features = self.point_cache.load(sample_id)
        base_landmarks = self.split.pred[item_idx].astype(np.float32)
        model_landmarks = base_landmarks.copy()
        experts = self.split.expert[item_idx].astype(np.float32)
        rng = np.random.default_rng(self.seed + item_idx + self.epoch * 100_003)
        if self.training and self.center_jitter_mm > 0:
            model_landmarks[list(HARD_LANDMARKS)] += rng.normal(
                0.0, self.center_jitter_mm, size=(len(HARD_LANDMARKS), 3)
            ).astype(np.float32)
        context = np.concatenate(
            [normalized_landmark_context(model_landmarks), metadata_vector(self.split.metadata[sample_id])]
        ).astype(np.float32)

        feature_rows = []
        candidate_rows = []
        target_rows = []
        nearest_rows = []
        oracle_rows = []
        for hard_pos, lm_idx in enumerate(HARD_LANDMARKS):
            center = model_landmarks[lm_idx].copy()
            selected = self._candidate_indices(points_world, center, rng)
            candidates = points_world[selected].copy()
            local_xyz = (candidates - center[None, :]) / max(self.local_radius_mm, 1e-6)
            if self.training and self.point_noise_mm > 0:
                local_xyz += rng.normal(
                    0.0,
                    self.point_noise_mm / max(self.local_radius_mm, 1e-6),
                    size=local_xyz.shape,
                ).astype(np.float32)
            intrinsic = surface_features[selected]
            relative_dists = np.linalg.norm(
                candidates[:, None, :] - model_landmarks[None, :, :], axis=-1
            ) / 100.0
            radial = np.linalg.norm(local_xyz, axis=1, keepdims=True)
            point_features = np.concatenate([local_xyz, intrinsic, relative_dists, radial], axis=1).astype(np.float32)
            if self.training and self.point_dropout > 0:
                drop = rng.random(len(point_features)) < self.point_dropout
                point_features[drop] = 0.0
            expert_dists = np.linalg.norm(candidates - experts[lm_idx][None, :], axis=1)
            sigma = max(self.heatmap_sigmas[hard_pos], 1e-6)
            target = np.exp(-(expert_dists**2) / (2.0 * sigma**2)).astype(np.float32)
            target /= max(float(target.sum()), 1e-12)
            feature_rows.append(point_features)
            candidate_rows.append(candidates.astype(np.float32))
            target_rows.append(target)
            nearest_rows.append(int(expert_dists.argmin()))
            oracle_rows.append(float(expert_dists.min()))

        return {
            "point_features": torch.tensor(np.stack(feature_rows), dtype=torch.float32),
            "candidate_points": torch.tensor(np.stack(candidate_rows), dtype=torch.float32),
            "target_distribution": torch.tensor(np.stack(target_rows), dtype=torch.float32),
            "nearest_index": torch.tensor(nearest_rows, dtype=torch.long),
            "expert": torch.tensor(experts[list(HARD_LANDMARKS)], dtype=torch.float32),
            "base_hard": torch.tensor(base_landmarks[list(HARD_LANDMARKS)], dtype=torch.float32),
            "context": torch.tensor(context, dtype=torch.float32),
            "oracle_error": torch.tensor(oracle_rows, dtype=torch.float32),
            "sample_position": torch.tensor(item_idx, dtype=torch.long),
        }


class Hard3SurfaceHeatmapRefiner(nn.Module):
    def __init__(self, point_dim, context_dim=74, width=192, landmark_dim=48, dropout=0.1):
        super().__init__()
        hidden = max(width // 2, 32)
        self.landmark_embedding = nn.Embedding(len(HARD_LANDMARKS), landmark_dim)
        self.point_encoder = nn.Sequential(
            nn.Conv1d(point_dim, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.landmark_projection = nn.Linear(landmark_dim, width)
        self.score_head = nn.Sequential(
            nn.Conv1d(width * 2, width, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 1),
        )
        self.log_var_head = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, point_features, context):
        batch_size, hard_count, point_count, feature_dim = point_features.shape
        flat = point_features.reshape(batch_size * hard_count, point_count, feature_dim).transpose(1, 2)
        encoded = self.point_encoder(flat)
        context_encoded = self.context_encoder(context)
        context_encoded = context_encoded[:, None, :].expand(-1, hard_count, -1).reshape(batch_size * hard_count, -1)
        lm_ids = torch.arange(hard_count, device=point_features.device)[None].expand(batch_size, -1).reshape(-1)
        conditioned = context_encoded + self.landmark_projection(self.landmark_embedding(lm_ids))
        conditioned_points = conditioned[:, :, None].expand(-1, -1, point_count)
        logits = self.score_head(torch.cat([encoded, conditioned_points], dim=1)).squeeze(1)
        pooled_max = encoded.max(dim=2).values
        pooled_mean = encoded.mean(dim=2)
        log_var = self.log_var_head(torch.cat([pooled_max, pooled_mean, conditioned], dim=1)).squeeze(1)
        return logits.reshape(batch_size, hard_count, point_count), log_var.reshape(batch_size, hard_count).clamp(-6, 6)


def weighted_coordinate(logits, candidates, temperature=1.0):
    weights = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
    return torch.einsum("bhk,bhkd->bhd", weights, candidates)


def compute_loss(logits, log_var, batch, args):
    log_probs = F.log_softmax(logits, dim=-1)
    heatmap = -(batch["target_distribution"] * log_probs).sum(dim=-1).mean()
    nearest = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["nearest_index"].reshape(-1))
    pred = weighted_coordinate(logits, batch["candidate_points"], args.train_temperature)
    coord_per = F.smooth_l1_loss(pred, batch["expert"], beta=1.0, reduction="none").mean(dim=-1)
    coord = coord_per.mean()
    euclidean = torch.linalg.norm(pred - batch["expert"], dim=-1)
    nll = (0.5 * torch.exp(-log_var) * euclidean.pow(2) + 0.5 * log_var).mean()
    clinical = F.softplus((euclidean - 2.0) / 0.5).mean()
    loss = (
        args.heatmap_weight * heatmap
        + args.nearest_ce_weight * nearest
        + args.coordinate_weight * coord
        + args.uncertainty_weight * nll
        + args.clinical_weight * clinical
    )
    return loss, pred, {
        "heatmap": float(heatmap.detach()),
        "nearest_ce": float(nearest.detach()),
        "coordinate": float(coord.detach()),
        "uncertainty": float(nll.detach()),
        "clinical": float(clinical.detach()),
    }


def autocast_context(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # Older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    totals = {"loss": 0.0, "count": 0}
    per_landmark = np.zeros((3, 2), dtype=np.float64)
    for batch in tqdm(loader, desc="hard3 train", leave=False, disable=args.no_tqdm):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.mixed_precision):
            logits, log_var = model(batch["point_features"], batch["context"])
            loss, pred, _ = compute_loss(logits, log_var, batch, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = batch["point_features"].shape[0]
        totals["loss"] += float(loss.detach()) * batch_size
        totals["count"] += batch_size
        errors = torch.linalg.norm(pred.detach() - batch["expert"], dim=-1).cpu().numpy()
        per_landmark[:, 0] += errors.sum(axis=0)
        per_landmark[:, 1] += len(errors)
    return totals["loss"] / max(totals["count"], 1), (per_landmark[:, 0] / np.maximum(per_landmark[:, 1], 1)).tolist()


@torch.no_grad()
def collect_outputs(model, loader, device, args):
    model.eval()
    outputs = {}
    for batch in tqdm(loader, desc="hard3 eval", leave=False, disable=args.no_tqdm):
        positions = batch["sample_position"].numpy().astype(int)
        device_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast_context(device, args.mixed_precision):
            logits, log_var = model(device_batch["point_features"], device_batch["context"])
        logits = logits.float().cpu().numpy()
        log_var = log_var.float().cpu().numpy()
        candidates = batch["candidate_points"].numpy()
        oracle = batch["oracle_error"].numpy()
        expert = batch["expert"].numpy()
        for row, position in enumerate(positions):
            outputs[int(position)] = {
                "logits": logits[row],
                "log_var": log_var[row],
                "candidates": candidates[row],
                "oracle": oracle[row],
                "expert": expert[row],
            }
    ordered = [outputs[idx] for idx in range(len(loader.dataset))]
    return {
        key: np.stack([item[key] for item in ordered])
        for key in ("logits", "log_var", "candidates", "oracle", "expert")
    }


def numpy_topk_coordinate(logits, candidates, topk, temperature):
    sample_count, hard_count, point_count = logits.shape
    output = np.zeros((sample_count, hard_count, 3), dtype=np.float32)
    use_count = point_count if topk <= 0 else min(int(topk), point_count)
    for sample_idx in range(sample_count):
        for hard_pos in range(hard_count):
            scores = logits[sample_idx, hard_pos]
            if use_count < point_count:
                selected = np.argpartition(scores, -use_count)[-use_count:]
            else:
                selected = np.arange(point_count)
            local_scores = scores[selected] / max(float(temperature), 1e-6)
            local_scores = local_scores - local_scores.max()
            weights = np.exp(local_scores)
            weights /= max(float(weights.sum()), 1e-12)
            output[sample_idx, hard_pos] = np.sum(candidates[sample_idx, hard_pos, selected] * weights[:, None], axis=0)
    return output


def confidence_from_log_var(log_var, calibration, power):
    median = calibration["median"]
    scale = calibration["scale"]
    z = (log_var - median[None, :]) / scale[None, :]
    confidence = 1.0 / (1.0 + np.exp(np.clip(z, -20, 20)))
    return np.power(np.clip(confidence, 1e-3, 1.0), float(power))


def fuse_predictions(initial, branch_hard, log_var, blend, fusion_mode, confidence_power, calibration):
    final = initial.copy()
    if fusion_mode == "fixed":
        effective = np.full(log_var.shape, float(blend), dtype=np.float32)
    else:
        confidence = confidence_from_log_var(log_var, calibration, confidence_power)
        effective = float(blend) * (0.5 + 0.5 * confidence)
    hard_initial = initial[:, HARD_LANDMARKS]
    final[:, HARD_LANDMARKS] = hard_initial + effective[:, :, None] * (branch_hard - hard_initial)
    return final, effective


def errors(pred, expert):
    return np.linalg.norm(np.asarray(pred) - np.asarray(expert), axis=-1)


def summarize(error_array):
    arr = np.asarray(error_array, dtype=np.float64)
    flat = arr.reshape(-1)
    result = {
        "n": int(flat.size),
        "ale": float(flat.mean()),
        "std": float(flat.std()),
        "median": float(np.median(flat)),
        "p75": float(np.percentile(flat, 75)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
    }
    for threshold in (2.0, 2.5, 3.0, 4.0):
        result[f"sdr_at_{str(threshold).replace('.', '_')}mm"] = float((flat <= threshold).mean())
    if arr.ndim == 2:
        result["per_landmark_mean"] = arr.mean(axis=0).astype(float).tolist()
        result["per_landmark_median"] = np.median(arr, axis=0).astype(float).tolist()
    return result


def bootstrap_comparison(initial_errors, final_errors, iters, seed):
    rng = np.random.default_rng(seed)
    initial_errors = np.asarray(initial_errors)
    final_errors = np.asarray(final_errors)
    deltas = []
    for _ in range(int(iters)):
        idx = rng.integers(0, len(initial_errors), size=len(initial_errors))
        deltas.append(float(final_errors[idx].mean() - initial_errors[idx].mean()))
    deltas = np.asarray(deltas)
    return {
        "delta_ale": float(final_errors.mean() - initial_errors.mean()),
        "delta_ale_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "probability_improved": float((deltas < 0).mean()),
    }


def group_rows(split, error_array):
    rows = []
    groups = {
        "overall:all": list(range(len(split.sample_ids))),
    }
    for sample_idx, sample_id in enumerate(split.sample_ids):
        meta = split.metadata[sample_id]
        groups.setdefault(f"class:{meta.get('class', '')}", []).append(sample_idx)
        groups.setdefault(f"gender:{meta.get('gender', '')}", []).append(sample_idx)
        groups.setdefault(f"class_gender:{meta.get('class', '')}_{meta.get('gender', '')}", []).append(sample_idx)
    for key, indices in groups.items():
        scope, group = key.split(":", 1)
        rows.append({"scope": scope, "group": group, "n_samples": len(indices), **summarize(error_array[indices])})
    return rows


def landmark_rows(initial_errors, final_errors):
    rows = []
    for lm_idx in range(NUM_LANDMARKS):
        before = initial_errors[:, lm_idx]
        after = final_errors[:, lm_idx]
        rows.append(
            {
                "landmark": lm_idx,
                "initial_mean": float(before.mean()),
                "final_mean": float(after.mean()),
                "delta_mean": float(after.mean() - before.mean()),
                "initial_median": float(np.median(before)),
                "final_median": float(np.median(after)),
                "initial_sdr_at_2mm": float((before <= 2.0).mean()),
                "final_sdr_at_2mm": float((after <= 2.0).mean()),
            }
        )
    return rows


def prediction_rows(split, base, initial, branch, final, confidence, final_errors):
    rows = []
    hard_position = {lm_idx: pos for pos, lm_idx in enumerate(HARD_LANDMARKS)}
    for sample_idx, sample_id in enumerate(split.sample_ids):
        meta = split.metadata[sample_id]
        for lm_idx in range(NUM_LANDMARKS):
            row = {
                "sample_id": sample_id,
                "class": meta.get("class", ""),
                "gender": meta.get("gender", ""),
                "subject_id": meta.get("subject_id", ""),
                "landmark": lm_idx,
            }
            for name, values in (
                ("expert", split.expert[sample_idx, lm_idx]),
                ("base", base[sample_idx, lm_idx]),
                ("initial", initial[sample_idx, lm_idx]),
                ("final", final[sample_idx, lm_idx]),
            ):
                for axis_idx, axis in enumerate(("x", "y", "z")):
                    row[f"{name}_{axis}"] = float(values[axis_idx])
            if lm_idx in hard_position:
                pos = hard_position[lm_idx]
                for axis_idx, axis in enumerate(("x", "y", "z")):
                    row[f"branch_{axis}"] = float(branch[sample_idx, pos, axis_idx])
                row["branch_confidence"] = float(confidence[sample_idx, pos])
            else:
                row.update({"branch_x": "", "branch_y": "", "branch_z": "", "branch_confidence": ""})
            row["final_error"] = float(final_errors[sample_idx, lm_idx])
            rows.append(row)
    return rows


def select_postprocess(val_outputs, val_initial, val_expert, args):
    log_var = val_outputs["log_var"]
    calibration = {
        "median": np.median(log_var, axis=0),
        "scale": np.maximum(np.percentile(log_var, 75, axis=0) - np.percentile(log_var, 25, axis=0), 0.25),
    }
    best = None
    sweep = []
    for topk in parse_int_grid(args.eval_topk_grid):
        for temperature in parse_float_grid(args.eval_temperature_grid):
            branch = numpy_topk_coordinate(val_outputs["logits"], val_outputs["candidates"], topk, temperature)
            for blend in parse_float_grid(args.blend_grid):
                for fusion_mode in args.fusion_modes.split(","):
                    fusion_mode = fusion_mode.strip()
                    powers = [0.0] if fusion_mode == "fixed" else parse_float_grid(args.confidence_power_grid)
                    for power in powers:
                        final, effective = fuse_predictions(
                            val_initial,
                            branch,
                            log_var,
                            blend,
                            fusion_mode,
                            power,
                            calibration,
                        )
                        score = float(errors(final, val_expert).mean())
                        hard_score = float(errors(final, val_expert)[:, HARD_LANDMARKS].mean())
                        row = {
                            "topk": topk,
                            "temperature": temperature,
                            "blend": blend,
                            "fusion_mode": fusion_mode,
                            "confidence_power": power,
                            "validation_ale": score,
                            "validation_hard3_ale": hard_score,
                            "mean_effective_blend": float(effective.mean()),
                        }
                        sweep.append(row)
                        if best is None or score < best["validation_ale"]:
                            best = dict(row)
    calibration = {key: value.astype(float).tolist() for key, value in calibration.items()}
    return best, sweep, calibration


def apply_postprocess(outputs, initial, config, calibration):
    calibration_np = {key: np.asarray(value, dtype=np.float32) for key, value in calibration.items()}
    branch = numpy_topk_coordinate(
        outputs["logits"], outputs["candidates"], int(config["topk"]), float(config["temperature"])
    )
    final, effective = fuse_predictions(
        initial,
        branch,
        outputs["log_var"],
        float(config["blend"]),
        config["fusion_mode"],
        float(config["confidence_power"]),
        calibration_np,
    )
    return branch, final, effective


def make_loader(dataset, batch_size, shuffle, args):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        # Workers are recreated each epoch so dataset.set_epoch() also changes augmentation seeds.
        persistent_workers=False,
        generator=generator,
    )


def main():
    parser = argparse.ArgumentParser(description="All-23 anatomy-guided cascade with a hard-landmark surface heatmap branch.")
    parser.add_argument("--base-prediction-dir", required=True)
    parser.add_argument("--initial-prediction-dir", default=None)
    parser.add_argument("--point-cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-prefix", default="stage2_raw")
    parser.add_argument("--initial-prefix", default="stacked")
    parser.add_argument("--candidate-points", type=int, default=4096)
    parser.add_argument("--local-fraction", type=float, default=0.75)
    parser.add_argument("--local-radius-mm", type=float, default=35.0)
    parser.add_argument("--heatmap-sigmas", default="6.0,4.5,4.5")
    parser.add_argument("--center-jitter-mm", type=float, default=1.0)
    parser.add_argument("--point-noise-mm", type=float, default=0.1)
    parser.add_argument("--point-dropout", type=float, default=0.03)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--landmark-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-temperature", type=float, default=0.75)
    parser.add_argument("--heatmap-weight", type=float, default=1.0)
    parser.add_argument("--nearest-ce-weight", type=float, default=0.1)
    parser.add_argument("--coordinate-weight", type=float, default=0.5)
    parser.add_argument("--uncertainty-weight", type=float, default=0.02)
    parser.add_argument("--clinical-weight", type=float, default=0.05)
    parser.add_argument("--eval-topk-grid", default="5,10,20,30,50")
    parser.add_argument("--eval-temperature-grid", default="0.35,0.5,0.75,1.0")
    parser.add_argument("--blend-grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--fusion-modes", default="fixed,confidence")
    parser.add_argument("--confidence-power-grid", default="0.5,1.0,2.0")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    point_cache_dir = Path(args.point_cache_dir) if args.point_cache_dir else Path(args.base_prediction_dir) / "stage1_point_cache"
    device = resolve_device(args.device)
    args.mixed_precision = bool(args.mixed_precision and device.type == "cuda")
    print(f"Device: {device}; mixed_precision={args.mixed_precision}", flush=True)

    train = load_base_split(args.base_prediction_dir, "train", args.base_prefix)
    val = load_base_split(args.base_prediction_dir, "val", args.base_prefix)
    test = load_base_split(args.base_prediction_dir, "test", args.base_prefix)
    if args.max_samples:
        for split in (train, val, test):
            limit = min(int(args.max_samples), len(split.sample_ids))
            split.sample_ids = split.sample_ids[:limit]
            split.pred = split.pred[:limit]
            split.expert = split.expert[:limit]
    val_initial = load_initial_split(args.initial_prediction_dir, "val", args.initial_prefix, val)
    test_initial = load_initial_split(args.initial_prediction_dir, "test", args.initial_prefix, test)
    print(f"Samples train/val/test: {len(train.sample_ids)}/{len(val.sample_ids)}/{len(test.sample_ids)}", flush=True)
    print(f"Initial all-23 source: {val_initial.source_path}", flush=True)

    point_cache = PointCache(point_cache_dir)
    dataset_kwargs = {
        "point_cache": point_cache,
        "candidate_points": args.candidate_points,
        "local_fraction": args.local_fraction,
        "local_radius_mm": args.local_radius_mm,
        "heatmap_sigmas": parse_float_grid(args.heatmap_sigmas),
        "seed": args.seed,
    }
    if len(dataset_kwargs["heatmap_sigmas"]) != 3:
        raise ValueError("--heatmap-sigmas must contain exactly three values for LM0, LM21 and LM22")
    train_dataset = Hard3SurfaceDataset(
        train,
        training=True,
        center_jitter_mm=args.center_jitter_mm,
        point_noise_mm=args.point_noise_mm,
        point_dropout=args.point_dropout,
        **dataset_kwargs,
    )
    val_dataset = Hard3SurfaceDataset(val, training=False, **dataset_kwargs)
    test_dataset = Hard3SurfaceDataset(test, training=False, **dataset_kwargs)
    train_loader = make_loader(train_dataset, args.batch_size, True, args)
    val_loader = make_loader(val_dataset, args.eval_batch_size, False, args)
    test_loader = make_loader(test_dataset, args.eval_batch_size, False, args)

    probe = train_dataset[0]
    point_dim = int(probe["point_features"].shape[-1])
    context_dim = int(probe["context"].shape[-1])
    model = Hard3SurfaceHeatmapRefiner(
        point_dim=point_dim,
        context_dim=context_dim,
        width=args.width,
        landmark_dim=args.landmark_dim,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Point feature dim: {point_dim}; context dim: {context_dim}; parameters: {parameter_count:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    scaler = make_grad_scaler(args.mixed_precision)
    best_score = math.inf
    best_epoch = 0
    stale = 0
    history = []
    checkpoint_path = output_dir / "best_hard3_surface_refiner.pth"
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loss, train_per_landmark = train_epoch(model, train_loader, optimizer, scaler, device, args)
        val_outputs_epoch = collect_outputs(model, val_loader, device, args)
        val_branch = numpy_topk_coordinate(
            val_outputs_epoch["logits"], val_outputs_epoch["candidates"], topk=30, temperature=args.train_temperature
        )
        val_hard_errors = errors(val_branch, val.expert[:, HARD_LANDMARKS])
        val_score = float(val_hard_errors.mean())
        scheduler.step(val_score)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_hard3_ale": val_score,
            "lr": optimizer.param_groups[0]["lr"],
            "train_lm0_ale": train_per_landmark[0],
            "train_lm21_ale": train_per_landmark[1],
            "train_lm22_ale": train_per_landmark[2],
            "val_lm0_ale": float(val_hard_errors[:, 0].mean()),
            "val_lm21_ale": float(val_hard_errors[:, 1].mean()),
            "val_lm22_ale": float(val_hard_errors[:, 2].mean()),
        }
        history.append(row)
        print(
            f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.5f} val_hard3={val_score:.4f} "
            f"LM0/21/22={row['val_lm0_ale']:.3f}/{row['val_lm21_ale']:.3f}/{row['val_lm22_ale']:.3f}",
            flush=True,
        )
        if val_score < best_score - 1e-5:
            best_score = val_score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_hard3_ale": val_score,
                    "point_dim": point_dim,
                    "context_dim": context_dim,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            stale += 1
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
            break

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_outputs = collect_outputs(model, val_loader, device, args)
    best_postprocess, sweep_rows, confidence_calibration = select_postprocess(
        val_outputs, val_initial.pred, val.expert, args
    )
    val_branch, val_final, val_effective = apply_postprocess(
        val_outputs, val_initial.pred, best_postprocess, confidence_calibration
    )

    # Test labels are first consumed after all model and postprocess choices are locked on validation.
    test_outputs = collect_outputs(model, test_loader, device, args)
    test_branch, test_final, test_effective = apply_postprocess(
        test_outputs, test_initial.pred, best_postprocess, confidence_calibration
    )
    val_initial_errors = errors(val_initial.pred, val.expert)
    val_final_errors = errors(val_final, val.expert)
    test_base_errors = errors(test.pred, test.expert)
    test_initial_errors = errors(test_initial.pred, test.expert)
    test_final_errors = errors(test_final, test.expert)

    write_csv(output_dir / "postprocess_sweep_val.csv", sweep_rows)
    write_csv(
        output_dir / "predictions_val.csv",
        prediction_rows(val, val.pred, val_initial.pred, val_branch, val_final, val_effective, val_final_errors),
    )
    write_csv(
        output_dir / "predictions_test.csv",
        prediction_rows(test, test.pred, test_initial.pred, test_branch, test_final, test_effective, test_final_errors),
    )
    write_csv(output_dir / "landmark_metrics_val.csv", landmark_rows(val_initial_errors, val_final_errors))
    write_csv(output_dir / "landmark_metrics_test.csv", landmark_rows(test_initial_errors, test_final_errors))
    write_csv(output_dir / "group_metrics_test.csv", group_rows(test, test_final_errors))

    metrics = {
        "model": "All-23 Anatomy-Guided Cascade",
        "hard_landmarks": list(HARD_LANDMARKS),
        "best_epoch": best_epoch,
        "best_training_validation_hard3_ale": best_score,
        "parameter_count_hard3_branch": parameter_count,
        "training_seconds": float(time.time() - started),
        "selected_postprocess": best_postprocess,
        "confidence_calibration": confidence_calibration,
        "candidate_oracle_validation_hard3": summarize(val_outputs["oracle"]),
        "candidate_oracle_test_hard3": summarize(test_outputs["oracle"]),
        "initial_validation_all23": summarize(val_initial_errors),
        "final_validation_all23": summarize(val_final_errors),
        "base_test_all23": summarize(test_base_errors),
        "initial_test_all23": summarize(test_initial_errors),
        "final_test_all23": summarize(test_final_errors),
        "initial_test_core20": summarize(test_initial_errors[:, CORE20]),
        "final_test_core20": summarize(test_final_errors[:, CORE20]),
        "initial_test_hard3": summarize(test_initial_errors[:, HARD_LANDMARKS]),
        "final_test_hard3": summarize(test_final_errors[:, HARD_LANDMARKS]),
        "bootstrap_final_vs_initial": bootstrap_comparison(
            test_initial_errors, test_final_errors, args.bootstrap_iters, args.seed
        ),
        "data_sources": {
            "base_train": train.source_path,
            "base_val": val.source_path,
            "base_test": test.source_path,
            "initial_val": val_initial.source_path,
            "initial_test": test_initial.source_path,
            "point_cache": str(point_cache_dir),
        },
    }
    (output_dir / "metrics_all23_cascade.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "config_all23_cascade.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"Initial all-23 ALE: {metrics['initial_test_all23']['ale']:.4f}", flush=True)
    print(f"Final all-23 ALE: {metrics['final_test_all23']['ale']:.4f}", flush=True)
    print(f"Final median: {metrics['final_test_all23']['median']:.4f}", flush=True)
    print(
        f"Hard3 ALE: {metrics['initial_test_hard3']['ale']:.4f} -> {metrics['final_test_hard3']['ale']:.4f}",
        flush=True,
    )
    print(f"Hard3 candidate oracle ALE: {metrics['candidate_oracle_test_hard3']['ale']:.4f}", flush=True)
    ci = metrics["bootstrap_final_vs_initial"]["delta_ale_ci95"]
    print(f"Final ALE delta CI95: [{ci[0]:.4f}, {ci[1]:.4f}]", flush=True)
    print(f"Selected postprocess: {best_postprocess}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
