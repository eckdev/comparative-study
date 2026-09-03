"""Low-capacity, leakage-safe candidate ranking for Trichion and bilateral Gonion."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold

from all23_rgb_geodesic_cascade.anatomy import HARD3, NUM_LANDMARKS
from all23_rgb_geodesic_cascade.metrics import bootstrap_delta, summarize


@dataclass(frozen=True)
class Hard3StructuredConfig:
    folds: int = 5
    epochs: int = 70
    min_epochs: int = 20
    patience: int = 12
    batch_size: int = 12
    width: int = 32
    dropout: float = 0.25
    pair_topk: int = 24
    lr: float = 5e-4
    weight_decay: float = 5e-3
    grad_clip: float = 1.0
    sigma_lm0: float = 2.5
    sigma_gonion: float = 3.0
    coordinate_weight: float = 0.25
    negative_weight: float = 0.15
    negative_margin: float = 0.5
    feature_noise: float = 0.02
    maximum_step_lm0: float = 12.0
    maximum_step_gonion: float = 15.0
    bootstrap_iters: int = 2000
    minimum_overall_gain_mm: float = 0.03
    minimum_hard3_gain_mm: float = 0.15
    minimum_improvement_probability: float = 0.90
    maximum_p95_regression_mm: float = 0.10
    seed: int = 42


@dataclass
class Hard3CandidateSet:
    sample_ids: list[str]
    strata: list[str]
    features: np.ndarray
    canonical: np.ndarray
    points: np.ndarray
    mask: np.ndarray
    expert: np.ndarray
    target_distance: np.ndarray

    def __len__(self):
        return len(self.sample_ids)


def _unit(vector, fallback):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (np.asarray(vector, dtype=np.float32) / norm).astype(np.float32)


def _canonical_frame(coarse):
    coarse = np.asarray(coarse, dtype=np.float32)
    left = coarse[[13, 14, 17, 19]].mean(axis=0)
    right = coarse[[16, 15, 18, 20]].mean(axis=0)
    lateral = _unit(right - left, (1.0, 0.0, 0.0))
    upper = coarse[[1, 2]].mean(axis=0)
    lower = coarse[[10, 11, 12]].mean(axis=0)
    vertical = upper - lower
    vertical = vertical - lateral * float(np.dot(vertical, lateral))
    vertical = _unit(vertical, (0.0, 1.0, 0.0))
    depth = _unit(np.cross(lateral, vertical), (0.0, 0.0, 1.0))
    origin = coarse[[2, 5, 10]].mean(axis=0)
    if float(np.dot(coarse[3] - origin, depth)) < 0:
        depth *= -1.0
    frame = np.stack([lateral, vertical, depth], axis=1).astype(np.float32)
    scale_values = [
        np.linalg.norm(coarse[13] - coarse[16]),
        np.linalg.norm(coarse[17] - coarse[18]),
        np.linalg.norm(coarse[19] - coarse[20]),
        np.linalg.norm(upper - lower),
    ]
    valid = [value for value in scale_values if np.isfinite(value) and value > 10.0]
    scale = float(np.median(valid)) if valid else 80.0
    return origin, frame, max(scale, 20.0)


def _neighbor_mean(values, edge_index):
    values = np.asarray(values, dtype=np.float32)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    src, dst = edge_index
    total = np.zeros_like(values, dtype=np.float64)
    count = np.zeros((len(values), 1), dtype=np.float64)
    np.add.at(total, dst, values[src])
    np.add.at(count, dst, 1.0)
    mean = total / np.maximum(count, 1.0)
    missing = count[:, 0] == 0
    mean[missing] = values[missing]
    return mean.astype(np.float32)


def _morphology(coarse, scale):
    pairs = ((13, 16), (14, 15), (17, 18), (19, 20))
    widths = [
        np.linalg.norm(coarse[left] - coarse[right]) / scale for left, right in pairs
    ]
    heights = [
        np.linalg.norm(coarse[2] - coarse[5]) / scale,
        np.linalg.norm(coarse[5] - coarse[10]) / scale,
        np.linalg.norm(coarse[10] - coarse[12]) / scale,
    ]
    asymmetry = np.mean(
        [
            abs(
                np.linalg.norm(coarse[left] - coarse[2])
                - np.linalg.norm(coarse[right] - coarse[2])
            )
            / scale
            for left, right in pairs
        ]
    )
    return np.asarray(widths + heights + [asymmetry], dtype=np.float32)


def _anchor_indices(landmark):
    if landmark == 0:
        return (1, 2, 3, 4, 5, 12)
    if landmark == 21:
        return (10, 11, 12, 13, 17, 19)
    return (10, 11, 12, 16, 18, 20)


def _side_sign(landmark, coarse, origin, frame):
    if landmark == 0:
        return 1.0
    indices = (13, 17, 19) if landmark == 21 else (16, 18, 20)
    lateral = float(((coarse[list(indices)].mean(axis=0) - origin) @ frame)[0])
    return 1.0 if lateral >= 0.0 else -1.0


def _candidate_features(item, mean, std):
    points = item["points"].numpy().astype(np.float32)
    normalized = item["features"].numpy().astype(np.float32)
    raw = normalized * np.asarray(std, dtype=np.float32) + np.asarray(
        mean, dtype=np.float32
    )
    edge_index = item["edge_index"].numpy().astype(np.int64)
    coarse = item["coarse"].numpy().astype(np.float32)
    expert = item["expert"].numpy().astype(np.float32)[list(HARD3)]
    roi_index = item["roi_index"].numpy().astype(np.int64)[list(HARD3)]
    mask = item["roi_mask"].numpy().astype(bool)[list(HARD3)]

    rgb = np.clip(raw[:, 3:6], 0.0, 1.0)
    rgb_median = np.median(rgb, axis=0)
    rgb_scale = np.maximum(
        np.percentile(rgb, 75, axis=0) - np.percentile(rgb, 25, axis=0), 0.05
    )
    rgb_normalized = np.clip((rgb - rgb_median) / rgb_scale, -8.0, 8.0)
    neighbor_one = _neighbor_mean(rgb, edge_index)
    neighbor_two = _neighbor_mean(neighbor_one, edge_index)
    contrast_one = np.clip((rgb - neighbor_one) / rgb_scale, -8.0, 8.0)
    contrast_two = np.clip((rgb - neighbor_two) / rgb_scale, -8.0, 8.0)
    point_laplacian = points - _neighbor_mean(points, edge_index)

    origin, frame, face_scale = _canonical_frame(coarse)
    morphology = _morphology(coarse, face_scale)
    candidate_rows, canonical_rows = [], []
    for local_index, landmark in enumerate(HARD3):
        indices = roi_index[local_index]
        candidate = points[indices]
        side = _side_sign(landmark, coarse, origin, frame)
        relative_center = ((candidate - coarse[landmark]) @ frame) / face_scale
        canonical = ((candidate - origin) @ frame) / face_scale
        normal = raw[indices, 9:12] @ frame
        laplacian = (point_laplacian[indices] @ frame) / np.maximum(
            raw[indices, 12:13], 1e-3
        )
        if landmark in (21, 22):
            relative_center[:, 0] *= side
            canonical[:, 0] *= side
            normal[:, 0] *= side
            laplacian[:, 0] *= side

        anchor_features = []
        for anchor_index in _anchor_indices(landmark):
            vector = ((candidate - coarse[anchor_index]) @ frame) / face_scale
            if landmark in (21, 22):
                vector[:, 0] *= side
            anchor_features.extend(
                [vector, np.linalg.norm(vector, axis=1, keepdims=True)]
            )

        candidate_rgb = rgb[indices]
        chromaticity = candidate_rgb / np.maximum(
            candidate_rgb.sum(axis=1, keepdims=True), 1e-3
        )
        intensity = candidate_rgb.mean(axis=1, keepdims=True)
        saturation = candidate_rgb.max(axis=1, keepdims=True) - candidate_rgb.min(
            axis=1, keepdims=True
        )
        c1 = contrast_one[indices]
        c2 = contrast_two[indices]
        color = np.concatenate(
            [
                rgb_normalized[indices],
                chromaticity,
                intensity,
                saturation,
                c1,
                np.linalg.norm(c1, axis=1, keepdims=True),
                c2,
                np.linalg.norm(c2, axis=1, keepdims=True),
            ],
            axis=1,
        )
        type_encoding = np.zeros((len(indices), 3), dtype=np.float32)
        type_encoding[:, local_index] = 1.0
        has_rgb = np.full(
            (len(indices), 1), float(np.std(rgb) > 1e-4), dtype=np.float32
        )
        row = np.concatenate(
            [
                relative_center,
                np.linalg.norm(relative_center, axis=1, keepdims=True),
                canonical,
                normal,
                laplacian,
                raw[indices, 12:14],
                color,
                *anchor_features,
                np.broadcast_to(morphology, (len(indices), len(morphology))),
                type_encoding,
                has_rgb,
            ],
            axis=1,
        ).astype(np.float32)
        row[~np.isfinite(row)] = 0.0
        candidate_rows.append(row)
        canonical_rows.append(canonical.astype(np.float32))

    candidate_points = points[roi_index]
    target_distance = np.linalg.norm(candidate_points - expert[:, None, :], axis=-1)
    target_distance[~mask] = np.inf
    return (
        np.stack(candidate_rows),
        np.stack(canonical_rows),
        candidate_points.astype(np.float32),
        mask,
        expert,
        target_distance.astype(np.float32),
    )


def extract_candidate_set(dataset, progress_label="Hard3 descriptors"):
    previous_training = dataset.training
    dataset.training = False
    features, canonical, points, masks, experts, distances = [], [], [], [], [], []
    sample_ids, strata = [], []
    try:
        for index in range(len(dataset)):
            item = dataset[index]
            rows = _candidate_features(item, dataset.mean, dataset.std)
            features.append(rows[0])
            canonical.append(rows[1])
            points.append(rows[2])
            masks.append(rows[3])
            experts.append(rows[4])
            distances.append(rows[5])
            sample_ids.append(item["sample_id"])
            strata.append(f"{item['class']}|{item['gender']}")
            if (index + 1) % 20 == 0 or index + 1 == len(dataset):
                print(f"{progress_label} {index + 1}/{len(dataset)}", flush=True)
    finally:
        dataset.training = previous_training
    return Hard3CandidateSet(
        sample_ids=sample_ids,
        strata=strata,
        features=np.stack(features),
        canonical=np.stack(canonical),
        points=np.stack(points),
        mask=np.stack(masks),
        expert=np.stack(experts),
        target_distance=np.stack(distances),
    )


class StructuredCandidateRanker(nn.Module):
    def __init__(self, feature_dim, width=32, dropout=0.25, pair_topk=24):
        super().__init__()
        self.pair_topk = max(2, int(pair_topk))
        self.trichion = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        self.gonion_encoder = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gonion_unary = nn.Linear(width, 1)
        pair_width = max(width // 2, 8)
        self.left_pair = nn.Linear(width, pair_width, bias=False)
        self.right_pair = nn.Linear(width, pair_width, bias=False)
        self.pair_geometry = nn.Sequential(
            nn.Linear(9, pair_width), nn.GELU(), nn.Linear(pair_width, pair_width)
        )
        self.pair_score = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout), nn.Linear(pair_width, 1)
        )

    @staticmethod
    def _gather(values, indices):
        return torch.gather(
            values,
            1,
            indices[..., None].expand(-1, -1, values.shape[-1]),
        )

    @staticmethod
    def _pair_features(left, right):
        difference = torch.abs(left - right)
        mean = 0.5 * (left + right)
        product = left * right
        return torch.cat([difference, mean, product], dim=-1)

    def forward(self, features, canonical, mask):
        trichion = self.trichion(features[:, 0]).squeeze(-1)
        gonion_features = features[:, 1:3]
        embedding = self.gonion_encoder(gonion_features)
        unary = self.gonion_unary(embedding).squeeze(-1)
        left_unary = unary[:, 0].masked_fill(~mask[:, 1], -torch.inf)
        right_unary = unary[:, 1].masked_fill(~mask[:, 2], -torch.inf)
        use_topk = min(self.pair_topk, unary.shape[-1])
        left_top = torch.topk(left_unary, use_topk, dim=-1).indices
        right_top = torch.topk(right_unary, use_topk, dim=-1).indices

        left_embedding, right_embedding = embedding[:, 0], embedding[:, 1]
        left_canonical, right_canonical = canonical[:, 1], canonical[:, 2]
        selected_left_embedding = self._gather(left_embedding, left_top)
        selected_right_embedding = self._gather(right_embedding, right_top)
        selected_left_canonical = self._gather(left_canonical, left_top)
        selected_right_canonical = self._gather(right_canonical, right_top)
        selected_left_unary = torch.gather(left_unary, 1, left_top)
        selected_right_unary = torch.gather(right_unary, 1, right_top)
        selected_left_mask = torch.gather(mask[:, 1], 1, left_top)
        selected_right_mask = torch.gather(mask[:, 2], 1, right_top)

        left_hidden = self.left_pair(left_embedding)[:, :, None, :]
        right_hidden = self.right_pair(selected_right_embedding)[:, None, :, :]
        geometry = self._pair_features(
            left_canonical[:, :, None, :], selected_right_canonical[:, None, :, :]
        )
        left_joint = (
            left_unary[:, :, None]
            + selected_right_unary[:, None, :]
            + self.pair_score(
                left_hidden + right_hidden + self.pair_geometry(geometry)
            ).squeeze(-1)
        )
        left_pair_mask = mask[:, 1, :, None] & selected_right_mask[:, None, :]
        left_joint = left_joint.masked_fill(~left_pair_mask, -torch.inf)
        left = torch.logsumexp(left_joint.float(), dim=-1) - torch.log(
            selected_right_mask.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
        )

        left_hidden = self.left_pair(selected_left_embedding)[:, :, None, :]
        right_hidden = self.right_pair(right_embedding)[:, None, :, :]
        geometry = self._pair_features(
            selected_left_canonical[:, :, None, :], right_canonical[:, None, :, :]
        )
        right_joint = (
            selected_left_unary[:, :, None]
            + right_unary[:, None, :]
            + self.pair_score(
                left_hidden + right_hidden + self.pair_geometry(geometry)
            ).squeeze(-1)
        )
        right_pair_mask = selected_left_mask[:, :, None] & mask[:, 2, None, :]
        right_joint = right_joint.masked_fill(~right_pair_mask, -torch.inf)
        right = torch.logsumexp(right_joint.float(), dim=1) - torch.log(
            selected_left_mask.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
        )
        return torch.stack(
            [trichion, left.to(trichion.dtype), right.to(trichion.dtype)], dim=1
        )


def _fit_scaler(candidate_set, indices, seed):
    values = candidate_set.features[indices]
    valid = candidate_set.mask[indices]
    values = values[valid]
    if len(values) > 250_000:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(len(values), 250_000, replace=False)]
    mean = values.mean(axis=0).astype(np.float32)
    std = np.maximum(values.std(axis=0), 1e-4).astype(np.float32)
    return mean, std


def _tensor_batch(candidate_set, indices, feature_mean, feature_std, device, noise=0.0):
    features = (candidate_set.features[indices] - feature_mean) / feature_std
    features = np.clip(features, -8.0, 8.0)
    result = {
        "features": torch.from_numpy(features).to(device),
        "canonical": torch.from_numpy(candidate_set.canonical[indices]).to(device),
        "points": torch.from_numpy(candidate_set.points[indices]).to(device),
        "mask": torch.from_numpy(candidate_set.mask[indices]).to(device),
        "expert": torch.from_numpy(candidate_set.expert[indices]).to(device),
        "distance": torch.from_numpy(candidate_set.target_distance[indices]).to(device),
    }
    if noise > 0:
        result["features"] = (
            result["features"] + torch.randn_like(result["features"]) * noise
        )
    return result


def _weighted_coordinate(logits, points, mask, topk=10, temperature=0.5):
    work = logits.float().masked_fill(~mask, -torch.inf)
    count = min(max(int(topk), 1), logits.shape[-1])
    values, indices = torch.topk(work, count, dim=-1)
    selected_mask = torch.gather(mask, -1, indices)
    values = values.masked_fill(~selected_mask, -torch.inf)
    weights = torch.softmax(values / max(float(temperature), 1e-4), dim=-1)
    selected = torch.gather(points, 2, indices[..., None].expand(-1, -1, -1, 3))
    return torch.sum(weights[..., None] * selected, dim=2)


def _ranking_loss(logits, batch, config):
    mask = batch["mask"]
    distance = batch["distance"].float()
    sigmas = distance.new_tensor(
        [config.sigma_lm0, config.sigma_gonion, config.sigma_gonion]
    )
    energy = -(distance**2) / (2.0 * sigmas[None, :, None] ** 2)
    energy = energy.masked_fill(~mask, -torch.inf)
    target = torch.softmax(energy, dim=-1)
    log_probability = torch.log_softmax(
        logits.float().masked_fill(~mask, -torch.inf), dim=-1
    )
    listwise = (
        torch.where(
            mask,
            target * (torch.log(target.clamp_min(1e-8)) - log_probability),
            torch.zeros_like(target),
        )
        .sum(dim=-1)
        .mean()
    )

    prediction = _weighted_coordinate(logits, batch["points"], mask)
    coordinate = F.smooth_l1_loss(prediction, batch["expert"].float(), beta=1.0)
    minimum = distance.amin(dim=-1, keepdim=True)
    negative_mask = mask & (distance > minimum + sigmas[None, :, None])
    negative_logits = logits.float().masked_fill(~negative_mask, -torch.inf)
    count = min(16, logits.shape[-1])
    negative, negative_indices = torch.topk(negative_logits, count, dim=-1)
    negative_valid = torch.gather(negative_mask, -1, negative_indices)
    positive = (target * logits.float().masked_fill(~mask, 0.0)).sum(
        dim=-1, keepdim=True
    )
    margin = F.softplus(negative - positive + config.negative_margin)
    margin = torch.where(negative_valid, margin, torch.zeros_like(margin))
    margin = (margin.sum(dim=-1) / negative_valid.sum(dim=-1).clamp_min(1)).mean()
    total = (
        listwise
        + config.coordinate_weight * coordinate
        + config.negative_weight * margin
    )
    return total, {"listwise": listwise, "coordinate": coordinate, "negative": margin}


@torch.no_grad()
def _predict_logits(
    model, candidate_set, indices, feature_mean, feature_std, config, device
):
    model.eval()
    chunks = []
    for start in range(0, len(indices), config.batch_size):
        selected = np.asarray(
            indices[start : start + config.batch_size], dtype=np.int64
        )
        batch = _tensor_batch(
            candidate_set, selected, feature_mean, feature_std, device
        )
        chunks.append(
            model(batch["features"], batch["canonical"], batch["mask"]).cpu().numpy()
        )
    return np.concatenate(chunks, axis=0)


def _decode_numpy(logits, points, mask, topk, temperature, snap):
    logits = np.asarray(logits, dtype=np.float64)
    masked = np.where(mask, logits, -np.inf)
    count = min(max(int(topk), 1), logits.shape[-1])
    indices = np.argpartition(masked, -count, axis=-1)[..., -count:]
    values = np.take_along_axis(masked, indices, axis=-1)
    selected_mask = np.take_along_axis(mask, indices, axis=-1)
    values = np.where(selected_mask, values, -np.inf)
    values = values / max(float(temperature), 1e-4)
    values -= np.max(values, axis=-1, keepdims=True)
    weights = np.exp(values) * selected_mask
    weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)
    selected = np.take_along_axis(points, indices[..., None], axis=2)
    coordinate = np.sum(weights[..., None] * selected, axis=2)
    if snap:
        distance = np.linalg.norm(points - coordinate[:, :, None, :], axis=-1)
        nearest = np.argmin(np.where(mask, distance, np.inf), axis=-1)
        coordinate = np.take_along_axis(
            points, nearest[..., None, None], axis=2
        ).squeeze(2)
    return coordinate.astype(np.float32)


def _select_coordinate_policy(candidate_set, logits):
    options = []
    for snap in (False, True):
        for topk in (1, 3, 5, 10, 20, 30):
            for temperature in (0.5,) if topk == 1 else (0.25, 0.5, 0.75, 1.0):
                prediction = _decode_numpy(
                    logits,
                    candidate_set.points,
                    candidate_set.mask,
                    topk,
                    temperature,
                    snap,
                )
                error = np.linalg.norm(prediction - candidate_set.expert, axis=-1)
                options.append(
                    {
                        "topk": topk,
                        "temperature": temperature,
                        "snap": snap,
                        "lm0_ale": float(error[:, 0].mean()),
                        "gonion_ale": float(error[:, 1:3].mean()),
                        "hard3_ale": float(error.mean()),
                    }
                )
    lm0 = min(options, key=lambda row: (row["lm0_ale"], row["hard3_ale"]))
    gonion = min(options, key=lambda row: (row["gonion_ale"], row["hard3_ale"]))
    return {"lm0": lm0, "gonion": gonion, "sweep": options}


def _decode_policy(candidate_set, logits, policy):
    lm0 = _decode_numpy(
        logits[:, 0:1],
        candidate_set.points[:, 0:1],
        candidate_set.mask[:, 0:1],
        policy["lm0"]["topk"],
        policy["lm0"]["temperature"],
        policy["lm0"]["snap"],
    )
    gonion = _decode_numpy(
        logits[:, 1:3],
        candidate_set.points[:, 1:3],
        candidate_set.mask[:, 1:3],
        policy["gonion"]["topk"],
        policy["gonion"]["temperature"],
        policy["gonion"]["snap"],
    )
    return np.concatenate([lm0, gonion], axis=1)


def _splitter(strata, folds, seed):
    strata = np.asarray(strata)
    unique, counts = np.unique(strata, return_counts=True)
    folds = min(max(2, int(folds)), len(strata))
    if len(unique) > 1 and int(counts.min()) >= folds:
        return StratifiedKFold(folds, shuffle=True, random_state=seed).split(
            np.zeros(len(strata)), strata
        )
    return KFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(strata)))


def _train_model(
    candidate_set, train_indices, val_indices, config, device, fold_number
):
    feature_mean, feature_std = _fit_scaler(
        candidate_set, train_indices, config.seed + fold_number
    )
    torch.manual_seed(config.seed + fold_number * 1009)
    model = StructuredCandidateRanker(
        candidate_set.features.shape[-1], config.width, config.dropout, config.pair_topk
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )
    rng = np.random.default_rng(config.seed + fold_number * 7919)
    best_score, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(train_indices)
        totals = {"total": 0.0, "listwise": 0.0, "coordinate": 0.0, "negative": 0.0}
        seen = 0
        for start in range(0, len(order), config.batch_size):
            selected = np.asarray(
                order[start : start + config.batch_size], dtype=np.int64
            )
            batch = _tensor_batch(
                candidate_set,
                selected,
                feature_mean,
                feature_std,
                device,
                config.feature_noise,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["features"], batch["canonical"], batch["mask"])
            loss, components = _ranking_loss(logits, batch, config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            count = len(selected)
            seen += count
            totals["total"] += float(loss.detach()) * count
            for name, value in components.items():
                totals[name] += float(value.detach()) * count
        val_logits = _predict_logits(
            model,
            candidate_set,
            list(val_indices),
            feature_mean,
            feature_std,
            config,
            device,
        )
        val_prediction = _decode_numpy(
            val_logits,
            candidate_set.points[val_indices],
            candidate_set.mask[val_indices],
            10,
            0.5,
            False,
        )
        val_error = np.linalg.norm(
            val_prediction - candidate_set.expert[val_indices], axis=-1
        )
        score = float(val_error.mean())
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "validation_hard3_ale": score,
            "validation_lm0_ale": float(val_error[:, 0].mean()),
            "validation_gonion_ale": float(val_error[:, 1:3].mean()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{name}": value / max(seen, 1) for name, value in totals.items()},
        }
        history.append(row)
        if score < best_score - 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        elif epoch >= config.min_epochs:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Hard3 OOF fold {fold_number} epoch {epoch:03d}/{config.epochs} "
                f"train={row['train_total']:.4f} val={score:.4f}",
                flush=True,
            )
        if epoch >= config.min_epochs and stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Hard3 structured ranker did not produce a checkpoint")
    model.load_state_dict(best_state)
    logits = _predict_logits(
        model,
        candidate_set,
        list(val_indices),
        feature_mean,
        feature_std,
        config,
        device,
    )
    return (
        logits,
        best_epoch,
        best_score,
        history,
        {key: value.detach().cpu() for key, value in best_state.items()},
        feature_mean,
        feature_std,
    )


class FittedHard3StructuredRefiner:
    def __init__(
        self,
        members,
        coordinate_policy,
        reliability_scales,
        report,
        config,
        device,
    ):
        self.members = []
        for model, feature_mean, feature_std in members:
            self.members.append(
                (
                    model.to(device).eval(),
                    np.asarray(feature_mean, dtype=np.float32),
                    np.asarray(feature_std, dtype=np.float32),
                )
            )
        self.coordinate_policy = coordinate_policy
        self.reliability_scales = np.asarray(reliability_scales, dtype=np.float32)
        self.report = report
        self.config = config
        self.device = device

    def predict(self, dataset, label="Hard3 inference"):
        candidates = extract_candidate_set(dataset, label)
        indices = list(range(len(candidates)))
        member_logits = [
            _predict_logits(
                model,
                candidates,
                indices,
                feature_mean,
                feature_std,
                self.config,
                self.device,
            )
            for model, feature_mean, feature_std in self.members
        ]
        logits = np.mean(np.stack(member_logits), axis=0)
        prediction = _decode_policy(candidates, logits, self.coordinate_policy)
        member_prediction = np.stack(
            [
                _decode_policy(candidates, values, self.coordinate_policy)
                for values in member_logits
            ]
        )
        ensemble_spread = np.linalg.norm(
            member_prediction - prediction[None], axis=-1
        ).mean(axis=0)
        reliability = 1.0 / (
            1.0 + (ensemble_spread / self.reliability_scales[None]) ** 2
        )
        probability = (
            np.exp(logits - np.max(logits, axis=-1, keepdims=True)) * candidates.mask
        )
        probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
        entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=-1)
        entropy /= np.log(np.maximum(candidates.mask.sum(axis=-1), 2))
        return {
            "sample_ids": candidates.sample_ids,
            "prediction": prediction,
            "expert": candidates.expert,
            "entropy": entropy.astype(np.float32),
            "ensemble_spread": ensemble_spread.astype(np.float32),
            "reliability": reliability.astype(np.float32),
            "oracle_error": np.min(candidates.target_distance, axis=-1),
        }


def _cache_signature(dataset, config):
    model_config = asdict(config)
    # These fields affect reporting/acceptance only, not the fitted ranker.
    for key in (
        "bootstrap_iters",
        "minimum_overall_gain_mm",
        "minimum_hard3_gain_mm",
        "minimum_improvement_probability",
        "maximum_p95_regression_mm",
    ):
        model_config.pop(key, None)
    sample_ids = [sample.sample_id for sample in dataset.samples]
    coarse_digest = hashlib.sha256()
    record_fingerprints = []
    for sample in dataset.samples:
        sample_id = sample.sample_id
        coarse_digest.update(sample_id.encode("utf-8"))
        coarse_digest.update(
            np.asarray(dataset._coarse(sample), dtype=np.float32).tobytes()
        )
        record_path = Path(dataset.records[sample_id])
        stat = record_path.stat()
        record_fingerprints.append(
            {
                "sample_id": sample_id,
                "name": record_path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = {
        "version": 3,
        "sample_ids": sample_ids,
        "records": record_fingerprints,
        "coarse_digest": coarse_digest.hexdigest(),
        "normalizer_mean": np.asarray(dataset.mean, dtype=np.float32).tolist(),
        "normalizer_std": np.asarray(dataset.std, dtype=np.float32).tolist(),
        "roi": {
            "points": int(dataset.roi_points),
            "radius_scale": float(dataset.roi_radius_scale),
            "mode": str(dataset.roi_mode),
            "euclidean_scale": float(dataset.roi_euclidean_scale),
            "multi_seeds": int(dataset.roi_multi_seeds),
        },
        "config": model_config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def fit_or_load_hard3_refiner(dataset, output_dir, config, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "hard3_structured_model.pth"
    report_path = output_dir / "hard3_training_report.json"
    signature = _cache_signature(dataset, config)
    if checkpoint_path.exists() and report_path.exists():
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("signature") == signature:
            members = []
            for saved in checkpoint["members"]:
                model = StructuredCandidateRanker(
                    checkpoint["feature_dim"],
                    config.width,
                    config.dropout,
                    config.pair_topk,
                ).to(device)
                model.load_state_dict(saved["model_state"])
                members.append((model, saved["feature_mean"], saved["feature_std"]))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print("Hard3 structured ranker cached", flush=True)
            return FittedHard3StructuredRefiner(
                members,
                checkpoint["coordinate_policy"],
                checkpoint["reliability_scales"],
                report,
                config,
                device,
            )

    train_candidates = extract_candidate_set(dataset, "Hard3 train descriptors")
    started = time.time()
    oof_logits = np.full(
        (len(train_candidates), 3, train_candidates.features.shape[-2]),
        -np.inf,
        dtype=np.float32,
    )
    fold_reports = []
    best_epochs = []
    saved_members = []
    for fold_number, (train_indices, val_indices) in enumerate(
        _splitter(train_candidates.strata, config.folds, config.seed), start=1
    ):
        (
            logits,
            best_epoch,
            best_score,
            history,
            model_state,
            feature_mean,
            feature_std,
        ) = _train_model(
            train_candidates,
            np.asarray(train_indices),
            np.asarray(val_indices),
            config,
            device,
            fold_number,
        )
        oof_logits[val_indices] = logits
        best_epochs.append(best_epoch)
        saved_members.append(
            {
                "model_state": model_state,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
            }
        )
        fold_reports.append(
            {
                "fold": fold_number,
                "train_sample_ids": [
                    train_candidates.sample_ids[i] for i in train_indices
                ],
                "validation_sample_ids": [
                    train_candidates.sample_ids[i] for i in val_indices
                ],
                "best_epoch": best_epoch,
                "best_validation_hard3_ale": best_score,
                "history": history,
            }
        )
    if not np.isfinite(oof_logits[train_candidates.mask]).all():
        raise RuntimeError("Hard3 OOF logits are incomplete")
    coordinate_policy = _select_coordinate_policy(train_candidates, oof_logits)
    oof_prediction = _decode_policy(train_candidates, oof_logits, coordinate_policy)
    oof_error = np.linalg.norm(oof_prediction - train_candidates.expert, axis=-1)
    members = []
    for saved in saved_members:
        model = StructuredCandidateRanker(
            train_candidates.features.shape[-1],
            config.width,
            config.dropout,
            config.pair_topk,
        ).to(device)
        model.load_state_dict(saved["model_state"])
        members.append((model, saved["feature_mean"], saved["feature_std"]))
    train_member_logits = [
        _predict_logits(
            model,
            train_candidates,
            list(range(len(train_candidates))),
            feature_mean,
            feature_std,
            config,
            device,
        )
        for model, feature_mean, feature_std in members
    ]
    train_ensemble_prediction = _decode_policy(
        train_candidates,
        np.mean(np.stack(train_member_logits), axis=0),
        coordinate_policy,
    )
    train_member_prediction = np.stack(
        [
            _decode_policy(train_candidates, logits, coordinate_policy)
            for logits in train_member_logits
        ]
    )
    train_spread = np.linalg.norm(
        train_member_prediction - train_ensemble_prediction[None], axis=-1
    ).mean(axis=0)
    reliability_scales = np.asarray(
        [
            max(float(np.percentile(train_spread[:, 0], 75)), 0.25),
            max(float(np.percentile(train_spread[:, 1:3], 75)), 0.25),
            max(float(np.percentile(train_spread[:, 1:3], 75)), 0.25),
        ],
        dtype=np.float32,
    )
    report = {
        "signature": signature,
        "method": "nested-OOF low-capacity Trichion texture and bilateral Gonion ranker",
        "uses_validation_labels_for_model_fit": False,
        "uses_test_labels": False,
        "sample_count": len(train_candidates),
        "feature_dim": int(train_candidates.features.shape[-1]),
        "parameter_count_per_member": sum(
            parameter.numel() for parameter in members[0][0].parameters()
        ),
        "ensemble_members": len(members),
        "folds": fold_reports,
        "best_epochs": best_epochs,
        "inference": "mean logits from nested-OOF inner-fold models",
        "coordinate_policy": coordinate_policy,
        "reliability": {
            "method": "inverse-quadratic inner-fold ensemble disagreement",
            "scale_mm": reliability_scales.tolist(),
            "train_spread": {
                "lm0": summarize(train_spread[:, 0]),
                "gonion": summarize(train_spread[:, 1:3]),
            },
        },
        "oof": {
            "hard3": summarize(oof_error),
            "lm0": summarize(oof_error[:, 0]),
            "gonion": summarize(oof_error[:, 1:3]),
            "per_landmark_ale": oof_error.mean(axis=0).tolist(),
            "candidate_oracle_ale": float(
                np.min(train_candidates.target_distance, axis=-1).mean()
            ),
        },
        "training_seconds": float(time.time() - started),
        "config": asdict(config),
    }
    torch.save(
        {
            "signature": signature,
            "feature_dim": int(train_candidates.features.shape[-1]),
            "coordinate_policy": coordinate_policy,
            "reliability_scales": reliability_scales,
            "members": saved_members,
        },
        checkpoint_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return FittedHard3StructuredRefiner(
        members,
        coordinate_policy,
        reliability_scales,
        report,
        config,
        device,
    )


def _ordered_result_field(outputs, candidate_result, name, default=None):
    if name not in candidate_result:
        if default is None:
            raise KeyError(f"Hard3 candidate result has no '{name}' field")
        return np.asarray(default)
    by_id = {
        sample_id: candidate_result[name][index]
        for index, sample_id in enumerate(candidate_result["sample_ids"])
    }
    missing = [
        sample_id for sample_id in outputs["sample_ids"] if sample_id not in by_id
    ]
    if missing:
        raise KeyError(f"Hard3 '{name}' values miss samples: {missing[:5]}")
    return np.stack([by_id[sample_id] for sample_id in outputs["sample_ids"]])


def _limited_hard3_candidate(base, raw_candidate, limits):
    base_hard3 = np.asarray(base, dtype=np.float32)[:, list(HARD3)]
    raw_candidate = np.asarray(raw_candidate, dtype=np.float32)
    limits = np.asarray(limits, dtype=np.float32)[None, :]
    delta = raw_candidate - base_hard3
    length = np.linalg.norm(delta, axis=-1)
    scale = np.minimum(1.0, limits / np.maximum(length, 1e-6))
    return base_hard3 + scale[..., None] * delta, length, scale


def _blend_prediction(base, candidate, reliability, row):
    confidence = (
        np.asarray(reliability, dtype=np.float32)
        if row["confidence_mode"] == "ensemble"
        else np.ones_like(reliability, dtype=np.float32)
    )
    alpha = np.asarray(
        [row["alpha_lm0"], row["alpha_gonion"], row["alpha_gonion"]],
        dtype=np.float32,
    )
    effective_alpha = confidence * alpha[None]
    prediction = np.asarray(base, dtype=np.float32).copy()
    base_hard3 = prediction[:, list(HARD3)].copy()
    prediction[:, list(HARD3)] = base_hard3 + effective_alpha[..., None] * (
        candidate - base_hard3
    )
    return prediction, effective_alpha


def calibrate_hard3_blend(outputs, candidate_result, config):
    raw_candidate = _ordered_result_field(outputs, candidate_result, "prediction")
    base = np.asarray(outputs["prediction"], dtype=np.float32)
    expert = np.asarray(outputs["expert"], dtype=np.float32)
    reliability = _ordered_result_field(
        outputs,
        candidate_result,
        "reliability",
        default=np.ones((len(base), 3), dtype=np.float32),
    )
    reliability = np.clip(np.asarray(reliability, dtype=np.float32), 0.05, 1.0)
    limits = (
        config.maximum_step_lm0,
        config.maximum_step_gonion,
        config.maximum_step_gonion,
    )
    candidate, raw_step, step_scale = _limited_hard3_candidate(
        base, raw_candidate, limits
    )
    base_error = np.linalg.norm(base - expert, axis=-1)
    base_p95 = float(np.percentile(base_error, 95))
    rows = []
    for confidence_mode in ("none", "ensemble"):
        for alpha_lm0 in (0.0, 0.25, 0.5, 0.75, 1.0):
            for alpha_gonion in (0.0, 0.25, 0.5, 0.75, 1.0):
                row = {
                    "confidence_mode": confidence_mode,
                    "alpha_lm0": alpha_lm0,
                    "alpha_gonion": alpha_gonion,
                }
                prediction, effective_alpha = _blend_prediction(
                    base, candidate, reliability, row
                )
                error = np.linalg.norm(prediction - expert, axis=-1)
                rows.append(
                    {
                        **row,
                        "overall_ale": float(error.mean()),
                        "hard3_ale": float(error[:, list(HARD3)].mean()),
                        "lm0_ale": float(error[:, 0].mean()),
                        "gonion_ale": float(error[:, 21:23].mean()),
                        "p95": float(np.percentile(error, 95)),
                        "mean_effective_alpha_lm0": float(effective_alpha[:, 0].mean()),
                        "mean_effective_alpha_gonion": float(
                            effective_alpha[:, 1:3].mean()
                        ),
                    }
                )
    eligible = [
        row for row in rows if row["p95"] <= base_p95 + config.maximum_p95_regression_mm
    ]
    proposed = min(
        eligible or rows,
        key=lambda row: (row["hard3_ale"], row["overall_ale"], row["p95"]),
    )
    proposed_prediction, _ = _blend_prediction(base, candidate, reliability, proposed)
    proposed_error = np.linalg.norm(proposed_prediction - expert, axis=-1)
    proposed_bootstrap = bootstrap_delta(
        base_error,
        proposed_error,
        config.bootstrap_iters,
        config.seed,
    )
    overall_gain = float(base_error.mean() - proposed["overall_ale"])
    hard3_gain = float(base_error[:, list(HARD3)].mean() - proposed["hard3_ale"])
    accepted = (
        overall_gain >= config.minimum_overall_gain_mm
        and hard3_gain >= config.minimum_hard3_gain_mm
        and proposed_bootstrap["probability_improved"]
        >= config.minimum_improvement_probability
        and proposed["p95"] <= base_p95 + config.maximum_p95_regression_mm
    )
    selected = proposed
    if not accepted:
        selected = next(
            row
            for row in rows
            if row["confidence_mode"] == "none"
            and row["alpha_lm0"] == 0.0
            and row["alpha_gonion"] == 0.0
        )
        overall_gain = 0.0
        hard3_gain = 0.0
    blended, effective_alpha = _blend_prediction(base, candidate, reliability, selected)
    blended_error = np.linalg.norm(blended - expert, axis=-1)
    bootstrap = bootstrap_delta(
        base_error,
        blended_error,
        config.bootstrap_iters,
        config.seed,
    )
    return {
        "accepted": accepted,
        "proposed": proposed,
        "selected": selected,
        "limits_mm": list(limits),
        "acceptance_thresholds": {
            "minimum_overall_gain_mm": config.minimum_overall_gain_mm,
            "minimum_hard3_gain_mm": config.minimum_hard3_gain_mm,
            "minimum_improvement_probability": config.minimum_improvement_probability,
            "maximum_p95_regression_mm": config.maximum_p95_regression_mm,
        },
        "base_overall": summarize(base_error),
        "base_hard3": summarize(base_error[:, list(HARD3)]),
        "raw_candidate_hard3": summarize(
            np.linalg.norm(raw_candidate - expert[:, list(HARD3)], axis=-1)
        ),
        "limited_candidate_hard3": summarize(
            np.linalg.norm(candidate - expert[:, list(HARD3)], axis=-1)
        ),
        "blended_overall": summarize(blended_error),
        "blended_hard3": summarize(blended_error[:, list(HARD3)]),
        "mean_reliability": reliability.mean(axis=0).tolist(),
        "mean_effective_alpha": effective_alpha.mean(axis=0).tolist(),
        "step_limit_fraction": np.mean(step_scale < 1.0, axis=0).tolist(),
        "raw_step_mm": {
            "lm0": summarize(raw_step[:, 0]),
            "gonion": summarize(raw_step[:, 1:3]),
        },
        "overall_gain_mm": overall_gain,
        "hard3_gain_mm": hard3_gain,
        "bootstrap_vs_base": bootstrap,
        "proposed_bootstrap_vs_base": proposed_bootstrap,
        "sweep": rows,
        "uses_validation_labels_for_blend_only": True,
        "uses_test_labels": False,
    }


def apply_hard3_blend(outputs, candidate_result, policy):
    result = dict(outputs)
    base = np.asarray(outputs["prediction"], dtype=np.float32)
    raw_candidate = _ordered_result_field(outputs, candidate_result, "prediction")
    reliability = _ordered_result_field(
        outputs,
        candidate_result,
        "reliability",
        default=np.ones((len(base), 3), dtype=np.float32),
    )
    reliability = np.clip(np.asarray(reliability, dtype=np.float32), 0.05, 1.0)
    candidate, _, _ = _limited_hard3_candidate(base, raw_candidate, policy["limits_mm"])
    selected = policy["selected"]
    prediction, effective_alpha = _blend_prediction(
        base, candidate, reliability, selected
    )
    full_candidate = base.copy()
    full_candidate[:, list(HARD3)] = candidate
    full_reliability = np.full((len(base), NUM_LANDMARKS), np.nan, dtype=np.float32)
    full_spread = np.full((len(base), NUM_LANDMARKS), np.nan, dtype=np.float32)
    full_effective_alpha = np.zeros((len(base), NUM_LANDMARKS), dtype=np.float32)
    full_reliability[:, list(HARD3)] = reliability
    full_effective_alpha[:, list(HARD3)] = effective_alpha
    if "ensemble_spread" in candidate_result:
        full_spread[:, list(HARD3)] = _ordered_result_field(
            outputs, candidate_result, "ensemble_spread"
        )
    result["pre_hard3_prediction"] = base.copy()
    result["hard3_candidate"] = full_candidate
    result["hard3_reliability"] = full_reliability
    result["hard3_ensemble_spread"] = full_spread
    result["hard3_effective_alpha"] = full_effective_alpha
    result["prediction"] = prediction
    result["errors"] = np.linalg.norm(prediction - result["expert"], axis=-1)
    return result
