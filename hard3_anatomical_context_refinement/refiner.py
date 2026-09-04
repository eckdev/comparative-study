"""Leakage-safe dual-view refinement for Trichion and bilateral Gonion."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agh_former_vnext_orthodontic_comparison.hard3_structured import (
    _blend_prediction,
    _decode_numpy,
    _decode_policy,
    _limited_hard3_candidate,
    _select_coordinate_policy,
    _splitter,
)
from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, NUM_LANDMARKS
from all23_rgb_geodesic_cascade.metrics import bootstrap_delta, summarize

from .atlas import TrainOnlyLocalHard3Atlas
from .model import DualViewHard3Net
from .patches import DualViewCandidateSet, extract_dual_view_set


@dataclass(frozen=True)
class Hard3DualViewConfig:
    folds: int = 5
    epochs: int = 90
    min_epochs: int = 30
    patience: int = 15
    batch_size: int = 8
    image_size: int = 64
    width: int = 24
    dropout: float = 0.10
    radius_scale: float = 1.0
    translation_pixels: int = 4
    color_noise: float = 0.025
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    sigma_lm0: float = 3.0
    sigma_gonion: float = 4.0
    heatmap_weight: float = 1.0
    poss_weight: float = 0.25
    poss_exponent: float = 2.0
    poss_temperature: float = 0.1
    ranking_weight: float = 0.5
    coordinate_weight: float = 0.25
    pair_weight: float = 0.10
    negative_weight: float = 0.15
    negative_margin: float = 0.5
    atlas_neighbors: int = 8
    atlas_temperature: float = 2.0
    final_ensemble_members: int = 3
    maximum_step_lm0: float = 12.0
    maximum_step_gonion: float = 15.0
    bootstrap_iters: int = 2000
    minimum_overall_gain_mm: float = 0.03
    minimum_hard3_gain_mm: float = 0.20
    minimum_improvement_probability: float = 0.90
    maximum_p95_regression_mm: float = 0.10
    target_hard3_ale: float = 4.0
    seed: int = 42


def _shift_without_wrap(values, shift_x, shift_y):
    shifted = torch.roll(values, shifts=(shift_y, shift_x), dims=(-2, -1))
    if shift_y > 0:
        shifted[..., :shift_y, :] = 0
    elif shift_y < 0:
        shifted[..., shift_y:, :] = 0
    if shift_x > 0:
        shifted[..., :, :shift_x] = 0
    elif shift_x < 0:
        shifted[..., :, shift_x:] = 0
    return shifted


def _tensor_batch(candidate_set, indices, device, config, training=False, rng=None):
    images = torch.from_numpy(candidate_set.images[indices].astype(np.float32)).to(device)
    targets = torch.from_numpy(candidate_set.targets[indices].astype(np.float32)).to(device)
    grids = torch.from_numpy(candidate_set.grids[indices]).to(device)
    result = {
        "images": images,
        "targets": targets,
        "grids": grids,
        "points": torch.from_numpy(candidate_set.points[indices]).to(device),
        "mask": torch.from_numpy(candidate_set.mask[indices]).to(device),
        "expert": torch.from_numpy(candidate_set.expert[indices]).to(device),
        "distance": torch.from_numpy(candidate_set.target_distance[indices]).to(device),
        "target_view_mask": torch.from_numpy(
            candidate_set.target_view_mask[indices]
        ).to(device),
    }
    if training and config.color_noise > 0:
        result["images"][:, :, :, :3] = torch.clamp(
            result["images"][:, :, :, :3]
            + torch.randn_like(result["images"][:, :, :, :3]) * config.color_noise,
            0.0,
            1.0,
        )
    if training and config.translation_pixels > 0:
        rng = rng or np.random.default_rng(config.seed)
        height, width = images.shape[-2:]
        for batch_index in range(len(indices)):
            for landmark in range(3):
                shift_x, shift_y = rng.integers(
                    -config.translation_pixels,
                    config.translation_pixels + 1,
                    size=2,
                )
                result["images"][batch_index, landmark] = _shift_without_wrap(
                    result["images"][batch_index, landmark], int(shift_x), int(shift_y)
                )
                result["targets"][batch_index, landmark] = _shift_without_wrap(
                    result["targets"][batch_index, landmark], int(shift_x), int(shift_y)
                )
                result["grids"][batch_index, landmark, :, :, 0] += 2.0 * float(
                    shift_x
                ) / max(width - 1, 1)
                result["grids"][batch_index, landmark, :, :, 1] += 2.0 * float(
                    shift_y
                ) / max(height - 1, 1)
        projected = torch.all(torch.abs(result["grids"]) <= 1.0, dim=-1).all(dim=2)
        result["mask"] &= projected
        for batch_index in range(len(indices)):
            for landmark in range(3):
                if not bool(result["mask"][batch_index, landmark].any()):
                    extent = torch.abs(
                        result["grids"][batch_index, landmark]
                    ).amax(dim=(0, 2))
                    result["mask"][batch_index, landmark, torch.argmin(extent)] = True
    return result


def _adaptive_wing_image_loss(
    logits,
    target,
    view_mask=None,
    omega=14.0,
    theta=0.5,
    epsilon=1.0,
    alpha=2.1,
):
    prediction = torch.sigmoid(logits.float())
    target = target.float()
    difference = torch.abs(target - prediction)
    exponent = alpha - target
    first = omega * torch.log1p(torch.pow(difference / epsilon, exponent))
    theta_value = torch.as_tensor(theta / epsilon, device=target.device)
    theta_ratio = torch.pow(theta_value, exponent)
    coefficient = (
        omega
        * (1.0 / (1.0 + theta_ratio))
        * exponent
        * torch.pow(theta_value, exponent - 1.0)
        / epsilon
    )
    constant = theta * coefficient - omega * torch.log1p(theta_ratio)
    second = coefficient * difference - constant
    weight = 1.0 + 4.0 * target
    loss = (torch.where(difference < theta, first, second) * weight).mean(
        dim=(-2, -1)
    )
    if view_mask is None:
        return loss.mean()
    valid = view_mask.float()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def _poss_image_loss(logits, target, view_mask=None, exponent=2.0, temperature=0.1):
    """Position-aware and sample-sensitive heatmap loss (Zhu, ICCV 2025)."""
    prediction = torch.sigmoid(logits.float()).clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    modulation = torch.abs(
        (target - prediction) / max(float(temperature), 1e-4)
    ).pow(float(exponent))
    loss = (-target * modulation * torch.log(prediction)).mean(dim=(-2, -1))
    if view_mask is None:
        return loss.mean()
    valid = view_mask.float()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def _weighted_coordinate(logits, points, mask, topk=10, temperature=0.5):
    logits = logits.float().masked_fill(~mask, -torch.inf)
    count = min(max(1, int(topk)), logits.shape[-1])
    values, indices = torch.topk(logits, count, dim=-1)
    valid = torch.gather(mask, -1, indices)
    values = values.masked_fill(~valid, -torch.inf)
    weight = torch.softmax(values / max(float(temperature), 1e-4), dim=-1)
    selected = torch.gather(points, 2, indices[..., None].expand(-1, -1, -1, 3))
    return torch.sum(weight[..., None] * selected, dim=2)


def _loss(heatmaps, candidate_logits, batch, config):
    heatmap = _adaptive_wing_image_loss(
        heatmaps, batch["targets"], batch.get("target_view_mask")
    )
    poss = _poss_image_loss(
        heatmaps,
        batch["targets"],
        batch.get("target_view_mask"),
        config.poss_exponent,
        config.poss_temperature,
    )
    distance = batch["distance"].float()
    mask = batch["mask"]
    sigmas = distance.new_tensor(
        [config.sigma_lm0, config.sigma_gonion, config.sigma_gonion]
    )[None, :, None]
    target_energy = -(distance**2) / (2.0 * sigmas**2)
    target_energy = target_energy.masked_fill(~mask, -torch.inf)
    target_probability = torch.softmax(target_energy, dim=-1)
    log_probability = torch.log_softmax(
        candidate_logits.float().masked_fill(~mask, -torch.inf), dim=-1
    )
    ranking = torch.where(
        mask,
        target_probability
        * (torch.log(target_probability.clamp_min(1e-8)) - log_probability),
        torch.zeros_like(target_probability),
    ).sum(dim=-1).mean()

    coordinate = _weighted_coordinate(candidate_logits, batch["points"], mask)
    coordinate_loss = F.smooth_l1_loss(coordinate, batch["expert"].float(), beta=1.0)
    predicted_midpoint = coordinate[:, 1:3].mean(dim=1)
    expert_midpoint = batch["expert"][:, 1:3].float().mean(dim=1)
    predicted_width = torch.linalg.norm(coordinate[:, 1] - coordinate[:, 2], dim=-1)
    expert_width = torch.linalg.norm(
        batch["expert"][:, 1].float() - batch["expert"][:, 2].float(), dim=-1
    )
    pair = F.smooth_l1_loss(predicted_midpoint, expert_midpoint, beta=1.0)
    pair = pair + F.smooth_l1_loss(predicted_width, expert_width, beta=1.0)

    minimum = distance.amin(dim=-1, keepdim=True)
    negative_mask = mask & (distance > minimum + sigmas)
    negative_logits = candidate_logits.float().masked_fill(~negative_mask, -torch.inf)
    count = min(16, candidate_logits.shape[-1])
    negatives, negative_indices = torch.topk(negative_logits, count, dim=-1)
    negative_valid = torch.gather(negative_mask, -1, negative_indices)
    positive = (target_probability * candidate_logits.float().masked_fill(~mask, 0.0)).sum(
        dim=-1, keepdim=True
    )
    margin = F.softplus(negatives - positive + config.negative_margin)
    margin = torch.where(negative_valid, margin, torch.zeros_like(margin))
    margin = (margin.sum(dim=-1) / negative_valid.sum(dim=-1).clamp_min(1)).mean()
    total = (
        config.heatmap_weight * heatmap
        + config.poss_weight * poss
        + config.ranking_weight * ranking
        + config.coordinate_weight * coordinate_loss
        + config.pair_weight * pair
        + config.negative_weight * margin
    )
    return total, {
        "heatmap": heatmap,
        "poss": poss,
        "ranking": ranking,
        "coordinate": coordinate_loss,
        "pair": pair,
        "negative": margin,
    }


@torch.no_grad()
def _predict_logits(model, candidate_set, indices, config, device):
    model.eval()
    chunks = []
    for start in range(0, len(indices), config.batch_size):
        selected = np.asarray(indices[start : start + config.batch_size], dtype=np.int64)
        batch = _tensor_batch(candidate_set, selected, device, config)
        heatmaps = model(batch["images"])
        chunks.append(
            model.candidate_logits(heatmaps, batch["grids"], batch["mask"])
            .float()
            .cpu()
            .numpy()
        )
    return np.concatenate(chunks, axis=0)


def _train_model(candidate_set, train_indices, val_indices, config, device, fold_number):
    torch.manual_seed(config.seed + fold_number * 1009)
    model = DualViewHard3Net(
        candidate_set.images.shape[3], config.width, config.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    rng = np.random.default_rng(config.seed + fold_number * 7919)
    best_score, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(train_indices)
        totals = {
            name: 0.0
            for name in (
                "total",
                "heatmap",
                "poss",
                "ranking",
                "coordinate",
                "pair",
                "negative",
            )
        }
        seen = 0
        for start in range(0, len(order), config.batch_size):
            selected = np.asarray(order[start : start + config.batch_size], dtype=np.int64)
            batch = _tensor_batch(candidate_set, selected, device, config, True, rng)
            optimizer.zero_grad(set_to_none=True)
            heatmaps = model(batch["images"])
            logits = model.candidate_logits(heatmaps, batch["grids"], batch["mask"])
            loss, components = _loss(heatmaps, logits, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite dual-view Hard3 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            count = len(selected)
            seen += count
            totals["total"] += float(loss.detach()) * count
            for name, value in components.items():
                totals[name] += float(value.detach()) * count
        val_logits = _predict_logits(
            model, candidate_set, list(val_indices), config, device
        )
        val_prediction = _decode_numpy(
            val_logits,
            candidate_set.points[val_indices],
            candidate_set.mask[val_indices],
            5,
            0.5,
            True,
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
                f"Hard3 dual-view OOF fold {fold_number} epoch {epoch:03d}/{config.epochs} "
                f"train={row['train_total']:.4f} val={score:.4f}",
                flush=True,
            )
        if epoch >= config.min_epochs and stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Dual-view Hard3 refiner did not produce a checkpoint")
    model.load_state_dict(best_state)
    logits = _predict_logits(model, candidate_set, list(val_indices), config, device)
    return logits, best_epoch, best_score, history, {
        key: value.detach().cpu() for key, value in best_state.items()
    }


def _train_fixed_model(candidate_set, epochs, config, device, member_number):
    torch.manual_seed(config.seed + 50_003 + member_number * 1009)
    model = DualViewHard3Net(
        candidate_set.images.shape[3], config.width, config.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=1e-6
    )
    rng = np.random.default_rng(config.seed + 70_001 + member_number * 7919)
    indices = np.arange(len(candidate_set), dtype=np.int64)
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(indices)
        total, seen = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            selected = np.asarray(order[start : start + config.batch_size], dtype=np.int64)
            batch = _tensor_batch(candidate_set, selected, device, config, True, rng)
            optimizer.zero_grad(set_to_none=True)
            heatmaps = model(batch["images"])
            logits = model.candidate_logits(heatmaps, batch["grids"], batch["mask"])
            loss, _ = _loss(heatmaps, logits, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite full-train dual-view Hard3 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total += float(loss.detach()) * len(selected)
            seen += len(selected)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(seen, 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            print(
                f"Hard3 final member {member_number} epoch {epoch:03d}/{epochs} "
                f"train={history[-1]['train_loss']:.4f}",
                flush=True,
            )
    return model, history, {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }


def _cache_signature(dataset, config):
    ignored = {
        "bootstrap_iters",
        "minimum_overall_gain_mm",
        "minimum_hard3_gain_mm",
        "minimum_improvement_probability",
        "maximum_p95_regression_mm",
        "target_hard3_ale",
    }
    model_config = {key: value for key, value in asdict(config).items() if key not in ignored}
    digest = hashlib.sha256()
    records = []
    for sample in dataset.samples:
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(np.asarray(dataset._coarse(sample), dtype=np.float32).tobytes())
        path = Path(dataset.records[sample.sample_id])
        stat = path.stat()
        records.append((sample.sample_id, path.name, int(stat.st_size), int(stat.st_mtime_ns)))
    payload = {
        "version": 2,
        "records": records,
        "coarse_digest": digest.hexdigest(),
        "normalizer_mean": np.asarray(dataset.mean, dtype=np.float32).tolist(),
        "normalizer_std": np.asarray(dataset.std, dtype=np.float32).tolist(),
        "roi": [
            int(dataset.roi_points),
            float(dataset.roi_radius_scale),
            str(dataset.roi_mode),
            float(dataset.roi_euclidean_scale),
            int(dataset.roi_multi_seeds),
        ],
        "config": model_config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ordered_baseline(candidate_set, outputs):
    by_id = {
        sample_id: np.asarray(outputs["prediction"][index], dtype=np.float32)
        for index, sample_id in enumerate(outputs["sample_ids"])
    }
    missing = [sample_id for sample_id in candidate_set.sample_ids if sample_id not in by_id]
    if missing:
        raise KeyError(f"Baseline predictions miss Hard3 samples: {missing[:5]}")
    return np.stack([by_id[sample_id] for sample_id in candidate_set.sample_ids])


def _robust_logit_scale(logits, mask):
    output = np.zeros_like(logits, dtype=np.float32)
    for sample in range(len(logits)):
        for landmark in range(3):
            valid = mask[sample, landmark]
            values = logits[sample, landmark, valid].astype(np.float64)
            median = np.median(values)
            scale = max(float(np.percentile(values, 75) - np.percentile(values, 25)), 1e-3)
            output[sample, landmark, valid] = np.clip(
                (values - median) / scale, -12.0, 12.0
            )
            output[sample, landmark, ~valid] = -np.inf
    return output


def _variant_predictions(candidate_set, logits, policy, atlas_prediction):
    variants = {
        "neural_policy": _decode_policy(candidate_set, logits, policy),
        "neural_argmax": _decode_numpy(
            logits, candidate_set.points, candidate_set.mask, 1, 1.0, True
        ),
        "atlas_direct": np.asarray(atlas_prediction, dtype=np.float32),
    }
    neural = _robust_logit_scale(logits, candidate_set.mask)
    atlas_distance = np.linalg.norm(
        candidate_set.points - atlas_prediction[:, :, None], axis=-1
    )
    for sigma in (3.0, 5.0, 8.0):
        atlas_logits = -(atlas_distance**2) / (2.0 * sigma**2)
        atlas_logits = np.where(candidate_set.mask, atlas_logits, -np.inf)
        variants[f"atlas_surface_s{sigma:g}"] = _decode_numpy(
            atlas_logits, candidate_set.points, candidate_set.mask, 1, 1.0, True
        )
        for weight in (0.25, 0.5, 1.0, 2.0):
            fused = neural + weight * atlas_logits
            prefix = f"fusion_w{weight:g}_s{sigma:g}"
            variants[f"{prefix}_policy"] = _decode_policy(
                candidate_set, fused, policy
            )
            variants[f"{prefix}_argmax"] = _decode_numpy(
                fused, candidate_set.points, candidate_set.mask, 1, 1.0, True
            )
    return variants


class FittedDualViewHard3Refiner:
    def __init__(self, models, policy, atlas, report, config, device):
        self.models = [model.to(device).eval() for model in models]
        self.policy = policy
        self.atlas = atlas
        self.report = report
        self.config = config
        self.device = device

    def predict(self, dataset, baseline_outputs, label="Hard3 dual-view inference"):
        centers = {
            sample_id: np.asarray(baseline_outputs["prediction"][index], dtype=np.float32)
            for index, sample_id in enumerate(baseline_outputs["sample_ids"])
        }
        candidate_set = extract_dual_view_set(
            dataset,
            self.config.image_size,
            self.config.radius_scale,
            centers,
            label,
        )
        indices = list(range(len(candidate_set)))
        member_logits = [
            _predict_logits(model, candidate_set, indices, self.config, self.device)
            for model in self.models
        ]
        logits = np.mean(np.stack(member_logits), axis=0)
        baseline = _ordered_baseline(candidate_set, baseline_outputs)
        atlas_result = self.atlas.predict(baseline, candidate_set.sample_ids)
        variants = _variant_predictions(
            candidate_set, logits, self.policy, atlas_result["prediction"]
        )
        member_coordinates = np.stack(
            [_decode_policy(candidate_set, values, self.policy) for values in member_logits]
        )
        neural_coordinate = variants["neural_policy"]
        spread = np.linalg.norm(
            member_coordinates - neural_coordinate[None], axis=-1
        ).mean(axis=0)
        probability = np.exp(
            logits - np.max(np.where(candidate_set.mask, logits, -np.inf), axis=-1, keepdims=True)
        ) * candidate_set.mask
        probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
        entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=-1)
        entropy /= np.log(np.maximum(candidate_set.mask.sum(axis=-1), 2))
        scale = np.maximum(
            np.asarray(self.report["reliability_scale_mm"], dtype=np.float32)[None],
            0.25,
        )
        reliability = 1.0 / (1.0 + (spread / scale) ** 2)
        reliability *= 1.0 / (
            1.0 + (atlas_result["dispersion"] / np.maximum(scale * 2.0, 1.0)) ** 2
        )
        return {
            "sample_ids": candidate_set.sample_ids,
            "prediction": neural_coordinate,
            "variant_predictions": variants,
            "expert": candidate_set.expert,
            "entropy": entropy.astype(np.float32),
            "ensemble_spread": spread.astype(np.float32),
            "atlas_prediction": atlas_result["prediction"],
            "atlas_dispersion": atlas_result["dispersion"],
            "reliability": np.clip(reliability, 0.05, 1.0).astype(np.float32),
            "oracle_error": np.min(candidate_set.target_distance, axis=-1),
        }


def fit_or_load_dual_view_refiner(dataset, output_dir, config, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "hard3_dual_view_model.pth"
    report_path = output_dir / "hard3_dual_view_training_report.json"
    signature = _cache_signature(dataset, config)
    if checkpoint_path.exists() and report_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("signature") == signature:
            models = []
            for state in checkpoint["model_states"]:
                model = DualViewHard3Net(
                    checkpoint["input_channels"], config.width, config.dropout
                )
                model.load_state_dict(state)
                models.append(model)
            atlas = TrainOnlyLocalHard3Atlas.from_state_dict(checkpoint["atlas"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print("Hard3 dual-view refiner cached", flush=True)
            return FittedDualViewHard3Refiner(
                models, checkpoint["coordinate_policy"], atlas, report, config, device
            )

    candidates = extract_dual_view_set(
        dataset, config.image_size, config.radius_scale, label="Hard3 train patches"
    )
    atlas = TrainOnlyLocalHard3Atlas(
        config.atlas_neighbors, config.atlas_temperature
    ).fit(candidates.expert_full, candidates.sample_ids)
    started = time.time()
    oof_logits = np.full(
        (len(candidates), 3, candidates.points.shape[-2]), -np.inf, dtype=np.float32
    )
    fold_reports, oof_models, best_epochs = [], [], []
    for fold_number, (train_indices, val_indices) in enumerate(
        _splitter(candidates.strata, config.folds, config.seed), start=1
    ):
        logits, best_epoch, best_score, history, state = _train_model(
            candidates,
            np.asarray(train_indices),
            np.asarray(val_indices),
            config,
            device,
            fold_number,
        )
        oof_logits[val_indices] = logits
        best_epochs.append(best_epoch)
        model = DualViewHard3Net(candidates.images.shape[3], config.width, config.dropout)
        model.load_state_dict(state)
        oof_models.append(model)
        fold_reports.append(
            {
                "fold": fold_number,
                "train_sample_ids": [candidates.sample_ids[i] for i in train_indices],
                "validation_sample_ids": [candidates.sample_ids[i] for i in val_indices],
                "best_epoch": best_epoch,
                "best_validation_hard3_ale": best_score,
                "history": history,
            }
        )
    if not np.isfinite(oof_logits[candidates.mask]).all():
        raise RuntimeError("Dual-view Hard3 OOF logits are incomplete")
    policy = _select_coordinate_policy(candidates, oof_logits)
    oof_prediction = _decode_policy(candidates, oof_logits, policy)
    oof_error = np.linalg.norm(oof_prediction - candidates.expert, axis=-1)
    member_predictions = np.stack(
        [
            _decode_policy(
                candidates,
                _predict_logits(
                    model.to(device), candidates, list(range(len(candidates))), config, device
                ),
                policy,
            )
            for model in oof_models
        ]
    )
    ensemble_prediction = member_predictions.mean(axis=0)
    spread = np.linalg.norm(member_predictions - ensemble_prediction[None], axis=-1).mean(axis=0)
    reliability_scale = np.maximum(np.percentile(spread, 75, axis=0), 0.25)
    fixed_epochs = int(np.clip(np.median(best_epochs), config.min_epochs, config.epochs))
    models, states, final_histories = [], [], []
    for member_number in range(1, max(1, config.final_ensemble_members) + 1):
        model, history, state = _train_fixed_model(
            candidates, fixed_epochs, config, device, member_number
        )
        models.append(model)
        states.append(state)
        final_histories.append(
            {"member": member_number, "epochs": fixed_epochs, "history": history}
        )
    parameter_count = sum(parameter.numel() for parameter in models[0].parameters())
    report = {
        "signature": signature,
        "method": "nested-OOF dual-view RGB-depth heatmap plus bilateral contour and train-only local atlas",
        "uses_validation_labels_for_model_fit": False,
        "uses_test_labels": False,
        "sample_count": len(candidates),
        "input_channels": int(candidates.images.shape[3]),
        "parameter_count_per_member": int(parameter_count),
        "ensemble_members": len(models),
        "folds": fold_reports,
        "oof_best_epochs": best_epochs,
        "final_training": {
            "selection": "median inner-fold best epoch; no outer-validation labels",
            "fixed_epochs": fixed_epochs,
            "members": final_histories,
        },
        "coordinate_policy": policy,
        "reliability_scale_mm": reliability_scale.tolist(),
        "oof": {
            "hard3": summarize(oof_error),
            "lm0": summarize(oof_error[:, 0]),
            "gonion": summarize(oof_error[:, 1:3]),
            "candidate_oracle_ale": float(np.min(candidates.target_distance, axis=-1).mean()),
        },
        "atlas": {
            "fit_sample_ids": candidates.sample_ids,
            "neighbors": config.atlas_neighbors,
            "temperature": config.atlas_temperature,
            "uses_outer_validation_labels": False,
            "uses_test_labels": False,
        },
        "patch_target_coverage": {
            "overall_fraction": float(candidates.target_view_mask.mean()),
            "lm0_frontal": float(candidates.target_view_mask[:, 0, 0].mean()),
            "lm0_profile": float(candidates.target_view_mask[:, 0, 1].mean()),
            "gonion_frontal": float(candidates.target_view_mask[:, 1:3, 0].mean()),
            "gonion_profile": float(candidates.target_view_mask[:, 1:3, 1].mean()),
        },
        "training_seconds": float(time.time() - started),
        "config": asdict(config),
    }
    torch.save(
        {
            "signature": signature,
            "input_channels": int(candidates.images.shape[3]),
            "coordinate_policy": policy,
            "model_states": states,
            "atlas": atlas.state_dict(),
        },
        checkpoint_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return FittedDualViewHard3Refiner(models, policy, atlas, report, config, device)


def _order_values(outputs, candidate_result, values):
    by_id = {
        sample_id: values[index]
        for index, sample_id in enumerate(candidate_result["sample_ids"])
    }
    return np.stack([by_id[sample_id] for sample_id in outputs["sample_ids"]])


def _candidate_for_variants(outputs, candidate_result, lm0_variant, gonion_variant):
    lm0_values = _order_values(
        outputs, candidate_result, candidate_result["variant_predictions"][lm0_variant]
    )
    gonion_values = _order_values(
        outputs, candidate_result, candidate_result["variant_predictions"][gonion_variant]
    )
    result = gonion_values.copy()
    result[:, 0] = lm0_values[:, 0]
    return result


def calibrate_dual_view_blend(outputs, candidate_result, config):
    base = np.asarray(outputs["prediction"], dtype=np.float32)
    expert = np.asarray(outputs["expert"], dtype=np.float32)
    base_error = np.linalg.norm(base - expert, axis=-1)
    base_p95 = float(np.percentile(base_error, 95))
    reliability = np.clip(
        _order_values(outputs, candidate_result, candidate_result["reliability"]),
        0.05,
        1.0,
    )
    variants = list(candidate_result["variant_predictions"])
    individual = {}
    for name in variants:
        prediction = _order_values(
            outputs, candidate_result, candidate_result["variant_predictions"][name]
        )
        error = np.linalg.norm(prediction - expert[:, list(HARD3)], axis=-1)
        individual[name] = {
            "lm0_ale": float(error[:, 0].mean()),
            "gonion_ale": float(error[:, 1:3].mean()),
            "hard3_ale": float(error.mean()),
        }
    # Restrict the joint sweep to the strongest predefined candidates to control
    # variance on the 48-sample outer validation fold.
    lm0_variants = sorted(variants, key=lambda name: individual[name]["lm0_ale"])[:8]
    gonion_variants = sorted(variants, key=lambda name: individual[name]["gonion_ale"])[:8]
    for anchor in ("neural_policy", "atlas_direct"):
        if anchor not in lm0_variants:
            lm0_variants.append(anchor)
        if anchor not in gonion_variants:
            gonion_variants.append(anchor)

    limits = (
        config.maximum_step_lm0,
        config.maximum_step_gonion,
        config.maximum_step_gonion,
    )
    rows = []
    for lm0_variant in lm0_variants:
        for gonion_variant in gonion_variants:
            raw = _candidate_for_variants(
                outputs, candidate_result, lm0_variant, gonion_variant
            )
            limited, _, _ = _limited_hard3_candidate(base, raw, limits)
            for confidence_mode in ("none", "ensemble"):
                for alpha_lm0 in (0.0, 0.25, 0.5, 0.75, 1.0):
                    for alpha_gonion in (0.0, 0.25, 0.5, 0.75, 1.0):
                        row = {
                            "lm0_variant": lm0_variant,
                            "gonion_variant": gonion_variant,
                            "confidence_mode": confidence_mode,
                            "alpha_lm0": alpha_lm0,
                            "alpha_gonion": alpha_gonion,
                        }
                        prediction, _ = _blend_prediction(
                            base, limited, reliability, row
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
                            }
                        )
    eligible = [
        row
        for row in rows
        if row["p95"] <= base_p95 + config.maximum_p95_regression_mm
    ]
    proposed = min(
        eligible or rows,
        key=lambda row: (row["hard3_ale"], row["overall_ale"], row["p95"]),
    )
    raw = _candidate_for_variants(
        outputs,
        candidate_result,
        proposed["lm0_variant"],
        proposed["gonion_variant"],
    )
    limited, raw_step, step_scale = _limited_hard3_candidate(base, raw, limits)
    proposed_prediction, _ = _blend_prediction(base, limited, reliability, proposed)
    proposed_error = np.linalg.norm(proposed_prediction - expert, axis=-1)
    proposed_bootstrap = bootstrap_delta(
        base_error, proposed_error, config.bootstrap_iters, config.seed
    )
    overall_gain = float(base_error.mean() - proposed_error.mean())
    hard3_gain = float(
        base_error[:, list(HARD3)].mean() - proposed_error[:, list(HARD3)].mean()
    )
    accepted = (
        overall_gain >= config.minimum_overall_gain_mm
        and hard3_gain >= config.minimum_hard3_gain_mm
        and proposed_bootstrap["probability_improved"]
        >= config.minimum_improvement_probability
        and proposed["p95"] <= base_p95 + config.maximum_p95_regression_mm
    )
    selected = proposed if accepted else {
        "lm0_variant": "neural_policy",
        "gonion_variant": "neural_policy",
        "confidence_mode": "none",
        "alpha_lm0": 0.0,
        "alpha_gonion": 0.0,
        "overall_ale": float(base_error.mean()),
        "hard3_ale": float(base_error[:, list(HARD3)].mean()),
        "lm0_ale": float(base_error[:, 0].mean()),
        "gonion_ale": float(base_error[:, 21:23].mean()),
        "p95": base_p95,
    }
    selected_raw = _candidate_for_variants(
        outputs,
        candidate_result,
        selected["lm0_variant"],
        selected["gonion_variant"],
    )
    selected_limited, _, _ = _limited_hard3_candidate(base, selected_raw, limits)
    blended, effective_alpha = _blend_prediction(
        base, selected_limited, reliability, selected
    )
    blended_error = np.linalg.norm(blended - expert, axis=-1)
    return {
        "accepted": bool(accepted),
        "proposed": proposed,
        "selected": selected,
        "target_hard3_ale": config.target_hard3_ale,
        "target_reached_on_validation": bool(
            accepted and float(blended_error[:, list(HARD3)].mean()) < config.target_hard3_ale
        ),
        "limits_mm": list(limits),
        "base_overall": summarize(base_error),
        "base_hard3": summarize(base_error[:, list(HARD3)]),
        "blended_overall": summarize(blended_error),
        "blended_hard3": summarize(blended_error[:, list(HARD3)]),
        "overall_gain_mm": float(base_error.mean() - blended_error.mean()),
        "hard3_gain_mm": float(
            base_error[:, list(HARD3)].mean()
            - blended_error[:, list(HARD3)].mean()
        ),
        "bootstrap_vs_base": bootstrap_delta(
            base_error, blended_error, config.bootstrap_iters, config.seed
        ),
        "proposed_bootstrap_vs_base": proposed_bootstrap,
        "candidate_metrics": individual,
        "mean_reliability": reliability.mean(axis=0).tolist(),
        "mean_effective_alpha": effective_alpha.mean(axis=0).tolist(),
        "step_limit_fraction": np.mean(step_scale < 1.0, axis=0).tolist(),
        "raw_step_mm": {
            "lm0": summarize(raw_step[:, 0]),
            "gonion": summarize(raw_step[:, 1:3]),
        },
        "acceptance_thresholds": {
            "minimum_overall_gain_mm": config.minimum_overall_gain_mm,
            "minimum_hard3_gain_mm": config.minimum_hard3_gain_mm,
            "minimum_improvement_probability": config.minimum_improvement_probability,
            "maximum_p95_regression_mm": config.maximum_p95_regression_mm,
        },
        "sweep": rows,
        "uses_validation_labels_for_selection_only": True,
        "uses_test_labels": False,
    }


def apply_dual_view_blend(outputs, candidate_result, policy):
    result = dict(outputs)
    base = np.asarray(outputs["prediction"], dtype=np.float32)
    selected = policy["selected"]
    raw = _candidate_for_variants(
        outputs,
        candidate_result,
        selected["lm0_variant"],
        selected["gonion_variant"],
    )
    reliability = np.clip(
        _order_values(outputs, candidate_result, candidate_result["reliability"]),
        0.05,
        1.0,
    )
    candidate, _, _ = _limited_hard3_candidate(base, raw, policy["limits_mm"])
    prediction, effective_alpha = _blend_prediction(
        base, candidate, reliability, selected
    )
    full_candidate = base.copy()
    full_candidate[:, list(HARD3)] = candidate
    full_reliability = np.full((len(base), NUM_LANDMARKS), np.nan, dtype=np.float32)
    full_alpha = np.zeros((len(base), NUM_LANDMARKS), dtype=np.float32)
    full_reliability[:, list(HARD3)] = reliability
    full_alpha[:, list(HARD3)] = effective_alpha
    result["pre_hard3_prediction"] = base.copy()
    result["hard3_candidate"] = full_candidate
    result["hard3_reliability"] = full_reliability
    result["hard3_effective_alpha"] = full_alpha
    result["prediction"] = prediction
    result["errors"] = np.linalg.norm(prediction - result["expert"], axis=-1)
    return result
