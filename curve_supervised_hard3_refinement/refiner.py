"""Leakage-safe OOF training for curve-first Hard3 refinement."""

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

from all23_rgb_geodesic_cascade.anatomy import HARD3
from all23_rgb_geodesic_cascade.metrics import summarize
from hard3_anatomical_context_refinement.patches import extract_dual_view_set

from .annotations import CurveAnnotationStore
from .model import CurveFirstHard3Net, probability_coordinate
from .targets import build_curve_targets
from .validate_annotations import validate_annotation_store


@dataclass(frozen=True)
class CurveHard3Config:
    run_mode: str = "pseudo"
    folds: int = 5
    epochs: int = 100
    min_epochs: int = 30
    patience: int = 15
    batch_size: int = 8
    image_size: int = 64
    width: int = 48
    blocks: int = 2
    dropout: float = 0.10
    radius_scale: float = 1.25
    neighbor_count: int = 12
    lr: float = 5e-4
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    curve_pretrain_epochs: int = 20
    curve_pretrain_lr: float = 5e-4
    real_sample_fraction: float = 0.50
    curve_checkpoint_weight: float = 0.05
    curve_sigma_mm: float = 2.0
    point_sigma_lm0_mm: float = 2.5
    point_sigma_gonion_mm: float = 3.0
    pseudo_radius_mm: float = 12.0
    pseudo_half_length_mm: float = 20.0
    pseudo_curve_weight: float = 0.20
    curve_bce_weight: float = 0.75
    curve_dice_weight: float = 0.25
    curve_distance_weight: float = 0.20
    listwise_weight: float = 1.0
    coordinate_weight: float = 0.25
    clinical_weight: float = 0.25
    bilateral_weight: float = 0.10
    confidence_weight: float = 0.05
    temperature: float = 0.75
    annotation_manifest: str | None = None
    minimum_annotated_samples: int = 0
    publication_minimum_annotated_samples: int = 60
    maximum_annotation_surface_distance_mm: float = 5.0
    maximum_landmark_curve_distance_mm: float = 5.0
    allow_pseudo_curves: bool = True
    final_model_policy: str = "inner_fold_ensemble"
    maximum_step_lm0: float = 12.0
    maximum_step_gonion: float = 15.0
    bootstrap_iters: int = 2000
    minimum_overall_gain_mm: float = 0.03
    minimum_hard3_gain_mm: float = 0.20
    minimum_improvement_probability: float = 0.90
    maximum_p95_regression_mm: float = 0.10
    target_hard3_ale: float = 4.0
    seed: int = 42


def _annotation_store(config):
    if config.annotation_manifest:
        return CurveAnnotationStore.load(config.annotation_manifest)
    return CurveAnnotationStore.empty()


def _batch(candidate_set, targets, indices, device):
    selected = np.asarray(indices, dtype=np.int64)
    batch = {
        "images": torch.from_numpy(
            candidate_set.images[selected].astype(np.float32)
        ).to(device),
        "canonical": torch.from_numpy(candidate_set.canonical[selected]).to(device),
        "neighbor_index": torch.from_numpy(candidate_set.neighbor_index[selected]).to(
            device
        ),
        "neighbor_mask": torch.from_numpy(candidate_set.neighbor_mask[selected]).to(
            device
        ),
        "mask": torch.from_numpy(candidate_set.mask[selected]).to(device),
        "points": torch.from_numpy(candidate_set.points[selected]).to(device),
        "shape_context": torch.from_numpy(candidate_set.shape_context[selected]).to(
            device
        ),
    }
    if targets is not None:
        batch.update(
            {
                "curve_distance": torch.from_numpy(targets.curve_distance[selected]).to(
                    device
                ),
                "point_distance": torch.from_numpy(targets.point_distance[selected]).to(
                    device
                ),
                "curve_source": torch.from_numpy(targets.source[selected]).to(device),
                "expert": torch.from_numpy(candidate_set.expert[selected]).to(device),
            }
        )
    return batch


def _forward(model, batch):
    return model(
        batch["images"],
        batch["canonical"],
        batch["neighbor_index"],
        batch["neighbor_mask"],
        batch["mask"],
        batch["shape_context"],
    )


def _masked_mean(values, mask):
    valid = mask.to(values.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def _curve_loss(output, batch, config):
    mask = batch["mask"]
    curve_target = torch.exp(
        -batch["curve_distance"].float().square()
        / (2.0 * max(float(config.curve_sigma_mm), 1e-4) ** 2)
    ).masked_fill(~mask, 0.0)
    curve_logits = output["curve_logits"].float().masked_fill(~mask, 0.0)
    source_weight = torch.where(
        batch["curve_source"].bool(),
        torch.ones_like(batch["curve_source"], dtype=torch.float32),
        torch.full_like(
            batch["curve_source"],
            float(config.pseudo_curve_weight),
            dtype=torch.float32,
        ),
    )
    point_weight = 1.0 + 4.0 * curve_target
    bce = F.binary_cross_entropy_with_logits(
        curve_logits, curve_target, reduction="none"
    )
    bce = (bce * point_weight * mask * source_weight[..., None]).sum() / (
        mask * source_weight[..., None]
    ).sum().clamp_min(1.0)
    predicted = torch.sigmoid(curve_logits) * mask
    intersection = (predicted * curve_target).sum(dim=-1)
    denominator = predicted.sum(dim=-1) + curve_target.sum(dim=-1)
    dice_rows = 1.0 - (2.0 * intersection + 1e-4) / (denominator + 1e-4)
    dice = (dice_rows * source_weight).sum() / source_weight.sum().clamp_min(1.0)
    safe_distance = torch.nan_to_num(
        batch["curve_distance"].float(), nan=0.0, posinf=0.0, neginf=0.0
    ).masked_fill(~mask, 0.0)
    expected_rows = (predicted * safe_distance).sum(dim=-1) / predicted.sum(
        dim=-1
    ).clamp_min(1e-6)
    expected_distance = (
        (expected_rows * source_weight).sum()
        / source_weight.sum().clamp_min(1.0)
        / 10.0
    )
    return bce, dice, expected_distance, curve_target


def _curve_only_loss(output, batch, config):
    bce, dice, expected_distance, _ = _curve_loss(output, batch, config)
    total = (
        config.curve_bce_weight * bce
        + config.curve_dice_weight * dice
        + config.curve_distance_weight * expected_distance
    )
    return total, {
        "curve_bce": bce,
        "curve_dice": dice,
        "curve_distance": expected_distance,
    }


def _loss(output, batch, config):
    mask = batch["mask"]
    bce, dice, curve_distance, _ = _curve_loss(output, batch, config)
    sigma = batch["point_distance"].new_tensor(
        [
            config.point_sigma_lm0_mm,
            config.point_sigma_gonion_mm,
            config.point_sigma_gonion_mm,
        ]
    )[None, :, None]
    target_energy = -batch["point_distance"].float().square() / (
        2.0 * sigma.square().clamp_min(1e-6)
    )
    target_energy = target_energy.masked_fill(~mask, -torch.inf)
    target_probability = torch.softmax(target_energy, dim=-1)
    log_probability = torch.log_softmax(
        output["final_logits"].float().masked_fill(~mask, -torch.inf)
        / max(float(config.temperature), 1e-4),
        dim=-1,
    ).masked_fill(~mask, 0.0)
    listwise = (
        target_probability
        * (torch.log(target_probability.clamp_min(1e-8)) - log_probability)
    ).masked_fill(~mask, 0.0)
    listwise = listwise.sum(dim=-1).mean()
    coordinate, probability = probability_coordinate(
        output["final_logits"], batch["points"], mask, config.temperature
    )
    coordinate_loss = F.smooth_l1_loss(coordinate, batch["expert"].float(), beta=1.0)
    finite_distance = batch["point_distance"].float().masked_fill(~mask, 0.0)
    clinical = (probability * finite_distance).sum(dim=-1).mean() / 10.0
    predicted_midpoint = coordinate[:, 1:3].mean(dim=1)
    expert_midpoint = batch["expert"][:, 1:3].float().mean(dim=1)
    predicted_width = torch.linalg.norm(coordinate[:, 1] - coordinate[:, 2], dim=-1)
    expert_width = torch.linalg.norm(
        batch["expert"][:, 1].float() - batch["expert"][:, 2].float(), dim=-1
    )
    bilateral = F.smooth_l1_loss(
        predicted_midpoint, expert_midpoint, beta=1.0
    ) + F.smooth_l1_loss(predicted_width, expert_width, beta=1.0)
    squared_error = (
        torch.linalg.norm(coordinate - batch["expert"].float(), dim=-1).square() / 16.0
    )
    log_variance = output["log_variance"].float()
    confidence = 0.5 * (torch.exp(-log_variance) * squared_error + log_variance).mean()
    total = (
        config.curve_bce_weight * bce
        + config.curve_dice_weight * dice
        + config.curve_distance_weight * curve_distance
        + config.listwise_weight * listwise
        + config.coordinate_weight * coordinate_loss
        + config.clinical_weight * clinical
        + config.bilateral_weight * bilateral
        + config.confidence_weight * confidence
    )
    return total, {
        "curve_bce": bce,
        "curve_dice": dice,
        "curve_distance": curve_distance,
        "listwise": listwise,
        "coordinate": coordinate_loss,
        "clinical": clinical,
        "bilateral": bilateral,
        "confidence": confidence,
    }


@torch.no_grad()
def _predict_model(model, candidate_set, targets, indices, config, device):
    model.eval()
    chunks = {
        "curve_logits": [],
        "final_logits": [],
        "landmark_logits": [],
        "log_variance": [],
    }
    indices = list(indices)
    for start in range(0, len(indices), config.batch_size):
        selected = indices[start : start + config.batch_size]
        batch = _batch(candidate_set, targets, selected, device)
        output = _forward(model, batch)
        for key in chunks:
            chunks[key].append(output[key].float().cpu().numpy())
    result = {key: np.concatenate(rows) for key, rows in chunks.items()}
    result["coordinate"] = _decode_numpy(
        result["final_logits"],
        candidate_set.points[np.asarray(indices)],
        candidate_set.mask[np.asarray(indices)],
        config.temperature,
        False,
    )
    return result


def _decode_numpy(logits, points, mask, temperature=1.0, argmax=False):
    safe = np.where(mask, logits, -np.inf).astype(np.float64)
    if argmax:
        index = np.argmax(safe, axis=-1)
        return np.take_along_axis(points, index[..., None, None], axis=2)[
            ..., 0, :
        ].astype(np.float32)
    maximum = np.max(safe, axis=-1, keepdims=True)
    probability = np.exp((safe - maximum) / max(float(temperature), 1e-4)) * mask
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
    return np.sum(probability[..., None] * points, axis=-2).astype(np.float32)


def _curve_diagnostics(logits, candidate_set, targets, indices, config):
    indices = np.asarray(indices, dtype=np.int64)
    mask = candidate_set.mask[indices]
    safe_logits = np.where(mask, logits, -30.0).astype(np.float64)
    predicted = 1.0 / (1.0 + np.exp(-safe_logits))
    predicted *= mask
    distance = np.nan_to_num(
        targets.curve_distance[indices], nan=0.0, posinf=0.0, neginf=0.0
    )
    expected_distance = (predicted * distance).sum(axis=-1) / np.maximum(
        predicted.sum(axis=-1), 1e-8
    )
    target = np.exp(
        -(distance**2) / (2.0 * max(float(config.curve_sigma_mm), 1e-4) ** 2)
    )
    target *= mask
    intersection = (predicted * target).sum(axis=-1)
    dice = (2.0 * intersection + 1e-4) / np.maximum(
        predicted.sum(axis=-1) + target.sum(axis=-1) + 1e-4, 1e-8
    )
    return expected_distance.astype(np.float32), dice.astype(np.float32)


def _new_model(candidate_set, config):
    return CurveFirstHard3Net(
        candidate_set.images.shape[3],
        candidate_set.canonical.shape[-1],
        candidate_set.shape_context.shape[-1],
        config.width,
        config.blocks,
        config.dropout,
    )


def _source_aware_splits(strata, source, folds, seed):
    fully_annotated = np.asarray(source, dtype=bool).all(axis=1)
    base = np.asarray(strata, dtype=str)
    folds = min(max(2, int(folds)), len(base))
    unique_strata = sorted(set(base.tolist()))
    stratum_index = {value: index for index, value in enumerate(unique_strata)}
    rng = np.random.default_rng(seed)
    validation_folds = [[] for _ in range(folds)]
    fold_size = np.zeros(folds, dtype=np.int64)
    source_count = np.zeros((folds, 2), dtype=np.int64)
    stratum_count = np.zeros((folds, len(unique_strata)), dtype=np.int64)
    groups = {}
    for index, (stratum, annotated) in enumerate(zip(base, fully_annotated)):
        groups.setdefault((int(annotated), stratum), []).append(index)

    # Rare, truly annotated groups are assigned first. Greedy deficits retain
    # class/gender balance even when each combined stratum has fewer rows than
    # the requested fold count (the common 24-annotation pilot case).
    keys = sorted(groups, key=lambda key: (key[0] == 0, len(groups[key]), key[1]))
    for source_value, stratum in keys:
        rows = rng.permutation(groups[(source_value, stratum)])
        tie_order = rng.permutation(folds)
        tie_rank = np.empty(folds, dtype=np.int64)
        tie_rank[tie_order] = np.arange(folds)
        column = stratum_index[stratum]
        for row in rows:
            destination = min(
                range(folds),
                key=lambda fold: (
                    int(source_count[fold, source_value]),
                    int(stratum_count[fold, column]),
                    int(fold_size[fold]),
                    int(tie_rank[fold]),
                ),
            )
            validation_folds[destination].append(int(row))
            source_count[destination, source_value] += 1
            stratum_count[destination, column] += 1
            fold_size[destination] += 1

    all_indices = np.arange(len(base), dtype=np.int64)
    splits = []
    for rows in validation_folds:
        validation = np.asarray(sorted(rows), dtype=np.int64)
        if len(validation) == 0:
            raise RuntimeError("Source-aware Curve-H3 split produced an empty fold")
        train = np.setdiff1d(all_indices, validation, assume_unique=True)
        splits.append((train, validation))

    strategy = "greedy_class_gender_and_curve_source"
    report = {
        "strategy": strategy,
        "fully_annotated_samples": int(fully_annotated.sum()),
        "folds": [
            {
                "fold": fold,
                "train_fully_annotated": int(fully_annotated[train].sum()),
                "validation_fully_annotated": int(fully_annotated[val].sum()),
                "train_samples": int(len(train)),
                "validation_samples": int(len(val)),
                "validation_class_gender": {
                    value: int((base[val] == value).sum()) for value in unique_strata
                },
            }
            for fold, (train, val) in enumerate(splits, start=1)
        ],
    }
    return splits, report


def _repeat_to_length(values, length, rng):
    values = np.asarray(values, dtype=np.int64)
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    rows = []
    while sum(len(row) for row in rows) < length:
        rows.append(rng.permutation(values))
    return np.concatenate(rows)[:length]


def _balanced_batches(indices, source, batch_size, real_fraction, rng):
    indices = np.asarray(indices, dtype=np.int64)
    batch_size = max(1, int(batch_size))
    real = indices[np.asarray(source, dtype=bool)[indices].any(axis=1)]
    pseudo = indices[~np.asarray(source, dtype=bool)[indices].any(axis=1)]
    if len(real) == 0 or len(pseudo) == 0 or not (0.0 < real_fraction < 1.0):
        order = rng.permutation(indices)
        return [
            order[start : start + batch_size]
            for start in range(0, len(order), batch_size)
        ]
    real_per_batch = int(np.clip(round(batch_size * real_fraction), 1, batch_size - 1))
    pseudo_per_batch = batch_size - real_per_batch
    batch_count = max(
        int(np.ceil(len(real) / real_per_batch)),
        int(np.ceil(len(pseudo) / pseudo_per_batch)),
    )
    real_rows = _repeat_to_length(real, batch_count * real_per_batch, rng)
    pseudo_rows = _repeat_to_length(pseudo, batch_count * pseudo_per_batch, rng)
    batches = []
    for batch_index in range(batch_count):
        selected = np.concatenate(
            [
                real_rows[
                    batch_index * real_per_batch : (batch_index + 1) * real_per_batch
                ],
                pseudo_rows[
                    batch_index
                    * pseudo_per_batch : (batch_index + 1)
                    * pseudo_per_batch
                ],
            ]
        )
        batches.append(rng.permutation(selected))
    return batches


def _set_training_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(stage == "joint")
    if stage == "curve":
        prefixes = (
            "view_encoder.",
            "view_projection.",
            "point_encoder.",
            "shape_encoder.",
            "landmark_embedding",
            "graph_blocks.",
            "context_fusion.",
            "bilateral_context.",
            "curve_head.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))


@torch.no_grad()
def _evaluate_curve_stage(model, candidate_set, targets, indices, config, device):
    model.eval()
    totals, seen = {}, 0
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(indices), config.batch_size):
        selected = indices[start : start + config.batch_size]
        batch = _batch(candidate_set, targets, selected, device)
        output = _forward(model, batch)
        loss, components = _curve_only_loss(output, batch, config)
        count = len(selected)
        totals["total"] = totals.get("total", 0.0) + float(loss) * count
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value) * count
        seen += count
    return {name: value / max(seen, 1) for name, value in totals.items()}


def _pretrain_curve_stage(
    model,
    candidate_set,
    targets,
    train_indices,
    val_indices,
    config,
    device,
    fold,
    rng,
):
    annotated_train = int(targets.source[np.asarray(train_indices)].any(axis=1).sum())
    if annotated_train == 0 or config.curve_pretrain_epochs <= 0:
        return []
    _set_training_stage(model, "curve")
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.curve_pretrain_lr,
        weight_decay=config.weight_decay,
    )
    history = []
    for epoch in range(1, config.curve_pretrain_epochs + 1):
        model.train()
        totals, seen = {}, 0
        batches = _balanced_batches(
            train_indices,
            targets.source,
            config.batch_size,
            config.real_sample_fraction,
            rng,
        )
        for selected in batches:
            batch = _batch(candidate_set, targets, selected, device)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            loss, components = _curve_only_loss(output, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Curve-H3 support pretraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()
            count = len(selected)
            totals["total"] = totals.get("total", 0.0) + float(loss.detach()) * count
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
            seen += count
        validation = _evaluate_curve_stage(
            model, candidate_set, targets, val_indices, config, device
        )
        row = {
            "stage": "curve_pretrain",
            "epoch": epoch,
            **{f"train_{name}": value / max(seen, 1) for name, value in totals.items()},
            **{f"val_{name}": value for name, value in validation.items()},
            "real_batch_fraction": float(config.real_sample_fraction),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == config.curve_pretrain_epochs:
            print(
                f"Curve-H3 fold {fold} pretrain {epoch:03d}/"
                f"{config.curve_pretrain_epochs} train={row['train_total']:.4f} "
                f"val={row['val_total']:.4f}",
                flush=True,
            )
    return history


def _fit_one(candidate_set, targets, train_indices, val_indices, config, device, fold):
    torch.manual_seed(config.seed + fold * 1009)
    model = _new_model(candidate_set, config).to(device)
    rng = np.random.default_rng(config.seed + fold * 7919)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    history = _pretrain_curve_stage(
        model,
        candidate_set,
        targets,
        train_indices,
        val_indices,
        config,
        device,
        fold,
        rng,
    )
    _set_training_stage(model, "joint")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    best_selection = float("inf")
    best_hard3 = float("inf")
    best_epoch, stale, best_state = 0, 0, None
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals, seen = {}, 0
        batches = _balanced_batches(
            train_indices,
            targets.source,
            config.batch_size,
            config.real_sample_fraction,
            rng,
        )
        for selected in batches:
            batch = _batch(candidate_set, targets, selected, device)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            loss, components = _loss(output, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite curve-supervised Hard3 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            count = len(selected)
            totals["total"] = totals.get("total", 0.0) + float(loss.detach()) * count
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
            seen += count
        prediction_output = _predict_model(
            model, candidate_set, targets, val_indices, config, device
        )
        prediction = prediction_output["coordinate"]
        error = np.linalg.norm(prediction - candidate_set.expert[val_indices], axis=-1)
        hard3_score = float(error.mean())
        real_curve_mask = targets.source[val_indices].astype(bool)
        curve_expected, curve_dice = _curve_diagnostics(
            prediction_output["curve_logits"],
            candidate_set,
            targets,
            val_indices,
            config,
        )
        if real_curve_mask.any():
            real_curve_distance = float(curve_expected[real_curve_mask].mean())
            real_curve_dice = float(curve_dice[real_curve_mask].mean())
            selection_score = (
                hard3_score
                + float(config.curve_checkpoint_weight) * real_curve_distance
            )
        else:
            real_curve_distance = None
            real_curve_dice = None
            selection_score = hard3_score
        scheduler.step(selection_score)
        row = {
            "stage": "joint",
            "epoch": epoch,
            **{f"train_{name}": value / max(seen, 1) for name, value in totals.items()},
            "val_hard3_ale": hard3_score,
            "val_lm0_ale": float(error[:, 0].mean()),
            "val_lm21_ale": float(error[:, 1].mean()),
            "val_lm22_ale": float(error[:, 2].mean()),
            "val_real_curve_expected_distance_mm": real_curve_distance,
            "val_real_curve_soft_dice": real_curve_dice,
            "val_selection_score": selection_score,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if selection_score < best_selection - 1e-4:
            best_selection = selection_score
            best_hard3 = hard3_score
            best_epoch, stale = epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        elif epoch >= config.min_epochs:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Curve-H3 fold {fold} joint {epoch:03d}/{config.epochs} "
                f"train={row['train_total']:.4f} val={hard3_score:.4f} "
                f"selected={selection_score:.4f}",
                flush=True,
            )
        if epoch >= config.min_epochs and stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Curve-supervised Hard3 did not produce a checkpoint")
    model.load_state_dict(best_state)
    output = _predict_model(model, candidate_set, targets, val_indices, config, device)
    return (
        model,
        output,
        best_epoch,
        best_hard3,
        best_selection,
        history,
        {key: value.detach().cpu() for key, value in best_state.items()},
    )


def _signature(candidate_set, config, annotation_store):
    payload = {
        "version": 2,
        "sample_ids": list(candidate_set.sample_ids),
        "centers_sha256": hashlib.sha256(
            np.asarray(candidate_set.centers, dtype=np.float32).tobytes()
        ).hexdigest(),
        "annotation_sha256": annotation_store.source_hash,
        "config": asdict(config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


class FittedCurveHard3Refiner:
    def __init__(self, models, report, config, device):
        self.models = [model.to(device).eval() for model in models]
        self.report = dict(report)
        self.config = config
        self.device = device

    def predict(self, dataset, baseline_outputs, label="Curve-H3 inference"):
        centers = {
            sample_id: np.asarray(
                baseline_outputs["prediction"][index], dtype=np.float32
            )
            for index, sample_id in enumerate(baseline_outputs["sample_ids"])
        }
        candidates = extract_dual_view_set(
            dataset,
            self.config.image_size,
            self.config.radius_scale,
            centers,
            label,
            neighbor_count=self.config.neighbor_count,
            include_contour_features=True,
        )
        indices = np.arange(len(candidates), dtype=np.int64)
        member = [
            _predict_model(model, candidates, None, indices, self.config, self.device)
            for model in self.models
        ]
        final_logits = np.mean(
            np.stack([row["final_logits"] for row in member]), axis=0
        )
        landmark_logits = np.mean(
            np.stack([row["landmark_logits"] for row in member]), axis=0
        )
        member_coordinate = np.stack([row["coordinate"] for row in member])
        curve_soft = _decode_numpy(
            final_logits,
            candidates.points,
            candidates.mask,
            self.config.temperature,
            False,
        )
        curve_argmax = _decode_numpy(
            final_logits,
            candidates.points,
            candidates.mask,
            self.config.temperature,
            True,
        )
        point_only = _decode_numpy(
            landmark_logits,
            candidates.points,
            candidates.mask,
            self.config.temperature,
            False,
        )
        baseline_by_id = {
            sample_id: baseline_outputs["prediction"][index, list(HARD3)]
            for index, sample_id in enumerate(baseline_outputs["sample_ids"])
        }
        baseline = np.stack([baseline_by_id[value] for value in candidates.sample_ids])
        spread = np.linalg.norm(
            member_coordinate - member_coordinate.mean(axis=0)[None], axis=-1
        ).mean(axis=0)
        predicted_variance = np.exp(
            np.mean(np.stack([row["log_variance"] for row in member]), axis=0)
        )
        scale = np.maximum(
            np.asarray(self.report["reliability_scale_mm"], dtype=np.float32), 0.25
        )
        reliability = 1.0 / (1.0 + (spread / scale[None]) ** 2)
        reliability *= 1.0 / (1.0 + predicted_variance / 4.0)
        variants = {
            "neural_policy": curve_soft,
            "curve_policy": curve_soft,
            "curve_argmax": curve_argmax,
            "curve_point_only": point_only,
            "atlas_direct": baseline.astype(np.float32),
        }
        oracle = np.min(candidates.target_distance, axis=-1)
        return {
            "sample_ids": candidates.sample_ids,
            "prediction": curve_soft,
            "variant_predictions": variants,
            "expert": candidates.expert,
            "reliability": np.clip(reliability, 0.05, 1.0).astype(np.float32),
            "oracle_error": oracle,
            "validation_diagnostics": {
                "curve_supervision": self.report["annotation_report"],
                "candidate_oracle": {
                    "hard3_ale": float(oracle.mean()),
                    "hard3_p95": float(np.percentile(oracle, 95)),
                    "hard3_sdr_at_2mm": float((oracle <= 2.0).mean()),
                },
            },
        }


def fit_or_load_curve_hard3_refiner(dataset, output_dir, config, device):
    if config.final_model_policy != "inner_fold_ensemble":
        raise ValueError("Curve-H3 currently requires inner_fold_ensemble")
    if config.run_mode not in ("pseudo", "pilot", "publication"):
        raise ValueError(f"Unknown Curve-H3 run mode: {config.run_mode}")
    if not 0.0 < float(config.real_sample_fraction) < 1.0:
        raise ValueError("Curve-H3 real_sample_fraction must be between 0 and 1")
    if float(config.curve_checkpoint_weight) < 0.0:
        raise ValueError("Curve-H3 curve_checkpoint_weight cannot be negative")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_sample_ids = [sample.sample_id for sample in dataset.samples]
    annotation_store = _annotation_store(config).subset(train_sample_ids)
    initial_coverage = annotation_store.coverage(train_sample_ids)
    annotated = int(initial_coverage["fully_curve_annotated"])
    required = int(config.minimum_annotated_samples)
    if config.run_mode == "publication":
        required = max(required, int(config.publication_minimum_annotated_samples))
    elif config.run_mode == "pilot":
        required = max(required, 1)
    if config.run_mode != "pseudo" and not config.annotation_manifest:
        raise RuntimeError(
            f"Curve-H3 {config.run_mode} mode requires a real curve annotation manifest"
        )
    annotation_geometry = None
    if config.run_mode != "pseudo":
        annotation_geometry = validate_annotation_store(
            dataset.samples,
            annotation_store,
            required,
            config.maximum_annotation_surface_distance_mm,
            config.maximum_landmark_curve_distance_mm,
        )
    preflight = {
        "version": "Curve-H3-v2",
        "run_mode": config.run_mode,
        "required_fully_annotated_training_samples": required,
        "publication_minimum_annotated_samples": int(
            config.publication_minimum_annotated_samples
        ),
        "coverage": initial_coverage,
        "annotation_geometry": annotation_geometry,
        "passed": bool(
            annotated >= required
            and (annotation_geometry is None or annotation_geometry["passed"])
        ),
        "train_sample_ids_only": True,
    }
    (output_dir / "annotation_preflight.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    if annotated < required:
        raise RuntimeError(
            f"Curve-H3 {config.run_mode} mode requires at least {required} fully "
            f"annotated outer-training samples; found {annotated}. The check ran "
            "before patch extraction or model training."
        )
    if annotation_geometry is not None and not annotation_geometry["passed"]:
        raise RuntimeError(
            "Curve-H3 annotation geometry preflight failed before patch extraction: "
            + "; ".join(annotation_geometry["failures"])
        )
    candidates = extract_dual_view_set(
        dataset,
        config.image_size,
        config.radius_scale,
        None,
        "Curve-H3 training patches",
        neighbor_count=config.neighbor_count,
        include_contour_features=True,
    )
    targets = build_curve_targets(
        dataset,
        candidates,
        annotation_store,
        config.allow_pseudo_curves,
        config.pseudo_radius_mm,
        config.pseudo_half_length_mm,
    )
    pilot_ready = bool(
        config.run_mode in ("pilot", "publication") and annotated >= required
    )
    publication_ready = bool(
        config.run_mode == "publication"
        and annotated >= int(config.publication_minimum_annotated_samples)
    )
    targets.annotation_report.update(
        {
            "run_mode": config.run_mode,
            "pilot_ready": pilot_ready,
            "publication_ready": publication_ready,
            "publication_minimum_annotated_samples": int(
                config.publication_minimum_annotated_samples
            ),
        }
    )
    signature = _signature(candidates, config, annotation_store)
    checkpoint_path = output_dir / "curve_hard3_model.pth"
    report_path = output_dir / "curve_hard3_training_report.json"
    if checkpoint_path.exists() and report_path.exists():
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("signature") == signature:
            models = []
            for state in checkpoint["model_states"]:
                model = _new_model(candidates, config)
                model.load_state_dict(state)
                models.append(model)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print("Curve-supervised Hard3 refiner cached", flush=True)
            return FittedCurveHard3Refiner(models, report, config, device)

    started = time.time()
    splits, source_split_report = _source_aware_splits(
        candidates.strata, targets.source, config.folds, config.seed
    )
    coverage = np.zeros(len(candidates), dtype=np.int64)
    oof_prediction = np.zeros_like(candidates.expert, dtype=np.float32)
    oof_log_variance = np.zeros((len(candidates), 3), dtype=np.float32)
    oof_curve_logits = np.full_like(
        candidates.target_distance, -np.inf, dtype=np.float32
    )
    models, states, fold_reports = [], [], []
    for fold, (train_indices, val_indices) in enumerate(splits, start=1):
        coverage[np.asarray(val_indices, dtype=np.int64)] += 1
        (
            model,
            output,
            best_epoch,
            best_hard3,
            best_selection,
            history,
            state,
        ) = _fit_one(
            candidates,
            targets,
            train_indices,
            val_indices,
            config,
            device,
            fold,
        )
        oof_prediction[np.asarray(val_indices)] = output["coordinate"]
        oof_log_variance[np.asarray(val_indices)] = output["log_variance"]
        oof_curve_logits[np.asarray(val_indices)] = output["curve_logits"]
        models.append(model)
        states.append(state)
        fold_reports.append(
            {
                "fold": fold,
                "train_sample_ids": [candidates.sample_ids[i] for i in train_indices],
                "validation_sample_ids": [
                    candidates.sample_ids[i] for i in val_indices
                ],
                "best_epoch": best_epoch,
                "best_validation_hard3_ale": best_hard3,
                "best_validation_selection_score": best_selection,
                "train_fully_annotated": int(
                    targets.source[np.asarray(train_indices)].all(axis=1).sum()
                ),
                "validation_fully_annotated": int(
                    targets.source[np.asarray(val_indices)].all(axis=1).sum()
                ),
                "history": history,
            }
        )
    if not np.all(coverage == 1):
        raise RuntimeError("Curve-H3 requires exactly one OOF prediction per sample")
    oof_error = np.linalg.norm(oof_prediction - candidates.expert, axis=-1)
    curve_expected_distance, curve_dice = _curve_diagnostics(
        oof_curve_logits,
        candidates,
        targets,
        np.arange(len(candidates)),
        config,
    )
    real_target = targets.source.astype(bool)
    fully_annotated = real_target.all(axis=1)

    def support_summary(mask):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return None
        return {
            "target_count": int(mask.sum()),
            "expected_distance_mm": summarize(curve_expected_distance[mask]),
            "mean_soft_dice": float(curve_dice[mask].mean()),
        }

    member_prediction = np.stack(
        [
            _predict_model(
                model,
                candidates,
                targets,
                np.arange(len(candidates)),
                config,
                device,
            )["coordinate"]
            for model in models
        ]
    )
    spread = np.linalg.norm(
        member_prediction - member_prediction.mean(axis=0)[None], axis=-1
    ).mean(axis=0)
    reliability_scale = np.maximum(np.percentile(spread, 75, axis=0), 0.25)
    parameter_count = sum(parameter.numel() for parameter in models[0].parameters())
    oof_report = {
        "hard3": summarize(oof_error),
        "lm0": summarize(oof_error[:, 0]),
        "lm21": summarize(oof_error[:, 1]),
        "lm22": summarize(oof_error[:, 2]),
        "candidate_oracle": summarize(np.min(candidates.target_distance, axis=-1)),
        "curve_support_all": support_summary(np.ones_like(real_target, dtype=bool)),
        "curve_support_real": support_summary(real_target),
        "curve_support_pseudo": support_summary(~real_target),
    }
    predicted_sigma_mm = 4.0 * np.exp(
        0.5 * np.clip(oof_log_variance.astype(np.float64), -4.0, 5.0)
    )

    def error_uncertainty_correlation(error, uncertainty):
        error = np.asarray(error, dtype=np.float64).reshape(-1)
        uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
        if np.std(error) < 1e-8 or np.std(uncertainty) < 1e-8:
            return None
        return float(np.corrcoef(error, uncertainty)[0, 1])

    oof_report["confidence_calibration"] = {
        "predicted_sigma_mm": {
            "mean": float(predicted_sigma_mm.mean()),
            "median": float(np.median(predicted_sigma_mm)),
            "std": float(predicted_sigma_mm.std()),
            "p95": float(np.percentile(predicted_sigma_mm, 95)),
            "max": float(predicted_sigma_mm.max()),
        },
        "error_uncertainty_correlation": error_uncertainty_correlation(
            oof_error, predicted_sigma_mm
        ),
        "per_landmark_correlation": {
            str(landmark): error_uncertainty_correlation(
                oof_error[:, local_index], predicted_sigma_mm[:, local_index]
            )
            for local_index, landmark in enumerate(HARD3)
        },
    }
    if fully_annotated.any():
        oof_report["fully_annotated_hard3"] = summarize(oof_error[fully_annotated])
    if (~fully_annotated).any():
        oof_report["remaining_samples_hard3"] = summarize(oof_error[~fully_annotated])
    report = {
        "version": "Curve-H3-v2",
        "method": (
            "source-balanced curve pretraining followed by a curve-conditioned "
            "surface-graph landmark decoder with bilateral Gonion context"
        ),
        "uses_validation_labels_for_model_fit": False,
        "uses_test_labels": False,
        "run_mode": config.run_mode,
        "pilot_ready": pilot_ready,
        "publication_ready": publication_ready,
        "annotation_report": targets.annotation_report,
        "source_aware_inner_splits": source_split_report,
        "sample_count": len(candidates),
        "parameter_count_per_member": int(parameter_count),
        "ensemble_members": len(models),
        "reliability_scale_mm": reliability_scale.tolist(),
        "oof": oof_report,
        "folds": fold_reports,
        "training_seconds": float(time.time() - started),
        "config": asdict(config),
    }
    torch.save(
        {
            "version": "Curve-H3-v2",
            "signature": signature,
            "model_states": states,
            "input_channels": int(candidates.images.shape[3]),
            "feature_dim": int(candidates.canonical.shape[-1]),
        },
        checkpoint_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return FittedCurveHard3Refiner(models, report, config, device)
