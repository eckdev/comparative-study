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
    _decode_numpy,
    _decode_policy,
    _limited_hard3_candidate,
    _select_coordinate_policy,
    _splitter,
)
from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, NUM_LANDMARKS
from all23_rgb_geodesic_cascade.metrics import bootstrap_delta, summarize

from .atlas import TrainOnlyLocalHard3Atlas
from .model import PROPOSAL_SOURCE_NAMES, DualViewHard3Net, diverse_topk_indices
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
    joint_pair_weight: float = 0.75
    joint_pair_negative_weight: float = 0.20
    decoder_mode: str = "contour_coordinate"
    proposal_topk: int = 96
    pair_topk: int = 96
    pair_temperature: float = 0.5
    proposal_neighbors: int = 12
    proposal_teacher_forcing_epochs: int = 0
    rerank_sigma: float = 2.0
    rerank_temperature: float = 0.75
    rerank_listwise_weight: float = 1.0
    rerank_expected_distance_weight: float = 0.5
    rerank_coordinate_weight: float = 0.25
    rerank_ordinal_weight: float = 0.25
    rerank_negative_count: int = 32
    rerank_negative_radius_mm: float = 3.0
    rerank_stage_epochs: int = 40
    rerank_stage_min_epochs: int = 10
    rerank_stage_patience: int = 10
    rerank_stage_lr: float = 5e-4
    pair_stage_epochs: int = 40
    pair_stage_min_epochs: int = 10
    pair_stage_patience: int = 10
    pair_stage_lr: float = 5e-4
    pair_target_sigma: float = 1.5
    pair_listwise_weight: float = 0.75
    pair_unary_listwise_weight: float = 0.50
    pair_clinical_radius_mm: float = 2.0
    pair_clinical_mass_weight: float = 1.0
    pair_expected_distance_weight: float = 0.5
    pair_hard_negative_weight: float = 0.25
    pair_hard_negative_radius_mm: float = 4.0
    pair_hard_negative_count: int = 64
    pair_validation_mode: str = "pair_argmax"
    contour_state_weight: float = 1.0
    contour_moment_weight: float = 0.5
    contour_state_scale: float = 0.03
    contour_residual_limit: float = 0.20
    distance_normalizer_mm: float = 10.0
    negative_weight: float = 0.15
    negative_margin: float = 0.5
    gonion_color_dropout: float = 0.25
    atlas_neighbors: int = 8
    atlas_temperature: float = 2.0
    final_ensemble_members: int = 3
    final_model_policy: str = "inner_fold_ensemble"
    diagnostic_topk: tuple[int, ...] = (32, 48, 96)
    minimum_proposal_recall: float = 0.90
    maximum_proposal_oracle_ale: float = 1.50
    minimum_proposal_sdr2: float = 0.75
    maximum_proposal_oracle_p95: float = 3.50
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
    images = torch.from_numpy(candidate_set.images[indices].astype(np.float32)).to(
        device
    )
    targets = torch.from_numpy(candidate_set.targets[indices].astype(np.float32)).to(
        device
    )
    grids = torch.from_numpy(candidate_set.grids[indices]).to(device)
    canonical = candidate_set.canonical[indices]
    shape_context = (
        candidate_set.shape_context[indices]
        if candidate_set.shape_context is not None
        else np.zeros((len(indices), 69), dtype=np.float32)
    )
    if candidate_set.base_gonion is not None:
        base_gonion = candidate_set.base_gonion[indices]
    else:
        valid = candidate_set.mask[indices, 1:3][..., None].astype(np.float32)
        global_geometry = canonical[:, 1:3, :, 3:6]
        base_gonion = (global_geometry * valid).sum(axis=2) / np.maximum(
            valid.sum(axis=2), 1.0
        )
    result = {
        "images": images,
        "targets": targets,
        "grids": grids,
        "points": torch.from_numpy(candidate_set.points[indices]).to(device),
        "canonical": torch.from_numpy(canonical).to(device),
        "shape_context": torch.from_numpy(
            np.asarray(shape_context, dtype=np.float32)
        ).to(device),
        "base_gonion": torch.from_numpy(np.asarray(base_gonion, dtype=np.float32)).to(
            device
        ),
        "neighbor_index": torch.from_numpy(candidate_set.neighbor_index[indices]).to(
            device
        ),
        "neighbor_mask": torch.from_numpy(candidate_set.neighbor_mask[indices]).to(
            device
        ),
        "mask": torch.from_numpy(candidate_set.mask[indices]).to(device),
        "expert": torch.from_numpy(candidate_set.expert[indices]).to(device),
        "distance": torch.from_numpy(candidate_set.target_distance[indices]).to(device),
        "target_view_mask": torch.from_numpy(
            candidate_set.target_view_mask[indices]
        ).to(device),
    }
    if candidate_set.expert_gonion_context is not None:
        result["expert_gonion_context"] = torch.from_numpy(
            candidate_set.expert_gonion_context[indices]
        ).to(device)
    if training and config.color_noise > 0:
        result["images"][:, :, :, :3] = torch.clamp(
            result["images"][:, :, :, :3]
            + torch.randn_like(result["images"][:, :, :, :3]) * config.color_noise,
            0.0,
            1.0,
        )
        if result["canonical"].shape[-1] >= 34:
            rgb = torch.clamp(
                result["canonical"][..., 26:29]
                + torch.randn_like(result["canonical"][..., 26:29])
                * config.color_noise,
                0.0,
                1.0,
            )
            result["canonical"][..., 26:29] = rgb
            result["canonical"][..., 32:33] = rgb.mean(dim=-1, keepdim=True)
            result["canonical"][..., 33:34] = rgb.amax(dim=-1, keepdim=True) - rgb.amin(
                dim=-1, keepdim=True
            )
    if training and config.gonion_color_dropout > 0:
        # Gonion is geometry/contour defined. Randomly withholding RGB and local
        # colour contrast prevents the network from keying on unilateral shadow.
        drop = torch.rand(len(indices), 2, device=device) < float(
            config.gonion_color_dropout
        )
        result["images"][:, 1:3, :, :6] *= (~drop)[:, :, None, None, None, None]
        if result["canonical"].shape[-1] >= 34:
            result["canonical"][:, 1:3, :, 26:34] *= (~drop)[:, :, None, None]
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
                result["grids"][batch_index, landmark, :, :, 0] += (
                    2.0 * float(shift_x) / max(width - 1, 1)
                )
                result["grids"][batch_index, landmark, :, :, 1] += (
                    2.0 * float(shift_y) / max(height - 1, 1)
                )
        projected = torch.all(torch.abs(result["grids"]) <= 1.0, dim=-1).all(dim=2)
        result["mask"] &= projected
        for batch_index in range(len(indices)):
            for landmark in range(3):
                if not bool(result["mask"][batch_index, landmark].any()):
                    extent = torch.abs(result["grids"][batch_index, landmark]).amax(
                        dim=(0, 2)
                    )
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
    loss = (torch.where(difference < theta, first, second) * weight).mean(dim=(-2, -1))
    if view_mask is None:
        return loss.mean()
    valid = view_mask.float()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def _poss_image_loss(logits, target, view_mask=None, exponent=2.0, temperature=0.1):
    """Position-aware and sample-sensitive heatmap loss (Zhu, ICCV 2025)."""
    prediction = torch.sigmoid(logits.float()).clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    modulation = torch.abs((target - prediction) / max(float(temperature), 1e-4)).pow(
        float(exponent)
    )
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


def _distance_distribution(logits, distance, mask, sigma, temperature=1.0):
    safe_distance = torch.where(mask, distance.float(), torch.zeros_like(distance))
    target_energy = -(safe_distance**2) / (2.0 * max(float(sigma), 1e-4) ** 2)
    target_energy = target_energy.masked_fill(~mask, -torch.inf)
    target = torch.softmax(target_energy, dim=-1)
    log_probability = torch.log_softmax(
        logits.float().masked_fill(~mask, -torch.inf) / max(float(temperature), 1e-4),
        dim=-1,
    ).masked_fill(~mask, 0.0)
    listwise = (
        target * (torch.log(target.clamp_min(1e-8)) - log_probability)
    ).masked_fill(~mask, 0.0)
    listwise = listwise.sum(dim=-1).mean()
    probability = torch.softmax(
        logits.float().masked_fill(~mask, -torch.inf) / max(float(temperature), 1e-4),
        dim=-1,
    )
    expected_distance = (probability * safe_distance).sum(dim=-1).mean()
    return listwise, expected_distance, probability


def _ordinal_distance_loss(
    logits,
    distance,
    mask,
    negative_radius_mm,
    negative_count,
    margin,
):
    minimum = distance.float().masked_fill(~mask, torch.inf).amin(dim=-1, keepdim=True)
    positive_mask = mask & (distance <= minimum + 1.0)
    positive_logits = logits.float().masked_fill(~positive_mask, -torch.inf)
    positive = torch.logsumexp(positive_logits, dim=-1, keepdim=True)
    positive = positive - torch.log(
        positive_mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
    )
    negative_mask = mask & (distance > minimum + float(negative_radius_mm))
    negatives = logits.float().masked_fill(~negative_mask, -torch.inf)
    count = min(max(1, int(negative_count)), logits.shape[-1])
    values, indices = torch.topk(negatives, count, dim=-1)
    valid = torch.gather(negative_mask, -1, indices)
    loss = F.softplus(values - positive + float(margin))
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return (loss.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)).mean()


def _teacher_force_probability(epoch, warmup_epochs):
    warmup = max(int(warmup_epochs), 0)
    if warmup == 0 or epoch is None:
        return 0.0
    return max(0.0, 1.0 - (max(int(epoch), 1) - 1) / max(warmup, 1))


def _uses_unary_reranker(config):
    if config.decoder_mode not in (
        "contour_coordinate",
        "full_pair",
        "sharp_pruned",
    ):
        raise ValueError(
            "decoder_mode must be contour_coordinate, full_pair or sharp_pruned"
        )
    return config.decoder_mode == "sharp_pruned"


def _median_best_epoch(best_epochs, maximum_epochs):
    if not best_epochs:
        raise ValueError("best_epochs cannot be empty")
    return int(np.clip(round(np.median(best_epochs)), 1, int(maximum_epochs)))


def _forward_model(
    model,
    batch,
    config,
    training=False,
    teacher_force_probability=0.0,
    compute_rerank=None,
    compute_pair=True,
):
    heatmaps, view_weights = model.forward_with_context(batch["images"])
    evidence = model.candidate_logits(
        heatmaps,
        batch["grids"],
        batch["mask"],
        view_weights,
        canonical=batch["canonical"],
        neighbor_index=batch["neighbor_index"],
        neighbor_mask=batch["neighbor_mask"],
        return_evidence=True,
    )
    proposal_logits = evidence["logits"]
    if compute_rerank is None:
        compute_rerank = _uses_unary_reranker(config)
    rerank = None
    candidate_logits = proposal_logits
    if compute_rerank:
        rerank = model.rerank_gonion(
            proposal_logits,
            batch["canonical"],
            batch["mask"],
            evidence["proposal_sources"],
        )
        candidate_logits = rerank["logits"]
    pair = None
    if compute_pair:
        pair = model.gonion_pair(
            candidate_logits,
            batch["canonical"],
            batch["points"],
            batch["mask"],
            config.pair_temperature,
            batch["distance"] if training else None,
            evidence["proposal_sources"],
            teacher_force_probability,
            batch.get("shape_context"),
            batch.get("base_gonion"),
        )
    return (
        heatmaps,
        candidate_logits,
        pair,
        view_weights,
        {
            "proposal_logits": proposal_logits,
            "proposal_sources": evidence["proposal_sources"],
            "rerank": rerank,
        },
    )


def _loss(heatmaps, candidate_logits, batch, config, pair_output=None):
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
    ).masked_fill(~mask, 0.0)
    ranking = (
        (
            target_probability
            * (torch.log(target_probability.clamp_min(1e-8)) - log_probability)
        )
        .masked_fill(~mask, 0.0)
        .sum(dim=-1)
        .mean()
    )

    unary_coordinate = _weighted_coordinate(candidate_logits, batch["points"], mask)
    coordinate = unary_coordinate
    pair_ranking = candidate_logits.new_zeros((), dtype=torch.float32)
    pair_negative = candidate_logits.new_zeros((), dtype=torch.float32)
    pair_expected = candidate_logits.new_zeros((), dtype=torch.float32)
    if pair_output is not None:
        coordinate = torch.cat(
            [unary_coordinate[:, 0:1], pair_output["soft_coordinate"]], dim=1
        )
        left_distance = torch.gather(distance[:, 1], 1, pair_output["left_indices"])
        right_distance = torch.gather(distance[:, 2], 1, pair_output["right_indices"])
        pair_energy = -(
            left_distance[:, :, None] ** 2 + right_distance[:, None, :] ** 2
        ) / (2.0 * max(float(config.pair_target_sigma), 1e-4) ** 2)
        pair_energy = pair_energy.masked_fill(~pair_output["mask"], -torch.inf)
        pair_target = torch.softmax(pair_energy.flatten(1), dim=-1).reshape_as(
            pair_energy
        )
        pair_log_probability = torch.log_softmax(
            pair_output["logits"].float().flatten(1), dim=-1
        ).reshape_as(pair_energy)
        pair_log_probability = pair_log_probability.masked_fill(
            ~pair_output["mask"], 0.0
        )
        pair_ranking = (
            pair_target
            * (torch.log(pair_target.clamp_min(1e-8)) - pair_log_probability)
        ).masked_fill(~pair_output["mask"], 0.0)
        pair_ranking = pair_ranking.sum(dim=(1, 2)).mean()
        pair_distance = 0.5 * (left_distance[:, :, None] + right_distance[:, None, :])
        pair_expected = (
            pair_output["probability"]
            * pair_distance.masked_fill(~pair_output["mask"], 0.0)
        ).sum(dim=(1, 2)).mean() / max(float(config.distance_normalizer_mm), 1e-4)
        pair_minimum = pair_distance.masked_fill(~pair_output["mask"], torch.inf).amin(
            dim=(1, 2), keepdim=True
        )
        pair_negative_mask = pair_output["mask"] & (
            pair_distance > pair_minimum + float(config.pair_target_sigma)
        )
        flattened_negative = (
            pair_output["logits"]
            .float()
            .flatten(1)
            .masked_fill(~pair_negative_mask.flatten(1), -torch.inf)
        )
        pair_negative_count = min(16, flattened_negative.shape[-1])
        pair_negative_logits, pair_negative_indices = torch.topk(
            flattened_negative, pair_negative_count, dim=-1
        )
        pair_negative_valid = torch.gather(
            pair_negative_mask.flatten(1), 1, pair_negative_indices
        )
        pair_positive = (
            pair_target
            * pair_output["logits"].float().masked_fill(~pair_output["mask"], 0.0)
        ).sum(dim=(1, 2), keepdim=False)[:, None]
        pair_margin = F.softplus(
            pair_negative_logits - pair_positive + config.negative_margin
        )
        pair_margin = torch.where(
            pair_negative_valid, pair_margin, torch.zeros_like(pair_margin)
        )
        pair_negative = (
            pair_margin.sum(dim=-1) / pair_negative_valid.sum(dim=-1).clamp_min(1)
        ).mean()
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
    positive = (
        target_probability * candidate_logits.float().masked_fill(~mask, 0.0)
    ).sum(dim=-1, keepdim=True)
    margin = F.softplus(negatives - positive + config.negative_margin)
    margin = torch.where(negative_valid, margin, torch.zeros_like(margin))
    margin = (margin.sum(dim=-1) / negative_valid.sum(dim=-1).clamp_min(1)).mean()
    total = (
        config.heatmap_weight * heatmap
        + config.poss_weight * poss
        + config.ranking_weight * ranking
        + config.coordinate_weight * coordinate_loss
        + config.pair_weight * pair
        + config.joint_pair_weight * pair_ranking
        + config.joint_pair_negative_weight * pair_negative
        + config.pair_expected_distance_weight * pair_expected
        + config.negative_weight * margin
    )
    return total, {
        "heatmap": heatmap,
        "poss": poss,
        "ranking": ranking,
        "coordinate": coordinate_loss,
        "pair": pair,
        "joint_pair_ranking": pair_ranking,
        "joint_pair_negative": pair_negative,
        "pair_expected_distance": pair_expected,
        "negative": margin,
    }


def _sharp_rerank_loss(rerank_output, batch, config):
    indices = rerank_output["proposal_indices"]
    mask = rerank_output["proposal_mask"]
    logits = rerank_output["shortlist_logits"]
    distance = torch.gather(batch["distance"][:, 1:3], 2, indices)
    points = torch.gather(
        batch["points"][:, 1:3],
        2,
        indices[..., None].expand(-1, -1, -1, 3),
    )
    listwise, expected, probability = _distance_distribution(
        logits,
        distance,
        mask,
        config.rerank_sigma,
        config.rerank_temperature,
    )
    coordinate = (probability[..., None] * points).sum(dim=2)
    coordinate_loss = F.smooth_l1_loss(
        coordinate, batch["expert"][:, 1:3].float(), beta=1.0
    )
    ordinal = _ordinal_distance_loss(
        logits,
        distance,
        mask,
        config.rerank_negative_radius_mm,
        config.rerank_negative_count,
        config.negative_margin,
    )
    normalized_expected = expected / max(float(config.distance_normalizer_mm), 1e-4)
    total = (
        config.rerank_listwise_weight * listwise
        + config.rerank_expected_distance_weight * normalized_expected
        + config.rerank_coordinate_weight * coordinate_loss
        + config.rerank_ordinal_weight * ordinal
    )
    return total, {
        "rerank_listwise": listwise,
        "rerank_expected_distance": normalized_expected,
        "rerank_coordinate": coordinate_loss,
        "rerank_ordinal": ordinal,
    }


def _adaptive_clinical_positive(distance, mask, radius_mm):
    fixed = mask & (distance <= float(radius_mm))
    minimum = distance.masked_fill(~mask, torch.inf).amin(dim=-1, keepdim=True)
    fallback = mask & (distance <= minimum + 0.5)
    return torch.where(fixed.any(dim=-1, keepdim=True), fixed, fallback)


def _clinical_full_pair_loss(pair_output, batch, config):
    left_distance = torch.gather(
        batch["distance"][:, 1], 1, pair_output["left_indices"]
    ).float()
    right_distance = torch.gather(
        batch["distance"][:, 2], 1, pair_output["right_indices"]
    ).float()
    distance = torch.stack([left_distance, right_distance], dim=1)
    unary_mask = torch.stack(
        [
            torch.gather(batch["mask"][:, 1], 1, pair_output["left_indices"]),
            torch.gather(batch["mask"][:, 2], 1, pair_output["right_indices"]),
        ],
        dim=1,
    )
    unary_listwise, _, _ = _distance_distribution(
        pair_output["unary_logits"],
        distance,
        unary_mask,
        config.pair_target_sigma,
        config.pair_temperature,
    )

    pair_mask = pair_output["mask"]
    pair_distance = 0.5 * (left_distance[:, :, None] + right_distance[:, None, :])
    target_energy = -(
        left_distance[:, :, None].square() + right_distance[:, None, :].square()
    ) / (2.0 * max(float(config.pair_target_sigma), 1e-4) ** 2)
    target_energy = target_energy.masked_fill(~pair_mask, -torch.inf)
    target = torch.softmax(target_energy.flatten(1), dim=-1).reshape_as(target_energy)
    probability = pair_output["probability"]
    pair_listwise = (
        target
        * (torch.log(target.clamp_min(1e-8)) - torch.log(probability.clamp_min(1e-8)))
    ).masked_fill(~pair_mask, 0.0)
    pair_listwise = pair_listwise.sum(dim=(1, 2)).mean()

    left_positive = _adaptive_clinical_positive(
        left_distance,
        unary_mask[:, 0],
        config.pair_clinical_radius_mm,
    )
    right_positive = _adaptive_clinical_positive(
        right_distance,
        unary_mask[:, 1],
        config.pair_clinical_radius_mm,
    )
    positive_pair = left_positive[:, :, None] & right_positive[:, None, :]
    positive_mass = (probability * positive_pair).sum(dim=(1, 2)).clamp_min(1e-8)
    clinical_mass = -torch.log(positive_mass).mean()

    expected_distance = (probability * pair_distance.masked_fill(~pair_mask, 0.0)).sum(
        dim=(1, 2)
    ).mean() / max(float(config.distance_normalizer_mm), 1e-4)

    pair_logits = pair_output["logits"].float()
    positive_logits = pair_logits.masked_fill(~positive_pair, -torch.inf)
    positive_score = torch.logsumexp(positive_logits.flatten(1), dim=-1)
    positive_count = positive_pair.flatten(1).sum(dim=-1).clamp_min(1).float()
    positive_score = positive_score - torch.log(positive_count)
    negative_mask = pair_mask & (
        pair_distance > float(config.pair_hard_negative_radius_mm)
    )
    flattened_negative = pair_logits.masked_fill(~negative_mask, -torch.inf).flatten(1)
    negative_count = min(
        max(1, int(config.pair_hard_negative_count)), flattened_negative.shape[-1]
    )
    negative_values, negative_indices = torch.topk(
        flattened_negative, negative_count, dim=-1
    )
    negative_valid = torch.gather(negative_mask.flatten(1), 1, negative_indices)
    hard_negative = F.softplus(
        negative_values - positive_score[:, None] + float(config.negative_margin)
    )
    hard_negative = torch.where(
        negative_valid, hard_negative, torch.zeros_like(hard_negative)
    )
    hard_negative = (
        hard_negative.sum(dim=-1) / negative_valid.sum(dim=-1).clamp_min(1)
    ).mean()

    total = (
        config.pair_listwise_weight * pair_listwise
        + config.pair_unary_listwise_weight * unary_listwise
        + config.pair_clinical_mass_weight * clinical_mass
        + config.pair_expected_distance_weight * expected_distance
        + config.pair_hard_negative_weight * hard_negative
    )
    return total, {
        "pair_listwise": pair_listwise,
        "pair_unary_listwise": unary_listwise,
        "pair_clinical_mass": clinical_mass,
        "pair_expected_distance": expected_distance,
        "pair_hard_negative": hard_negative,
    }


def _contour_coordinate_loss(pair_output, batch, config):
    """Supervise a joint mean/asymmetry state in canonical jaw coordinates."""
    clinical, components = _clinical_full_pair_loss(pair_output, batch, config)
    left_distance = torch.gather(
        batch["distance"][:, 1], 1, pair_output["left_indices"]
    ).float()
    right_distance = torch.gather(
        batch["distance"][:, 2], 1, pair_output["right_indices"]
    ).float()
    if "expert_gonion_context" in batch:
        left_target = batch["expert_gonion_context"][:, 0].float()
        right_target = batch["expert_gonion_context"][:, 1].float()
    else:
        left_target_index = left_distance.argmin(dim=-1)
        right_target_index = right_distance.argmin(dim=-1)
        left_target = torch.gather(
            pair_output["left_global"],
            1,
            left_target_index[:, None, None].expand(-1, 1, 3),
        ).squeeze(1)
        right_target = torch.gather(
            pair_output["right_global"],
            1,
            right_target_index[:, None, None].expand(-1, 1, 3),
        ).squeeze(1)
    target_mean = 0.5 * (left_target + right_target)
    target_asymmetry = 0.5 * (left_target - right_target)
    scale = max(float(config.contour_state_scale), 1e-4)
    state = F.smooth_l1_loss(
        pair_output["predicted_mean"] / scale,
        target_mean / scale,
    ) + F.smooth_l1_loss(
        pair_output["predicted_asymmetry"] / scale,
        target_asymmetry / scale,
    )

    probability = pair_output["probability"]
    left_weight = probability.sum(dim=2)
    right_weight = probability.sum(dim=1)
    expected_left = (left_weight[..., None] * pair_output["left_global"]).sum(dim=1)
    expected_right = (right_weight[..., None] * pair_output["right_global"]).sum(dim=1)
    expected_mean = 0.5 * (expected_left + expected_right)
    expected_asymmetry = 0.5 * (expected_left - expected_right)
    moment = F.smooth_l1_loss(
        expected_mean / scale,
        target_mean / scale,
    ) + F.smooth_l1_loss(
        expected_asymmetry / scale,
        target_asymmetry / scale,
    )
    total = (
        clinical
        + float(config.contour_state_weight) * state
        + float(config.contour_moment_weight) * moment
    )
    return total, {
        **components,
        "contour_state": state,
        "contour_moment": moment,
    }


def _new_model(candidate_set, config):
    return DualViewHard3Net(
        candidate_set.images.shape[3],
        config.width,
        config.dropout,
        geometry_dim=candidate_set.canonical.shape[-1],
        proposal_topk=config.proposal_topk,
        pair_topk=config.pair_topk,
        enable_unary_reranker=_uses_unary_reranker(config),
        decoder_mode=config.decoder_mode,
        shape_context_dim=(
            candidate_set.shape_context.shape[-1]
            if candidate_set.shape_context is not None
            else 69
        ),
        contour_residual_limit=config.contour_residual_limit,
    )


@torch.no_grad()
def _predict_outputs(model, candidate_set, indices, config, device):
    model.eval()
    chunks = {
        "logits": [],
        "proposal_logits": [],
        "proposal_sources": [],
        "pair_soft": [],
        "pair_argmax": [],
        "pair_snapped": [],
        "pair_left_indices": [],
        "pair_right_indices": [],
        "view_weights": [],
    }
    for start in range(0, len(indices), config.batch_size):
        selected = np.asarray(
            indices[start : start + config.batch_size], dtype=np.int64
        )
        batch = _tensor_batch(candidate_set, selected, device, config)
        _, logits, pair, view_weights, evidence = _forward_model(model, batch, config)
        chunks["logits"].append(logits.float().cpu().numpy())
        chunks["proposal_logits"].append(
            evidence["proposal_logits"].float().cpu().numpy()
        )
        chunks["proposal_sources"].append(
            evidence["proposal_sources"].float().cpu().numpy()
        )
        chunks["pair_soft"].append(pair["soft_coordinate"].float().cpu().numpy())
        chunks["pair_argmax"].append(pair["argmax_coordinate"].float().cpu().numpy())
        chunks["pair_snapped"].append(pair["snapped_coordinate"].float().cpu().numpy())
        chunks["pair_left_indices"].append(pair["left_indices"].cpu().numpy())
        chunks["pair_right_indices"].append(pair["right_indices"].cpu().numpy())
        chunks["view_weights"].append(view_weights.float().cpu().numpy())
        for name in (
            "predicted_mean",
            "predicted_asymmetry",
            "base_mean",
            "base_asymmetry",
        ):
            if name in pair:
                chunks.setdefault(name, []).append(pair[name].float().cpu().numpy())
    return {
        name: np.concatenate(values, axis=0)
        for name, values in chunks.items()
        if values
    }


def _predict_logits(model, candidate_set, indices, config, device):
    """Compatibility wrapper retained for downstream diagnostic imports."""
    return _predict_outputs(model, candidate_set, indices, config, device)["logits"]


def _proposal_indices(source_logits, mask, topk):
    with torch.no_grad():
        return (
            diverse_topk_indices(
                torch.from_numpy(np.asarray(source_logits, dtype=np.float32)),
                torch.from_numpy(np.asarray(mask, dtype=np.bool_)),
                topk,
            )
            .cpu()
            .numpy()
        )


def _learned_proposal_indices(source_logits, mask, topk):
    values = np.where(
        np.asarray(mask, dtype=np.bool_),
        np.asarray(source_logits, dtype=np.float32)[:, 0],
        -np.inf,
    )
    count = min(max(1, int(topk)), values.shape[-1])
    return np.argsort(values, axis=-1)[:, -count:]


def _shortlist_metrics(distances, selected, nearest):
    hits = np.any(selected == nearest[..., None], axis=-1)
    shortlisted_distance = np.take_along_axis(distances, selected, axis=-1).min(axis=-1)
    return {
        "lm21_recall": float(hits[:, 0].mean()),
        "lm22_recall": float(hits[:, 1].mean()),
        "both_recall": float(np.all(hits, axis=1).mean()),
        "either_missed_fraction": float((~np.all(hits, axis=1)).mean()),
        "lm21_oracle_ale": float(shortlisted_distance[:, 0].mean()),
        "lm22_oracle_ale": float(shortlisted_distance[:, 1].mean()),
        "gonion_oracle_ale": float(shortlisted_distance.mean()),
        "gonion_oracle_p95": float(np.percentile(shortlisted_distance, 95)),
        "gonion_oracle_sdr_at_2mm": float((shortlisted_distance <= 2.0).mean()),
        "gonion_oracle_sdr_at_3mm": float((shortlisted_distance <= 3.0).mean()),
    }


def _proposal_diagnostics(candidate_set, proposal_sources, requested_topk):
    distances = np.asarray(candidate_set.target_distance[:, 1:3], dtype=np.float32)
    masks = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
    sources = np.asarray(proposal_sources[:, 1:3], dtype=np.float32)
    nearest = np.argmin(distances, axis=-1)
    candidate_count = distances.shape[-1]
    topk_values = sorted(
        {min(max(1, int(value)), candidate_count) for value in requested_topk}
    )
    at_k, diverse_at_k = {}, {}
    for topk in topk_values:
        learned = np.stack(
            [
                _learned_proposal_indices(
                    sources[:, landmark], masks[:, landmark], topk
                )
                for landmark in range(2)
            ],
            axis=1,
        )
        diverse = np.stack(
            [
                _proposal_indices(sources[:, landmark], masks[:, landmark], topk)
                for landmark in range(2)
            ],
            axis=1,
        )
        at_k[str(topk)] = _shortlist_metrics(distances, learned, nearest)
        diverse_at_k[str(topk)] = _shortlist_metrics(distances, diverse, nearest)

    source_recall = {}
    diagnostic_k = topk_values[-1]
    for source_index, source_name in enumerate(PROPOSAL_SOURCE_NAMES):
        side_hits = []
        for landmark in range(2):
            values = np.where(
                masks[:, landmark], sources[:, landmark, source_index], -np.inf
            )
            count = min(diagnostic_k, values.shape[-1])
            selected = np.argsort(values, axis=-1)[:, -count:]
            side_hits.append(np.any(selected == nearest[:, landmark, None], axis=-1))
        side_hits = np.stack(side_hits, axis=1)
        source_recall[source_name] = {
            "lm21": float(side_hits[:, 0].mean()),
            "lm22": float(side_hits[:, 1].mean()),
            "both": float(np.all(side_hits, axis=1).mean()),
        }
    return {
        "source_names": list(PROPOSAL_SOURCE_NAMES),
        "candidate_count": int(candidate_count),
        "selection": "learned_surface_context_score",
        "at_k": at_k,
        "diverse_rank_union_at_k": diverse_at_k,
        "source_recall_at_largest_k": {
            "topk": int(diagnostic_k),
            "sources": source_recall,
        },
    }


def _select_dual_coordinate_policy(candidate_set, logits, pair_outputs):
    policy = _select_coordinate_policy(candidate_set, logits)
    unary = _decode_policy(candidate_set, logits, policy)[:, 1:3]
    options = {"unary": unary}
    options.update(
        {
            name: np.asarray(pair_outputs[name], dtype=np.float32)
            for name in ("pair_soft", "pair_argmax", "pair_snapped")
        }
    )
    rows = []
    for name, coordinate in options.items():
        error = np.linalg.norm(coordinate - candidate_set.expert[:, 1:3], axis=-1)
        rows.append(
            {
                "mode": name,
                "gonion_ale": float(error.mean()),
                "lm21_ale": float(error[:, 0].mean()),
                "lm22_ale": float(error[:, 1].mean()),
            }
        )
    selected = min(rows, key=lambda row: (row["gonion_ale"], row["mode"]))
    policy["gonion_pair"] = {**selected, "sweep": rows}
    return policy


def _decode_dual_policy(candidate_set, logits, pair_outputs, policy):
    unary = _decode_policy(candidate_set, logits, policy)
    mode = policy.get("gonion_pair", {}).get("mode", "unary")
    if mode != "unary":
        unary[:, 1:3] = np.asarray(pair_outputs[mode], dtype=np.float32)
    return unary


def _set_training_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "proposal":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        if model.gonion_unary_reranker is not None:
            for parameter in model.gonion_unary_reranker.parameters():
                parameter.requires_grad_(False)
        for parameter in model.gonion_pair_ranker.parameters():
            parameter.requires_grad_(False)
        model.train()
        if model.gonion_unary_reranker is not None:
            model.gonion_unary_reranker.eval()
        model.gonion_pair_ranker.eval()
    elif stage == "rerank":
        if model.gonion_unary_reranker is None:
            raise RuntimeError("Unary reranker stage requested while it is disabled")
        for parameter in model.gonion_unary_reranker.parameters():
            parameter.requires_grad_(True)
        model.eval()
        model.gonion_unary_reranker.train()
    elif stage == "pair":
        for parameter in model.gonion_pair_ranker.parameters():
            parameter.requires_grad_(True)
        model.eval()
        model.gonion_pair_ranker.train()
    else:
        raise ValueError(f"Unknown Hard3 training stage: {stage}")


def _validation_stage_metrics(model, candidate_set, val_indices, config, device, stage):
    outputs = _predict_outputs(model, candidate_set, list(val_indices), config, device)
    logits = outputs["proposal_logits"] if stage == "proposal" else outputs["logits"]
    unary = _decode_numpy(
        logits,
        candidate_set.points[val_indices],
        candidate_set.mask[val_indices],
        5,
        0.5,
        True,
    )
    prediction = unary.copy()
    if stage == "pair":
        if config.pair_validation_mode not in (
            "pair_soft",
            "pair_argmax",
            "pair_snapped",
        ):
            raise ValueError(
                "pair_validation_mode must be pair_soft, pair_argmax, or pair_snapped"
            )
        prediction[:, 1:3] = outputs[config.pair_validation_mode]
    error = np.linalg.norm(prediction - candidate_set.expert[val_indices], axis=-1)
    return outputs, error


def _pair_stage_loss(heatmaps, logits, batch, config, pair_output):
    if config.decoder_mode == "contour_coordinate":
        return _contour_coordinate_loss(pair_output, batch, config)
    if config.decoder_mode == "full_pair":
        return _clinical_full_pair_loss(pair_output, batch, config)
    _, components = _loss(heatmaps, logits, batch, config, pair_output)
    total = (
        config.coordinate_weight * components["coordinate"]
        + config.pair_weight * components["pair"]
        + config.joint_pair_weight * components["joint_pair_ranking"]
        + config.joint_pair_negative_weight * components["joint_pair_negative"]
        + config.pair_expected_distance_weight * components["pair_expected_distance"]
    )
    return total, components


def _fit_stage(
    model,
    candidate_set,
    train_indices,
    val_indices,
    config,
    device,
    fold_number,
    stage,
):
    is_pair = stage == "pair"
    is_rerank = stage == "rerank"
    if is_pair:
        epochs, min_epochs = config.pair_stage_epochs, config.pair_stage_min_epochs
        patience, lr = config.pair_stage_patience, config.pair_stage_lr
    elif is_rerank:
        epochs, min_epochs = (
            config.rerank_stage_epochs,
            config.rerank_stage_min_epochs,
        )
        patience, lr = config.rerank_stage_patience, config.rerank_stage_lr
    else:
        epochs, min_epochs = config.epochs, config.min_epochs
        patience, lr = config.patience, config.lr
    _set_training_stage(model, stage)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    offset = 40_009 if is_pair else (20_003 if is_rerank else 0)
    rng = np.random.default_rng(config.seed + fold_number * 7919 + offset)
    best_score, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, int(epochs) + 1):
        _set_training_stage(model, stage)
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
                "joint_pair_ranking",
                "joint_pair_negative",
                "pair_expected_distance",
                "pair_listwise",
                "pair_unary_listwise",
                "pair_clinical_mass",
                "pair_hard_negative",
                "negative",
                "rerank_listwise",
                "rerank_expected_distance",
                "rerank_coordinate",
                "rerank_ordinal",
            )
        }
        seen = 0
        for start in range(0, len(order), config.batch_size):
            selected = np.asarray(
                order[start : start + config.batch_size], dtype=np.int64
            )
            batch = _tensor_batch(
                candidate_set,
                selected,
                device,
                config,
                training=stage == "proposal",
                rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            heatmaps, logits, pair_output, _, evidence = _forward_model(
                model,
                batch,
                config,
                training=False,
                teacher_force_probability=0.0,
                compute_rerank=(_uses_unary_reranker(config) and stage != "proposal"),
                compute_pair=is_pair,
            )
            if is_pair:
                loss, components = _pair_stage_loss(
                    heatmaps, logits, batch, config, pair_output
                )
            elif is_rerank:
                loss, components = _sharp_rerank_loss(evidence["rerank"], batch, config)
            else:
                loss, components = _loss(heatmaps, logits, batch, config, None)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite dual-view Hard3 {stage} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()
            count = len(selected)
            seen += count
            totals["total"] += float(loss.detach()) * count
            for name, value in components.items():
                totals.setdefault(name, 0.0)
                totals[name] += float(value.detach()) * count
        _, val_error = _validation_stage_metrics(
            model, candidate_set, val_indices, config, device, stage
        )
        score = float(val_error[:, 1:3].mean()) if is_pair else float(val_error.mean())
        hard3_score = float(val_error.mean())
        scheduler.step(score)
        row = {
            "stage": stage,
            "epoch": epoch,
            "validation_selection_ale": score,
            "validation_hard3_ale": hard3_score,
            "validation_lm0_ale": float(val_error[:, 0].mean()),
            "validation_gonion_ale": float(val_error[:, 1:3].mean()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{name}": value / max(seen, 1) for name, value in totals.items()},
        }
        history.append(row)
        if score < best_score - 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        elif epoch >= min_epochs:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Hard3 {stage} OOF fold {fold_number} epoch {epoch:03d}/{epochs} "
                f"train={row['train_total']:.4f} val={score:.4f}",
                flush=True,
            )
        if epoch >= min_epochs and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"Hard3 {stage} stage did not produce a checkpoint")
    model.load_state_dict(best_state)
    return best_epoch, best_score, history, best_state


def _train_model(
    candidate_set, train_indices, val_indices, config, device, fold_number
):
    torch.manual_seed(config.seed + fold_number * 1009)
    model = _new_model(candidate_set, config).to(device)
    proposal_epoch, proposal_score, proposal_history, proposal_state = _fit_stage(
        model,
        candidate_set,
        train_indices,
        val_indices,
        config,
        device,
        fold_number,
        "proposal",
    )
    model.load_state_dict(proposal_state)
    if _uses_unary_reranker(config):
        rerank_epoch, rerank_score, rerank_history, rerank_state = _fit_stage(
            model,
            candidate_set,
            train_indices,
            val_indices,
            config,
            device,
            fold_number,
            "rerank",
        )
    else:
        rerank_epoch, rerank_score, rerank_history = 0, proposal_score, []
        rerank_state = proposal_state
    model.load_state_dict(rerank_state)
    pair_epoch, pair_score, pair_history, pair_state = _fit_stage(
        model,
        candidate_set,
        train_indices,
        val_indices,
        config,
        device,
        fold_number,
        "pair",
    )
    model.load_state_dict(pair_state)
    outputs = _predict_outputs(model, candidate_set, list(val_indices), config, device)
    stage_epochs = {
        "proposal": proposal_epoch,
        "rerank": rerank_epoch,
        "pair": pair_epoch,
    }
    stage_scores = {
        "proposal": proposal_score,
        "rerank": rerank_score,
        "pair": pair_score,
    }
    return (
        outputs,
        stage_epochs,
        stage_scores,
        proposal_history + rerank_history + pair_history,
        {key: value.detach().cpu() for key, value in pair_state.items()},
    )


def _train_fixed_model(
    candidate_set,
    proposal_epochs,
    rerank_epochs,
    pair_epochs,
    config,
    device,
    member_number,
):
    torch.manual_seed(config.seed + 50_003 + member_number * 1009)
    model = _new_model(candidate_set, config).to(device)
    rng = np.random.default_rng(config.seed + 70_001 + member_number * 7919)
    indices = np.arange(len(candidate_set), dtype=np.int64)
    history = []
    stages = [("proposal", int(proposal_epochs), config.lr)]
    if _uses_unary_reranker(config):
        stages.append(("rerank", int(rerank_epochs), config.rerank_stage_lr))
    stages.append(("pair", int(pair_epochs), config.pair_stage_lr))
    for stage, epochs, lr in stages:
        _set_training_stage(model, stage)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=lr, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=1e-6
        )
        for epoch in range(1, epochs + 1):
            _set_training_stage(model, stage)
            order = rng.permutation(indices)
            total, seen = 0.0, 0
            for start in range(0, len(order), config.batch_size):
                selected = np.asarray(
                    order[start : start + config.batch_size], dtype=np.int64
                )
                batch = _tensor_batch(
                    candidate_set,
                    selected,
                    device,
                    config,
                    training=stage == "proposal",
                    rng=rng,
                )
                optimizer.zero_grad(set_to_none=True)
                heatmaps, logits, pair_output, _, evidence = _forward_model(
                    model,
                    batch,
                    config,
                    compute_rerank=(
                        _uses_unary_reranker(config) and stage != "proposal"
                    ),
                    compute_pair=stage == "pair",
                )
                if stage == "pair":
                    loss, _ = _pair_stage_loss(
                        heatmaps, logits, batch, config, pair_output
                    )
                elif stage == "rerank":
                    loss, _ = _sharp_rerank_loss(evidence["rerank"], batch, config)
                else:
                    loss, _ = _loss(heatmaps, logits, batch, config, None)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite full-train dual-view Hard3 {stage} loss"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
                optimizer.step()
                total += float(loss.detach()) * len(selected)
                seen += len(selected)
            scheduler.step()
            history.append(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "train_loss": total / max(seen, 1),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(
                    f"Hard3 final member {member_number} {stage} "
                    f"epoch {epoch:03d}/{epochs} train={history[-1]['train_loss']:.4f}",
                    flush=True,
                )
    return (
        model,
        history,
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )


def _cache_signature(dataset, config, centers_by_id=None):
    ignored = {
        "bootstrap_iters",
        "minimum_overall_gain_mm",
        "minimum_hard3_gain_mm",
        "minimum_improvement_probability",
        "maximum_p95_regression_mm",
        "target_hard3_ale",
        "minimum_proposal_recall",
        "maximum_proposal_oracle_ale",
        "minimum_proposal_sdr2",
        "maximum_proposal_oracle_p95",
    }
    is_contour = config.decoder_mode == "contour_coordinate"
    if not is_contour:
        ignored.update(
            {
                "contour_state_weight",
                "contour_moment_weight",
                "contour_state_scale",
                "contour_residual_limit",
            }
        )
    model_config = {
        key: value for key, value in asdict(config).items() if key not in ignored
    }
    digest = hashlib.sha256()
    records = []
    for sample in dataset.samples:
        digest.update(sample.sample_id.encode("utf-8"))
        center = (
            centers_by_id[sample.sample_id]
            if centers_by_id is not None
            else dataset._coarse(sample)
        )
        digest.update(np.asarray(center, dtype=np.float32).tobytes())
        path = Path(dataset.records[sample.sample_id])
        stat = path.stat()
        records.append(
            (sample.sample_id, path.name, int(stat.st_size), int(stat.st_mtime_ns))
        )
    payload = {
        "version": 13 if is_contour else 11,
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
    missing = [
        sample_id for sample_id in candidate_set.sample_ids if sample_id not in by_id
    ]
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
            scale = max(
                float(np.percentile(values, 75) - np.percentile(values, 25)), 1e-3
            )
            output[sample, landmark, valid] = np.clip(
                (values - median) / scale, -12.0, 12.0
            )
            output[sample, landmark, ~valid] = -np.inf
    return output


def _variant_predictions(
    candidate_set, logits, policy, atlas_prediction, joint_predictions=None
):
    unary_policy = _decode_policy(candidate_set, logits, policy)
    neural_policy = unary_policy.copy()
    if joint_predictions is not None:
        mode = policy.get("gonion_pair", {}).get("mode", "unary")
        if mode != "unary":
            neural_policy[:, 1:3] = joint_predictions[mode]
    variants = {
        "neural_policy": neural_policy,
        "unary_policy": unary_policy,
        "neural_argmax": _decode_numpy(
            logits, candidate_set.points, candidate_set.mask, 1, 1.0, True
        ),
        "atlas_direct": np.asarray(atlas_prediction, dtype=np.float32),
    }
    if joint_predictions is not None:
        for name in ("pair_soft", "pair_argmax", "pair_snapped"):
            coordinate = unary_policy.copy()
            coordinate[:, 1:3] = joint_predictions[name]
            variants[f"joint_{name.removeprefix('pair_')}"] = coordinate
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
            variants[f"{prefix}_policy"] = _decode_policy(candidate_set, fused, policy)
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
            sample_id: np.asarray(
                baseline_outputs["prediction"][index], dtype=np.float32
            )
            for index, sample_id in enumerate(baseline_outputs["sample_ids"])
        }
        candidate_set = extract_dual_view_set(
            dataset,
            self.config.image_size,
            self.config.radius_scale,
            centers,
            label,
            neighbor_count=self.config.proposal_neighbors,
            include_contour_features=(self.config.decoder_mode == "contour_coordinate"),
        )
        indices = list(range(len(candidate_set)))
        member_outputs = [
            _predict_outputs(model, candidate_set, indices, self.config, self.device)
            for model in self.models
        ]
        member_logits = [values["logits"] for values in member_outputs]
        logits = np.mean(np.stack(member_logits), axis=0)
        joint_predictions = {
            name: np.mean(np.stack([values[name] for values in member_outputs]), axis=0)
            for name in ("pair_soft", "pair_argmax", "pair_snapped")
        }
        baseline = _ordered_baseline(candidate_set, baseline_outputs)
        atlas_result = self.atlas.predict(baseline, candidate_set.sample_ids)
        variants = _variant_predictions(
            candidate_set,
            logits,
            self.policy,
            atlas_result["prediction"],
            joint_predictions,
        )
        member_coordinates = np.stack(
            [
                _decode_dual_policy(
                    candidate_set, values["logits"], values, self.policy
                )
                for values in member_outputs
            ]
        )
        neural_coordinate = variants["neural_policy"]
        spread = np.linalg.norm(
            member_coordinates - neural_coordinate[None], axis=-1
        ).mean(axis=0)
        probability = (
            np.exp(
                logits
                - np.max(
                    np.where(candidate_set.mask, logits, -np.inf),
                    axis=-1,
                    keepdims=True,
                )
            )
            * candidate_set.mask
        )
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
            "mean_view_weights": np.mean(
                np.stack([values["view_weights"] for values in member_outputs]),
                axis=0,
            ).astype(np.float32),
        }


def fit_or_load_dual_view_refiner(
    dataset,
    output_dir,
    config,
    device,
    training_baseline_outputs=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "hard3_dual_view_model.pth"
    report_path = output_dir / "hard3_dual_view_training_report.json"
    training_centers = None
    if training_baseline_outputs is not None:
        training_centers = {
            sample_id: np.asarray(
                training_baseline_outputs["prediction"][index], dtype=np.float32
            )
            for index, sample_id in enumerate(training_baseline_outputs["sample_ids"])
        }
        missing = [
            sample.sample_id
            for sample in dataset.samples
            if sample.sample_id not in training_centers
        ]
        if missing:
            raise KeyError(
                "Training baseline outputs miss Hard3 samples: "
                + ", ".join(missing[:5])
            )
    signature = _cache_signature(dataset, config, training_centers)
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
                model = DualViewHard3Net(
                    checkpoint["input_channels"],
                    config.width,
                    config.dropout,
                    geometry_dim=checkpoint["geometry_dim"],
                    proposal_topk=config.proposal_topk,
                    pair_topk=config.pair_topk,
                    enable_unary_reranker=_uses_unary_reranker(config),
                    decoder_mode=config.decoder_mode,
                    shape_context_dim=checkpoint.get("shape_context_dim", 69),
                    contour_residual_limit=config.contour_residual_limit,
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
        dataset,
        config.image_size,
        config.radius_scale,
        centers_by_id=training_centers,
        label="Hard3 train patches",
        neighbor_count=config.proposal_neighbors,
        include_contour_features=(config.decoder_mode == "contour_coordinate"),
    )
    atlas = TrainOnlyLocalHard3Atlas(
        config.atlas_neighbors, config.atlas_temperature
    ).fit(candidates.expert_full, candidates.sample_ids)
    started = time.time()
    oof_logits = np.full(
        (len(candidates), 3, candidates.points.shape[-2]), -np.inf, dtype=np.float32
    )
    oof_proposal_sources = np.full(
        (len(candidates), 3, len(PROPOSAL_SOURCE_NAMES), candidates.points.shape[-2]),
        -np.inf,
        dtype=np.float32,
    )
    oof_pair = {
        name: np.zeros((len(candidates), 2, 3), dtype=np.float32)
        for name in ("pair_soft", "pair_argmax", "pair_snapped")
    }
    pair_count = min(
        config.pair_topk,
        config.proposal_topk,
        candidates.points.shape[-2],
    )
    oof_pair_indices = {
        name: np.full((len(candidates), pair_count), -1, dtype=np.int64)
        for name in ("pair_left_indices", "pair_right_indices")
    }
    oof_view_weights = np.zeros((len(candidates), 3, 2), dtype=np.float32)
    fold_reports, oof_models, oof_states, best_epochs = [], [], [], []
    for fold_number, (train_indices, val_indices) in enumerate(
        _splitter(candidates.strata, config.folds, config.seed), start=1
    ):
        fold_outputs, stage_epochs, stage_scores, history, state = _train_model(
            candidates,
            np.asarray(train_indices),
            np.asarray(val_indices),
            config,
            device,
            fold_number,
        )
        oof_logits[val_indices] = fold_outputs["logits"]
        oof_proposal_sources[val_indices] = fold_outputs["proposal_sources"]
        for name in oof_pair:
            oof_pair[name][val_indices] = fold_outputs[name]
        for name in oof_pair_indices:
            oof_pair_indices[name][val_indices] = fold_outputs[name]
        oof_view_weights[val_indices] = fold_outputs["view_weights"]
        best_epochs.append(stage_epochs)
        model = _new_model(candidates, config)
        model.load_state_dict(state)
        oof_models.append(model)
        oof_states.append(state)
        fold_reports.append(
            {
                "fold": fold_number,
                "train_sample_ids": [candidates.sample_ids[i] for i in train_indices],
                "validation_sample_ids": [
                    candidates.sample_ids[i] for i in val_indices
                ],
                "best_epochs": stage_epochs,
                "best_validation_pair_selection_ale": stage_scores["pair"],
                "best_validation_proposal_ale": stage_scores["proposal"],
                "best_validation_rerank_ale": stage_scores["rerank"],
                "history": history,
            }
        )
    if not np.isfinite(oof_logits[candidates.mask]).all():
        raise RuntimeError("Dual-view Hard3 OOF logits are incomplete")
    expanded_mask = np.broadcast_to(
        candidates.mask[:, :, None], oof_proposal_sources.shape
    )
    if not np.isfinite(oof_proposal_sources[expanded_mask]).all():
        raise RuntimeError("Dual-view Hard3 OOF proposal sources are incomplete")
    policy = _select_dual_coordinate_policy(candidates, oof_logits, oof_pair)
    oof_prediction = _decode_dual_policy(candidates, oof_logits, oof_pair, policy)
    oof_error = np.linalg.norm(oof_prediction - candidates.expert, axis=-1)
    oof_axis_error = np.abs(oof_prediction - candidates.expert).mean(axis=0)
    member_predictions = np.stack(
        [
            _decode_dual_policy(
                candidates,
                (
                    member_output := _predict_outputs(
                        model.to(device),
                        candidates,
                        list(range(len(candidates))),
                        config,
                        device,
                    )
                )["logits"],
                member_output,
                policy,
            )
            for model in oof_models
        ]
    )
    ensemble_prediction = member_predictions.mean(axis=0)
    spread = np.linalg.norm(
        member_predictions - ensemble_prediction[None], axis=-1
    ).mean(axis=0)
    reliability_scale = np.maximum(np.percentile(spread, 75, axis=0), 0.25)
    median_proposal_epoch = _median_best_epoch(
        [values["proposal"] for values in best_epochs], config.epochs
    )
    median_rerank_epoch = (
        _median_best_epoch(
            [values["rerank"] for values in best_epochs],
            config.rerank_stage_epochs,
        )
        if _uses_unary_reranker(config)
        else 0
    )
    median_pair_epoch = _median_best_epoch(
        [values["pair"] for values in best_epochs], config.pair_stage_epochs
    )
    if config.final_model_policy == "inner_fold_ensemble":
        models = oof_models
        states = oof_states
        fixed_epochs = None
        final_histories = [
            {
                "member": fold["fold"],
                "proposal_epochs": fold["best_epochs"]["proposal"],
                "rerank_epochs": fold["best_epochs"]["rerank"],
                "pair_epochs": fold["best_epochs"]["pair"],
                "source": "inner_fold_best_checkpoint",
            }
            for fold in fold_reports
        ]
        final_selection = (
            "ensemble of inner-fold best checkpoints; no outer-validation labels"
        )
    elif config.final_model_policy == "median_best_refit":
        fixed_epochs = {
            "proposal": median_proposal_epoch,
            "rerank": median_rerank_epoch,
            "pair": median_pair_epoch,
        }
        models, states, final_histories = [], [], []
        for member_number in range(1, max(1, config.final_ensemble_members) + 1):
            model, history, state = _train_fixed_model(
                candidates,
                median_proposal_epoch,
                median_rerank_epoch,
                median_pair_epoch,
                config,
                device,
                member_number,
            )
            models.append(model)
            states.append(state)
            final_histories.append(
                {
                    "member": member_number,
                    "proposal_epochs": median_proposal_epoch,
                    "rerank_epochs": median_rerank_epoch,
                    "pair_epochs": median_pair_epoch,
                    "history": history,
                }
            )
        final_selection = (
            "full-train refit at median inner-fold best epoch; "
            "no outer-validation labels"
        )
    else:
        raise ValueError(
            "final_model_policy must be inner_fold_ensemble or median_best_refit"
        )
    diagnostics = _proposal_diagnostics(
        candidates,
        oof_proposal_sources,
        tuple(config.diagnostic_topk) + (config.proposal_topk, config.pair_topk),
    )
    diagnostic_key = str(min(config.proposal_topk, candidates.points.shape[-2]))
    diagnostic_row = diagnostics["at_k"][diagnostic_key]
    revision = {
        "contour_coordinate": "H3-DVAR-v7",
        "full_pair": "H3-DVAR-v6",
        "sharp_pruned": "H3-DVAR-v5",
    }[config.decoder_mode]
    print(
        f"{revision} broad proposal@{diagnostic_key}: "
        f"LM21={diagnostic_row['lm21_recall']:.3f} "
        f"LM22={diagnostic_row['lm22_recall']:.3f} "
        f"both={diagnostic_row['both_recall']:.3f} "
        f"oracle={diagnostic_row['gonion_oracle_ale']:.3f} mm "
        f"p95={diagnostic_row['gonion_oracle_p95']:.3f} "
        f"SDR2={diagnostic_row['gonion_oracle_sdr_at_2mm']:.3f}",
        flush=True,
    )
    nearest_left = np.argmin(candidates.target_distance[:, 1], axis=-1)
    nearest_right = np.argmin(candidates.target_distance[:, 2], axis=-1)
    left_pair_hit = np.any(
        oof_pair_indices["pair_left_indices"] == nearest_left[:, None], axis=1
    )
    right_pair_hit = np.any(
        oof_pair_indices["pair_right_indices"] == nearest_right[:, None], axis=1
    )
    pair_shortlist_distance = np.stack(
        [
            np.take_along_axis(
                candidates.target_distance[:, 1],
                oof_pair_indices["pair_left_indices"],
                axis=1,
            ).min(axis=1),
            np.take_along_axis(
                candidates.target_distance[:, 2],
                oof_pair_indices["pair_right_indices"],
                axis=1,
            ).min(axis=1),
        ],
        axis=1,
    )
    left_clinical_hit = pair_shortlist_distance[:, 0] <= config.pair_clinical_radius_mm
    right_clinical_hit = pair_shortlist_distance[:, 1] <= config.pair_clinical_radius_mm
    search_name = {
        "contour_coordinate": "shape-conditioned contour search",
        "full_pair": "full-pair search",
        "sharp_pruned": "reranked shortlist",
    }[config.decoder_mode]
    print(
        f"{revision} "
        f"{search_name}"
        f"@{pair_count}: "
        f"LM21={left_pair_hit.mean():.3f} "
        f"LM22={right_pair_hit.mean():.3f} "
        f"both={np.mean(left_pair_hit & right_pair_hit):.3f} "
        f"oracle={pair_shortlist_distance.mean():.3f} mm "
        f"p95={np.percentile(pair_shortlist_distance, 95):.3f} "
        f"SDR2={(pair_shortlist_distance <= 2.0).mean():.3f} "
        f"clinical_both={np.mean(left_clinical_hit & right_clinical_hit):.3f}",
        flush=True,
    )
    parameter_count = sum(parameter.numel() for parameter in models[0].parameters())
    report = {
        "signature": signature,
        "version": revision,
        "method": (
            "nested-OOF broad surface-context proposal, all-23 shape-conditioned "
            "contour-state regression, and non-separable bilateral Gonion decoding"
            if config.decoder_mode == "contour_coordinate"
            else (
                "nested-OOF broad surface-context proposal and recall-preserving "
                "clinical full-pair Gonion decoding"
                if config.decoder_mode == "full_pair"
                else "nested-OOF broad surface-context proposal, frozen sharp unary "
                "reranking, and bilateral Gonion pair decoding"
            )
        ),
        "uses_validation_labels_for_model_fit": False,
        "uses_test_labels": False,
        "sample_count": len(candidates),
        "input_channels": int(candidates.images.shape[3]),
        "parameter_count_per_member": int(parameter_count),
        "ensemble_members": len(models),
        "folds": fold_reports,
        "oof_best_epochs": best_epochs,
        "final_training": {
            "selection": final_selection,
            "policy": config.final_model_policy,
            "reranker_enabled": _uses_unary_reranker(config),
            "median_inner_fold_best_epochs": {
                "proposal": median_proposal_epoch,
                "rerank": median_rerank_epoch,
                "pair": median_pair_epoch,
            },
            "fixed_epochs": fixed_epochs,
            "members": final_histories,
        },
        "coordinate_policy": policy,
        "reliability_scale_mm": reliability_scale.tolist(),
        "oof": {
            "hard3": summarize(oof_error),
            "lm0": summarize(oof_error[:, 0]),
            "gonion": summarize(oof_error[:, 1:3]),
            "candidate_oracle_ale": float(
                np.min(candidates.target_distance, axis=-1).mean()
            ),
            "proposal_diagnostics": diagnostics,
            "gonion_pair_topk_recall": {
                "topk": int(pair_count),
                "lm21": float(left_pair_hit.mean()),
                "lm22": float(right_pair_hit.mean()),
                "both": float(np.mean(left_pair_hit & right_pair_hit)),
                "oracle_ale": float(pair_shortlist_distance.mean()),
                "oracle_p95": float(np.percentile(pair_shortlist_distance, 95)),
                "oracle_sdr_at_2mm": float((pair_shortlist_distance <= 2.0).mean()),
                "oracle_sdr_at_3mm": float((pair_shortlist_distance <= 3.0).mean()),
                "clinical_radius_mm": float(config.pair_clinical_radius_mm),
                "clinical_both_coverage": float(
                    np.mean(left_clinical_hit & right_clinical_hit)
                ),
            },
            "selection_diagnostics": {
                "actual_gonion_ale": float(oof_error[:, 1:3].mean()),
                "oracle_gonion_ale": float(pair_shortlist_distance.mean()),
                "selector_regret_mm": float(
                    oof_error[:, 1:3].mean() - pair_shortlist_distance.mean()
                ),
                "oracle_to_actual_ratio": float(
                    pair_shortlist_distance.mean()
                    / max(float(oof_error[:, 1:3].mean()), 1e-8)
                ),
                "axis_mae_xyz": {
                    "lm0": oof_axis_error[0].tolist(),
                    "lm21": oof_axis_error[1].tolist(),
                    "lm22": oof_axis_error[2].tolist(),
                },
                "pair_policy_selected": policy["gonion_pair"]["mode"],
            },
            "mean_dynamic_view_weights": oof_view_weights.mean(axis=0).tolist(),
            "std_dynamic_view_weights": oof_view_weights.std(axis=0).tolist(),
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
        "training_center_source": (
            "stage2_shape_prior_prediction"
            if training_centers is not None
            else "dataset_stage1_coarse"
        ),
        "training_center_metrics": {
            "hard3_ale": float(
                np.linalg.norm(
                    candidates.centers[:, list(HARD3)] - candidates.expert,
                    axis=-1,
                ).mean()
            ),
            "gonion_ale": float(
                np.linalg.norm(
                    candidates.centers[:, [21, 22]] - candidates.expert[:, 1:3],
                    axis=-1,
                ).mean()
            ),
        },
        "training_seconds": float(time.time() - started),
        "config": asdict(config),
    }
    torch.save(
        {
            "signature": signature,
            "input_channels": int(candidates.images.shape[3]),
            "geometry_dim": int(candidates.canonical.shape[-1]),
            "shape_context_dim": int(candidates.shape_context.shape[-1]),
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
        outputs,
        candidate_result,
        candidate_result["variant_predictions"][gonion_variant],
    )
    result = gonion_values.copy()
    result[:, 0] = lm0_values[:, 0]
    return result


def _dual_blend_prediction(base, candidate, reliability, row):
    confidence = (
        np.asarray(reliability, dtype=np.float32)
        if row["confidence_mode"] == "ensemble"
        else np.ones_like(reliability, dtype=np.float32)
    )
    left_alpha = row.get("alpha_gonion_left", row.get("alpha_gonion", 0.0))
    right_alpha = row.get("alpha_gonion_right", row.get("alpha_gonion", 0.0))
    alpha = np.asarray([row["alpha_lm0"], left_alpha, right_alpha], dtype=np.float32)
    effective_alpha = confidence * alpha[None]
    prediction = np.asarray(base, dtype=np.float32).copy()
    base_hard3 = prediction[:, list(HARD3)].copy()
    prediction[:, list(HARD3)] = base_hard3 + effective_alpha[..., None] * (
        candidate - base_hard3
    )
    return prediction, effective_alpha


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
            "lm21_ale": float(error[:, 1].mean()),
            "lm22_ale": float(error[:, 2].mean()),
            "gonion_ale": float(error[:, 1:3].mean()),
            "hard3_ale": float(error.mean()),
        }
    # Restrict the joint sweep to the strongest predefined candidates to control
    # variance on the 48-sample outer validation fold.
    lm0_variants = sorted(variants, key=lambda name: individual[name]["lm0_ale"])[:8]
    # Atlas coordinates were strongly harmful in Fold 1. They remain available
    # as a weak logit regularizer, but are no longer eligible as direct Gonion
    # predictions.
    gonion_candidates = [name for name in variants if not name.startswith("atlas_")]
    gonion_variants = sorted(
        gonion_candidates, key=lambda name: individual[name]["gonion_ale"]
    )[:8]
    for anchor in ("neural_policy", "atlas_direct"):
        if anchor not in lm0_variants:
            lm0_variants.append(anchor)
    for anchor in ("neural_policy", "joint_soft", "joint_argmax", "joint_snapped"):
        if anchor in variants and anchor not in gonion_variants:
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
                    for alpha_left in (0.0, 0.25, 0.5, 0.75, 1.0):
                        for alpha_right in (0.0, 0.25, 0.5, 0.75, 1.0):
                            alpha_gonion = 0.5 * (alpha_left + alpha_right)
                            row = {
                                "lm0_variant": lm0_variant,
                                "gonion_variant": gonion_variant,
                                "confidence_mode": confidence_mode,
                                "alpha_lm0": alpha_lm0,
                                "alpha_gonion": alpha_gonion,
                                "alpha_gonion_left": alpha_left,
                                "alpha_gonion_right": alpha_right,
                            }
                            prediction, _ = _dual_blend_prediction(
                                base, limited, reliability, row
                            )
                            error = np.linalg.norm(prediction - expert, axis=-1)
                            rows.append(
                                {
                                    **row,
                                    "overall_ale": float(error.mean()),
                                    "hard3_ale": float(error[:, list(HARD3)].mean()),
                                    "lm0_ale": float(error[:, 0].mean()),
                                    "lm21_ale": float(error[:, 21].mean()),
                                    "lm22_ale": float(error[:, 22].mean()),
                                    "gonion_ale": float(error[:, 21:23].mean()),
                                    "p95": float(np.percentile(error, 95)),
                                }
                            )
    eligible = [
        row for row in rows if row["p95"] <= base_p95 + config.maximum_p95_regression_mm
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
    proposed_prediction, _ = _dual_blend_prediction(
        base, limited, reliability, proposed
    )
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
    selected = (
        proposed
        if accepted
        else {
            "lm0_variant": "neural_policy",
            "gonion_variant": "neural_policy",
            "confidence_mode": "none",
            "alpha_lm0": 0.0,
            "alpha_gonion": 0.0,
            "alpha_gonion_left": 0.0,
            "alpha_gonion_right": 0.0,
            "overall_ale": float(base_error.mean()),
            "hard3_ale": float(base_error[:, list(HARD3)].mean()),
            "lm0_ale": float(base_error[:, 0].mean()),
            "gonion_ale": float(base_error[:, 21:23].mean()),
            "p95": base_p95,
        }
    )
    selected_raw = _candidate_for_variants(
        outputs,
        candidate_result,
        selected["lm0_variant"],
        selected["gonion_variant"],
    )
    selected_limited, _, _ = _limited_hard3_candidate(base, selected_raw, limits)
    blended, effective_alpha = _dual_blend_prediction(
        base, selected_limited, reliability, selected
    )
    blended_error = np.linalg.norm(blended - expert, axis=-1)
    return {
        "accepted": bool(accepted),
        "proposed": proposed,
        "selected": selected,
        "target_hard3_ale": config.target_hard3_ale,
        "target_reached_on_validation": bool(
            accepted
            and float(blended_error[:, list(HARD3)].mean()) < config.target_hard3_ale
        ),
        "limits_mm": list(limits),
        "base_overall": summarize(base_error),
        "base_hard3": summarize(base_error[:, list(HARD3)]),
        "blended_overall": summarize(blended_error),
        "blended_hard3": summarize(blended_error[:, list(HARD3)]),
        "overall_gain_mm": float(base_error.mean() - blended_error.mean()),
        "hard3_gain_mm": float(
            base_error[:, list(HARD3)].mean() - blended_error[:, list(HARD3)].mean()
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
    prediction, effective_alpha = _dual_blend_prediction(
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
