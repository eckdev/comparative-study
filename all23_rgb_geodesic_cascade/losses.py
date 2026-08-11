from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .anatomy import ANATOMICAL_EDGES, HARD3, MIDLINE, NUM_LANDMARKS, SYMMETRY_PAIRS


@dataclass
class LossWeights:
    heatmap: float = 1.0
    region: float = 0.25
    coordinate: float = 0.5
    coarse: float = 0.2
    anatomy: float = 0.08
    symmetry: float = 0.04
    uncertainty: float = 0.02
    clinical: float = 0.05
    gate: float = 0.10
    hard_landmark: float = 1.5
    hard_rank: float = 0.0


def landmark_loss_weights(reference, hard_weight=1.0):
    values = reference.new_ones(NUM_LANDMARKS)
    values[list(HARD3)] = float(hard_weight)
    return values


def weighted_landmark_mean(values, weights):
    weights = weights[None, :].expand(values.shape[0], -1)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def adaptive_wing_loss(
    logits,
    target,
    mask,
    omega=14.0,
    theta=0.5,
    epsilon=1.0,
    alpha=2.1,
    landmark_weights=None,
):
    logits = logits.float()
    target = target.float()
    prediction = torch.sigmoid(logits.masked_fill(~mask, 0.0))
    difference = torch.abs(target - prediction)
    exponent = alpha - target
    first = omega * torch.log1p(torch.pow(difference / epsilon, exponent))
    theta_ratio = torch.pow(torch.as_tensor(theta / epsilon, device=target.device), exponent)
    a = omega * (1.0 / (1.0 + theta_ratio)) * exponent * torch.pow(
        torch.as_tensor(theta / epsilon, device=target.device), exponent - 1.0
    ) / epsilon
    c = theta * a - omega * torch.log1p(theta_ratio)
    second = a * difference - c
    loss = torch.where(difference < theta, first, second)
    weights = 1.0 + 4.0 * target
    per_landmark = (loss * weights * mask.float()).sum(dim=-1)
    per_landmark = per_landmark / mask.float().sum(dim=-1).clamp_min(1.0)
    if landmark_weights is None:
        return per_landmark.mean()
    return weighted_landmark_mean(per_landmark, landmark_weights)


def region_loss(logits, target, mask, positive_weight=20.0, landmark_weights=None):
    logits = logits.float()
    target = target.float()
    weight = torch.where(target > 0.5, target.new_tensor(positive_weight), target.new_tensor(1.0))
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * weight * mask.float()).sum(dim=-1)
    bce = bce / mask.float().sum(dim=-1).clamp_min(1.0)
    probability = torch.sigmoid(logits) * mask.float()
    intersection = (probability * target).sum(dim=-1)
    denominator = probability.sum(dim=-1) + target.sum(dim=-1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    combined = bce + dice
    if landmark_weights is None:
        return combined.mean()
    return weighted_landmark_mean(combined, landmark_weights)


def pairwise_anatomy_loss(prediction, expert):
    prediction = prediction.float()
    expert = expert.float()
    losses = []
    for left, right in ANATOMICAL_EDGES:
        predicted_distance = torch.linalg.norm(prediction[:, left] - prediction[:, right], dim=-1)
        expert_distance = torch.linalg.norm(expert[:, left] - expert[:, right], dim=-1)
        # Relative error keeps long mandibular edges from dominating short ocular edges.
        losses.append(F.smooth_l1_loss(predicted_distance / expert_distance.clamp_min(1.0), torch.ones_like(expert_distance)))
    return torch.stack(losses).mean()


def symmetry_loss(prediction):
    prediction = prediction.float()
    midline_x = prediction[:, MIDLINE, 0].mean(dim=1)
    terms = []
    for left, right in SYMMETRY_PAIRS:
        lateral_left = torch.abs(prediction[:, left, 0] - midline_x)
        lateral_right = torch.abs(prediction[:, right, 0] - midline_x)
        terms.append(F.smooth_l1_loss(lateral_left, lateral_right))
        terms.append(F.smooth_l1_loss(prediction[:, left, 1:], prediction[:, right, 1:]))
    return torch.stack(terms).mean()


def coarse_nearest_loss(coarse_logits, batch, expert):
    coarse_logits = coarse_logits.float()
    expert = expert.float()
    losses = []
    for batch_index in range(expert.shape[0]):
        selected = (batch["batch"] == batch_index) & batch["vertex_mask"]
        points = batch["points"][selected]
        logits = coarse_logits[selected].transpose(0, 1)
        nearest = torch.argmin(torch.cdist(expert[batch_index], points), dim=1)
        losses.append(F.cross_entropy(logits, nearest))
    return torch.stack(losses).mean()


def hard_candidate_ranking_loss(logits, candidates, mask, expert):
    """Classify the expert-nearest surface candidate for LM0/21/22."""
    if logits is None:
        return expert.new_zeros(())
    hard_expert = expert[:, HARD3].float()
    hard_candidates = candidates[:, HARD3].float()
    hard_mask = mask[:, HARD3]
    distances = torch.linalg.norm(
        hard_candidates - hard_expert[:, :, None, :], dim=-1
    ).masked_fill(~hard_mask, torch.inf)
    nearest = torch.argmin(distances, dim=-1)
    masked_logits = logits.float().masked_fill(~hard_mask, -torch.inf)
    loss = F.cross_entropy(
        masked_logits.reshape(-1, masked_logits.shape[-1]),
        nearest.reshape(-1),
    )
    return loss / torch.log(
        loss.new_tensor(max(masked_logits.shape[-1], 2), dtype=torch.float32)
    )


def compute_loss(outputs, batch, weights, positive_weight=20.0):
    expert = batch["expert"].float()
    prediction = outputs["final_coordinates"].float()
    landmark_weights = landmark_loss_weights(expert, weights.hard_landmark)
    heatmap = adaptive_wing_loss(
        outputs["local_logits"],
        batch["heatmap_target"],
        batch["roi_mask"],
        landmark_weights=landmark_weights,
    )
    region = region_loss(
        outputs["region_logits"],
        batch["region_target"],
        batch["roi_mask"],
        positive_weight,
        landmark_weights=landmark_weights,
    )
    coordinate_per_axis = F.smooth_l1_loss(
        prediction, expert, beta=1.0, reduction="none"
    ).mean(dim=-1)
    coordinate = weighted_landmark_mean(coordinate_per_axis, landmark_weights)
    coarse_coordinate = F.smooth_l1_loss(outputs["coarse_coordinates"].float(), expert, beta=2.0)
    coarse_ce = coarse_nearest_loss(outputs["coarse_logits"], batch, expert)
    coarse = coarse_coordinate + 0.1 * coarse_ce
    anatomy = pairwise_anatomy_loss(prediction, expert)
    symmetric = symmetry_loss(prediction)
    euclidean = torch.linalg.norm(prediction - expert, dim=-1)
    log_var = outputs["log_var"].float().clamp(-6.0, 6.0)
    uncertainty = weighted_landmark_mean(
        0.5 * torch.exp(-log_var) * euclidean.pow(2) + 0.5 * log_var,
        landmark_weights,
    )
    clinical = weighted_landmark_mean(
        F.softplus((euclidean - 2.0) / 0.5), landmark_weights
    )
    gate = prediction.new_zeros(())
    if (
        "refinement_alpha" in outputs
        and "refined_coordinates" in outputs
        and outputs["refinement_alpha"].requires_grad
    ):
        external_coarse = batch["coarse"].float()
        refined = outputs["refined_coordinates"].float().detach()
        direction = refined - external_coarse
        denominator = direction.pow(2).sum(dim=-1)
        optimal_alpha = (
            ((expert - external_coarse) * direction).sum(dim=-1)
            / denominator.clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        gate_error = F.smooth_l1_loss(
            outputs["refinement_alpha"].float(),
            optimal_alpha,
            beta=0.1,
            reduction="none",
        )
        valid = denominator > 1e-4
        gate_error = torch.where(valid, gate_error, torch.zeros_like(gate_error))
        optimal_coordinate = external_coarse + optimal_alpha[..., None] * direction
        gated_coordinate = external_coarse + outputs["refinement_alpha"].float()[
            ..., None
        ] * direction
        gate_coordinate_error = F.smooth_l1_loss(
            gated_coordinate,
            optimal_coordinate,
            beta=1.0,
            reduction="none",
        ).mean(dim=-1)
        coarse_error = torch.linalg.norm(external_coarse - expert, dim=-1)
        gated_error = torch.linalg.norm(gated_coordinate - expert, dim=-1)
        gate_regret = torch.relu(gated_error - coarse_error)
        gate = weighted_landmark_mean(
            gate_error + 0.25 * gate_coordinate_error + 0.25 * gate_regret,
            landmark_weights,
        )
    hard_rank = hard_candidate_ranking_loss(
        outputs.get("hard_rank_logits"),
        outputs["candidate_points"],
        batch["roi_mask"],
        expert,
    )
    total = (
        weights.heatmap * heatmap
        + weights.region * region
        + weights.coordinate * coordinate
        + weights.coarse * coarse
        + weights.anatomy * anatomy
        + weights.symmetry * symmetric
        + weights.uncertainty * uncertainty
        + weights.clinical * clinical
        + weights.gate * gate
        + weights.hard_rank * hard_rank
    )
    components = {
        "total": total,
        "heatmap": heatmap,
        "region": region,
        "coordinate": coordinate,
        "coarse": coarse,
        "anatomy": anatomy,
        "symmetry": symmetric,
        "uncertainty": uncertainty,
        "clinical": clinical,
        "gate": gate,
        "hard_rank": hard_rank,
    }
    return total, euclidean, components
