from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .anatomy import ANATOMICAL_EDGES, MIDLINE, SYMMETRY_PAIRS


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


def adaptive_wing_loss(logits, target, mask, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
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
    return (loss * weights * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def region_loss(logits, target, mask, positive_weight=20.0):
    logits = logits.float()
    target = target.float()
    weight = torch.where(target > 0.5, target.new_tensor(positive_weight), target.new_tensor(1.0))
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * weight * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    probability = torch.sigmoid(logits) * mask.float()
    intersection = (probability * target).sum(dim=-1)
    denominator = probability.sum(dim=-1) + target.sum(dim=-1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


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


def compute_loss(outputs, batch, weights, positive_weight=20.0):
    expert = batch["expert"].float()
    prediction = outputs["final_coordinates"].float()
    heatmap = adaptive_wing_loss(outputs["local_logits"], batch["heatmap_target"], batch["roi_mask"])
    region = region_loss(outputs["region_logits"], batch["region_target"], batch["roi_mask"], positive_weight)
    coordinate_per_axis = F.smooth_l1_loss(prediction, expert, beta=1.0, reduction="none")
    coordinate = coordinate_per_axis.mean()
    coarse_coordinate = F.smooth_l1_loss(outputs["coarse_coordinates"].float(), expert, beta=2.0)
    coarse_ce = coarse_nearest_loss(outputs["coarse_logits"], batch, expert)
    coarse = coarse_coordinate + 0.1 * coarse_ce
    anatomy = pairwise_anatomy_loss(prediction, expert)
    symmetric = symmetry_loss(prediction)
    euclidean = torch.linalg.norm(prediction - expert, dim=-1)
    log_var = outputs["log_var"].float().clamp(-6.0, 6.0)
    uncertainty = (0.5 * torch.exp(-log_var) * euclidean.pow(2) + 0.5 * log_var).mean()
    clinical = F.softplus((euclidean - 2.0) / 0.5).mean()
    total = (
        weights.heatmap * heatmap
        + weights.region * region
        + weights.coordinate * coordinate
        + weights.coarse * coarse
        + weights.anatomy * anatomy
        + weights.symmetry * symmetric
        + weights.uncertainty * uncertainty
        + weights.clinical * clinical
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
    }
    return total, euclidean, components
