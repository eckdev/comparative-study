"""Core20 multi-view spatial-configuration candidate ranker."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anatomy import GROUP_NAMES


def _groups(channels):
    groups = min(8, int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class ConvGN(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, dropout=0.0):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Conv2d(
                input_channels, output_channels, 1, stride=stride, bias=False
            )
        )

    def forward(self, values):
        return self.main(values) + self.skip(values)


class CompactFeaturePyramid(nn.Module):
    """Small-batch-safe FPN; all normalization is GroupNorm."""

    def __init__(self, input_channels, width=32, dropout=0.10):
        super().__init__()
        self.level_one = ConvGN(input_channels, width, dropout=dropout)
        self.level_two = ConvGN(width, width * 2, stride=2, dropout=dropout)
        self.level_three = ConvGN(width * 2, width * 4, stride=2, dropout=dropout)
        self.lateral_three = nn.Conv2d(width * 4, width, 1)
        self.lateral_two = nn.Conv2d(width * 2, width, 1)
        self.lateral_one = nn.Conv2d(width, width, 1)
        self.smooth_two = ConvGN(width, width, dropout=dropout)
        self.smooth_one = ConvGN(width, width, dropout=dropout)
        self.output_channels = width

    def forward(self, image):
        first = self.level_one(image)
        second = self.level_two(first)
        third = self.level_three(second)
        pyramid = self.lateral_three(third)
        pyramid = F.interpolate(
            pyramid, size=second.shape[-2:], mode="bilinear", align_corners=False
        ) + self.lateral_two(second)
        pyramid = self.smooth_two(pyramid)
        pyramid = F.interpolate(
            pyramid, size=first.shape[-2:], mode="bilinear", align_corners=False
        ) + self.lateral_one(first)
        return self.smooth_one(pyramid)


class Core20MVSCNet(nn.Module):
    def __init__(
        self,
        image_channels,
        geometry_dim,
        width=48,
        landmark_embedding_dim=24,
        group_embedding_dim=12,
        dropout=0.10,
        coordinate_topk=8,
        coordinate_temperature=0.5,
        use_images=True,
        use_spatial_prior=True,
        use_confidence_gate=True,
    ):
        super().__init__()
        self.coordinate_topk = int(coordinate_topk)
        self.coordinate_temperature = float(coordinate_temperature)
        self.use_images = bool(use_images)
        self.use_spatial_prior = bool(use_spatial_prior)
        self.use_confidence_gate = bool(use_confidence_gate)
        image_width = max(16, int(width) // 2)
        hidden = max(32, int(width))
        self.image_encoder = CompactFeaturePyramid(image_channels, image_width, dropout)
        self.landmark_embedding = nn.Embedding(20, landmark_embedding_dim)
        self.group_embedding = nn.Embedding(len(GROUP_NAMES), group_embedding_dim)
        self.view_attention = nn.Sequential(
            nn.Linear(image_width + landmark_embedding_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.geometry_encoder = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        candidate_dim = (
            image_width + hidden + landmark_embedding_dim + group_embedding_dim
        )
        self.group_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(candidate_dim),
                    nn.Linear(candidate_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
                for _ in GROUP_NAMES
            ]
        )
        self.prior_weights = nn.Parameter(torch.full((len(GROUP_NAMES),), -0.7))
        context_dim = (
            image_width + hidden + landmark_embedding_dim + group_embedding_dim + 4
        )
        self.log_variance_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _normalize_prior(score, mask):
        valid = mask.float()
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        safe = torch.where(mask, score.float(), torch.zeros_like(score.float()))
        mean = safe.sum(dim=-1, keepdim=True) / count
        variance = ((safe - mean).square() * valid).sum(dim=-1, keepdim=True) / count
        result = (safe - mean) / torch.sqrt(variance + 1e-5)
        return result.clamp(-6.0, 6.0).masked_fill(~mask, 0.0)

    def _sample_views(self, images, grids, landmark_embedding):
        batch, views, channels, height, width = images.shape
        candidates = grids.shape[2]
        feature_map = self.image_encoder(
            images.reshape(batch * views, channels, height, width)
        )
        feature_channels = feature_map.shape[1]
        sampled = F.grid_sample(
            feature_map,
            grids.reshape(batch * views, 1, candidates, 2).float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = (
            sampled.squeeze(2)
            .transpose(1, 2)
            .reshape(batch, views, candidates, feature_channels)
        )
        pooled = feature_map.mean(dim=(-2, -1)).reshape(batch, views, feature_channels)
        embedded = landmark_embedding[:, None].expand(-1, views, -1)
        view_weight = torch.softmax(
            self.view_attention(torch.cat([pooled, embedded], dim=-1)).squeeze(-1),
            dim=-1,
        )
        fused = (sampled * view_weight[:, :, None, None]).sum(dim=1)
        pooled_fused = (pooled * view_weight[..., None]).sum(dim=1)
        return fused, pooled_fused, view_weight

    def forward(self, batch):
        mask = batch["mask"].bool()
        local_landmark = batch["local_landmark"].long()
        group = batch["group"].long()
        landmark_embedding = self.landmark_embedding(local_landmark)
        group_embedding = self.group_embedding(group)
        if self.use_images:
            image_candidate, image_context, view_weight = self._sample_views(
                batch["images"].float(), batch["grids"].float(), landmark_embedding
            )
        else:
            image_width = self.image_encoder.output_channels
            image_candidate = torch.zeros(
                (*batch["features"].shape[:2], image_width),
                device=batch["features"].device,
            )
            image_context = torch.zeros(
                (len(image_candidate), image_width), device=image_candidate.device
            )
            view_weight = torch.full(
                (len(image_candidate), 3), 1.0 / 3.0, device=image_candidate.device
            )
        geometry = self.geometry_encoder(batch["features"].float())
        candidates = geometry.shape[1]
        embeddings = torch.cat([landmark_embedding, group_embedding], dim=-1)
        embeddings = embeddings[:, None].expand(-1, candidates, -1)
        candidate_state = torch.cat([image_candidate, geometry, embeddings], dim=-1)
        logits = torch.zeros(
            candidate_state.shape[:2],
            device=candidate_state.device,
            dtype=torch.float32,
        )
        for group_index, head in enumerate(self.group_heads):
            selected = group == group_index
            if selected.any():
                group_logits = head(candidate_state[selected].float()).squeeze(-1)
                logits[selected] = group_logits.to(dtype=logits.dtype)
        prior = self._normalize_prior(batch["prior_score"], mask)
        prior_weight = F.softplus(self.prior_weights[group])[:, None]
        if not self.use_spatial_prior:
            prior_weight = torch.zeros_like(prior_weight)
        logits = logits + prior_weight * prior
        logits = logits.masked_fill(~mask, -torch.inf)

        count = min(max(1, self.coordinate_topk), logits.shape[-1])
        top_logits, top_index = torch.topk(logits, count, dim=-1)
        top_points = torch.gather(
            batch["points"].float(),
            1,
            top_index[..., None].expand(-1, -1, 3),
        )
        top_probability = torch.softmax(
            top_logits / max(self.coordinate_temperature, 1e-4), dim=-1
        )
        proposal = (top_probability[..., None] * top_points).sum(dim=1)

        probability = torch.softmax(logits.float(), dim=-1)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=-1)
        valid_count = mask.sum(dim=-1).float().clamp_min(2.0)
        entropy = entropy / torch.log(valid_count)
        if logits.shape[-1] > 1:
            top_two = torch.topk(logits, 2, dim=-1).values
            margin = torch.sigmoid(top_two[:, 0] - top_two[:, 1])
        else:
            margin = torch.ones_like(entropy)
        prior_delta = proposal - batch["prior_mean"].float()
        prior_inverse = torch.linalg.pinv(batch["prior_covariance"].float())
        prior_disagreement = torch.sqrt(
            torch.einsum(
                "bi,bij,bj->b", prior_delta, prior_inverse, prior_delta
            ).clamp_min(0.0)
            + 1e-6
        )
        base_displacement = torch.linalg.norm(proposal - batch["base"].float(), dim=-1)
        valid = mask.float()[..., None]
        geometry_context = (geometry * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(
            1.0
        )
        diagnostics = torch.stack(
            [entropy, margin, prior_disagreement / 3.0, base_displacement / 6.0],
            dim=-1,
        )
        context = torch.cat(
            [
                image_context,
                geometry_context,
                landmark_embedding,
                group_embedding,
                diagnostics,
            ],
            dim=-1,
        )
        log_variance = self.log_variance_head(context).squeeze(-1).clamp(-4.0, 4.0)
        gate_alpha = torch.sigmoid(self.gate_head(context).squeeze(-1))
        if not self.use_confidence_gate:
            gate_alpha = torch.ones_like(gate_alpha)
        final = batch["base"].float() + gate_alpha[:, None] * (
            proposal - batch["base"].float()
        )
        confidence = (1.0 - entropy).clamp(0.0, 1.0) * torch.sigmoid(
            -0.5 * log_variance
        )
        return {
            "logits": logits,
            "probability": probability,
            "proposal": proposal,
            "final": final,
            "gate_alpha": gate_alpha,
            "confidence": confidence,
            "log_variance": log_variance,
            "entropy": entropy,
            "margin": margin,
            "prior_disagreement": prior_disagreement,
            "view_weights": view_weight,
        }
