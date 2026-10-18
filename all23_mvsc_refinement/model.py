"""All-23 multi-view candidate scorer with landmark-type heads.

The image encoder and view fusion are those of Core20-MVSC. The scorer differs
in three ways: it covers all 23 landmarks, it uses one head per landmark type
(corner / contour / texture boundary) conditioned on landmark and anatomical
group embeddings, and it has no internal gate. Gating is done afterwards by a
separate gate trained on cross-fitted out-of-fold evidence (gate.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from all23_rgb_geodesic_cascade.anatomy import NUM_LANDMARKS
from core20_mvsc_refinement.model import CompactFeaturePyramid

from .anatomy import GROUP_NAMES, TYPE_NAMES


class All23MVSCNet(nn.Module):
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
    ):
        super().__init__()
        self.coordinate_topk = int(coordinate_topk)
        self.coordinate_temperature = float(coordinate_temperature)
        self.use_images = bool(use_images)
        self.use_spatial_prior = bool(use_spatial_prior)
        image_width = max(16, int(width) // 2)
        hidden = max(32, int(width))
        self.image_encoder = CompactFeaturePyramid(image_channels, image_width, dropout)
        self.landmark_embedding = nn.Embedding(NUM_LANDMARKS, landmark_embedding_dim)
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
        self.type_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(candidate_dim),
                    nn.Linear(candidate_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
                for _ in TYPE_NAMES
            ]
        )
        self.prior_weights = nn.Parameter(torch.full((len(TYPE_NAMES),), -0.7))
        context_dim = (
            image_width + hidden + landmark_embedding_dim + group_embedding_dim + 3
        )
        self.log_variance_head = nn.Sequential(
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
            grids.reshape(batch * views, 1, candidates, 2).to(feature_map.dtype),
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
        embedded = landmark_embedding[:, None].expand(-1, views, -1).to(pooled.dtype)
        view_weight = torch.softmax(
            self.view_attention(torch.cat([pooled, embedded], dim=-1))
            .squeeze(-1)
            .float(),
            dim=-1,
        )
        fused = (sampled.float() * view_weight[:, :, None, None]).sum(dim=1)
        pooled_fused = (pooled.float() * view_weight[..., None]).sum(dim=1)
        return fused, pooled_fused, view_weight

    def forward(self, batch):
        mask = batch["mask"].bool()
        landmark = batch["landmark"].long()
        group = batch["group"].long()
        landmark_type = batch["type"].long()
        landmark_embedding = self.landmark_embedding(landmark)
        group_embedding = self.group_embedding(group)
        views = batch["images"].shape[1]
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
                (len(image_candidate), views),
                1.0 / views,
                device=image_candidate.device,
            )
        geometry = self.geometry_encoder(batch["features"].float())
        candidates = geometry.shape[1]
        embeddings = torch.cat([landmark_embedding, group_embedding], dim=-1)
        embeddings = embeddings[:, None].expand(-1, candidates, -1)
        candidate_state = torch.cat(
            [image_candidate.to(geometry.dtype), geometry, embeddings.to(geometry.dtype)],
            dim=-1,
        )
        logits = torch.zeros(
            candidate_state.shape[:2],
            device=candidate_state.device,
            dtype=torch.float32,
        )
        for type_index, head in enumerate(self.type_heads):
            selected = landmark_type == type_index
            if selected.any():
                type_logits = head(candidate_state[selected].float()).squeeze(-1)
                logits[selected] = type_logits.to(dtype=logits.dtype)
        prior = self._normalize_prior(batch["prior_score"], mask)
        prior_weight = F.softplus(self.prior_weights[landmark_type])[:, None]
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
        top_probability = torch.nan_to_num(top_probability, nan=0.0)
        proposal = (top_probability[..., None] * top_points).sum(dim=1)

        probability = torch.softmax(logits.float(), dim=-1)
        entropy = -(
            probability * torch.log(probability.clamp_min(1e-8))
        ).sum(dim=-1)
        valid_count = mask.sum(dim=-1).float().clamp_min(2.0)
        entropy = entropy / torch.log(valid_count)
        if logits.shape[-1] > 1:
            top_two = torch.topk(logits, 2, dim=-1).values
            margin = torch.sigmoid(top_two[:, 0] - top_two[:, 1])
            margin = torch.nan_to_num(margin, nan=1.0)
        else:
            margin = torch.ones_like(entropy)
        prior_delta = proposal - batch["prior_mean"].float()
        prior_inverse = torch.linalg.pinv(batch["prior_covariance"].float())
        prior_disagreement = torch.sqrt(
            torch.einsum("bi,bij,bj->b", prior_delta, prior_inverse, prior_delta)
            .clamp_min(0.0)
            + 1e-6
        )
        valid = mask.float()[..., None]
        geometry_context = (geometry.float() * valid).sum(dim=1) / valid.sum(
            dim=1
        ).clamp_min(1.0)
        diagnostics = torch.stack(
            [entropy, margin, prior_disagreement / 3.0], dim=-1
        )
        context = torch.cat(
            [
                image_context.float(),
                geometry_context,
                landmark_embedding.float(),
                group_embedding.float(),
                diagnostics,
            ],
            dim=-1,
        )
        log_variance = self.log_variance_head(context).squeeze(-1).clamp(-4.0, 4.0)
        confidence = (1.0 - entropy).clamp(0.0, 1.0) * torch.sigmoid(
            -0.5 * log_variance
        )
        return {
            "logits": logits,
            "probability": probability,
            "proposal": proposal,
            "confidence": confidence,
            "log_variance": log_variance,
            "entropy": entropy,
            "margin": margin,
            "prior_disagreement": prior_disagreement,
            "view_weights": view_weight,
        }
