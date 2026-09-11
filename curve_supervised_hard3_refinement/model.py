"""Curve-first candidate network for peripheral 3D facial landmarks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewEncoder(nn.Module):
    def __init__(self, input_channels, width, dropout):
        super().__init__()
        hidden = max(int(width), 16)
        groups = min(8, hidden)
        while hidden % groups:
            groups -= 1
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output_dim = hidden * 2

    def forward(self, image):
        return self.network(image).flatten(1)


class SurfaceGraphBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(width * 2 + 3, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.update = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)

    @staticmethod
    def _gather(values, indices):
        batch, landmarks, candidates, channels = values.shape
        flat = values.reshape(batch * landmarks, candidates, channels)
        flat_index = indices.reshape(batch * landmarks, candidates, -1)
        batch_index = torch.arange(batch * landmarks, device=values.device)[
            :, None, None
        ]
        gathered = flat[batch_index, flat_index]
        return gathered.reshape(
            batch, landmarks, candidates, flat_index.shape[-1], channels
        )

    def forward(self, hidden, geometry, neighbor_index, neighbor_mask, mask):
        neighbor = self._gather(hidden, neighbor_index.long())
        neighbor_xyz = self._gather(geometry.float(), neighbor_index.long())
        center = hidden[..., None, :].expand_as(neighbor)
        delta = neighbor_xyz - geometry[..., None, :].float()
        edge = self.edge(torch.cat([center, neighbor - center, delta], dim=-1))
        gathered_mask = (
            self._gather(mask[..., None].float(), neighbor_index.long())
            .squeeze(-1)
            .bool()
        )
        valid = neighbor_mask.bool() & gathered_mask & mask[..., None]
        valid_float = valid[..., None].to(edge.dtype)
        mean = (edge * valid_float).sum(dim=-2) / valid_float.sum(dim=-2).clamp_min(1.0)
        maximum = edge.masked_fill(~valid[..., None], -torch.inf).amax(dim=-2)
        maximum = torch.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0)
        update = self.update(torch.cat([hidden, mean, maximum], dim=-1))
        return self.norm(hidden + update).masked_fill(~mask[..., None], 0.0)


class CurveFirstHard3Net(nn.Module):
    """Predict a support curve before selecting each landmark on that curve."""

    def __init__(
        self,
        input_channels,
        feature_dim,
        shape_context_dim=69,
        width=48,
        blocks=2,
        dropout=0.10,
    ):
        super().__init__()
        hidden = max(int(width), 24)
        self.hidden = hidden
        self.view_encoder = ViewEncoder(input_channels, hidden // 2, dropout)
        self.view_projection = nn.Sequential(
            nn.Linear(self.view_encoder.output_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.shape_encoder = nn.Sequential(
            nn.LayerNorm(shape_context_dim),
            nn.Linear(shape_context_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.landmark_embedding = nn.Parameter(torch.zeros(3, hidden))
        self.graph_blocks = nn.ModuleList(
            [SurfaceGraphBlock(hidden, dropout) for _ in range(max(1, int(blocks)))]
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden * 5, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.bilateral_context = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.curve_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.support_encoder = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.support_graph = SurfaceGraphBlock(hidden, dropout)
        self.landmark_head = nn.Sequential(
            nn.Linear(hidden * 3 + 1, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.curve_scale = nn.Parameter(torch.full((3,), 0.5))

    @staticmethod
    def _masked_pool(values, mask):
        valid = mask[..., None].to(values.dtype)
        mean = (values * valid).sum(dim=-2) / valid.sum(dim=-2).clamp_min(1.0)
        maximum = values.masked_fill(~mask[..., None], -torch.inf).amax(dim=-2)
        maximum = torch.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.cat([mean, maximum], dim=-1)

    def forward(
        self,
        images,
        canonical,
        neighbor_index,
        neighbor_mask,
        candidate_mask,
        shape_context,
    ):
        batch, landmarks, views, channels, height, width = images.shape
        view = self.view_encoder(
            images.float().reshape(batch * landmarks * views, channels, height, width)
        ).reshape(batch, landmarks, views, -1)
        appearance = self.view_projection(
            torch.cat([view.mean(dim=2), view.amax(dim=2)], dim=-1)
        )
        shape = self.shape_encoder(torch.nan_to_num(shape_context.float()))
        encoded = self.point_encoder(torch.nan_to_num(canonical.float()))
        encoded = encoded.masked_fill(~candidate_mask[..., None], 0.0)
        geometry = canonical[..., 3:6]
        for block in self.graph_blocks:
            encoded = block(
                encoded,
                geometry,
                neighbor_index,
                neighbor_mask,
                candidate_mask,
            )
        pooled = self._masked_pool(encoded, candidate_mask)
        shape_rows = shape[:, None].expand(-1, landmarks, -1)
        landmark_rows = self.landmark_embedding[None].expand(batch, -1, -1)
        context = self.context_fusion(
            torch.cat([pooled, appearance, shape_rows, landmark_rows], dim=-1)
        )
        bilateral = self.bilateral_context(
            torch.cat([pooled[:, 1], pooled[:, 2], shape], dim=-1)
        ).reshape(batch, 2, self.hidden)
        context = context.clone()
        context[:, 1:3] = context[:, 1:3] + bilateral
        expanded_context = context[:, :, None].expand(-1, -1, encoded.shape[2], -1)
        curve_logits = self.curve_head(encoded + expanded_context).squeeze(-1)
        curve_probability = torch.sigmoid(curve_logits).masked_fill(
            ~candidate_mask, 0.0
        )
        support = self.support_encoder(
            torch.cat([encoded, expanded_context, curve_probability[..., None]], dim=-1)
        )
        # The second graph pass is explicitly conditioned on predicted curve
        # membership. It lets the landmark head reason along the support curve
        # instead of treating the complete 2D ROI as an unordered candidate set.
        support = support * (0.25 + curve_probability[..., None])
        support = self.support_graph(
            support,
            geometry,
            neighbor_index,
            neighbor_mask,
            candidate_mask,
        )
        landmark_logits = self.landmark_head(
            torch.cat(
                [
                    encoded,
                    support,
                    expanded_context,
                    curve_probability[..., None],
                ],
                dim=-1,
            )
        ).squeeze(-1)
        final_logits = landmark_logits + F.softplus(self.curve_scale)[None, :, None] * (
            curve_logits
        )
        final_logits = final_logits.masked_fill(~candidate_mask, -torch.inf)
        curve_logits = curve_logits.masked_fill(~candidate_mask, -torch.inf)
        log_variance = self.confidence_head(
            torch.cat([pooled, context], dim=-1)
        ).squeeze(-1)
        return {
            "curve_logits": curve_logits,
            "landmark_logits": landmark_logits.masked_fill(~candidate_mask, -torch.inf),
            "final_logits": final_logits,
            "log_variance": torch.clamp(log_variance, -4.0, 5.0),
        }


def probability_coordinate(logits, points, mask, temperature=1.0):
    probability = torch.softmax(
        logits.float().masked_fill(~mask, -torch.inf) / max(float(temperature), 1e-4),
        dim=-1,
    )
    coordinate = (probability[..., None] * points.float()).sum(dim=-2)
    return coordinate, probability
