import math

import torch
import torch.nn as nn

from all23_rgb_geodesic_cascade.anatomy import (
    HARD3,
    NUM_LANDMARKS,
    graph_attention_mask,
)
from all23_rgb_geodesic_cascade.model import (
    All23RGBGeodesicCascade,
    scatter_max,
    scatter_mean,
)


class AnatomicalTokenBlock(nn.Module):
    """Surface cross-attention followed by canonical landmark-graph attention."""

    def __init__(self, width, heads, dropout=0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.surface_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.graph_norm = nn.LayerNorm(width)
        self.graph_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, tokens, surfaces, graph_mask):
        rows = []
        for sample_index, surface in enumerate(surfaces):
            query = self.token_norm(tokens[sample_index : sample_index + 1])
            key_value = self.surface_norm(surface[None])
            attended, _ = self.cross_attention(
                query, key_value, key_value, need_weights=False
            )
            rows.append(tokens[sample_index : sample_index + 1] + attended)
        tokens = torch.cat(rows, dim=0)
        normalized = self.graph_norm(tokens)
        graph, _ = self.graph_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=graph_mask,
            need_weights=False,
        )
        tokens = tokens + graph
        return tokens + self.ffn(tokens)


class AGHFormerVNext(All23RGBGeodesicCascade):
    """AGH tokens plus mesh-geometric, RGB-geodesic coarse-to-fine refinement.

    The external Stage 1 coordinate is label-safe and out-of-fold for outer-train
    samples. A global heatmap and bounded residual propose an alternative center;
    a landmark/sample-specific gate blends both before geodesic refinement.
    """

    def __init__(
        self,
        *args,
        token_blocks=2,
        token_surface_points=4096,
        fusion_residual_limit_mm=8.0,
        fusion_hard_residual_limit_mm=15.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.token_surface_points = max(256, int(token_surface_points))
        self.coarse_position_projection = nn.Sequential(
            nn.Linear(3, self.width),
            nn.LayerNorm(self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )
        self.token_blocks = nn.ModuleList(
            [
                AnatomicalTokenBlock(
                    self.width,
                    kwargs.get("heads", 4),
                    kwargs.get("dropout", 0.1),
                )
                for _ in range(max(1, int(token_blocks)))
            ]
        )
        self.global_residual_head = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, 3),
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(self.width + 2),
            nn.Linear(self.width + 2, self.width // 2),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(self.width // 2, 1),
        )
        self.fusion_log_var_head = nn.Sequential(
            nn.LayerNorm(self.width), nn.Linear(self.width, 1)
        )
        nn.init.zeros_(self.global_residual_head[-1].weight)
        nn.init.zeros_(self.global_residual_head[-1].bias)
        nn.init.zeros_(self.fusion_gate[-1].weight)
        nn.init.constant_(self.fusion_gate[-1].bias, -1.5)
        residual_limits = torch.full(
            (NUM_LANDMARKS,), float(fusion_residual_limit_mm), dtype=torch.float32
        )
        residual_limits[list(HARD3)] = float(fusion_hard_residual_limit_mm)
        self.register_buffer(
            "fusion_residual_limits", residual_limits, persistent=True
        )
        self.register_buffer(
            "token_graph_mask", graph_attention_mask(), persistent=False
        )

    def _surface_rows(self, encoded, graph_batch):
        rows = []
        batch_count = int(graph_batch.max().item()) + 1
        for batch_index in range(batch_count):
            row = encoded[graph_batch == batch_index]
            if len(row) > self.token_surface_points:
                indices = torch.linspace(
                    0,
                    len(row) - 1,
                    self.token_surface_points,
                    device=row.device,
                ).round().long()
                row = row[indices]
            rows.append(row)
        return rows

    @staticmethod
    def _normalize_coarse(coarse):
        center = coarse.mean(dim=1, keepdim=True)
        scale = torch.linalg.norm(coarse - center, dim=-1).mean(
            dim=1, keepdim=True
        )
        return (coarse - center) / scale.clamp_min(1.0)[..., None]

    def coarse_heatmaps(self, encoded, batch, batch_count, external_coarse=None):
        pooled = torch.cat(
            [
                scatter_mean(encoded, batch, batch_count),
                scatter_max(encoded, batch, batch_count),
            ],
            dim=-1,
        )
        context = self.global_context(pooled)
        tokens = context[:, None, :] + self.landmark_embedding[None, :, :]
        if external_coarse is not None:
            tokens = tokens + self.coarse_position_projection(
                self._normalize_coarse(external_coarse.float())
            ).to(tokens.dtype)
        surfaces = self._surface_rows(encoded, batch)
        for block in self.token_blocks:
            tokens = block(tokens, surfaces, self.token_graph_mask)
        point_keys = self.coarse_point_key(encoded)
        token_queries = self.coarse_token_query(tokens)
        logits = torch.sum(
            point_keys[:, None, :] * token_queries[batch], dim=-1
        ) / math.sqrt(self.width)
        return logits, tokens

    @staticmethod
    def _normalized_entropy(coarse_logits, graph_batch, vertex_mask):
        rows = []
        batch_count = int(graph_batch.max().item()) + 1
        for batch_index in range(batch_count):
            selected = graph_batch == batch_index
            logits = coarse_logits[selected].transpose(0, 1).float()
            mask = vertex_mask[selected][None].expand(NUM_LANDMARKS, -1)
            probabilities = torch.softmax(
                logits.masked_fill(~mask, -torch.inf), dim=-1
            )
            probabilities = torch.where(
                mask, probabilities, torch.zeros_like(probabilities)
            )
            entropy = -(
                probabilities * torch.log(probabilities.clamp_min(1e-8))
            ).sum(dim=-1)
            entropy = entropy / torch.log(mask.sum(dim=-1).float().clamp_min(2.0))
            rows.append(entropy)
        return torch.stack(rows)

    def fuse_coarse_coordinates(self, tokens, heatmap_coordinates, coarse_logits, batch):
        residual = torch.tanh(self.global_residual_head(tokens).float())
        residual = residual * self.fusion_residual_limits[None, :, None]
        heatmap_residual = heatmap_coordinates.float() + residual
        external = batch["coarse"].float()
        scale = torch.linalg.norm(
            external - external.mean(dim=1, keepdim=True), dim=-1
        ).mean(dim=1, keepdim=True)
        disagreement = torch.linalg.norm(
            heatmap_residual - external, dim=-1
        ) / scale.clamp_min(1.0)
        entropy = self._normalized_entropy(
            coarse_logits, batch["batch"], batch["vertex_mask"]
        )
        gate_input = torch.cat(
            [tokens.float(), disagreement[..., None], entropy[..., None]], dim=-1
        )
        alpha = torch.sigmoid(self.fusion_gate(gate_input).squeeze(-1))
        fused = external + alpha[..., None] * (heatmap_residual - external)
        return fused, {
            "coarse_loss_coordinates": fused,
            "fusion_alpha": alpha,
            "fusion_log_var": self.fusion_log_var_head(tokens).squeeze(-1).clamp(-6.0, 6.0),
            "heatmap_residual_coordinates": heatmap_residual,
        }

    def forward(self, batch, coordinate_mode="topk"):
        outputs = super().forward(batch, coordinate_mode=coordinate_mode)
        if "fusion_log_var" in outputs:
            local_log_var = outputs["log_var"].float()
            outputs["local_log_var"] = local_log_var
            outputs["log_var"] = torch.logaddexp(
                local_log_var, outputs["fusion_log_var"].float()
            ).clamp(-6.0, 6.0)
        return outputs
