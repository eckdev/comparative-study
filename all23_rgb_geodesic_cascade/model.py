import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anatomy import (
    CONTOUR_LANDMARKS,
    NUM_LANDMARKS,
    TEXTURE_LANDMARKS,
    graph_attention_mask,
    heatmap_sigma_mm,
    roi_radius_mm,
)


def segment_softmax(scores, index, segment_count):
    """Softmax over sparse edges grouped by destination vertex."""
    expanded = index[:, None].expand(-1, scores.shape[1])
    maximum = scores.new_full((segment_count, scores.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, expanded, scores, reduce="amax", include_self=True)
    exponent = torch.exp(scores - maximum[index])
    denominator = scores.new_zeros((segment_count, scores.shape[1]))
    denominator.index_add_(0, index, exponent)
    return exponent / denominator[index].clamp_min(1e-12)


def scatter_mean(values, index, count):
    output = values.new_zeros((count, values.shape[-1]))
    output.index_add_(0, index, values)
    denominator = values.new_zeros((count, 1))
    denominator.index_add_(0, index, values.new_ones((len(values), 1)))
    return output / denominator.clamp_min(1.0)


def scatter_max(values, index, count):
    expanded = index[:, None].expand(-1, values.shape[1])
    output = values.new_full((count, values.shape[1]), -torch.inf)
    output.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


class SparsePointTransformerBlock(nn.Module):
    def __init__(self, width, heads=4, dropout=0.1):
        super().__init__()
        if width % heads:
            raise ValueError("Point-transformer width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.position = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.attention = nn.Sequential(nn.Linear(self.head_dim, self.head_dim), nn.GELU(), nn.Linear(self.head_dim, 1))
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width)
        )

    def forward(self, features, points, edge_index):
        src, dst = edge_index
        normalized = self.norm1(features)
        query = self.query(normalized)[dst].reshape(-1, self.heads, self.head_dim)
        key = self.key(normalized)[src].reshape(-1, self.heads, self.head_dim)
        value = self.value(normalized)[src].reshape(-1, self.heads, self.head_dim)
        relative = points[src] - points[dst]
        position = self.position(relative).reshape(-1, self.heads, self.head_dim)
        score = self.attention(query - key + position).squeeze(-1) / math.sqrt(self.head_dim)
        weights = segment_softmax(score, dst, len(features))
        messages = (value + position) * weights[..., None]
        aggregated = features.new_zeros((len(features), self.heads, self.head_dim))
        aggregated.index_add_(0, dst, messages)
        features = features + self.dropout(self.output(aggregated.reshape(len(features), self.width)))
        return features + self.dropout(self.ffn(self.norm2(features)))


def topk_soft_coordinate(logits, candidates, mask, topk=30, temperature=0.75):
    masked = logits.masked_fill(~mask, -torch.inf)
    use_count = min(int(topk), logits.shape[-1]) if int(topk) > 0 else logits.shape[-1]
    values, indices = torch.topk(masked, use_count, dim=-1)
    valid = torch.gather(mask, -1, indices)
    values = values.masked_fill(~valid, -torch.inf)
    weights = torch.softmax(values / max(float(temperature), 1e-6), dim=-1)
    selected = torch.gather(candidates, 2, indices[..., None].expand(-1, -1, -1, 3))
    return torch.sum(weights[..., None] * selected, dim=2)


@torch.no_grad()
def mse_over_mesh_coordinate(logits, candidates, mask, sigmas):
    """Select the ROI vertex whose Gaussian best matches the predicted heatmap."""
    xyz = candidates.float()
    probabilities = torch.sigmoid(logits).float()
    pairwise = torch.cdist(xyz, xyz)
    sigma = sigmas[None, :, None, None].float().clamp_min(1e-6)
    gaussian = torch.exp(-(pairwise**2) / (2.0 * sigma**2))
    value_mask = mask[:, :, None, :].float()
    score = ((gaussian - probabilities[:, :, None, :]) ** 2 * value_mask).sum(dim=-1)
    score = score / value_mask.sum(dim=-1).clamp_min(1.0)
    score = score.masked_fill(~mask, torch.inf)
    selected = torch.argmin(score, dim=-1)
    return torch.gather(candidates, 2, selected[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)


class All23RGBGeodesicCascade(nn.Module):
    def __init__(
        self,
        input_dim=14,
        width=128,
        global_blocks=4,
        heads=4,
        dropout=0.1,
        coordinate_topk=30,
        coordinate_temperature=0.75,
        use_anatomical_attention=True,
        use_specialized_heads=True,
        use_local_refiner=True,
    ):
        super().__init__()
        self.width = int(width)
        self.coordinate_topk = int(coordinate_topk)
        self.coordinate_temperature = float(coordinate_temperature)
        self.use_anatomical_attention = bool(use_anatomical_attention)
        self.use_specialized_heads = bool(use_specialized_heads)
        self.use_local_refiner = bool(use_local_refiner)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width)
        )
        self.global_blocks = nn.ModuleList(
            [SparsePointTransformerBlock(width, heads, dropout) for _ in range(int(global_blocks))]
        )
        self.landmark_embedding = nn.Parameter(torch.randn(NUM_LANDMARKS, width) * 0.02)
        self.global_context = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width))
        self.coarse_point_key = nn.Linear(width, width, bias=False)
        self.coarse_token_query = nn.Linear(width, width, bias=False)

        self.relative_position = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.intra_norm = nn.LayerNorm(width)
        self.intra_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.intra_ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.inter_norm = nn.LayerNorm(width)
        self.inter_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.inter_ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.local_shared = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout))
        self.score_heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
                for name in ("texture", "contour", "generic")
            }
        )
        self.region_head = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.confidence_head = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.register_buffer("anatomical_attention_mask", graph_attention_mask(), persistent=False)
        self.register_buffer(
            "roi_radii", torch.tensor([roi_radius_mm(index) for index in range(NUM_LANDMARKS)]), persistent=False
        )
        self.register_buffer(
            "heatmap_sigmas", torch.tensor([heatmap_sigma_mm(index) for index in range(NUM_LANDMARKS)]), persistent=False
        )

    def encode_surface(self, features, points, edge_index):
        encoded = self.input_projection(features)
        for block in self.global_blocks:
            encoded = block(encoded, points, edge_index)
        return encoded

    def coarse_heatmaps(self, encoded, batch, batch_count):
        pooled = torch.cat([scatter_mean(encoded, batch, batch_count), scatter_max(encoded, batch, batch_count)], dim=-1)
        context = self.global_context(pooled)
        tokens = context[:, None, :] + self.landmark_embedding[None, :, :]
        point_keys = self.coarse_point_key(encoded)
        token_queries = self.coarse_token_query(tokens)
        logits = torch.sum(point_keys[:, None, :] * token_queries[batch], dim=-1) / math.sqrt(self.width)
        return logits, tokens

    def _coarse_coordinates(self, logits, points, batch, batch_count, vertex_mask):
        coordinates = []
        for batch_index in range(batch_count):
            selected = batch == batch_index
            local_logits = logits[selected].transpose(0, 1)[None]
            local_points = points[selected][None, None].expand(1, NUM_LANDMARKS, -1, 3)
            local_mask = vertex_mask[selected][None, None].expand(1, NUM_LANDMARKS, -1)
            coordinates.append(
                topk_soft_coordinate(
                    local_logits,
                    local_points,
                    local_mask,
                    self.coordinate_topk,
                    self.coordinate_temperature,
                )[0]
            )
        return torch.stack(coordinates)

    def forward(self, batch, coordinate_mode="topk"):
        points = batch["points"]
        graph_batch = batch["batch"]
        batch_count = int(batch["coarse"].shape[0])
        encoded = self.encode_surface(batch["features"], points, batch["edge_index"])
        coarse_logits, global_tokens = self.coarse_heatmaps(encoded, graph_batch, batch_count)
        coarse_coordinates = self._coarse_coordinates(
            coarse_logits, points, graph_batch, batch_count, batch["vertex_mask"]
        )

        roi_index = batch["roi_index"]
        roi_mask = batch["roi_mask"]
        if not self.use_local_refiner:
            all_local_logits = coarse_logits[roi_index]
            landmark_index = torch.arange(NUM_LANDMARKS, device=points.device)[None, :, None, None]
            landmark_index = landmark_index.expand(batch_count, -1, roi_index.shape[-1], 1)
            local_logits = torch.gather(all_local_logits, -1, landmark_index).squeeze(-1)
            local_logits = local_logits.masked_fill(~roi_mask, -torch.inf)
            return {
                "coarse_logits": coarse_logits,
                "coarse_coordinates": coarse_coordinates,
                "local_logits": local_logits,
                "region_logits": local_logits.masked_fill(~roi_mask, -20.0),
                "final_coordinates": coarse_coordinates,
                "log_var": self.confidence_head(global_tokens).squeeze(-1).clamp(-6.0, 6.0),
                "candidate_points": points[roi_index],
                "tokens": global_tokens,
            }
        local_features = encoded[roi_index]
        candidates = points[roi_index]
        relative = candidates - batch["coarse"][:, :, None, :]
        radii = self.roi_radii[None, :, None, None].clamp_min(1e-6)
        radial = torch.linalg.norm(relative, dim=-1, keepdim=True) / radii
        local_features = local_features + self.relative_position(torch.cat([relative / radii, radial], dim=-1))
        local_features = local_features + self.landmark_embedding[None, :, None, :]

        flat = local_features.reshape(-1, local_features.shape[2], self.width)
        flat_mask = roi_mask.reshape(-1, roi_mask.shape[-1])
        normalized = self.intra_norm(flat)
        attended, _ = self.intra_attention(
            normalized, normalized, normalized, key_padding_mask=~flat_mask, need_weights=False
        )
        flat = flat + attended
        flat = flat + self.intra_ffn(flat)
        local_features = flat.reshape(batch_count, NUM_LANDMARKS, -1, self.width)
        mask_float = roi_mask[..., None].float()
        tokens = (local_features * mask_float).sum(dim=2) / mask_float.sum(dim=2).clamp_min(1.0)
        tokens = tokens + global_tokens
        normalized_tokens = self.inter_norm(tokens)
        if self.use_anatomical_attention:
            graph_tokens, _ = self.inter_attention(
                normalized_tokens,
                normalized_tokens,
                normalized_tokens,
                attn_mask=self.anatomical_attention_mask,
                need_weights=False,
            )
            tokens = tokens + graph_tokens
            tokens = tokens + self.inter_ffn(tokens)
        conditioned = self.local_shared(torch.cat([local_features, tokens[:, :, None, :].expand_as(local_features)], dim=-1))

        local_logits = conditioned.new_zeros(conditioned.shape[:-1])
        texture = torch.tensor(TEXTURE_LANDMARKS, device=conditioned.device)
        contour = torch.tensor(CONTOUR_LANDMARKS, device=conditioned.device)
        all_indices = set(range(NUM_LANDMARKS))
        generic = torch.tensor(sorted(all_indices - set(TEXTURE_LANDMARKS) - set(CONTOUR_LANDMARKS)), device=conditioned.device)
        if self.use_specialized_heads:
            for name, indices in (("texture", texture), ("contour", contour), ("generic", generic)):
                local_logits[:, indices] = self.score_heads[name](conditioned[:, indices]).squeeze(-1)
        else:
            local_logits = self.score_heads["generic"](conditioned).squeeze(-1)
        local_logits = local_logits.masked_fill(~roi_mask, -torch.inf)
        region_logits = self.region_head(conditioned).squeeze(-1).masked_fill(~roi_mask, -20.0)
        log_var = self.confidence_head(tokens).squeeze(-1).clamp(-6.0, 6.0)

        soft_coordinates = topk_soft_coordinate(
            local_logits,
            candidates,
            roi_mask,
            self.coordinate_topk,
            self.coordinate_temperature,
        )
        if coordinate_mode == "mse_over_mesh":
            hard_coordinates = mse_over_mesh_coordinate(local_logits, candidates, roi_mask, self.heatmap_sigmas)
            final_coordinates = hard_coordinates + soft_coordinates - soft_coordinates.detach()
        elif coordinate_mode == "topk":
            final_coordinates = soft_coordinates
        else:
            raise ValueError(f"Unknown coordinate mode: {coordinate_mode}")
        return {
            "coarse_logits": coarse_logits,
            "coarse_coordinates": coarse_coordinates,
            "local_logits": local_logits,
            "region_logits": region_logits,
            "final_coordinates": final_coordinates,
            "log_var": log_var,
            "candidate_points": candidates,
            "tokens": tokens,
        }
