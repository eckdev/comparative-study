import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anatomy import (
    CONTOUR_LANDMARKS,
    HARD3,
    NUM_LANDMARKS,
    TEXTURE_LANDMARKS,
    graph_attention_mask,
    heatmap_sigma_mm,
    roi_radius_mm,
)


def segment_softmax(scores, index, segment_count):
    """Softmax over sparse edges grouped by destination vertex."""
    # CUDA scatter operations may promote exp() to float32 under autocast while
    # new_zeros() remains float16. Accumulate the normalization in float32 for
    # both dtype consistency and numerical stability, then restore score dtype.
    work_scores = scores.float()
    expanded = index[:, None].expand(-1, work_scores.shape[1])
    maximum = work_scores.new_full((segment_count, work_scores.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, expanded, work_scores, reduce="amax", include_self=True)
    exponent = torch.exp(work_scores - maximum[index])
    denominator = work_scores.new_zeros((segment_count, work_scores.shape[1]))
    denominator.index_add_(0, index, exponent)
    return (exponent / denominator[index].clamp_min(1e-12)).to(scores.dtype)


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
def nearest_candidate_coordinate(coordinates, candidates, mask):
    """Project coordinates onto the closest valid ROI vertex."""
    distances = torch.linalg.norm(candidates.float() - coordinates.float()[:, :, None], dim=-1)
    distances = distances.masked_fill(~mask, torch.inf)
    selected = torch.argmin(distances, dim=-1)
    return torch.gather(
        candidates,
        2,
        selected[..., None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)


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


class GlobalCoarseNetwork(nn.Module):
    """Stage 1 full-surface network used to center dynamic Stage 2 ROIs."""

    def __init__(
        self,
        input_dim=14,
        width=96,
        global_blocks=3,
        heads=4,
        dropout=0.1,
        coordinate_topk=50,
        coordinate_temperature=0.75,
    ):
        super().__init__()
        self.width = int(width)
        self.coordinate_topk = int(coordinate_topk)
        self.coordinate_temperature = float(coordinate_temperature)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.global_blocks = nn.ModuleList(
            [SparsePointTransformerBlock(width, heads, dropout) for _ in range(int(global_blocks))]
        )
        self.landmark_embedding = nn.Parameter(torch.randn(NUM_LANDMARKS, width) * 0.02)
        self.global_context = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width)
        )
        self.point_key = nn.Linear(width, width, bias=False)
        self.token_query = nn.Linear(width, width, bias=False)
        self.confidence_head = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )

    def forward(self, batch):
        points = batch["points"]
        graph_batch = batch["batch"]
        batch_count = len(batch["sample_id"])
        encoded = self.input_projection(batch["features"])
        for block in self.global_blocks:
            encoded = block(encoded, points, batch["edge_index"])
        pooled = torch.cat(
            [
                scatter_mean(encoded, graph_batch, batch_count),
                scatter_max(encoded, graph_batch, batch_count),
            ],
            dim=-1,
        )
        tokens = self.global_context(pooled)[:, None, :] + self.landmark_embedding[None, :, :]
        point_keys = self.point_key(encoded)
        token_queries = self.token_query(tokens)
        logits = torch.sum(point_keys[:, None, :] * token_queries[graph_batch], dim=-1)
        logits = logits / math.sqrt(self.width)
        coordinates = []
        for batch_index in range(batch_count):
            selected = graph_batch == batch_index
            local_logits = logits[selected].transpose(0, 1)[None]
            local_points = points[selected][None, None].expand(1, NUM_LANDMARKS, -1, 3)
            local_mask = batch["vertex_mask"][selected][None, None].expand(
                1, NUM_LANDMARKS, -1
            )
            coordinates.append(
                topk_soft_coordinate(
                    local_logits,
                    local_points,
                    local_mask,
                    self.coordinate_topk,
                    self.coordinate_temperature,
                )[0]
            )
        return {
            "coarse_logits": logits,
            "coordinates": torch.stack(coordinates),
            "log_var": self.confidence_head(tokens).squeeze(-1).clamp(-6.0, 6.0),
        }


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
        use_refinement_gate=False,
        use_hard_candidate_ranker=False,
        use_e10_rankers=False,
        hard_coordinate_topk=8,
        hard_coordinate_temperature=0.35,
        gonion_pair_topk=32,
        roi_radius_scale=1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.coordinate_topk = int(coordinate_topk)
        self.coordinate_temperature = float(coordinate_temperature)
        self.use_anatomical_attention = bool(use_anatomical_attention)
        self.use_specialized_heads = bool(use_specialized_heads)
        self.use_local_refiner = bool(use_local_refiner)
        self.use_refinement_gate = bool(use_refinement_gate)
        self.use_hard_candidate_ranker = bool(use_hard_candidate_ranker)
        self.use_e10_rankers = bool(use_e10_rankers)
        if self.use_e10_rankers and not self.use_hard_candidate_ranker:
            raise ValueError("E10 rankers require use_hard_candidate_ranker=True")
        self.hard_coordinate_topk = int(hard_coordinate_topk)
        self.hard_coordinate_temperature = float(hard_coordinate_temperature)
        self.gonion_pair_topk = int(gonion_pair_topk)
        self.roi_radius_scale = float(roi_radius_scale)
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
        self.trichion_context = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.gonion_pair_context = nn.Sequential(
            nn.Linear(width * 3, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2),
        )
        self.local_shared = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout))
        self.score_heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
                for name in ("texture", "contour", "generic")
            }
        )
        self.region_head = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.confidence_head = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
        self.refinement_gate = nn.Sequential(
            nn.Linear(width + 5, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
        )
        if self.use_hard_candidate_ranker and not self.use_e10_rankers:
            # Four stable coarse anchors describe each difficult candidate in
            # an anatomy-relative coordinate frame. Trichion uses upper/midline
            # anchors; each Gonion uses the lower midline and its contralateral mate.
            self.hard_anchor_projection = nn.Sequential(
                nn.Linear(4 * 4, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.hard_candidate_ranker = nn.Sequential(
                nn.Linear(width * 2, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, width // 2),
                nn.GELU(),
                nn.Linear(width // 2, 1),
            )
        if self.use_e10_rankers:
            # Trichion is primarily a texture-boundary landmark. Its E10 head
            # sees RGB, local colour contrast and geometry relative to refined
            # upper-midline landmarks rather than sharing the Gonion ranker.
            self.trichion_texture_projection = nn.Sequential(
                nn.Linear(7, width // 2),
                nn.LayerNorm(width // 2),
                nn.GELU(),
            )
            self.trichion_anchor_projection = nn.Sequential(
                nn.Linear(4 * 4, width // 2),
                nn.LayerNorm(width // 2),
                nn.GELU(),
            )
            self.trichion_candidate_ranker = nn.Sequential(
                nn.Linear(width * 2, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, width // 2),
                nn.GELU(),
                nn.Linear(width // 2, 1),
            )

            # Gonion candidates are scored jointly. Unary geometry uses only
            # refined LM10-12 lower-midline anchors; no contralateral coarse
            # landmark is exposed to this head.
            self.gonion_anchor_projection = nn.Sequential(
                nn.Linear(3 * 4, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.gonion_unary_ranker = nn.Sequential(
                nn.Linear(width * 2, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, 1),
            )
            pair_width = max(width // 2, 16)
            self.gonion_pair_left = nn.Linear(width * 2, pair_width)
            self.gonion_pair_right = nn.Linear(width * 2, pair_width)
            self.gonion_pair_geometry = nn.Sequential(
                nn.Linear(10, pair_width), nn.GELU(), nn.Linear(pair_width, pair_width)
            )
            self.gonion_pair_score = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pair_width, 1),
            )
        if self.use_hard_candidate_ranker:
            self.hard_refinement_gate = nn.Sequential(
                nn.Linear(width + 5, width // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width // 2, 1),
            )
        self.register_buffer(
            "hard_indices",
            torch.tensor(HARD3, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "hard_anchor_indices",
            torch.tensor(
                [
                    [1, 2, 3, 12],
                    [10, 11, 12, 22],
                    [10, 11, 12, 21],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer("anatomical_attention_mask", graph_attention_mask(), persistent=False)
        self.register_buffer(
            "roi_radii",
            torch.tensor([roi_radius_mm(index) * self.roi_radius_scale for index in range(NUM_LANDMARKS)]),
            persistent=False,
        )
        self.register_buffer(
            "heatmap_sigmas", torch.tensor([heatmap_sigma_mm(index) for index in range(NUM_LANDMARKS)]), persistent=False
        )

    def encode_surface(self, features, points, edge_index):
        encoded = self.input_projection(features)
        for block in self.global_blocks:
            encoded = block(encoded, points, edge_index)
        return encoded

    def refinement_gate_modules(self):
        modules = [self.refinement_gate]
        if hasattr(self, "hard_refinement_gate"):
            modules.append(self.hard_refinement_gate)
        return modules

    def set_refinement_gate_trainable(self, trainable):
        for module in self.refinement_gate_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(bool(trainable))

    def refinement_gate_is_trainable(self):
        return any(
            parameter.requires_grad
            for module in self.refinement_gate_modules()
            for parameter in module.parameters()
        )

    def refinement_gate_state_dict(self):
        state = {"refinement_gate": self.refinement_gate.state_dict()}
        if hasattr(self, "hard_refinement_gate"):
            state["hard_refinement_gate"] = self.hard_refinement_gate.state_dict()
        return state

    def load_refinement_gate_state_dict(self, state):
        self.refinement_gate.load_state_dict(state["refinement_gate"])
        if hasattr(self, "hard_refinement_gate") and "hard_refinement_gate" in state:
            self.hard_refinement_gate.load_state_dict(state["hard_refinement_gate"])

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

    def _hard_anchor_features(self, candidates, coarse, radii):
        hard_candidates = candidates[:, self.hard_indices]
        anchors = coarse[:, self.hard_anchor_indices]
        vectors = hard_candidates[:, :, :, None, :] - anchors[:, :, None, :, :]
        hard_radii = radii[:, self.hard_indices, 0, 0][:, :, None, None, None]
        vectors = vectors / hard_radii.clamp_min(1e-6)
        distances = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        return torch.cat([vectors, distances], dim=-1).flatten(start_dim=-2)

    @staticmethod
    def _candidate_anchor_features(candidates, anchors, radius):
        vectors = candidates[:, :, None, :] - anchors[:, None, :, :]
        vectors = vectors / radius[:, None, None, None].clamp_min(1e-6)
        distances = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        return torch.cat([vectors, distances], dim=-1).flatten(start_dim=-2)

    @staticmethod
    def _gather_candidates(values, indices):
        return torch.gather(
            values,
            1,
            indices[..., None].expand(-1, -1, values.shape[-1]),
        )

    @staticmethod
    def _bilateral_geometry(left, right, lower_midline, radius):
        left = left[:, :, None, :]
        right = right[:, None, :, :]
        scale = radius[:, None, None, None].clamp_min(1e-6)
        pair_vector = (right - left) / scale
        pair_distance = torch.linalg.norm(pair_vector, dim=-1, keepdim=True)
        midpoint = (0.5 * (left + right) - lower_midline[:, None, None, :]) / scale
        midline_x = lower_midline[:, None, None, 0:1]
        lateral_balance = ((left[..., 0:1] - midline_x) + (right[..., 0:1] - midline_x)) / scale
        yz_difference = (left[..., 1:3] - right[..., 1:3]) / scale
        return torch.cat(
            [pair_vector, pair_distance, midpoint, lateral_balance, yz_difference],
            dim=-1,
        )

    def _trichion_e10_logits(
        self, conditioned, candidate_features, candidates, refined_anchors, radii
    ):
        if candidate_features.shape[-1] < 9:
            raise ValueError("E10 Trichion ranker requires XYZ + RGB + RGB-contrast features")
        rgb = candidate_features[:, 0, :, 3:6]
        contrast = candidate_features[:, 0, :, 6:9]
        texture = torch.cat(
            [rgb, contrast, torch.linalg.norm(contrast, dim=-1, keepdim=True)],
            dim=-1,
        )
        texture_embedding = self.trichion_texture_projection(texture)
        anchors = refined_anchors[:, [1, 2, 3, 12]]
        geometry = self._candidate_anchor_features(
            candidates[:, 0], anchors, radii[:, 0, 0, 0]
        )
        anchor_embedding = self.trichion_anchor_projection(geometry)
        ranker_input = torch.cat(
            [conditioned[:, 0], texture_embedding, anchor_embedding], dim=-1
        )
        return self.trichion_candidate_ranker(ranker_input).squeeze(-1)

    def _gonion_e10_logits(self, conditioned, candidates, mask, refined_anchors, radii):
        anchors = refined_anchors[:, 10:13]
        lower_midline = anchors.mean(dim=1)
        radius = 0.5 * (radii[:, 21, 0, 0] + radii[:, 22, 0, 0])
        descriptors = []
        unary_logits = []
        for landmark in (21, 22):
            geometry = self._candidate_anchor_features(
                candidates[:, landmark], anchors, radius
            )
            anchor_embedding = self.gonion_anchor_projection(geometry)
            descriptor = torch.cat(
                [conditioned[:, landmark], anchor_embedding], dim=-1
            )
            descriptors.append(descriptor)
            unary_logits.append(self.gonion_unary_ranker(descriptor).squeeze(-1))

        left_descriptor, right_descriptor = descriptors
        left_unary, right_unary = unary_logits
        left_mask, right_mask = mask[:, 21], mask[:, 22]
        pair_count = min(max(self.gonion_pair_topk, 1), left_unary.shape[-1])
        left_top = torch.topk(
            left_unary.masked_fill(~left_mask, -torch.inf), pair_count, dim=-1
        ).indices
        right_top = torch.topk(
            right_unary.masked_fill(~right_mask, -torch.inf), pair_count, dim=-1
        ).indices

        selected_left_descriptor = self._gather_candidates(left_descriptor, left_top)
        selected_right_descriptor = self._gather_candidates(right_descriptor, right_top)
        selected_left_points = self._gather_candidates(candidates[:, 21], left_top)
        selected_right_points = self._gather_candidates(candidates[:, 22], right_top)
        selected_left_unary = torch.gather(left_unary, 1, left_top)
        selected_right_unary = torch.gather(right_unary, 1, right_top)
        selected_left_mask = torch.gather(left_mask, 1, left_top)
        selected_right_mask = torch.gather(right_mask, 1, right_top)

        left_hidden = self.gonion_pair_left(left_descriptor)[:, :, None, :]
        right_hidden = self.gonion_pair_right(selected_right_descriptor)[:, None, :, :]
        left_geometry = self._bilateral_geometry(
            candidates[:, 21], selected_right_points, lower_midline, radius
        )
        left_pair_score = self.gonion_pair_score(
            left_hidden + right_hidden + self.gonion_pair_geometry(left_geometry)
        ).squeeze(-1)
        left_joint = (
            left_unary[:, :, None] + selected_right_unary[:, None, :] + left_pair_score
        )
        left_pair_mask = left_mask[:, :, None] & selected_right_mask[:, None, :]
        left_joint = left_joint.masked_fill(~left_pair_mask, -torch.inf)
        left_marginal = torch.logsumexp(left_joint.float(), dim=-1)
        left_marginal = left_marginal - torch.log(
            selected_right_mask.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
        )

        left_hidden = self.gonion_pair_left(selected_left_descriptor)[:, :, None, :]
        right_hidden = self.gonion_pair_right(right_descriptor)[:, None, :, :]
        right_geometry = self._bilateral_geometry(
            selected_left_points, candidates[:, 22], lower_midline, radius
        )
        right_pair_score = self.gonion_pair_score(
            left_hidden + right_hidden + self.gonion_pair_geometry(right_geometry)
        ).squeeze(-1)
        right_joint = (
            selected_left_unary[:, :, None] + right_unary[:, None, :] + right_pair_score
        )
        right_pair_mask = selected_left_mask[:, :, None] & right_mask[:, None, :]
        right_joint = right_joint.masked_fill(~right_pair_mask, -torch.inf)
        right_marginal = torch.logsumexp(right_joint.float(), dim=1)
        right_marginal = right_marginal - torch.log(
            selected_left_mask.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
        )
        return left_marginal.to(left_unary.dtype), right_marginal.to(right_unary.dtype)

    def coordinates_from_logits(self, logits, candidates, mask, coordinate_mode):
        soft_coordinates = topk_soft_coordinate(
            logits,
            candidates,
            mask,
            self.coordinate_topk,
            self.coordinate_temperature,
        )
        if coordinate_mode == "mse_over_mesh":
            surface_coordinates = mse_over_mesh_coordinate(
                logits, candidates, mask, self.heatmap_sigmas
            )
            refined = surface_coordinates + soft_coordinates - soft_coordinates.detach()
        elif coordinate_mode == "topk":
            refined = soft_coordinates
        else:
            raise ValueError(f"Unknown coordinate mode: {coordinate_mode}")

        if self.use_hard_candidate_ranker:
            hard_logits = logits[:, self.hard_indices]
            hard_candidates = candidates[:, self.hard_indices]
            hard_mask = mask[:, self.hard_indices]
            hard_soft = topk_soft_coordinate(
                hard_logits,
                hard_candidates,
                hard_mask,
                self.hard_coordinate_topk,
                self.hard_coordinate_temperature,
            )
            hard_surface = nearest_candidate_coordinate(
                hard_soft, hard_candidates, hard_mask
            )
            hard_straight_through = hard_surface + hard_soft - hard_soft.detach()
            refined = refined.clone()
            refined[:, self.hard_indices] = hard_straight_through
        return refined

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
                "refined_coordinates": coarse_coordinates,
                "refinement_alpha": coarse_coordinates.new_ones(
                    (batch_count, NUM_LANDMARKS)
                ),
                "log_var": self.confidence_head(global_tokens).squeeze(-1).clamp(-6.0, 6.0),
                "candidate_points": points[roi_index],
                "tokens": global_tokens,
            }
        local_features = encoded[roi_index]
        candidates = points[roi_index]
        relative = candidates - batch["coarse"][:, :, None, :]
        sample_radius_scale = batch.get("sample_radius_scale")
        if sample_radius_scale is None:
            sample_radius_scale = self.roi_radii.new_ones((batch_count,))
        radii = (
            self.roi_radii[None, :, None, None]
            * sample_radius_scale[:, None, None, None]
        ).clamp_min(1e-6)
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
        if self.use_specialized_heads:
            hard_context = torch.zeros_like(tokens)
            hard_context[:, 0] = self.trichion_context(
                torch.cat([tokens[:, 0], tokens[:, 1], tokens[:, 2]], dim=-1)
            )
            lower_midline = tokens[:, 10:13].mean(dim=1)
            gonion_context = self.gonion_pair_context(
                torch.cat([tokens[:, 21], tokens[:, 22], lower_midline], dim=-1)
            ).reshape(batch_count, 2, self.width)
            hard_context[:, 21:23] = gonion_context
            tokens = tokens + hard_context
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
        if self.use_e10_rankers:
            base_logits = local_logits.masked_fill(~roi_mask, -torch.inf)
            # These anchors are refined predictions from the same frozen-in-path
            # forward pass. Detaching prevents the hard-landmark objective from
            # degrading already stable LM1-3 and LM10-12 estimates.
            refined_anchors = topk_soft_coordinate(
                base_logits,
                candidates,
                roi_mask,
                self.coordinate_topk,
                self.coordinate_temperature,
            ).detach()
            candidate_features = batch["features"][roi_index]
            trichion_logits = self._trichion_e10_logits(
                conditioned,
                candidate_features,
                candidates,
                refined_anchors,
                radii,
            )
            gonion_left_logits, gonion_right_logits = self._gonion_e10_logits(
                conditioned,
                candidates,
                roi_mask,
                refined_anchors,
                radii,
            )
            local_logits = local_logits.clone()
            local_logits[:, 0] = trichion_logits
            local_logits[:, 21] = gonion_left_logits
            local_logits[:, 22] = gonion_right_logits
        elif self.use_hard_candidate_ranker:
            anchor_features = self._hard_anchor_features(candidates, batch["coarse"], radii)
            anchor_embedding = self.hard_anchor_projection(anchor_features)
            hard_conditioned = torch.cat(
                [conditioned[:, self.hard_indices], anchor_embedding], dim=-1
            )
            hard_logits = self.hard_candidate_ranker(hard_conditioned).squeeze(-1)
            local_logits = local_logits.clone()
            local_logits[:, self.hard_indices] = hard_logits
        local_logits = local_logits.masked_fill(~roi_mask, -torch.inf)
        region_logits = self.region_head(conditioned).squeeze(-1).masked_fill(~roi_mask, -20.0)
        log_var = self.confidence_head(tokens).squeeze(-1).clamp(-6.0, 6.0)

        refined_coordinates = self.coordinates_from_logits(
            local_logits,
            candidates,
            roi_mask,
            coordinate_mode,
        )

        if self.use_refinement_gate:
            work_logits = local_logits.float().masked_fill(~roi_mask, -torch.inf)
            probability = torch.softmax(work_logits, dim=-1)
            probability = torch.where(roi_mask, probability, torch.zeros_like(probability))
            entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=-1)
            valid_count = roi_mask.sum(dim=-1).float().clamp_min(2.0)
            entropy = entropy / torch.log(valid_count)
            top_values = torch.topk(probability, k=min(2, probability.shape[-1]), dim=-1).values
            if top_values.shape[-1] == 1:
                margin = top_values[..., 0]
            else:
                margin = top_values[..., 0] - top_values[..., 1]
            scalar_radius = radii.squeeze(-1).squeeze(-1)
            delta_norm = torch.linalg.norm(
                refined_coordinates.detach() - batch["coarse"], dim=-1
            ) / scalar_radius
            coarse_disagreement = torch.linalg.norm(
                coarse_coordinates.detach() - batch["coarse"], dim=-1
            ) / scalar_radius
            diagnostics = torch.stack(
                [
                    log_var.detach(),
                    delta_norm,
                    entropy.detach(),
                    margin.detach(),
                    coarse_disagreement,
                ],
                dim=-1,
            )
            gate_input = torch.cat([tokens, diagnostics], dim=-1)
            if not self.refinement_gate_is_trainable():
                gate_input = gate_input.detach()
            gate_logits = self.refinement_gate(gate_input).squeeze(-1)
            if self.use_hard_candidate_ranker:
                gate_logits = gate_logits.clone()
                gate_logits[:, self.hard_indices] = self.hard_refinement_gate(
                    gate_input[:, self.hard_indices]
                ).squeeze(-1)
            refinement_alpha = torch.sigmoid(gate_logits)
        else:
            gate_logits = refined_coordinates.new_zeros(
                (batch_count, NUM_LANDMARKS)
            )
            refinement_alpha = refined_coordinates.new_ones(
                (batch_count, NUM_LANDMARKS)
            )
        final_coordinates = batch["coarse"] + refinement_alpha[..., None] * (
            refined_coordinates - batch["coarse"]
        )
        return {
            "coarse_logits": coarse_logits,
            "coarse_coordinates": coarse_coordinates,
            "local_logits": local_logits,
            "region_logits": region_logits,
            "final_coordinates": final_coordinates,
            "refined_coordinates": refined_coordinates,
            "refinement_alpha": refinement_alpha,
            "refinement_gate_logits": gate_logits,
            "log_var": log_var,
            "candidate_points": candidates,
            "tokens": tokens,
            "hard_rank_logits": (
                local_logits[:, self.hard_indices]
                if self.use_hard_candidate_ranker
                else None
            ),
        }
