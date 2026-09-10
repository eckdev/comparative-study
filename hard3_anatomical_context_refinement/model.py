"""Small dual-view heatmap network for the three peripheral landmarks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


PROPOSAL_SOURCE_NAMES = ("geometry", "fused", "frontal", "profile")


def diverse_topk_indices(source_logits, candidate_mask, topk):
    """Select a balanced union of candidates ranked by multiple proposal sources."""
    if source_logits.ndim != 3:
        raise ValueError("source_logits must have shape [B, sources, candidates]")
    if candidate_mask.shape != source_logits.shape[:1] + source_logits.shape[2:]:
        raise ValueError("candidate_mask must have shape [B, candidates]")
    candidates = source_logits.shape[-1]
    count = min(max(1, int(topk)), candidates)
    valid = candidate_mask[:, None].expand_as(source_logits)
    safe = source_logits.float().masked_fill(~valid, -torch.inf)
    order = torch.argsort(safe, dim=-1, descending=True)
    positions = torch.arange(candidates, device=source_logits.device).view(1, 1, -1)
    positions = positions.expand_as(order)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order, positions)
    ranks = ranks.masked_fill(~valid, candidates + 1)

    # Ranking by the best source rank gives each source an equal proposal quota.
    # Mean rank and the learned geometry score only provide deterministic ties.
    best_rank = ranks.amin(dim=1).float()
    mean_rank = ranks.float().mean(dim=1)
    geometry_tie = torch.nan_to_num(
        source_logits[:, 0].float(), nan=0.0, posinf=1e4, neginf=-1e4
    )
    priority = -best_rank - 1e-3 * mean_rank / max(candidates, 1)
    priority = priority + 1e-6 * geometry_tie
    priority = priority.masked_fill(~candidate_mask, -torch.inf)
    return torch.topk(priority, count, dim=-1).indices


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0.0):
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )

    def forward(self, values):
        return self.block(values) + self.skip(values)


class CompactUNet(nn.Module):
    def __init__(self, input_channels, width=24, dropout=0.10):
        super().__init__()
        self.encoder_one = ConvBlock(input_channels, width, dropout)
        self.encoder_two = ConvBlock(width, width * 2, dropout)
        self.bottleneck = ConvBlock(width * 2, width * 4, dropout)
        self.decoder_two = ConvBlock(width * 6, width * 2, dropout)
        self.decoder_one = ConvBlock(width * 3, width, dropout)
        self.output = nn.Conv2d(width, 1, 1)

    @property
    def embedding_dim(self):
        return self.bottleneck.block[0].out_channels * 2

    def forward(self, image, return_embedding=False):
        first = self.encoder_one(image)
        second = self.encoder_two(F.avg_pool2d(first, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(second, 2))
        embedding = torch.cat(
            [bottleneck.mean(dim=(-2, -1)), bottleneck.amax(dim=(-2, -1))], dim=-1
        )
        hidden = F.interpolate(
            bottleneck, size=second.shape[-2:], mode="bilinear", align_corners=False
        )
        hidden = self.decoder_two(torch.cat([hidden, second], dim=1))
        hidden = F.interpolate(
            hidden, size=first.shape[-2:], mode="bilinear", align_corners=False
        )
        heatmap = self.output(self.decoder_one(torch.cat([hidden, first], dim=1)))
        if return_embedding:
            return heatmap, embedding
        return heatmap


class SurfaceContextRanker(nn.Module):
    """Score each candidate from its features and a fixed local surface graph."""

    def __init__(self, input_dim, width=48, dropout=0.10):
        super().__init__()
        hidden = max(int(width), 32)
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Start from the fused heatmap ranking. Context is introduced by the
        # all-candidate listwise objective during the proposal stage.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    @staticmethod
    def _gather_neighbors(values, indices):
        batch, landmarks, candidates, channels = values.shape
        flat = values.reshape(batch * landmarks, candidates, channels)
        flat_indices = indices.reshape(batch * landmarks, candidates, -1)
        batch_index = torch.arange(batch * landmarks, device=values.device)[
            :, None, None
        ]
        return flat[batch_index, flat_indices].reshape(
            batch, landmarks, candidates, flat_indices.shape[-1], channels
        )

    def forward(
        self,
        features,
        geometry,
        neighbor_index,
        neighbor_mask,
        candidate_mask,
    ):
        encoded = self.point_encoder(features.float())
        neighbors = self._gather_neighbors(encoded, neighbor_index.long())
        neighbor_geometry = self._gather_neighbors(
            geometry[..., :3].float(), neighbor_index.long()
        )
        center = encoded[..., None, :].expand_as(neighbors)
        delta = neighbor_geometry - geometry[..., None, :3].float()
        edge = self.edge_encoder(torch.cat([center, neighbors - center, delta], dim=-1))
        gathered_valid = (
            self._gather_neighbors(
                candidate_mask[..., None].float(), neighbor_index.long()
            )
            .squeeze(-1)
            .bool()
        )
        valid = neighbor_mask.bool() & gathered_valid & candidate_mask[..., None]
        maximum = edge.masked_fill(~valid[..., None], -torch.inf).amax(dim=-2)
        maximum = torch.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0)
        valid_float = valid[..., None].to(edge.dtype)
        mean = (edge * valid_float).sum(dim=-2) / valid_float.sum(dim=-2).clamp_min(1.0)
        score = self.fusion(torch.cat([encoded, maximum, mean], dim=-1)).squeeze(-1)
        return score.masked_fill(~candidate_mask, -torch.inf)


class SharpUnaryReranker(nn.Module):
    """Rerank a recall-preserving proposal set with sample-level context."""

    def __init__(self, input_dim, width=48, dropout=0.10):
        super().__init__()
        hidden = max(int(width), 32)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # The broad proposal remains the initial ranking. The sharp stage learns
        # only a residual after its own loss becomes active.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, features, candidate_mask):
        encoded = self.candidate_encoder(features.float())
        valid = candidate_mask[..., None].to(encoded.dtype)
        mean = (encoded * valid).sum(dim=-2) / valid.sum(dim=-2).clamp_min(1.0)
        maximum = encoded.masked_fill(~candidate_mask[..., None], -torch.inf).amax(
            dim=-2
        )
        maximum = torch.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0)
        context = torch.cat([mean, maximum], dim=-1)[..., None, :].expand(
            *encoded.shape[:-1], mean.shape[-1] * 2
        )
        raw = self.score(torch.cat([encoded, context], dim=-1)).squeeze(-1)
        correction = 3.0 * torch.tanh(raw / 3.0)
        return correction.masked_fill(~candidate_mask, -torch.inf)


class ClinicalBilateralPairRanker(nn.Module):
    """Condition both Gonion candidate sets before scoring every retained pair."""

    def __init__(self, input_dim, geometry_dim, width=48, dropout=0.10):
        super().__init__()
        hidden = max(int(width), 32)
        heads = 4 if hidden % 4 == 0 else 1
        self.geometry_dim = int(geometry_dim)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.side_embedding = nn.Parameter(torch.zeros(2, hidden))
        self.cross_attention = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden)
        self.unary_score = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        pair_dim = max(hidden // 2, 16)
        self.left_query = nn.Linear(hidden, pair_dim, bias=False)
        self.right_key = nn.Linear(hidden, pair_dim, bias=False)
        self.pair_scale = nn.Parameter(torch.tensor(0.0))
        self.symmetry_scale = nn.Parameter(torch.tensor(-2.0))
        self.symmetry_weights = nn.Parameter(torch.zeros(min(18, geometry_dim)))

        # Start from the recall-preserving broad unary distribution. Pair context
        # is introduced only after the dedicated clinical-ranking stage begins.
        nn.init.zeros_(self.unary_score[-1].weight)
        nn.init.zeros_(self.unary_score[-1].bias)

    def forward(
        self,
        left_features,
        right_features,
        left_geometry,
        right_geometry,
        left_mask,
        right_mask,
    ):
        left = self.candidate_encoder(torch.nan_to_num(left_features.float()))
        right = self.candidate_encoder(torch.nan_to_num(right_features.float()))
        left = left + self.side_embedding[0]
        right = right + self.side_embedding[1]
        left_context, _ = self.cross_attention(
            left,
            right,
            right,
            key_padding_mask=~right_mask,
            need_weights=False,
        )
        right_context, _ = self.cross_attention(
            right,
            left,
            left,
            key_padding_mask=~left_mask,
            need_weights=False,
        )
        left_context = self.cross_norm(left + left_context)
        right_context = self.cross_norm(right + right_context)
        left_correction = self.unary_score(
            torch.cat([left, left_context, left - left_context], dim=-1)
        ).squeeze(-1)
        right_correction = self.unary_score(
            torch.cat([right, right_context, right - right_context], dim=-1)
        ).squeeze(-1)
        left_correction = 3.0 * torch.tanh(left_correction / 3.0)
        right_correction = 3.0 * torch.tanh(right_correction / 3.0)
        left_correction = left_correction.masked_fill(~left_mask, 0.0)
        right_correction = right_correction.masked_fill(~right_mask, 0.0)

        query = F.normalize(self.left_query(left_context), dim=-1)
        key = F.normalize(self.right_key(right_context), dim=-1)
        compatibility = torch.einsum("bih,bjh->bij", query, key)
        compatibility = torch.tanh(self.pair_scale) * compatibility

        symmetry_dim = self.symmetry_weights.numel()
        difference = torch.abs(
            left_geometry[:, :, None, :symmetry_dim]
            - right_geometry[:, None, :, :symmetry_dim]
        )
        symmetry_weights = torch.softmax(self.symmetry_weights, dim=0)
        symmetry_distance = (difference * symmetry_weights).sum(dim=-1)
        # Mirrored canonical coordinates provide a weak prior, not a hard
        # symmetry constraint; the learned scale may suppress it for asymmetric
        # faces.
        symmetry_penalty = F.softplus(self.symmetry_scale) * symmetry_distance
        pair_correction = compatibility - symmetry_penalty
        pair_mask = left_mask[:, :, None] & right_mask[:, None, :]
        return {
            "left_correction": left_correction,
            "right_correction": right_correction,
            "pair_correction": pair_correction.masked_fill(~pair_mask, 0.0),
        }


class ShapeConditionedContourPairRanker(nn.Module):
    """Decode Gonion as a shared contour state plus patient-specific asymmetry."""

    def __init__(
        self,
        input_dim,
        geometry_dim,
        shape_context_dim=69,
        width=48,
        dropout=0.10,
        residual_limit=0.20,
    ):
        super().__init__()
        hidden = max(int(width), 32)
        heads = 4 if hidden % 4 == 0 else 1
        self.geometry_dim = int(geometry_dim)
        self.shape_context_dim = int(shape_context_dim)
        self.residual_limit = float(residual_limit)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.side_embedding = nn.Parameter(torch.zeros(2, hidden))
        self.self_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(hidden)
        self.cross_norm = nn.LayerNorm(hidden)
        self.shape_encoder = nn.Sequential(
            nn.LayerNorm(self.shape_context_dim),
            nn.Linear(self.shape_context_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
        )
        self.local_score = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.pair_query = nn.Linear(hidden, hidden, bias=False)
        self.pair_key = nn.Linear(hidden, hidden, bias=False)
        self.candidate_axis_weights = nn.Parameter(torch.zeros(3))
        self.mean_axis_weights = nn.Parameter(torch.zeros(3))
        self.asymmetry_axis_weights = nn.Parameter(torch.zeros(3))
        self.candidate_scale = nn.Parameter(torch.tensor(-1.0))
        # Different initial strengths make the mean/asymmetry energy genuinely
        # bilateral. With one shared scale, the squared terms algebraically
        # collapse into two independent unary distances.
        self.mean_scale = nn.Parameter(torch.tensor(-0.75))
        self.asymmetry_scale = nn.Parameter(torch.tensor(-1.25))
        self.compatibility_scale = nn.Parameter(torch.tensor(0.5))

        # The initial state is the upstream all-23 prediction. Learning begins as
        # a bounded residual correction, which is substantially more stable than
        # predicting an absolute jaw location from 192 samples.
        nn.init.zeros_(self.state_head[-1].weight)
        nn.init.zeros_(self.state_head[-1].bias)
        nn.init.zeros_(self.local_score[-1].weight)
        nn.init.zeros_(self.local_score[-1].bias)

    @staticmethod
    def _masked_pool(values, mask):
        valid = mask[..., None].to(values.dtype)
        mean = (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        maximum = values.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        maximum = torch.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.cat([mean, maximum], dim=-1)

    def forward(
        self,
        left_features,
        right_features,
        left_geometry,
        right_geometry,
        left_mask,
        right_mask,
        shape_context,
        base_gonion,
    ):
        left = self.candidate_encoder(torch.nan_to_num(left_features.float()))
        right = self.candidate_encoder(torch.nan_to_num(right_features.float()))
        left = left + self.side_embedding[0]
        right = right + self.side_embedding[1]

        left_self, _ = self.self_attention(
            left, left, left, key_padding_mask=~left_mask, need_weights=False
        )
        right_self, _ = self.self_attention(
            right, right, right, key_padding_mask=~right_mask, need_weights=False
        )
        left = self.self_norm(left + left_self)
        right = self.self_norm(right + right_self)
        left_cross, _ = self.cross_attention(
            left, right, right, key_padding_mask=~right_mask, need_weights=False
        )
        right_cross, _ = self.cross_attention(
            right, left, left, key_padding_mask=~left_mask, need_weights=False
        )
        left = self.cross_norm(left + left_cross)
        right = self.cross_norm(right + right_cross)

        shape = self.shape_encoder(torch.nan_to_num(shape_context.float()))
        context = self.context_encoder(
            torch.cat(
                [
                    self._masked_pool(left, left_mask),
                    self._masked_pool(right, right_mask),
                    shape,
                ],
                dim=-1,
            )
        )
        state_delta = self.residual_limit * torch.tanh(self.state_head(context))
        base_mean = 0.5 * (base_gonion[:, 0] + base_gonion[:, 1])
        base_asymmetry = 0.5 * (base_gonion[:, 0] - base_gonion[:, 1])
        predicted_mean = base_mean + state_delta[:, :3]
        predicted_asymmetry = base_asymmetry + state_delta[:, 3:]
        predicted_left = predicted_mean + predicted_asymmetry
        predicted_right = predicted_mean - predicted_asymmetry

        expanded_context_left = context[:, None].expand(-1, left.shape[1], -1)
        expanded_context_right = context[:, None].expand(-1, right.shape[1], -1)
        left_local = self.local_score(
            torch.cat([left, expanded_context_left], dim=-1)
        ).squeeze(-1)
        right_local = self.local_score(
            torch.cat([right, expanded_context_right], dim=-1)
        ).squeeze(-1)
        left_local = 3.0 * torch.tanh(left_local / 3.0)
        right_local = 3.0 * torch.tanh(right_local / 3.0)

        if left_geometry.shape[-1] < 6 or right_geometry.shape[-1] < 6:
            raise ValueError("Contour decoder requires canonical global XYZ features")
        left_global = left_geometry[..., 3:6].float()
        right_global = right_geometry[..., 3:6].float()
        candidate_weights = torch.softmax(self.candidate_axis_weights, dim=0) * 3.0
        left_distance = (
            (left_global - predicted_left[:, None]).square() * candidate_weights
        ).sum(dim=-1)
        right_distance = (
            (right_global - predicted_right[:, None]).square() * candidate_weights
        ).sum(dim=-1)
        candidate_scale = F.softplus(self.candidate_scale)
        left_correction = left_local - candidate_scale * left_distance
        right_correction = right_local - candidate_scale * right_distance

        pair_mean = 0.5 * (left_global[:, :, None] + right_global[:, None, :])
        pair_asymmetry = 0.5 * (left_global[:, :, None] - right_global[:, None, :])
        mean_weights = torch.softmax(self.mean_axis_weights, dim=0) * 3.0
        asymmetry_weights = torch.softmax(self.asymmetry_axis_weights, dim=0) * 3.0
        mean_error = (
            (pair_mean - predicted_mean[:, None, None]).square() * mean_weights
        ).sum(dim=-1)
        asymmetry_error = (
            (pair_asymmetry - predicted_asymmetry[:, None, None]).square()
            * asymmetry_weights
        ).sum(dim=-1)
        query = F.normalize(self.pair_query(left), dim=-1)
        key = F.normalize(self.pair_key(right), dim=-1)
        compatibility = torch.einsum("bih,bjh->bij", query, key)
        pair_correction = (
            -(
                F.softplus(self.mean_scale) * mean_error
                + F.softplus(self.asymmetry_scale) * asymmetry_error
            )
            + 2.0 * torch.tanh(self.compatibility_scale) * compatibility
        )
        pair_mask = left_mask[:, :, None] & right_mask[:, None, :]
        return {
            "left_correction": left_correction.masked_fill(~left_mask, 0.0),
            "right_correction": right_correction.masked_fill(~right_mask, 0.0),
            "pair_correction": pair_correction.masked_fill(~pair_mask, 0.0),
            "predicted_mean": predicted_mean,
            "predicted_asymmetry": predicted_asymmetry,
            "base_mean": base_mean,
            "base_asymmetry": base_asymmetry,
            "left_global": left_global,
            "right_global": right_global,
        }


class DualViewHard3Net(nn.Module):
    """Dual-view heatmaps plus a jointly decoded bilateral Gonion pair."""

    def __init__(
        self,
        input_channels,
        width=24,
        dropout=0.10,
        geometry_dim=18,
        proposal_topk=96,
        pair_topk=96,
        enable_unary_reranker=False,
        decoder_mode="full_pair",
        shape_context_dim=69,
        contour_residual_limit=0.20,
        enable_pair_ranker=True,
    ):
        super().__init__()
        self.proposal_topk = max(2, int(proposal_topk))
        self.pair_topk = max(2, int(pair_topk))
        self.geometry_dim = int(geometry_dim)
        self.decoder_mode = str(decoder_mode)
        self.trichion = CompactUNet(input_channels, width, dropout)
        self.gonion = CompactUNet(input_channels, width, dropout)
        # These biases preserve the useful initialization from v1. The small
        # quality heads then adapt the frontal/profile balance per sample.
        self.trichion_view_logits = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.gonion_view_logits = nn.Parameter(torch.tensor([0.0, 1.0]))
        gate_dim = self.trichion.embedding_dim + 3
        gate_width = max(width, 16)
        self.trichion_view_gate = nn.Sequential(
            nn.Linear(gate_dim, gate_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_width, 1),
        )
        self.gonion_view_gate = nn.Sequential(
            nn.Linear(gate_dim, gate_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_width, 1),
        )
        # Start as v1 and let ranking supervision introduce dynamic weighting.
        nn.init.zeros_(self.trichion_view_gate[-1].weight)
        nn.init.zeros_(self.trichion_view_gate[-1].bias)
        nn.init.zeros_(self.gonion_view_gate[-1].weight)
        nn.init.zeros_(self.gonion_view_gate[-1].bias)

        proposal_input_dim = self.geometry_dim + 5
        proposal_width = max(width * 2, 48)
        self.gonion_geometry_proposal = SurfaceContextRanker(
            proposal_input_dim,
            proposal_width,
            dropout,
        )
        reranker_input_dim = self.geometry_dim + len(PROPOSAL_SOURCE_NAMES)
        self.gonion_unary_reranker = (
            SharpUnaryReranker(
                reranker_input_dim,
                proposal_width,
                dropout,
            )
            if enable_unary_reranker
            else None
        )

        pair_width = max(width * 2, 48)
        pair_input_dim = self.geometry_dim + len(PROPOSAL_SOURCE_NAMES)
        if not enable_pair_ranker:
            self.gonion_pair_ranker = None
        elif self.decoder_mode in (
            "contour_coordinate",
            "crossfit_calibrated",
            "crossfit_interaction",
        ):
            self.gonion_pair_ranker = ShapeConditionedContourPairRanker(
                pair_input_dim,
                self.geometry_dim,
                shape_context_dim,
                pair_width,
                dropout,
                contour_residual_limit,
            )
        else:
            self.gonion_pair_ranker = ClinicalBilateralPairRanker(
                pair_input_dim,
                self.geometry_dim,
                pair_width,
                dropout,
            )

    @staticmethod
    def _heatmap_statistics(heatmaps):
        probability = torch.sigmoid(heatmaps.float()).flatten(start_dim=-2)
        return torch.stack(
            [
                probability.mean(dim=-1),
                probability.std(dim=-1, unbiased=False),
                probability.amax(dim=-1),
            ],
            dim=-1,
        )

    @staticmethod
    def _gather(values, indices):
        return torch.gather(
            values,
            1,
            indices[..., None].expand(-1, -1, values.shape[-1]),
        )

    @staticmethod
    def _masked_standardize(values, mask):
        valid = mask.to(values.dtype)
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        # candidate_logits intentionally contains -inf at padded vertices. They
        # must be removed before subtraction/division; masking only afterwards
        # leaves an inf intermediate whose backward pass can produce NaN.
        safe = values.masked_fill(~mask, 0.0)
        mean = (safe * valid).sum(dim=-1, keepdim=True) / count
        centered = (safe - mean) * valid
        variance = (centered.square()).sum(dim=-1, keepdim=True) / count
        return (centered / torch.sqrt(variance + 1e-4)).masked_fill(~mask, -torch.inf)

    def forward_with_context(self, images):
        batch, landmarks, views, channels, height, width = images.shape
        if landmarks != 3 or views != 2:
            raise ValueError("DualViewHard3Net expects [B,3,2,C,H,W]")
        trichion, trichion_embedding = self.trichion(
            images[:, 0].reshape(batch * views, channels, height, width),
            return_embedding=True,
        )
        gonion, gonion_embedding = self.gonion(
            images[:, 1:3].reshape(batch * 2 * views, channels, height, width),
            return_embedding=True,
        )
        trichion = trichion.reshape(batch, 1, views, height, width)
        gonion = gonion.reshape(batch, 2, views, height, width)
        heatmaps = torch.cat([trichion, gonion], dim=1)
        embeddings = torch.cat(
            [
                trichion_embedding.reshape(batch, 1, views, -1),
                gonion_embedding.reshape(batch, 2, views, -1),
            ],
            dim=1,
        )
        statistics = self._heatmap_statistics(heatmaps)
        quality = torch.cat([embeddings, statistics], dim=-1)
        trichion_quality = self.trichion_view_gate(quality[:, 0]).squeeze(-1)
        gonion_quality = self.gonion_view_gate(quality[:, 1:3]).squeeze(-1)
        view_logits = torch.cat([trichion_quality[:, None], gonion_quality], dim=1)
        prior = torch.stack(
            [
                self.trichion_view_logits,
                self.gonion_view_logits,
                self.gonion_view_logits,
            ]
        )
        view_weights = torch.softmax(view_logits.float() + prior[None].float(), dim=-1)
        return heatmaps, view_weights.to(heatmaps.dtype)

    def forward(self, images):
        heatmaps, _ = self.forward_with_context(images)
        return heatmaps

    def candidate_logits(
        self,
        heatmaps,
        grids,
        candidate_mask,
        view_weights=None,
        canonical=None,
        neighbor_index=None,
        neighbor_mask=None,
        return_evidence=False,
    ):
        batch, landmarks, views, height, width = heatmaps.shape
        candidates = grids.shape[-2]
        sampled = F.grid_sample(
            heatmaps.reshape(batch * landmarks * views, 1, height, width),
            grids.reshape(batch * landmarks * views, candidates, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch, landmarks, views, candidates)
        if view_weights is None:
            trichion_weights = torch.softmax(self.trichion_view_logits.float(), dim=0)
            gonion_weights = torch.softmax(self.gonion_view_logits.float(), dim=0)
            view_weights = torch.stack(
                [trichion_weights, gonion_weights, gonion_weights], dim=0
            )[None].expand(batch, -1, -1)
        fused = (sampled * view_weights[..., None].to(sampled.dtype)).sum(dim=2)
        fused = fused.float().masked_fill(~candidate_mask, -torch.inf)
        view_mask = candidate_mask[:, :, None].expand_as(sampled)
        standardized_views = self._masked_standardize(sampled.float(), view_mask)
        standardized_fused = self._masked_standardize(fused, candidate_mask)

        proposal = fused.clone()
        if canonical is not None:
            gonion_mask = candidate_mask[:, 1:3]
            image_evidence = standardized_views[:, 1:3].permute(0, 1, 3, 2)
            image_evidence = image_evidence.masked_fill(~gonion_mask[..., None], 0.0)
            fused_evidence = standardized_fused[:, 1:3, :, None].masked_fill(
                ~gonion_mask[..., None], 0.0
            )
            weights = view_weights[:, 1:3, None].expand(-1, -1, candidates, -1)
            proposal_features = torch.cat(
                [
                    canonical[:, 1:3].float(),
                    image_evidence,
                    fused_evidence,
                    weights.float(),
                ],
                dim=-1,
            )
            if neighbor_index is None:
                self_index = torch.arange(candidates, device=canonical.device)
                neighbor_index = self_index.view(1, 1, candidates, 1).expand(
                    batch, landmarks, -1, -1
                )
                neighbor_mask = candidate_mask[..., None]
            elif neighbor_mask is None:
                neighbor_mask = torch.ones_like(neighbor_index, dtype=torch.bool)
            correction = self.gonion_geometry_proposal(
                proposal_features,
                canonical[:, 1:3],
                neighbor_index[:, 1:3],
                neighbor_mask[:, 1:3],
                gonion_mask,
            )
            proposal[:, 1:3] = standardized_fused[:, 1:3] + correction
        proposal = proposal.masked_fill(~candidate_mask, -torch.inf)
        standardized_proposal = self._masked_standardize(proposal, candidate_mask)
        sources = torch.stack(
            [
                standardized_proposal,
                standardized_fused,
                standardized_views[:, :, 0],
                standardized_views[:, :, 1],
            ],
            dim=2,
        ).masked_fill(~candidate_mask[:, :, None], -torch.inf)
        evidence = {
            "logits": proposal,
            "heatmap_logits": fused,
            "view_logits": sampled.float().masked_fill(~view_mask, -torch.inf),
            "proposal_sources": sources,
        }
        return evidence if return_evidence else proposal

    def rerank_gonion(
        self,
        candidate_logits,
        canonical,
        candidate_mask,
        proposal_sources,
    ):
        """Preserve broad recall, then produce a sharper Gonion distribution."""
        if self.gonion_unary_reranker is None:
            raise RuntimeError("Unary reranker is disabled for this model")
        gonion_mask = candidate_mask[:, 1:3]
        broad = self._masked_standardize(candidate_logits[:, 1:3].float(), gonion_mask)
        proposal_count = min(self.proposal_topk, candidate_logits.shape[-1])
        proposal_indices = torch.topk(
            broad.masked_fill(~gonion_mask, -torch.inf), proposal_count, dim=-1
        ).indices
        proposal_mask = torch.gather(gonion_mask, 2, proposal_indices)
        geometry = torch.gather(
            canonical[:, 1:3],
            2,
            proposal_indices[..., None].expand(-1, -1, -1, canonical.shape[-1]),
        )
        source_features = torch.gather(
            proposal_sources[:, 1:3].permute(0, 1, 3, 2),
            2,
            proposal_indices[..., None].expand(-1, -1, -1, proposal_sources.shape[2]),
        )
        broad_selected = torch.gather(broad, 2, proposal_indices).masked_fill(
            ~proposal_mask, 0.0
        )
        features = torch.cat([geometry.float(), source_features.float()], dim=-1)
        correction = self.gonion_unary_reranker(features, proposal_mask)
        reranked_selected = broad_selected + correction.masked_fill(~proposal_mask, 0.0)

        # Retain broad-score variation for ensemble calibration. A five-logit
        # penalty is larger than the bounded reranker correction, so candidates
        # outside the broad shortlist cannot re-enter the bilateral top-k.
        reranked_gonion = broad - 5.0
        reranked_gonion = reranked_gonion.scatter(
            2, proposal_indices, reranked_selected
        ).masked_fill(~gonion_mask, -torch.inf)
        logits = candidate_logits.clone()
        logits[:, 1:3] = reranked_gonion
        return {
            "logits": logits,
            "proposal_indices": proposal_indices,
            "proposal_mask": proposal_mask,
            "shortlist_logits": reranked_selected,
        }

    def gonion_pair(
        self,
        candidate_logits,
        canonical,
        points,
        candidate_mask,
        temperature=0.5,
        target_distance=None,
        proposal_sources=None,
        teacher_force_probability=0.0,
        shape_context=None,
        base_gonion=None,
    ):
        """Rank LM21/22 jointly and return differentiable pair coordinates."""
        if self.gonion_pair_ranker is None:
            raise RuntimeError("The neural bilateral pair ranker is disabled")
        left_mask, right_mask = candidate_mask[:, 1], candidate_mask[:, 2]
        left_logits = self._masked_standardize(
            candidate_logits[:, 1].float(), left_mask
        )
        right_logits = self._masked_standardize(
            candidate_logits[:, 2].float(), right_mask
        )
        use_topk = min(
            self.pair_topk,
            self.proposal_topk,
            candidate_logits.shape[-1],
        )
        # V5 consumes only the sharp reranker's shortlist. The broad top-96
        # proposal is retained one stage earlier and is never pruned directly to
        # the bilateral search space.
        left_indices = torch.topk(
            left_logits.masked_fill(~left_mask, -torch.inf), use_topk, dim=-1
        ).indices
        right_indices = torch.topk(
            right_logits.masked_fill(~right_mask, -torch.inf), use_topk, dim=-1
        ).indices

        # A short, decaying warmup may expose the nearest candidate. After the
        # warmup training uses the same proposal distribution as inference.
        force_probability = float(teacher_force_probability)
        if target_distance is not None and force_probability > 0.0:
            nearest_left = target_distance[:, 1].argmin(dim=-1)
            nearest_right = target_distance[:, 2].argmin(dim=-1)
            if force_probability >= 1.0:
                force = torch.ones(
                    len(left_indices), dtype=torch.bool, device=left_indices.device
                )
            else:
                force = (
                    torch.rand(len(left_indices), device=left_indices.device)
                    < force_probability
                )
            left_indices = left_indices.clone()
            right_indices = right_indices.clone()
            left_missing = force & ~torch.any(
                left_indices == nearest_left[:, None], dim=1
            )
            right_missing = force & ~torch.any(
                right_indices == nearest_right[:, None], dim=1
            )
            left_indices[left_missing, -1] = nearest_left[left_missing]
            right_indices[right_missing, -1] = nearest_right[right_missing]

        left_valid = torch.gather(left_mask, 1, left_indices)
        right_valid = torch.gather(right_mask, 1, right_indices)
        left_geometry = self._gather(canonical[:, 1], left_indices)
        right_geometry = self._gather(canonical[:, 2], right_indices)
        left_points = self._gather(points[:, 1], left_indices)
        right_points = self._gather(points[:, 2], right_indices)
        left_unary = torch.gather(left_logits, 1, left_indices).masked_fill(
            ~left_valid, 0.0
        )
        right_unary = torch.gather(right_logits, 1, right_indices).masked_fill(
            ~right_valid, 0.0
        )

        if proposal_sources is None:
            proposal_sources = canonical.new_zeros(
                canonical.shape[0], 3, len(PROPOSAL_SOURCE_NAMES), canonical.shape[2]
            )
        source_values = proposal_sources[:, 1:3].permute(0, 1, 3, 2)
        left_sources = self._gather(source_values[:, 0], left_indices)
        right_sources = self._gather(source_values[:, 1], right_indices)
        left_sources = torch.nan_to_num(left_sources).masked_fill(
            ~left_valid[..., None], 0.0
        )
        right_sources = torch.nan_to_num(right_sources).masked_fill(
            ~right_valid[..., None], 0.0
        )
        pair_arguments = (
            torch.cat([left_geometry.float(), left_sources.float()], dim=-1),
            torch.cat([right_geometry.float(), right_sources.float()], dim=-1),
            left_geometry.float(),
            right_geometry.float(),
            left_valid,
            right_valid,
        )
        if self.decoder_mode in (
            "contour_coordinate",
            "crossfit_calibrated",
            "crossfit_interaction",
        ):
            if shape_context is None:
                shape_context = canonical.new_zeros(
                    canonical.shape[0], self.gonion_pair_ranker.shape_context_dim
                )
            if base_gonion is None:
                left_base = left_geometry[..., 3:6].mean(dim=1)
                right_base = right_geometry[..., 3:6].mean(dim=1)
                base_gonion = torch.stack([left_base, right_base], dim=1)
            ranked = self.gonion_pair_ranker(
                *pair_arguments,
                shape_context,
                base_gonion,
            )
        else:
            ranked = self.gonion_pair_ranker(*pair_arguments)
        left_unary = left_unary + ranked["left_correction"]
        right_unary = right_unary + ranked["right_correction"]
        pair_logits = (
            left_unary[:, :, None] + right_unary[:, None, :] + ranked["pair_correction"]
        )
        pair_mask = left_valid[:, :, None] & right_valid[:, None, :]
        pair_logits = pair_logits.masked_fill(~pair_mask, -torch.inf)

        flat_logits = pair_logits.flatten(1)
        flat_mask = pair_mask.flatten(1)
        probability = torch.softmax(
            flat_logits.float() / max(float(temperature), 1e-4), dim=-1
        ).reshape_as(pair_logits)
        probability = probability * pair_mask
        probability = probability / probability.sum(dim=(1, 2), keepdim=True).clamp_min(
            1e-8
        )
        left_weight = probability.sum(dim=2)
        right_weight = probability.sum(dim=1)
        soft_coordinate = torch.stack(
            [
                (left_weight[..., None] * left_points).sum(dim=1),
                (right_weight[..., None] * right_points).sum(dim=1),
            ],
            dim=1,
        )
        flat_index = flat_logits.masked_fill(~flat_mask, -torch.inf).argmax(dim=-1)
        left_choice = torch.div(flat_index, use_topk, rounding_mode="floor")
        right_choice = flat_index.remainder(use_topk)
        argmax_coordinate = torch.stack(
            [
                torch.gather(
                    left_points, 1, left_choice[:, None, None].expand(-1, 1, 3)
                ).squeeze(1),
                torch.gather(
                    right_points, 1, right_choice[:, None, None].expand(-1, 1, 3)
                ).squeeze(1),
            ],
            dim=1,
        )
        soft_to_left = torch.linalg.norm(
            left_points - soft_coordinate[:, 0, None], dim=-1
        ).masked_fill(~left_valid, torch.inf)
        soft_to_right = torch.linalg.norm(
            right_points - soft_coordinate[:, 1, None], dim=-1
        ).masked_fill(~right_valid, torch.inf)
        left_snap = soft_to_left.argmin(dim=-1)
        right_snap = soft_to_right.argmin(dim=-1)
        snapped_coordinate = torch.stack(
            [
                torch.gather(
                    left_points, 1, left_snap[:, None, None].expand(-1, 1, 3)
                ).squeeze(1),
                torch.gather(
                    right_points, 1, right_snap[:, None, None].expand(-1, 1, 3)
                ).squeeze(1),
            ],
            dim=1,
        )
        result = {
            "logits": pair_logits,
            "mask": pair_mask,
            "probability": probability,
            "unary_logits": torch.stack([left_unary, right_unary], dim=1),
            "left_indices": left_indices,
            "right_indices": right_indices,
            "soft_coordinate": soft_coordinate,
            "argmax_coordinate": argmax_coordinate,
            "snapped_coordinate": snapped_coordinate,
            "proposal_sources": proposal_sources,
        }
        for name in (
            "predicted_mean",
            "predicted_asymmetry",
            "base_mean",
            "base_asymmetry",
            "left_global",
            "right_global",
        ):
            if name in ranked:
                result[name] = ranked[name]
        return result
