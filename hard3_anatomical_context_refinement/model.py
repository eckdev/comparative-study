"""Small dual-view heatmap network for the three peripheral landmarks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DualViewHard3Net(nn.Module):
    """Dual-view heatmaps plus a jointly decoded bilateral Gonion pair."""

    def __init__(
        self,
        input_channels,
        width=24,
        dropout=0.10,
        geometry_dim=18,
        pair_topk=32,
    ):
        super().__init__()
        self.pair_topk = max(2, int(pair_topk))
        self.geometry_dim = int(geometry_dim)
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

        pair_input_dim = 4 * self.geometry_dim + 4
        pair_width = max(width * 2, 48)
        self.gonion_pair_ranker = nn.Sequential(
            nn.Linear(pair_input_dim, pair_width),
            nn.LayerNorm(pair_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_width, pair_width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_width // 2, 1),
        )
        nn.init.zeros_(self.gonion_pair_ranker[-1].weight)
        nn.init.zeros_(self.gonion_pair_ranker[-1].bias)

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

    def candidate_logits(self, heatmaps, grids, candidate_mask, view_weights=None):
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
        logits = (sampled * view_weights[..., None].to(sampled.dtype)).sum(dim=2)
        return logits.masked_fill(~candidate_mask, -torch.inf)

    def gonion_pair(
        self,
        candidate_logits,
        canonical,
        points,
        candidate_mask,
        temperature=0.5,
        target_distance=None,
    ):
        """Rank LM21/22 jointly and return differentiable pair coordinates."""
        left_mask, right_mask = candidate_mask[:, 1], candidate_mask[:, 2]
        left_logits = self._masked_standardize(
            candidate_logits[:, 1].float(), left_mask
        )
        right_logits = self._masked_standardize(
            candidate_logits[:, 2].float(), right_mask
        )
        use_topk = min(self.pair_topk, candidate_logits.shape[-1])
        left_indices = torch.topk(left_logits, use_topk, dim=-1).indices
        right_indices = torch.topk(right_logits, use_topk, dim=-1).indices

        # During training, always expose the closest expert candidate. At
        # inference this branch is unavailable and the learned unary proposal is
        # solely responsible for recall.
        if target_distance is not None:
            nearest_left = target_distance[:, 1].argmin(dim=-1)
            nearest_right = target_distance[:, 2].argmin(dim=-1)
            left_indices = left_indices.clone()
            right_indices = right_indices.clone()
            left_indices[:, -1] = nearest_left
            right_indices[:, -1] = nearest_right

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

        left = left_geometry[:, :, None, :].expand(-1, -1, use_topk, -1)
        right = right_geometry[:, None, :, :].expand(-1, use_topk, -1, -1)
        unary_features = torch.stack(
            [
                left_unary[:, :, None].expand(-1, -1, use_topk),
                right_unary[:, None, :].expand(-1, use_topk, -1),
                left_unary[:, :, None] + right_unary[:, None, :],
                torch.abs(left_unary[:, :, None] - right_unary[:, None, :]),
            ],
            dim=-1,
        )
        pair_features = torch.cat(
            [
                left,
                right,
                torch.abs(left - right),
                0.5 * (left + right),
                unary_features,
            ],
            dim=-1,
        )
        correction = self.gonion_pair_ranker(pair_features).squeeze(-1)
        pair_logits = left_unary[:, :, None] + right_unary[:, None, :] + correction
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
        return {
            "logits": pair_logits,
            "mask": pair_mask,
            "left_indices": left_indices,
            "right_indices": right_indices,
            "soft_coordinate": soft_coordinate,
            "argmax_coordinate": argmax_coordinate,
            "snapped_coordinate": snapped_coordinate,
        }
