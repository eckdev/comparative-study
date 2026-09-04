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

    def forward(self, image):
        first = self.encoder_one(image)
        second = self.encoder_two(F.avg_pool2d(first, 2))
        hidden = self.bottleneck(F.avg_pool2d(second, 2))
        hidden = F.interpolate(hidden, size=second.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.decoder_two(torch.cat([hidden, second], dim=1))
        hidden = F.interpolate(hidden, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.decoder_one(torch.cat([hidden, first], dim=1)))


class DualViewHard3Net(nn.Module):
    """Separate appearance and contour networks with shared bilateral weights."""

    def __init__(self, input_channels, width=24, dropout=0.10):
        super().__init__()
        self.trichion = CompactUNet(input_channels, width, dropout)
        self.gonion = CompactUNet(input_channels, width, dropout)
        # Frontal/profile fusion is learned, but shared by the bilateral pair.
        self.trichion_view_logits = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.gonion_view_logits = nn.Parameter(torch.tensor([0.0, 1.0]))

    def forward(self, images):
        batch, landmarks, views, channels, height, width = images.shape
        if landmarks != 3 or views != 2:
            raise ValueError("DualViewHard3Net expects [B,3,2,C,H,W]")
        trichion = self.trichion(images[:, 0].reshape(batch * views, channels, height, width))
        gonion = self.gonion(images[:, 1:3].reshape(batch * 2 * views, channels, height, width))
        trichion = trichion.reshape(batch, 1, views, height, width)
        gonion = gonion.reshape(batch, 2, views, height, width)
        return torch.cat([trichion, gonion], dim=1)

    def candidate_logits(self, heatmaps, grids, candidate_mask):
        batch, landmarks, views, height, width = heatmaps.shape
        candidates = grids.shape[-2]
        sampled = F.grid_sample(
            heatmaps.reshape(batch * landmarks * views, 1, height, width),
            grids.reshape(batch * landmarks * views, candidates, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch, landmarks, views, candidates)
        trichion_weights = torch.softmax(self.trichion_view_logits.float(), dim=0)
        gonion_weights = torch.softmax(self.gonion_view_logits.float(), dim=0)
        weights = torch.stack(
            [trichion_weights, gonion_weights, gonion_weights], dim=0
        ).to(sampled.dtype)
        logits = (sampled * weights[None, :, :, None]).sum(dim=2)
        return logits.masked_fill(~candidate_mask, -torch.inf)
