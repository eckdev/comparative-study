import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def anatomical_adjacency(num_landmarks=23):
    adjacency = torch.eye(num_landmarks, dtype=torch.float32)
    groups = [range(0, 5), range(5, 13), range(13, 21), range(21, 23)]
    for group in groups:
        for i in group:
            for j in group:
                adjacency[i, j] = 1.0
    for i, j in [(13, 16), (14, 15), (17, 18), (19, 20), (21, 22)]:
        adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency


class SurfaceEncoder(nn.Module):
    def __init__(self, input_dim, width=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Linear(width * 3, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, features):
        x = self.net(features)
        mean = x.mean(dim=1, keepdim=True).expand_as(x)
        maxv = x.max(dim=1, keepdim=True).values.expand_as(x)
        return x + self.mix(torch.cat([x, mean, maxv], dim=-1))


class GraphBlock(nn.Module):
    def __init__(self, width, heads=4, dropout=0.1, adjacency=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width))
        adjacency = anatomical_adjacency() if adjacency is None else adjacency
        mask = torch.zeros_like(adjacency)
        mask[adjacency <= 0] = -10000.0
        self.register_buffer("mask", mask)

    def forward(self, tokens):
        x = self.norm1(tokens)
        y, _ = self.attn(x, x, x, attn_mask=self.mask, need_weights=False)
        tokens = tokens + y
        return tokens + self.mlp(self.norm2(tokens))


class AtlasSPNet(nn.Module):
    def __init__(
        self,
        input_dim,
        num_landmarks=23,
        width=128,
        heads=4,
        graph_blocks=2,
        patch_points=256,
        dropout=0.1,
        use_refinement=True,
        use_shape_prior=True,
    ):
        super().__init__()
        self.num_landmarks = int(num_landmarks)
        self.width = int(width)
        self.patch_points = int(patch_points)
        self.use_refinement = bool(use_refinement)
        self.use_shape_prior = bool(use_shape_prior)
        self.surface = SurfaceEncoder(input_dim, width=width, dropout=dropout)
        self.landmark_tokens = nn.Parameter(torch.randn(num_landmarks, width) * 0.02)
        self.cross = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.token_norm = nn.LayerNorm(width)
        self.surface_norm = nn.LayerNorm(width)
        self.coarse_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.coarse_log_var = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.patch_mlp = nn.Sequential(
            nn.Linear(width + 3, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.local_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.graph_blocks = nn.ModuleList([GraphBlock(width, heads=heads, dropout=dropout) for _ in range(graph_blocks)])
        self.shape_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.confidence_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        nn.init.zeros_(self.coarse_head[-1].weight)
        nn.init.zeros_(self.coarse_head[-1].bias)
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)
        nn.init.zeros_(self.shape_head[-1].weight)
        nn.init.zeros_(self.shape_head[-1].bias)

    def forward(self, points_norm, features):
        surface = self.surface(features)
        batch = points_norm.shape[0]
        tokens = self.landmark_tokens.unsqueeze(0).expand(batch, -1, -1)
        cross, _ = self.cross(self.token_norm(tokens), self.surface_norm(surface), self.surface_norm(surface), need_weights=False)
        tokens = tokens + cross
        logits = torch.einsum("bnc,blc->bnl", F.normalize(surface, dim=-1), F.normalize(tokens, dim=-1)) * math.sqrt(self.width)
        heat_weights = torch.softmax(logits, dim=1)
        heat_coords = torch.einsum("bnl,bnd->bld", heat_weights, points_norm)
        coarse_delta = torch.tanh(self.coarse_head(tokens)) * 0.25
        coarse = heat_coords + coarse_delta
        local_delta = torch.zeros_like(coarse)
        if self.use_refinement and self.patch_points > 0:
            dists = torch.cdist(coarse, points_norm)
            k = min(self.patch_points, points_norm.shape[1])
            patch_idx = dists.topk(k=k, largest=False).indices
            idx_exp = patch_idx.unsqueeze(-1).expand(-1, -1, -1, self.width)
            surf_exp = surface.unsqueeze(1).expand(-1, self.num_landmarks, -1, -1)
            patch_embed = torch.gather(surf_exp, 2, idx_exp)
            point_exp = points_norm.unsqueeze(1).expand(-1, self.num_landmarks, -1, -1)
            rel = torch.gather(point_exp, 2, patch_idx.unsqueeze(-1).expand(-1, -1, -1, 3)) - coarse.unsqueeze(2)
            patch_feat = self.patch_mlp(torch.cat([patch_embed, rel], dim=-1)).max(dim=2).values
            local_delta = torch.tanh(self.local_head(patch_feat)) * 0.12
            tokens = tokens + patch_feat
        shape_delta = torch.zeros_like(coarse)
        if self.use_shape_prior:
            shape_tokens = tokens
            for block in self.graph_blocks:
                shape_tokens = block(shape_tokens)
            shape_delta = torch.tanh(self.shape_head(shape_tokens)) * 0.10
            tokens = shape_tokens
        final = coarse + local_delta + shape_delta
        log_vars = self.coarse_log_var(tokens).squeeze(-1).clamp(-6.0, 6.0)
        confidence = torch.sigmoid(self.confidence_head(tokens).squeeze(-1))
        return {
            "logits": logits,
            "heat_coords": heat_coords,
            "coarse_norm": coarse,
            "local_delta_norm": local_delta,
            "shape_delta_norm": shape_delta,
            "pred_norm": final,
            "log_vars": log_vars,
            "confidence": confidence,
        }
