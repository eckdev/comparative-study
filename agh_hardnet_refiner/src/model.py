import torch
from torch import nn
import torch.nn.functional as F


class AGHHardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=192, landmark_embedding_dim=32, residual_limit_mm=8.0, dropout=0.1):
        super().__init__()
        self.residual_limit_mm = float(residual_limit_mm)
        self.landmark_embedding = nn.Embedding(23, landmark_embedding_dim)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(input_dim + landmark_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim + landmark_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(hidden_dim, 3)
        self.log_var_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        candidate_features,
        candidate_points,
        landmark,
        temperature=1.0,
        topk=20,
        center_distance=None,
        center_prior_weight=0.0,
        base=None,
        candidate_blend=1.0,
    ):
        b, n, _ = candidate_features.shape
        emb = self.landmark_embedding(landmark).unsqueeze(1).expand(-1, n, -1)
        hidden = self.candidate_mlp(torch.cat([candidate_features, emb], dim=-1))
        raw_logits = self.score_head(hidden).squeeze(-1)
        logits = raw_logits
        if center_distance is not None and float(center_prior_weight) != 0.0:
            logits = logits - float(center_prior_weight) * center_distance
        k = min(int(topk), n)
        if k < n:
            top_vals, top_idx = torch.topk(logits, k=k, dim=1)
            masked = torch.full_like(logits, -1e4)
            logits = masked.scatter(1, top_idx, top_vals)
        probs = F.softmax(logits / max(float(temperature), 1e-6), dim=1)
        weighted = torch.sum(probs.unsqueeze(-1) * candidate_points, dim=1)
        context = torch.sum(probs.unsqueeze(-1) * hidden, dim=1)
        context = self.context_mlp(torch.cat([context, self.landmark_embedding(landmark)], dim=-1))
        residual = torch.tanh(self.residual_head(context)) * self.residual_limit_mm
        if base is None:
            pred = weighted + residual
        else:
            pred = base + float(candidate_blend) * (weighted - base) + residual
        return {
            "logits": logits,
            "raw_logits": logits,
            "mlp_logits": raw_logits,
            "probs": probs,
            "weighted": weighted,
            "residual": residual,
            "pred": pred,
            "log_var": self.log_var_head(context).squeeze(-1).clamp(-6.0, 6.0),
            "confidence": torch.sigmoid(self.confidence_head(context).squeeze(-1)),
        }


def hardnet_loss(outputs, batch, args):
    expert = batch["expert"]
    log_probs = F.log_softmax(outputs["raw_logits"], dim=1)
    soft_labels = batch["soft_labels"]
    heatmap = -(soft_labels * log_probs).sum(dim=1).mean()
    hard_ce = F.cross_entropy(outputs["raw_logits"], batch["hard_target_idx"])
    coord = F.smooth_l1_loss(outputs["pred"], expert, beta=float(args.coord_beta_mm))
    weighted = F.smooth_l1_loss(outputs["weighted"], expert, beta=float(args.coord_beta_mm))
    err = torch.linalg.norm(outputs["pred"] - expert, dim=1)
    weighted_err = torch.linalg.norm(outputs["weighted"] - expert, dim=1)
    clinical = F.softplus((err - float(args.clinical_threshold_mm)) / max(float(args.clinical_margin_mm), 1e-6)).mean()
    nll = (0.5 * torch.exp(-outputs["log_var"]) * err.pow(2) + 0.5 * outputs["log_var"]).mean()
    residual_reg = torch.linalg.norm(outputs["residual"], dim=1).mean()
    entropy = -(outputs["probs"] * torch.log(outputs["probs"].clamp_min(1e-8))).sum(dim=1).mean()
    loss = (
        float(args.coord_weight) * coord
        + float(args.heatmap_weight) * heatmap
        + float(args.hard_ce_weight) * hard_ce
        + float(args.weighted_coord_weight) * weighted
        + float(args.clinical_weight) * clinical
        + float(args.nll_weight) * nll
        + float(args.residual_reg_weight) * residual_reg
        + float(args.entropy_weight) * entropy
    )
    return loss, {
        "coord": float(coord.detach().cpu()),
        "heatmap": float(heatmap.detach().cpu()),
        "hard_ce": float(hard_ce.detach().cpu()),
        "weighted_coord": float(weighted.detach().cpu()),
        "clinical": float(clinical.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "residual_reg": float(residual_reg.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "mean_error": float(err.detach().mean().cpu()),
        "weighted_mean_error": float(weighted_err.detach().mean().cpu()),
    }
