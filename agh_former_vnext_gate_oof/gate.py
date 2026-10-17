from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from all23_rgb_geodesic_cascade.anatomy import HARD3, NUM_LANDMARKS
from .analysis import paired_comparison, write_json


@dataclass
class Evidence:
    sample_ids: list
    center: np.ndarray
    local: np.ndarray
    diagnostics: np.ndarray
    token_alpha: np.ndarray
    expert: np.ndarray | None = None

    def validate(self, require_expert=False):
        count = len(self.sample_ids)
        if len(set(self.sample_ids)) != count:
            raise ValueError("Duplicate evidence sample IDs")
        expected_xyz = (count, NUM_LANDMARKS, 3)
        if self.center.shape != expected_xyz or self.local.shape != expected_xyz:
            raise ValueError("Evidence coordinates must be [sample, 23, 3]")
        if self.diagnostics.shape != (count, NUM_LANDMARKS, 5):
            raise ValueError("Evidence diagnostics must be [sample, 23, 5]")
        if self.token_alpha.shape != (count, NUM_LANDMARKS):
            raise ValueError("Token alpha must be [sample, 23]")
        if require_expert and (self.expert is None or self.expert.shape != expected_xyz):
            raise ValueError("Expert coordinates are required for fitting/evaluation")
        arrays = [self.center, self.local, self.diagnostics, self.token_alpha]
        if self.expert is not None:
            arrays.append(self.expert)
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("Non-finite evidence values")
        if np.any((self.token_alpha < 0) | (self.token_alpha > 1)):
            raise ValueError("Token alpha is outside [0, 1]")
        return self

    def attach_expert(self, expert_by_id):
        self.expert = np.stack([expert_by_id[sample_id] for sample_id in self.sample_ids]).astype(np.float32)
        return self.validate(require_expert=True)

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        payload = {
            "sample_ids": np.asarray(self.sample_ids),
            "center": self.center,
            "local": self.local,
            "diagnostics": self.diagnostics,
            "token_alpha": self.token_alpha,
        }
        if self.expert is not None:
            payload["expert"] = self.expert
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as record:
            evidence = cls(
                sample_ids=record["sample_ids"].tolist(),
                center=record["center"].copy(),
                local=record["local"].copy(),
                diagnostics=record["diagnostics"].copy(),
                token_alpha=record["token_alpha"].copy(),
                expert=record["expert"].copy() if "expert" in record.files else None,
            )
        return evidence.validate()


class DiagnosticGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.landmark_embedding = nn.Embedding(NUM_LANDMARKS, 4)
        self.network = nn.Sequential(
            nn.Linear(9, 16), nn.GELU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, 1.0)

    def forward(self, diagnostics):
        ids = torch.arange(NUM_LANDMARKS, device=diagnostics.device)
        embedding = self.landmark_embedding(ids)[None].expand(len(diagnostics), -1, -1)
        return torch.sigmoid(self.network(torch.cat([diagnostics, embedding], dim=-1)).squeeze(-1))


def _coordinate_prediction(evidence, alpha):
    return evidence.center + alpha[..., None] * (evidence.local - evidence.center)


def predict_alpha(model, evidence, mean, std, device, batch_size=64):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(evidence.sample_ids), batch_size):
            values = (evidence.diagnostics[start:start + batch_size] - mean) / std
            tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
            pieces.append(model(tensor).cpu().numpy())
    return np.concatenate(pieces, axis=0)


def _selection_score(comparison):
    hard = comparison["hard3"]
    core = comparison["core20"]
    regression = max(-core["gain_mm"] - 0.05, 0.0)
    tail = max(hard["candidate"]["p95"] - hard["incoming"]["p95"], 0.0)
    return hard["candidate"]["ale"] + 4.0 * regression + tail


def fit_gate(train, validation, output_dir, device, *, seed=42, epochs=100, patience=15):
    train.validate(require_expert=True)
    validation.validate(require_expert=True)
    if set(train.sample_ids) & set(validation.sample_ids):
        raise ValueError("Gate train and validation participants overlap")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    mean = train.diagnostics.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(train.diagnostics.std(axis=(0, 1), keepdims=True), 1e-4).astype(np.float32)
    model = DiagnosticGate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    center = torch.as_tensor(train.center, device=device, dtype=torch.float32)
    local = torch.as_tensor(train.local, device=device, dtype=torch.float32)
    expert = torch.as_tensor(train.expert, device=device, dtype=torch.float32)
    features = torch.as_tensor((train.diagnostics - mean) / std, device=device, dtype=torch.float32)
    importance = torch.ones(NUM_LANDMARKS, device=device)
    importance[list(HARD3)] = 4.0
    delta = local - center
    optimum = ((expert - center) * delta).sum(dim=-1) / delta.square().sum(dim=-1).clamp_min(1e-6)
    optimum = optimum.clamp(0.0, 1.0)
    history = []
    best_score, best_epoch, stale = float("inf"), 0, 0
    checkpoint_path = output_dir / "best_gate.pth"
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        ordering = torch.randperm(len(train.sample_ids), generator=generator).tolist()
        losses = []
        for start in range(0, len(ordering), 32):
            batch = ordering[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            alpha = model(features[batch])
            predicted = center[batch] + alpha[..., None] * delta[batch]
            radial = torch.linalg.norm(predicted - expert[batch], dim=-1)
            target = torch.nn.functional.smooth_l1_loss(
                alpha, optimum[batch], reduction="none"
            )
            loss = ((radial + 0.1 * target) * importance).sum() / (len(batch) * importance.sum())
            if not torch.isfinite(loss):
                raise RuntimeError("Diagnostic gate loss became non-finite")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_alpha = predict_alpha(model, validation, mean, std, device)
        val_prediction = _coordinate_prediction(validation, val_alpha)
        comparison = paired_comparison(
            validation.expert, validation.local, val_prediction,
            validation.sample_ids, seed=seed, iterations=200,
        )
        score = _selection_score(comparison)
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "selection_score": score,
            "val_hard3_gain_mm": comparison["hard3"]["gain_mm"],
            "val_core20_gain_mm": comparison["core20"]["gain_mm"],
        })
        print(
            f"Gate epoch {epoch:03d}/{epochs} loss={history[-1]['train_loss']:.4f} "
            f"val_hard3_gain={comparison['hard3']['gain_mm']:.4f} "
            f"val_core20_gain={comparison['core20']['gain_mm']:.4f}", flush=True,
        )
        if score < best_score - 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({
                "state": model.state_dict(), "mean": mean, "std": std,
                "epoch": epoch, "seed": seed, "train_ids": train.sample_ids,
            }, temporary)
            temporary.replace(checkpoint_path)
        else:
            stale += 1
        write_json(output_dir / "history.json", history)
        if epoch >= 10 and stale >= patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    val_alpha = predict_alpha(model, validation, mean, std, device)
    result = paired_comparison(
        validation.expert, validation.local,
        _coordinate_prediction(validation, val_alpha), validation.sample_ids,
        seed=seed, iterations=2000,
    )
    accepted = (
        result["hard3"]["gain_mm"] > 0.0
        and result["core20"]["gain_mm"] >= -0.05
        and result["hard3"]["candidate"]["p95"] <= result["hard3"]["incoming"]["p95"]
    )
    summary = {
        "accepted_on_validation": bool(accepted), "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "comparison": result,
    }
    write_json(output_dir / "gate_selection.json", summary)
    return model, mean, std, summary


def load_gate(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DiagnosticGate().to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint["mean"], checkpoint["std"]


def gate_predictions(evidence, model, mean, std, device, accepted=True):
    alpha = predict_alpha(model, evidence, mean, std, device)
    if not accepted:
        alpha = np.ones_like(alpha)
    return _coordinate_prediction(evidence, alpha), alpha
