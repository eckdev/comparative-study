"""Per-prediction gate fitted on cross-fitted out-of-fold refiner evidence.

The refiner proposes a coordinate for every landmark; the gate decides how far
to move from the incoming estimate towards that proposal:

    p_out = p_in + alpha * (p_proposal - p_in),  0 <= alpha <= 1.

Unlike the Stage 2/Stage 4 gates of AGH-Former vNext, the evidence used to fit
this gate comes from refiners that never saw the evidence participants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from all23_rgb_geodesic_cascade.anatomy import NUM_LANDMARKS, SYMMETRY_PAIRS

from .anatomy import TYPE_NAMES
from .patches import type_indices


DIAGNOSTIC_NAMES = (
    "entropy",
    "peak_margin",
    "prior_disagreement_over_3",
    "log_variance_over_4",
    "displacement_over_type_scale",
    "iteration_step_over_type_scale",
    "mc_dropout_spread_over_type_scale",
    "anatomical_inconsistency_over_type_scale",
)
# Typical incoming error per type (mm) used to put displacements on one scale.
TYPE_SCALE_MM = {"corner": 2.0, "contour": 4.0, "texture_boundary": 5.0}
MIDLINE_PLANE_LANDMARKS = tuple(range(1, 13))


def type_scale_vector():
    return np.asarray(
        [TYPE_SCALE_MM[TYPE_NAMES[index]] for index in type_indices()],
        dtype=np.float32,
    )


def midsagittal_planes(shapes):
    """Per-sample plane (centroid, unit normal) through the midline landmarks.

    Fitted on the incoming estimate, never on expert landmarks. The normal is
    the least-variance direction of LM1-LM12, i.e. the lateral axis.
    """
    midline = np.asarray(shapes, dtype=np.float64)[:, list(MIDLINE_PLANE_LANDMARKS)]
    centroid = midline.mean(axis=1)
    normals = []
    for points, center in zip(midline, centroid):
        _, _, vt = np.linalg.svd(points - center, full_matrices=False)
        normals.append(vt[-1])
    return centroid, np.stack(normals)


def anatomical_inconsistency(proposal, base):
    """Distance of each proposal from its anatomical expectation (mm).

    Midline landmarks: absolute distance to the midsagittal plane.
    Bilateral landmarks: distance to the partner proposal mirrored through it.
    Faces are not perfectly symmetric, so this is an evidence signal for the
    gate, not a constraint on the coordinates.
    """
    proposal = np.asarray(proposal, dtype=np.float64)
    centroid, normal = midsagittal_planes(base)
    signed = np.einsum("nlk,nk->nl", proposal - centroid[:, None], normal)
    mirrored = proposal - 2.0 * signed[..., None] * normal[:, None]
    result = np.abs(signed)
    for left, right in SYMMETRY_PAIRS:
        result[:, left] = np.linalg.norm(proposal[:, left] - mirrored[:, right], axis=-1)
        result[:, right] = np.linalg.norm(proposal[:, right] - mirrored[:, left], axis=-1)
    return result.astype(np.float32)


def diagnostics_from_iterations(base, iterations):
    """Stack gate inputs from the last refiner iteration, shape [N,23,D]."""
    last = iterations[-1]
    previous = iterations[-2]["proposal"] if len(iterations) > 1 else base
    scale = type_scale_vector()[None]
    displacement = np.linalg.norm(last["proposal"] - base, axis=-1) / scale
    step = np.linalg.norm(last["proposal"] - previous, axis=-1) / scale
    if len(iterations) == 1:
        step = np.zeros_like(step)
    spread = last.get("mc_spread")
    if spread is None:
        spread = np.zeros_like(displacement)
    return np.stack(
        [
            last["entropy"],
            last["margin"],
            last["prior_disagreement"] / 3.0,
            last["log_variance"] / 4.0,
            displacement,
            step,
            spread / scale,
            anatomical_inconsistency(last["proposal"], base) / scale,
        ],
        axis=-1,
    ).astype(np.float32)


@dataclass
class GateEvidence:
    sample_ids: list
    base: np.ndarray
    proposal: np.ndarray
    diagnostics: np.ndarray
    expert: np.ndarray | None = None

    def validate(self, require_expert=False):
        count = len(self.sample_ids)
        if len(set(self.sample_ids)) != count:
            raise ValueError("Duplicate gate-evidence sample IDs")
        expected = (count, NUM_LANDMARKS, 3)
        if self.base.shape != expected or self.proposal.shape != expected:
            raise ValueError("Gate evidence coordinates must be [sample, 23, 3]")
        if self.diagnostics.shape != (count, NUM_LANDMARKS, len(DIAGNOSTIC_NAMES)):
            raise ValueError("Gate diagnostics have an unexpected shape")
        if require_expert and (self.expert is None or self.expert.shape != expected):
            raise ValueError("Expert coordinates are required to fit the gate")
        arrays = [self.base, self.proposal, self.diagnostics]
        if self.expert is not None:
            arrays.append(self.expert)
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("Non-finite gate evidence")
        return self

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sample_ids": np.asarray(self.sample_ids),
            "base": self.base,
            "proposal": self.proposal,
            "diagnostics": self.diagnostics,
        }
        if self.expert is not None:
            payload["expert"] = self.expert
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as record:
            return cls(
                sample_ids=record["sample_ids"].tolist(),
                base=record["base"].copy(),
                proposal=record["proposal"].copy(),
                diagnostics=record["diagnostics"].copy(),
                expert=record["expert"].copy() if "expert" in record.files else None,
            ).validate()

    @classmethod
    def concatenate(cls, rows):
        rows = list(rows)
        expert = (
            np.concatenate([row.expert for row in rows])
            if all(row.expert is not None for row in rows)
            else None
        )
        return cls(
            sample_ids=[sample_id for row in rows for sample_id in row.sample_ids],
            base=np.concatenate([row.base for row in rows]),
            proposal=np.concatenate([row.proposal for row in rows]),
            diagnostics=np.concatenate([row.diagnostics for row in rows]),
            expert=expert,
        ).validate()


class OOFGate(nn.Module):
    def __init__(self, diagnostic_dim=len(DIAGNOSTIC_NAMES), hidden=24):
        super().__init__()
        self.landmark_embedding = nn.Embedding(NUM_LANDMARKS, 4)
        self.type_embedding = nn.Embedding(len(TYPE_NAMES), 3)
        self.register_buffer(
            "landmark_types", torch.as_tensor(type_indices(), dtype=torch.long)
        )
        self.network = nn.Sequential(
            nn.Linear(diagnostic_dim + 7, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        # Start from a mild trust in the proposal (alpha ~= 0.73).
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, 1.0)

    def forward(self, diagnostics):
        ids = torch.arange(NUM_LANDMARKS, device=diagnostics.device)
        embedding = torch.cat(
            [self.landmark_embedding(ids), self.type_embedding(self.landmark_types)],
            dim=-1,
        )
        embedding = embedding[None].expand(len(diagnostics), -1, -1)
        values = torch.cat([diagnostics, embedding], dim=-1)
        return torch.sigmoid(self.network(values).squeeze(-1))


def _split_by_sample(sample_ids, fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sample_ids))
    count = max(1, int(round(fraction * len(sample_ids))))
    validation = np.sort(order[:count])
    train = np.sort(order[count:])
    if len(train) == 0:
        train = validation
    return train, validation


def _gated(evidence_base, evidence_proposal, alpha):
    return evidence_base + alpha[..., None] * (evidence_proposal - evidence_base)


def fit_oof_gate(evidence, output_dir, device, seed=42, epochs=300, patience=30):
    """Fit the gate to minimize Euclidean error on OOF evidence."""
    evidence.validate(require_expert=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    train_rows, validation_rows = _split_by_sample(evidence.sample_ids, 0.2, seed)
    diagnostics = evidence.diagnostics
    mean = diagnostics[train_rows].mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(diagnostics[train_rows].std(axis=(0, 1), keepdims=True), 1e-4)
    std = std.astype(np.float32)

    def tensors(rows):
        return (
            torch.as_tensor((diagnostics[rows] - mean) / std, device=device),
            torch.as_tensor(evidence.base[rows], device=device, dtype=torch.float32),
            torch.as_tensor(evidence.proposal[rows], device=device, dtype=torch.float32),
            torch.as_tensor(evidence.expert[rows], device=device, dtype=torch.float32),
        )

    train = tensors(train_rows)
    validation = tensors(validation_rows)
    model = OOFGate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    best_ale, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        features, base, proposal, expert = train
        alpha = model(features.float())
        loss = torch.linalg.norm(_gated(base, proposal, alpha) - expert, dim=-1).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            features, base, proposal, expert = validation
            alpha = model(features.float())
            validation_ale = float(
                torch.linalg.norm(_gated(base, proposal, alpha) - expert, dim=-1).mean()
            )
        history.append(
            {"epoch": epoch, "train_ale": float(loss.detach()), "validation_ale": validation_ale}
        )
        if validation_ale < best_ale - 1e-5:
            best_ale, best_epoch, stale = validation_ale, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch >= 30 and stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    payload = {
        "state": model.state_dict(),
        "mean": mean,
        "std": std,
        "best_epoch": best_epoch,
        "best_holdout_ale": best_ale,
        "evidence_sample_ids": list(evidence.sample_ids),
        "seed": seed,
    }
    torch.save(payload, output_dir / "oof_gate.pth")
    report = {
        "best_epoch": best_epoch,
        "best_holdout_ale": best_ale,
        "evidence_samples": len(evidence.sample_ids),
        "holdout_samples": int(len(validation_rows)),
        "diagnostics": list(DIAGNOSTIC_NAMES),
        "history": history,
    }
    return model, mean, std, report


def load_oof_gate(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    model = OOFGate().to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload["mean"], payload["std"]


def predict_gate_alpha(model, mean, std, diagnostics, device):
    with torch.no_grad():
        values = torch.as_tensor((diagnostics - mean) / std, device=device).float()
        return model(values).cpu().numpy().astype(np.float32)
