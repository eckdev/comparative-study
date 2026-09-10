"""Relational set ranking for bilateral Gonion candidates.

The broad surface proposal has high recall, but candidate-wise selectors cannot
identify the anatomical jaw corner reliably. H3-RSCR v10 treats every side as a
set-valued query. Each candidate is scored against summaries of its own side,
the contralateral side, and a Core20-only facial context. Training and model
selection use subject-level out-of-fold evidence only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from all23_rgb_geodesic_cascade.anatomy import CORE20

from .interaction_selector import (
    _gather_candidates,
    _one_hot_strata,
    _query_standardize,
    _shortlist_indices,
)
from .statistical_selector import (
    _decode,
    _metrics,
    _ridge_family,
    _shortlist_mask,
    _standardize_per_sample,
    _state_features,
    _state_scores,
    selector_validation_diagnostics,
)


def _set_data(
    candidate_set,
    proposal_sources,
    state_prediction,
    categories,
    count,
):
    full_mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
    proposal = _standardize_per_sample(
        np.asarray(proposal_sources[:, 1:3, 0], dtype=np.float32), full_mask
    )
    indices = _shortlist_indices(proposal, full_mask, count)
    mask = _gather_candidates(full_mask, indices).astype(np.bool_)
    canonical = _gather_candidates(
        np.asarray(candidate_set.canonical[:, 1:3], dtype=np.float32), indices
    )
    sources = np.asarray(proposal_sources[:, 1:3], dtype=np.float32).transpose(
        0, 1, 3, 2
    )
    sources = _gather_candidates(sources, indices)
    sources = np.nan_to_num(sources, nan=0.0, posinf=8.0, neginf=-8.0)

    # Local XYZ (0:3) is measured from the noisy upstream Gonion center. Keep
    # only center-invariant coordinates and explicit surface/appearance cues.
    positional = np.nan_to_num(canonical[..., 3:18], nan=0.0, posinf=8.0, neginf=-8.0)
    positional = _query_standardize(positional, mask)
    intrinsic = np.nan_to_num(canonical[..., 18:], nan=0.0, posinf=8.0, neginf=-8.0)
    state = np.asarray(state_prediction, dtype=np.float32)
    global_coordinate = canonical[..., 3:6]
    delta = global_coordinate - state[:, :, None]
    normal = (
        canonical[..., 18:21] if canonical.shape[-1] >= 21 else np.zeros_like(delta)
    )
    relational = np.concatenate(
        [
            delta,
            np.abs(delta),
            np.sign(delta) * np.square(delta),
            np.linalg.norm(delta, axis=-1, keepdims=True),
            delta * normal,
            np.sum(delta * normal, axis=-1, keepdims=True),
        ],
        axis=-1,
    )
    relational = _query_standardize(relational, mask)
    features = np.concatenate(
        [sources, positional, intrinsic, relational], axis=-1
    ).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0)
    features[~mask] = 0.0

    samples = len(candidate_set)
    shape = np.asarray(candidate_set.shape_context, dtype=np.float32).reshape(
        samples, 23, 3
    )
    core = np.nan_to_num(
        shape[:, list(CORE20)].reshape(samples, -1),
        nan=0.0,
        posinf=4.0,
        neginf=-4.0,
    )
    bilateral = np.concatenate(
        [state.reshape(samples, -1), state.mean(axis=1), state[:, 1] - state[:, 0]],
        axis=1,
    )
    strata = _one_hot_strata(candidate_set.strata, categories)
    context = np.concatenate([core, bilateral, strata], axis=1).astype(np.float32)
    return {
        "indices": indices,
        "features": features,
        "context": context,
        "mask": mask,
        "proposal": _gather_candidates(proposal, indices).astype(np.float32),
        "points": _gather_candidates(candidate_set.points[:, 1:3], indices).astype(
            np.float32
        ),
        "distance": _gather_candidates(
            candidate_set.target_distance[:, 1:3], indices
        ).astype(np.float32),
        "expert": np.asarray(candidate_set.expert[:, 1:3], dtype=np.float32),
    }


class _RelationalSetRanker(nn.Module):
    """Permutation-equivariant candidate scorer with bilateral set context."""

    def __init__(self, input_dim, context_dim, width, dropout):
        super().__init__()
        hidden = max(8, int(width))
        score_width = max(8, hidden * 2)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(int(context_dim)),
            nn.Linear(int(context_dim), hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # candidate + own(mean,max) + opposite(mean,max) + facial context
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 6, score_width),
            nn.LayerNorm(score_width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(score_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    @staticmethod
    def _summary(encoded, mask):
        valid = mask[..., None].to(encoded.dtype)
        mean = (encoded * valid).sum(dim=-2) / valid.sum(dim=-2).clamp_min(1.0)
        maximum = encoded.masked_fill(~mask[..., None], -torch.inf).amax(dim=-2)
        maximum = torch.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.cat([mean, maximum], dim=-1)

    def forward(self, features, context, proposal, mask):
        encoded = self.candidate_encoder(features.float())
        summary = self._summary(encoded, mask)
        own = summary[:, :, None].expand(-1, -1, encoded.shape[2], -1)
        opposite = summary.flip(1)[:, :, None].expand_as(own)
        face = self.context_encoder(context.float())
        face = face[:, None, None].expand(-1, 2, encoded.shape[2], -1)
        correction = self.scorer(
            torch.cat([encoded, own, opposite, face], dim=-1)
        ).squeeze(-1)
        return (proposal.float() + correction).masked_fill(~mask, -torch.inf)


def _training_mask(mask, distance, probability, generator):
    if probability <= 0.0:
        return mask
    keep = torch.rand(mask.shape, generator=generator).to(mask.device) >= probability
    keep &= mask
    nearest = distance.masked_fill(~mask, torch.inf).argmin(dim=-1)
    keep.scatter_(-1, nearest[..., None], True)
    return keep & mask


def _ranker_loss(
    model,
    features,
    context,
    proposal,
    points,
    expert,
    distance,
    mask,
    config,
    generator,
):
    active = _training_mask(
        mask.bool(),
        distance.float(),
        float(config.set_candidate_dropout),
        generator,
    )
    logits = model(features, context, proposal, active).float()
    safe_distance = torch.where(
        active, distance.float(), torch.full_like(distance, 1e4)
    )
    sigma = max(float(config.set_sigma), 1e-4)
    target_logits = -torch.clamp(safe_distance, max=15.0) / sigma
    target_logits = target_logits.masked_fill(~active, -torch.inf)
    target = torch.softmax(target_logits, dim=-1)
    clinical = 1.0 + (safe_distance <= 2.0).float()
    target = target * clinical
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probability = torch.log_softmax(logits, dim=-1)
    listwise = -(target * log_probability.masked_fill(~active, 0.0)).sum(dim=-1).mean()

    probability = torch.softmax(logits, dim=-1).masked_fill(~active, 0.0)
    expected = (probability * torch.clamp(safe_distance, max=15.0)).sum(
        dim=-1
    ).mean() / 10.0
    soft_coordinate = (probability[..., None] * points.float()).sum(dim=-2)
    coordinate = F.smooth_l1_loss(soft_coordinate / 10.0, expert.float() / 10.0)
    predicted_pair = torch.cat(
        [soft_coordinate.mean(dim=1), soft_coordinate[:, 1] - soft_coordinate[:, 0]],
        dim=-1,
    )
    expert_pair = torch.cat(
        [expert.float().mean(dim=1), expert.float()[:, 1] - expert.float()[:, 0]],
        dim=-1,
    )
    pair = F.smooth_l1_loss(predicted_pair / 10.0, expert_pair / 10.0)

    positive_mask = active & (safe_distance <= 2.0)
    nearest = safe_distance.argmin(dim=-1)
    fallback = torch.zeros_like(active).scatter(-1, nearest[..., None], True)
    positive_mask = torch.where(
        positive_mask.any(dim=-1, keepdim=True), positive_mask, fallback
    )
    positive_mass = (probability * positive_mask).sum(dim=-1).clamp_min(1e-8)
    clinical_mass = -torch.log(positive_mass).mean()

    positive = logits.masked_fill(~positive_mask, -torch.inf).amax(dim=-1)
    best_distance = safe_distance.amin(dim=-1)
    negative_mask = active & (
        safe_distance >= best_distance[..., None] + float(config.set_negative_radius_mm)
    )
    fallback_negative = active & ~fallback
    negative_mask = torch.where(
        negative_mask.any(dim=-1, keepdim=True), negative_mask, fallback_negative
    )
    negative = logits.masked_fill(~negative_mask, -torch.inf).amax(dim=-1)
    valid_negative = torch.isfinite(negative)
    hard_negative = (
        F.softplus(
            float(config.set_negative_margin)
            - positive[valid_negative]
            + negative[valid_negative]
        ).mean()
        if valid_negative.any()
        else logits.new_zeros(())
    )
    return (
        listwise
        + float(config.set_expected_distance_weight) * expected
        + float(config.set_coordinate_weight) * coordinate
        + float(config.set_pair_weight) * pair
        + float(config.set_clinical_mass_weight) * clinical_mass
        + float(config.set_hard_negative_weight) * hard_negative
    )


def _model_logits(model, data, indices, device, batch_size):
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), max(1, int(batch_size))):
            selected = np.asarray(indices[start : start + batch_size], dtype=np.int64)
            rows.append(
                model(
                    torch.from_numpy(data["features"][selected]).to(device),
                    torch.from_numpy(data["context"][selected]).to(device),
                    torch.from_numpy(data["proposal"][selected]).to(device),
                    torch.from_numpy(data["mask"][selected]).to(device),
                )
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(rows, axis=0)


def _decode_logits(data, logits, indices, topk=1, temperature=1.0):
    selected = np.asarray(indices, dtype=np.int64)
    return _decode(
        logits,
        data["points"][selected],
        data["mask"][selected],
        topk,
        temperature,
    )


def _new_ranker(data, config):
    return _RelationalSetRanker(
        data["features"].shape[-1],
        data["context"].shape[-1],
        config.set_width,
        config.set_dropout,
    )


def _train_ranker(
    data, train_indices, validation_indices, expert, config, device, fold_number
):
    seed = int(config.seed) + 2029 * int(fold_number)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _new_ranker(data, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.set_lr),
        weight_decay=float(config.set_weight_decay),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_epoch, stale = float("inf"), 0, 0
    generator = torch.Generator().manual_seed(seed)
    history = []
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    for epoch in range(1, int(config.set_epochs) + 1):
        model.train()
        permutation = train_indices[
            torch.randperm(len(train_indices), generator=generator).numpy()
        ]
        losses = []
        for start in range(0, len(permutation), max(1, int(config.batch_size))):
            selected = permutation[start : start + int(config.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss = _ranker_loss(
                model,
                torch.from_numpy(data["features"][selected]).to(device),
                torch.from_numpy(data["context"][selected]).to(device),
                torch.from_numpy(data["proposal"][selected]).to(device),
                torch.from_numpy(data["points"][selected]).to(device),
                torch.from_numpy(data["expert"][selected]).to(device),
                torch.from_numpy(data["distance"][selected]).to(device),
                torch.from_numpy(data["mask"][selected]).to(device),
                config,
                generator,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite H3 relational set loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_logits = _model_logits(
            model, data, validation_indices, device, config.batch_size
        )
        coordinate = _decode_logits(data, validation_logits, validation_indices)
        score = float(
            np.linalg.norm(coordinate - expert[validation_indices], axis=-1).mean()
        )
        row = {"epoch": epoch, "train": float(np.mean(losses)), "val_ale": score}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(config.set_epochs):
            print(
                f"H3-RSCR-v10 fold {fold_number} epoch {epoch:03d}/"
                f"{config.set_epochs} train={row['train']:.4f} val_ALE={score:.4f}",
                flush=True,
            )
        if score < best_score - 1e-5:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= int(config.set_min_epochs) and stale >= int(config.set_patience):
            break
    model.load_state_dict(best_state)
    return model.eval(), best_epoch, best_score, history


def _train_fixed(data, epochs, config, device):
    seed = int(config.seed) + 10007
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _new_ranker(data, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.set_lr),
        weight_decay=float(config.set_weight_decay),
    )
    indices = np.arange(len(data["features"]), dtype=np.int64)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        permutation = indices[torch.randperm(len(indices), generator=generator).numpy()]
        losses = []
        for start in range(0, len(permutation), max(1, int(config.batch_size))):
            selected = permutation[start : start + int(config.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss = _ranker_loss(
                model,
                torch.from_numpy(data["features"][selected]).to(device),
                torch.from_numpy(data["context"][selected]).to(device),
                torch.from_numpy(data["proposal"][selected]).to(device),
                torch.from_numpy(data["points"][selected]).to(device),
                torch.from_numpy(data["expert"][selected]).to(device),
                torch.from_numpy(data["distance"][selected]).to(device),
                torch.from_numpy(data["mask"][selected]).to(device),
                config,
                generator,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite final H3 relational set loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            print(
                f"H3-RSCR-v10 final epoch {epoch:03d}/{epochs} "
                f"train={np.mean(losses):.4f}",
                flush=True,
            )
    return model.eval()


def _cpu_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


@dataclass
class _StoredSetRanker:
    input_dim: int
    context_dim: int
    width: int
    dropout: float
    state: dict

    @classmethod
    def from_model(cls, model):
        return cls(
            int(model.candidate_encoder[1].in_features),
            int(model.context_encoder[1].in_features),
            int(model.candidate_encoder[1].out_features),
            float(model.candidate_encoder[3].p),
            _cpu_state(model),
        )

    def build(self, device):
        model = _RelationalSetRanker(
            self.input_dim, self.context_dim, self.width, self.dropout
        )
        model.load_state_dict(self.state)
        return model.to(device).eval()

    def state_dict(self):
        return {
            "input_dim": self.input_dim,
            "context_dim": self.context_dim,
            "width": self.width,
            "dropout": self.dropout,
            "state": self.state,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            int(state["input_dim"]),
            int(state["context_dim"]),
            int(state["width"]),
            float(state["dropout"]),
            state["state"],
        )


class CrossFittedRelationalSetSelector:
    version = "H3-RSCR-v10"
    primary_key = "crossfit_set_context"
    coordinate_keys = (
        "crossfit_set_context",
        "set_context_refit",
        "set_context_state_only",
    )

    def __init__(
        self, state_model, member_rankers, final_ranker, policy, categories, report
    ):
        self.state_model = state_model
        self.member_rankers = list(member_rankers)
        self.final_ranker = final_ranker
        self.policy = dict(policy)
        self.categories = tuple(categories)
        self.report = dict(report)

    @classmethod
    def fit(cls, candidate_set, proposal_sources, splits, config, device):
        l2_values = tuple(float(value) for value in config.statistical_l2_grid)
        categories = tuple(sorted(set(candidate_set.strata)))
        state_features = _state_features(candidate_set, categories)
        state_target = np.asarray(
            candidate_set.expert_gonion_context, dtype=np.float32
        ).reshape(len(candidate_set), -1)
        mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
        proposal = _standardize_per_sample(
            np.asarray(proposal_sources[:, 1:3, 0], dtype=np.float32), mask
        )
        shortlist = min(int(config.set_shortlist), candidate_set.points.shape[-2])
        shortlist_mask = _shortlist_mask(proposal, mask, shortlist)
        state_oof = {
            l2: np.zeros((len(candidate_set), 2, 3), dtype=np.float32)
            for l2 in l2_values
        }
        for train_indices, validation_indices in splits:
            train_indices = np.asarray(train_indices, dtype=np.int64)
            validation_indices = np.asarray(validation_indices, dtype=np.int64)
            family = _ridge_family(
                state_features[train_indices], state_target[train_indices], l2_values
            )
            for l2, model in family.items():
                state_oof[l2][validation_indices] = model.predict(
                    state_features[validation_indices]
                ).reshape(-1, 2, 3)
        state_rows = []
        for l2, prediction in state_oof.items():
            score = _standardize_per_sample(
                _state_scores(candidate_set, prediction), mask
            )
            coordinate = _decode(
                score,
                candidate_set.points[:, 1:3],
                shortlist_mask,
                1,
                1.0,
            )
            state_rows.append(
                {"l2": l2, **_metrics(coordinate, candidate_set.expert[:, 1:3])}
            )
        state_row = min(
            state_rows, key=lambda row: (row["ale"], row["p95"], -row["sdr_at_2mm"])
        )
        state_l2 = float(state_row["l2"])
        data = _set_data(
            candidate_set,
            proposal_sources,
            state_oof[state_l2],
            categories,
            shortlist,
        )
        oof_logits = np.full(data["proposal"].shape, -np.inf, dtype=np.float32)
        member_rankers, fold_rows, best_epochs = [], [], []
        for fold_number, (train_indices, validation_indices) in enumerate(
            splits, start=1
        ):
            model, best_epoch, best_score, history = _train_ranker(
                data,
                train_indices,
                validation_indices,
                candidate_set.expert[:, 1:3],
                config,
                device,
                fold_number,
            )
            validation_indices = np.asarray(validation_indices, dtype=np.int64)
            oof_logits[validation_indices] = _model_logits(
                model, data, validation_indices, device, config.batch_size
            )
            member_rankers.append(_StoredSetRanker.from_model(model))
            best_epochs.append(int(best_epoch))
            fold_rows.append(
                {
                    "fold": fold_number,
                    "best_epoch": int(best_epoch),
                    "best_val_ale": float(best_score),
                    "history": history,
                }
            )
        if not np.isfinite(oof_logits[data["mask"]]).all():
            raise RuntimeError("H3 relational set OOF logits are incomplete")
        decoder_rows = []
        for topk, temperature in ((1, 1.0), (3, 0.25), (3, 0.5), (5, 0.5)):
            coordinate = _decode(
                oof_logits, data["points"], data["mask"], topk, temperature
            )
            decoder_rows.append(
                {
                    "coordinate_topk": topk,
                    "temperature": temperature,
                    **_metrics(coordinate, candidate_set.expert[:, 1:3]),
                }
            )
        selected = min(
            decoder_rows,
            key=lambda row: (row["ale"], row["p95"], -row["sdr_at_2mm"]),
        )
        selected.update(
            {"shortlist": int(data["points"].shape[2]), "state_l2": state_l2}
        )
        oof_prediction = _decode(
            oof_logits,
            data["points"],
            data["mask"],
            selected["coordinate_topk"],
            selected["temperature"],
        )
        state_model = _ridge_family(state_features, state_target, (state_l2,))[state_l2]
        fixed_epochs = max(1, int(np.median(best_epochs)))
        final_model = _train_fixed(data, fixed_epochs, config, device)
        report = {
            "version": cls.version,
            "method": (
                "bilateral permutation-equivariant set ranker conditioned on "
                "Core20-only facial context"
            ),
            "uses_outer_validation_labels": False,
            "uses_test_labels": False,
            "input_dim": int(data["features"].shape[-1]),
            "context_dim": int(data["context"].shape[-1]),
            "parameter_count": int(sum(p.numel() for p in final_model.parameters())),
            "selected": selected,
            "state_l2_sweep": state_rows,
            "decoder_sweep": decoder_rows,
            "folds": fold_rows,
            "fixed_epochs": fixed_epochs,
        }
        selector = cls(
            state_model,
            member_rankers,
            _StoredSetRanker.from_model(final_model),
            selected,
            categories,
            report,
        )
        selector.oof_prediction = oof_prediction
        return selector

    def _predict_logits(self, candidate_set, proposal_sources):
        state_prediction = self.state_model.predict(
            _state_features(candidate_set, self.categories)
        ).reshape(-1, 2, 3)
        data = _set_data(
            candidate_set,
            proposal_sources,
            state_prediction,
            self.categories,
            self.policy["shortlist"],
        )
        device = torch.device("cpu")
        indices = np.arange(len(candidate_set), dtype=np.int64)
        members = [
            _model_logits(
                stored.build(device), data, indices, device, max(8, len(indices))
            )
            for stored in self.member_rankers
        ]
        ensemble = np.mean(np.stack(members), axis=0)
        refit = _model_logits(
            self.final_ranker.build(device), data, indices, device, max(8, len(indices))
        )
        state = _standardize_per_sample(
            _state_scores(candidate_set, state_prediction),
            candidate_set.mask[:, 1:3],
        )
        state = _gather_candidates(state, data["indices"])
        return data, ensemble, refit, state, members

    def predict(self, candidate_set, proposal_sources):
        data, ensemble, refit, state, members = self._predict_logits(
            candidate_set, proposal_sources
        )
        policy = self.policy
        decode = lambda values: _decode(
            values,
            data["points"],
            data["mask"],
            policy["coordinate_topk"],
            policy["temperature"],
        )
        return {
            "crossfit_set_context": decode(ensemble),
            "set_context_refit": decode(refit),
            "set_context_state_only": decode(state),
            "member_coordinate": np.stack(
                [decode(values) for values in members]
            ).astype(np.float32),
        }

    def validation_diagnostics(self, candidate_set, prediction, proposal_sources):
        return selector_validation_diagnostics(
            candidate_set,
            prediction,
            proposal_sources,
            self.policy["shortlist"],
        )

    def state_dict(self):
        return {
            "version": self.version,
            "state_model": self.state_model.state_dict(),
            "member_rankers": [model.state_dict() for model in self.member_rankers],
            "final_ranker": self.final_ranker.state_dict(),
            "policy": self.policy,
            "categories": list(self.categories),
            "report": self.report,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != cls.version:
            raise ValueError(
                f"Unsupported relational set selector: {state.get('version')}"
            )
        from .statistical_selector import _RidgeModel

        return cls(
            _RidgeModel.from_state_dict(state["state_model"]),
            [
                _StoredSetRanker.from_state_dict(values)
                for values in state["member_rankers"]
            ],
            _StoredSetRanker.from_state_dict(state["final_ranker"]),
            state["policy"],
            state["categories"],
            state["report"],
        )
