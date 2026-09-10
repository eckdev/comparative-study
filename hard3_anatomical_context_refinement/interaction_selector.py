"""Subject-wise nonlinear ranking for high-recall Gonion proposals.

H3-CFCS v8 established that the correct surface neighborhood is usually present
in the broad proposal, while an additive linear selector cannot identify it.
This module keeps the proposal frozen and trains a compact query-normalized
ranker on out-of-fold subjects. Candidate features are conditioned on a
Core20-only bilateral state prediction, but never on expert validation or test
landmarks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from all23_rgb_geodesic_cascade.anatomy import CORE20

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


def _gather_candidates(values, indices):
    values = np.asarray(values)
    return np.take_along_axis(
        values,
        indices[..., None] if values.ndim == 4 else indices,
        axis=2,
    )


def _query_standardize(values, mask):
    values = np.asarray(values, dtype=np.float32)
    output = np.zeros_like(values)
    for sample in range(values.shape[0]):
        for side in range(values.shape[1]):
            valid = np.asarray(mask[sample, side], dtype=np.bool_)
            row = values[sample, side, valid]
            if not len(row):
                continue
            median = np.median(row, axis=0)
            scale = np.percentile(row, 75, axis=0) - np.percentile(row, 25, axis=0)
            scale = np.maximum(scale, 1e-4)
            output[sample, side, valid] = np.clip((row - median) / scale, -8.0, 8.0)
    return output


def _one_hot_strata(strata, categories):
    output = np.zeros((len(strata), len(categories)), dtype=np.float32)
    lookup = {name: index for index, name in enumerate(categories)}
    for row, name in enumerate(strata):
        if name in lookup:
            output[row, lookup[name]] = 1.0
    return output


def _shortlist_indices(proposal, mask, count):
    selected = _shortlist_mask(proposal, mask, count)
    minimum_valid = max(1, int(selected.sum(axis=-1).min()))
    safe = np.where(selected, proposal, -np.inf)
    indices = np.argpartition(safe, -minimum_valid, axis=-1)[..., -minimum_valid:]
    scores = np.take_along_axis(safe, indices, axis=-1)
    order = np.argsort(scores, axis=-1)[..., ::-1]
    return np.take_along_axis(indices, order, axis=-1)


def _interaction_features(
    candidate_set,
    proposal_sources,
    state_prediction,
    categories,
    indices,
):
    canonical = _gather_candidates(candidate_set.canonical[:, 1:3], indices).astype(
        np.float32
    )
    sources = np.asarray(proposal_sources[:, 1:3], dtype=np.float32).transpose(
        0, 1, 3, 2
    )
    sources = _gather_candidates(sources, indices)
    sources = np.nan_to_num(sources, nan=0.0, posinf=8.0, neginf=-8.0)
    mask = _gather_candidates(candidate_set.mask[:, 1:3], indices).astype(np.bool_)

    # local XYZ (0:3) is tied to the upstream Gonion center and is deliberately
    # excluded. The remaining geometry is expressed in a Core20-derived frame.
    center_invariant = np.nan_to_num(
        canonical[..., 3:], nan=0.0, posinf=8.0, neginf=-8.0
    )
    global_coordinate = canonical[..., 3:6]
    state = np.asarray(state_prediction, dtype=np.float32)
    delta = global_coordinate - state[:, :, None]
    delta_norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    normal = (
        canonical[..., 18:21] if canonical.shape[-1] >= 21 else np.zeros_like(delta)
    )
    normal_alignment = np.sum(delta * normal, axis=-1, keepdims=True)
    interaction = np.concatenate(
        [
            delta,
            np.abs(delta),
            np.sign(delta) * np.square(delta),
            delta_norm,
            delta * normal,
            normal_alignment,
        ],
        axis=-1,
    )
    candidate_block = np.concatenate(
        [sources, center_invariant, interaction], axis=-1
    ).astype(np.float32)
    standardized = _query_standardize(candidate_block, mask)

    samples, sides, candidates = mask.shape
    bilateral_state = np.concatenate(
        [state.reshape(samples, -1), state.mean(axis=1), state[:, 1] - state[:, 0]],
        axis=1,
    )
    core_context = np.asarray(candidate_set.shape_context, dtype=np.float32).reshape(
        samples, 23, 3
    )[:, list(CORE20)]
    core_context = np.nan_to_num(
        core_context.reshape(samples, -1), nan=0.0, posinf=4.0, neginf=-4.0
    )
    bilateral_state = np.concatenate([bilateral_state, core_context], axis=1)
    state_context = np.broadcast_to(
        bilateral_state[:, None, None],
        (samples, sides, candidates, bilateral_state.shape[-1]),
    )
    side_code = np.broadcast_to(
        np.asarray([-1.0, 1.0], dtype=np.float32)[None, :, None, None],
        (samples, sides, candidates, 1),
    )
    strata = _one_hot_strata(candidate_set.strata, categories)
    strata = np.broadcast_to(
        strata[:, None, None],
        (samples, sides, candidates, strata.shape[-1]),
    )
    features = np.concatenate(
        [candidate_block, standardized, state_context, side_code, strata], axis=-1
    )
    features = np.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )
    features[~mask] = 0.0
    return features, mask


def _shortlist_data(
    candidate_set,
    proposal_sources,
    state_prediction,
    categories,
    count,
):
    mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
    proposal = _standardize_per_sample(
        np.asarray(proposal_sources[:, 1:3, 0], dtype=np.float32), mask
    )
    indices = _shortlist_indices(proposal, mask, count)
    features, selected_mask = _interaction_features(
        candidate_set,
        proposal_sources,
        state_prediction,
        categories,
        indices,
    )
    return {
        "indices": indices,
        "features": features,
        "mask": selected_mask,
        "proposal": _gather_candidates(proposal, indices).astype(np.float32),
        "points": _gather_candidates(candidate_set.points[:, 1:3], indices).astype(
            np.float32
        ),
        "distance": _gather_candidates(
            candidate_set.target_distance[:, 1:3], indices
        ).astype(np.float32),
        "expert": np.asarray(candidate_set.expert[:, 1:3], dtype=np.float32),
    }


class _QueryInteractionRanker(nn.Module):
    def __init__(self, input_dim, width, dropout):
        super().__init__()
        hidden = max(8, int(width))
        reduced = max(8, hidden // 2)
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, reduced),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(reduced, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, proposal):
        correction = self.network(features).squeeze(-1)
        return proposal + correction


def _ranker_loss(model, features, proposal, points, expert, distance, mask, config):
    logits = model(features, proposal).float()
    mask = mask.bool()
    safe_distance = torch.where(mask, distance.float(), torch.full_like(distance, 1e4))
    target_logits = -torch.clamp(safe_distance, max=15.0) / max(
        float(config.interaction_sigma), 1e-4
    )
    target_logits = target_logits.masked_fill(~mask, -torch.inf)
    target = torch.softmax(target_logits, dim=-1)
    clinical = 1.0 + (safe_distance <= 2.0).float()
    target = target * clinical
    target = target / torch.clamp(target.sum(dim=-1, keepdim=True), min=1e-8)
    log_probability = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    listwise = -(target * log_probability).sum(dim=-1).mean()

    probability = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    expected = (probability * torch.clamp(safe_distance, max=15.0)).sum(dim=-1)
    expected = expected.mean() / 10.0
    soft_coordinate = (probability[..., None] * points.float()).sum(dim=-2)
    coordinate = F.smooth_l1_loss(soft_coordinate / 10.0, expert.float() / 10.0)
    predicted_pair = torch.cat(
        [
            soft_coordinate.mean(dim=1),
            soft_coordinate[:, 1] - soft_coordinate[:, 0],
        ],
        dim=-1,
    )
    expert_pair = torch.cat(
        [expert.float().mean(dim=1), expert.float()[:, 1] - expert.float()[:, 0]],
        dim=-1,
    )
    pair = F.smooth_l1_loss(predicted_pair / 10.0, expert_pair / 10.0)

    best_index = safe_distance.argmin(dim=-1)
    positive = torch.gather(logits, -1, best_index[..., None]).squeeze(-1)
    best_distance = torch.gather(safe_distance, -1, best_index[..., None]).squeeze(-1)
    negative_mask = mask & (
        safe_distance
        >= best_distance[..., None] + float(config.interaction_negative_radius_mm)
    )
    fallback = mask.scatter(-1, best_index[..., None], False)
    negative_mask = torch.where(
        negative_mask.any(dim=-1, keepdim=True), negative_mask, fallback
    )
    negative = logits.masked_fill(~negative_mask, -torch.inf).max(dim=-1).values
    valid_negative = torch.isfinite(negative)
    if valid_negative.any():
        hard_negative = F.softplus(
            float(config.interaction_negative_margin)
            - positive[valid_negative]
            + negative[valid_negative]
        ).mean()
    else:
        hard_negative = logits.new_zeros(())
    total = (
        listwise
        + float(config.interaction_expected_distance_weight) * expected
        + float(config.interaction_coordinate_weight) * coordinate
        + float(config.interaction_pair_weight) * pair
        + float(config.interaction_hard_negative_weight) * hard_negative
    )
    return total


def _model_logits(model, data, indices, device, batch_size):
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), max(1, int(batch_size))):
            selected = np.asarray(indices[start : start + batch_size], dtype=np.int64)
            features = torch.from_numpy(data["features"][selected]).to(device)
            proposal = torch.from_numpy(data["proposal"][selected]).to(device)
            rows.append(model(features, proposal).float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _coordinate_from_logits(data, logits, indices, topk=1, temperature=1.0):
    selected = np.asarray(indices, dtype=np.int64)
    return _decode(
        logits,
        data["points"][selected],
        data["mask"][selected],
        topk,
        temperature,
    )


def _train_ranker(
    data,
    train_indices,
    validation_indices,
    expert,
    config,
    device,
    fold_number,
):
    seed = int(config.seed) + 1709 * int(fold_number)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _QueryInteractionRanker(
        data["features"].shape[-1],
        config.interaction_width,
        config.interaction_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.interaction_lr),
        weight_decay=float(config.interaction_weight_decay),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    generator = torch.Generator().manual_seed(seed)
    history = []
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    for epoch in range(1, int(config.interaction_epochs) + 1):
        model.train()
        permutation = train_indices[
            torch.randperm(len(train_indices), generator=generator).numpy()
        ]
        losses = []
        for start in range(0, len(permutation), max(1, int(config.batch_size))):
            selected = permutation[start : start + int(config.batch_size)]
            features = torch.from_numpy(data["features"][selected]).to(device)
            proposal = torch.from_numpy(data["proposal"][selected]).to(device)
            points = torch.from_numpy(data["points"][selected]).to(device)
            batch_expert = torch.from_numpy(data["expert"][selected]).to(device)
            distance = torch.from_numpy(data["distance"][selected]).to(device)
            mask = torch.from_numpy(data["mask"][selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _ranker_loss(
                model,
                features,
                proposal,
                points,
                batch_expert,
                distance,
                mask,
                config,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite H3 query interaction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_logits = _model_logits(
            model,
            data,
            validation_indices,
            device,
            config.batch_size,
        )
        coordinate = _coordinate_from_logits(
            data, validation_logits, validation_indices, 1, 1.0
        )
        score = float(
            np.linalg.norm(coordinate - expert[validation_indices], axis=-1).mean()
        )
        row = {"epoch": epoch, "train": float(np.mean(losses)), "val_ale": score}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(config.interaction_epochs):
            print(
                f"H3-QIR-v9 fold {fold_number} epoch {epoch:03d}/"
                f"{config.interaction_epochs} train={row['train']:.4f} "
                f"val_ALE={score:.4f}",
                flush=True,
            )
        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= int(config.interaction_min_epochs) and stale >= int(
            config.interaction_patience
        ):
            break
    model.load_state_dict(best_state)
    return model.eval(), best_epoch, best_score, history


def _train_fixed_ranker(data, epochs, config, device):
    seed = int(config.seed) + 9199
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _QueryInteractionRanker(
        data["features"].shape[-1],
        config.interaction_width,
        config.interaction_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.interaction_lr),
        weight_decay=float(config.interaction_weight_decay),
    )
    indices = np.arange(len(data["features"]), dtype=np.int64)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        permutation = indices[torch.randperm(len(indices), generator=generator).numpy()]
        losses = []
        for start in range(0, len(permutation), max(1, int(config.batch_size))):
            selected = permutation[start : start + int(config.batch_size)]
            features = torch.from_numpy(data["features"][selected]).to(device)
            proposal = torch.from_numpy(data["proposal"][selected]).to(device)
            points = torch.from_numpy(data["points"][selected]).to(device)
            batch_expert = torch.from_numpy(data["expert"][selected]).to(device)
            distance = torch.from_numpy(data["distance"][selected]).to(device)
            mask = torch.from_numpy(data["mask"][selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _ranker_loss(
                model,
                features,
                proposal,
                points,
                batch_expert,
                distance,
                mask,
                config,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite final H3 query interaction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            print(
                f"H3-QIR-v9 final epoch {epoch:03d}/{epochs} "
                f"train={np.mean(losses):.4f}",
                flush=True,
            )
    return model.eval()


def _cpu_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


@dataclass
class _StoredRanker:
    input_dim: int
    width: int
    dropout: float
    state: dict

    @classmethod
    def from_model(cls, model):
        first = model.network[0]
        return cls(
            input_dim=int(first.in_features),
            width=int(first.out_features),
            dropout=float(model.network[3].p),
            state=_cpu_state(model),
        )

    def build(self, device):
        model = _QueryInteractionRanker(self.input_dim, self.width, self.dropout)
        model.load_state_dict(self.state)
        return model.to(device).eval()

    def state_dict(self):
        return {
            "input_dim": self.input_dim,
            "width": self.width,
            "dropout": self.dropout,
            "state": self.state,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            int(state["input_dim"]),
            int(state["width"]),
            float(state["dropout"]),
            state["state"],
        )


class CrossFittedInteractionSelector:
    """Compact subject-wise ranker for a frozen high-recall Gonion proposal."""

    version = "H3-QIR-v9"
    primary_key = "crossfit_interaction"
    coordinate_keys = (
        "crossfit_interaction",
        "interaction_refit",
        "interaction_state_only",
    )

    def __init__(
        self,
        state_model,
        member_rankers,
        final_ranker,
        policy,
        categories,
        report,
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
        shortlist = min(
            int(config.interaction_shortlist), candidate_set.points.shape[-2]
        )
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
            state_score = _standardize_per_sample(
                _state_scores(candidate_set, prediction), mask
            )
            coordinate = _decode(
                state_score,
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
        selected_state_oof = state_oof[state_l2]

        data = _shortlist_data(
            candidate_set,
            proposal_sources,
            selected_state_oof,
            categories,
            shortlist,
        )
        oof_logits = np.full(data["proposal"].shape, -np.inf, dtype=np.float32)
        member_rankers = []
        fold_rows = []
        best_epochs = []
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
            oof_logits[np.asarray(validation_indices, dtype=np.int64)] = _model_logits(
                model,
                data,
                np.asarray(validation_indices, dtype=np.int64),
                device,
                config.batch_size,
            )
            member_rankers.append(_StoredRanker.from_model(model))
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
            raise RuntimeError("H3 query interaction OOF logits are incomplete")

        decoder_rows = []
        for topk, temperature in ((1, 1.0), (3, 0.25), (3, 0.5), (5, 0.5)):
            coordinate = _decode(
                oof_logits,
                data["points"],
                data["mask"],
                topk,
                temperature,
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

        state_model = _ridge_family(
            state_features,
            state_target,
            (state_l2,),
        )[state_l2]
        fixed_epochs = max(1, int(np.median(best_epochs)))
        final_model = _train_fixed_ranker(data, fixed_epochs, config, device)
        report = {
            "version": cls.version,
            "method": (
                "subject-wise query-normalized listwise ranker conditioned on a "
                "Core20-only bilateral state prediction"
            ),
            "uses_outer_validation_labels": False,
            "uses_test_labels": False,
            "input_dim": int(data["features"].shape[-1]),
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
            _StoredRanker.from_model(final_model),
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
        data = _shortlist_data(
            candidate_set,
            proposal_sources,
            state_prediction,
            self.categories,
            self.policy["shortlist"],
        )
        device = torch.device("cpu")
        indices = np.arange(len(candidate_set), dtype=np.int64)
        member_logits = [
            _model_logits(
                stored.build(device), data, indices, device, max(8, len(indices))
            )
            for stored in self.member_rankers
        ]
        ensemble_logits = np.mean(np.stack(member_logits), axis=0)
        refit_logits = _model_logits(
            self.final_ranker.build(device), data, indices, device, max(8, len(indices))
        )
        state_logits = _standardize_per_sample(
            _state_scores(candidate_set, state_prediction),
            candidate_set.mask[:, 1:3],
        )
        state_logits = _gather_candidates(state_logits, data["indices"])
        return data, ensemble_logits, refit_logits, state_logits, member_logits

    def predict(self, candidate_set, proposal_sources):
        data, ensemble, refit, state, member_logits = self._predict_logits(
            candidate_set, proposal_sources
        )
        policy = self.policy
        decode = lambda scores: _decode(
            scores,
            data["points"],
            data["mask"],
            policy["coordinate_topk"],
            policy["temperature"],
        )
        member_coordinate = np.stack([decode(values) for values in member_logits])
        return {
            "crossfit_interaction": decode(ensemble),
            "interaction_refit": decode(refit),
            "interaction_state_only": decode(state),
            "member_coordinate": member_coordinate.astype(np.float32),
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
                f"Unsupported interaction selector: {state.get('version')}"
            )
        from .statistical_selector import _RidgeModel

        return cls(
            _RidgeModel.from_state_dict(state["state_model"]),
            [
                _StoredRanker.from_state_dict(values)
                for values in state["member_rankers"]
            ],
            _StoredRanker.from_state_dict(state["final_ranker"]),
            state["policy"],
            state["categories"],
            state["report"],
        )
