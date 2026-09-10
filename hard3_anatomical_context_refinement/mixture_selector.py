"""Leakage-free coordinate-mixture estimator for bilateral Gonion.

The broad proposal has high recall, while no single vertex descriptor has
generalized across subjects. This selector changes the target: it summarizes
the full score distributions from the frozen OOF proposal experts and predicts
one bilateral Gonion state. Hyperparameters and the coordinate decoder are
selected only from nested out-of-fold predictions.
"""

from __future__ import annotations

import numpy as np

from all23_rgb_geodesic_cascade.anatomy import CORE20

from .statistical_selector import (
    _RidgeModel,
    _metrics,
    _ridge_family,
    _shortlist_mask,
    _standardize_per_sample,
    selector_validation_diagnostics,
)


def _masked_distribution(scores, mask, temperature=1.0):
    safe = np.where(mask, scores, -np.inf).astype(np.float64)
    maximum = np.max(safe, axis=-1, keepdims=True)
    probability = np.exp((safe - maximum) / max(float(temperature), 1e-4))
    probability *= mask
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
    return probability.astype(np.float32)


def _weighted_coordinate(scores, points, mask, topk, temperature):
    safe = np.where(mask, scores, -np.inf)
    valid_count = max(1, int(mask.sum(axis=-1).min()))
    count = min(max(1, int(topk)), scores.shape[-1], valid_count)
    indices = np.argpartition(safe, -count, axis=-1)[..., -count:]
    selected_scores = np.take_along_axis(safe, indices, axis=-1)
    selected_points = np.take_along_axis(points, indices[..., None], axis=2)
    if count == 1:
        return selected_points[..., 0, :].astype(np.float32)
    weights = _masked_distribution(
        selected_scores,
        np.ones_like(selected_scores, dtype=np.bool_),
        temperature,
    )
    return np.sum(weights[..., None] * selected_points, axis=2).astype(np.float32)


def _score_summary(scores, canonical, mask):
    standardized = _standardize_per_sample(scores, mask)
    safe = np.where(mask, standardized, -np.inf)
    probability = _masked_distribution(standardized, mask, 1.0)
    mean = np.sum(probability[..., None] * canonical, axis=2)
    variance = np.sum(
        probability[..., None] * np.square(canonical - mean[:, :, None]), axis=2
    )
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=-1)
    entropy /= np.log(np.maximum(mask.sum(axis=-1), 2))
    ordered = np.sort(safe, axis=-1)
    margin = ordered[..., -1] - ordered[..., -2]
    return np.concatenate(
        [
            _weighted_coordinate(standardized, canonical, mask, 1, 1.0),
            _weighted_coordinate(standardized, canonical, mask, 5, 0.5),
            _weighted_coordinate(standardized, canonical, mask, 20, 0.75),
            mean,
            np.sqrt(np.maximum(variance, 0.0)),
            entropy[..., None],
            margin[..., None],
        ],
        axis=-1,
    ).astype(np.float32)


def _one_hot_strata(strata, categories):
    output = np.zeros((len(strata), len(categories)), dtype=np.float32)
    lookup = {value: index for index, value in enumerate(categories)}
    for row, value in enumerate(strata):
        if value in lookup:
            output[row, lookup[value]] = 1.0
    return output


def _mixture_features(candidate_set, proposal_sources, categories):
    mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
    canonical = np.nan_to_num(
        np.asarray(candidate_set.canonical[:, 1:3, :, 3:6], dtype=np.float32),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    )
    sources = np.asarray(proposal_sources[:, 1:3], dtype=np.float32)
    summaries = []
    coordinate_summaries = []
    for source in range(sources.shape[2]):
        summary = _score_summary(sources[:, :, source], canonical, mask)
        summaries.append(summary)
        coordinate_summaries.append(summary[..., :12])
    summary = np.concatenate(summaries, axis=-1).reshape(len(mask), -1)
    coordinates = np.concatenate(coordinate_summaries, axis=-1).reshape(len(mask), -1)
    base = np.asarray(candidate_set.base_gonion, dtype=np.float32).reshape(
        len(mask), -1
    )
    bilateral = np.concatenate(
        [
            candidate_set.base_gonion.mean(axis=1),
            candidate_set.base_gonion[:, 0] - candidate_set.base_gonion[:, 1],
        ],
        axis=-1,
    ).astype(np.float32)
    category = _one_hot_strata(candidate_set.strata, categories)
    context = np.asarray(candidate_set.shape_context, dtype=np.float32).reshape(
        len(mask), 23, 3
    )
    core = context[:, list(CORE20)].reshape(len(mask), -1)
    core_quadratic = np.sign(core) * np.square(core)
    common = np.concatenate([base, bilateral, category], axis=-1)
    return {
        "coordinate": np.concatenate([common, coordinates], axis=-1),
        "evidence": np.concatenate([common, summary], axis=-1),
        "evidence_shape": np.concatenate([common, summary, core], axis=-1),
        "evidence_shape_quadratic": np.concatenate(
            [common, summary, core, core_quadratic], axis=-1
        ),
    }


def _affine_world_coordinate(candidate_set, state):
    """Invert each label-free mirrored canonical frame by local affine fit."""
    canonical = np.asarray(candidate_set.canonical[:, 1:3, :, 3:6], dtype=np.float64)
    points = np.asarray(candidate_set.points[:, 1:3], dtype=np.float64)
    mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
    state = np.asarray(state, dtype=np.float64).reshape(-1, 2, 3)
    output = np.zeros_like(state, dtype=np.float64)
    for sample in range(len(state)):
        for side in range(2):
            valid = mask[sample, side]
            design = np.concatenate(
                [canonical[sample, side, valid], np.ones((valid.sum(), 1))], axis=1
            )
            transform, _, _, _ = np.linalg.lstsq(
                design, points[sample, side, valid], rcond=None
            )
            output[sample, side] = np.append(state[sample, side], 1.0) @ transform
    return output.astype(np.float32)


def _decode_state(
    candidate_set, proposal_sources, state, alpha, decoder, shortlist_count=96
):
    state = np.asarray(state, dtype=np.float32).reshape(-1, 2, 3)
    base = np.asarray(candidate_set.base_gonion, dtype=np.float32)
    state = base + float(alpha) * (state - base)
    continuous = _affine_world_coordinate(candidate_set, state)
    if decoder == "continuous":
        return continuous

    mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
    proposal = _standardize_per_sample(
        np.asarray(proposal_sources[:, 1:3, 0], dtype=np.float32), mask
    )
    shortlist = _shortlist_mask(proposal, mask, shortlist_count)
    points = np.asarray(candidate_set.points[:, 1:3], dtype=np.float32)
    distance = np.linalg.norm(points - continuous[:, :, None], axis=-1)
    distance = np.where(shortlist, distance, np.inf)
    nearest = np.argmin(distance, axis=-1)
    surface = np.take_along_axis(points, nearest[..., None, None], axis=2)[..., 0, :]
    if decoder == "surface":
        return surface.astype(np.float32)
    if decoder == "hybrid":
        return (0.5 * continuous + 0.5 * surface).astype(np.float32)
    if decoder == "soft_surface":
        score = -np.square(distance) / (2.0 * 2.0**2)
        return _weighted_coordinate(score, points, shortlist, 5, 1.0)
    raise ValueError(f"Unknown mixture decoder: {decoder}")


class CrossFittedMixtureStateSelector:
    """Joint bilateral state stacker fitted only from OOF proposal evidence."""

    version = "H3-CMSE-v13"
    primary_key = "crossfit_mixture_state"
    coordinate_keys = (
        "crossfit_mixture_state",
        "mixture_state_refit",
        "mixture_state_continuous",
        "mixture_state_surface",
    )

    def __init__(self, final_model, member_models, policy, categories, report):
        self.final_model = final_model
        self.member_models = tuple(member_models)
        self.policy = dict(policy)
        self.categories = tuple(categories)
        self.report = dict(report)

    @classmethod
    def fit(cls, candidate_set, proposal_sources, splits, config):
        categories = tuple(sorted(set(candidate_set.strata)))
        families = _mixture_features(candidate_set, proposal_sources, categories)
        requested = tuple(config.mixture_feature_modes)
        unknown = sorted(set(requested) - set(families))
        if unknown:
            raise ValueError(f"Unknown mixture feature modes: {unknown}")
        l2_values = tuple(float(value) for value in config.mixture_l2_grid)
        if not l2_values:
            raise ValueError("mixture_l2_grid cannot be empty")
        target = np.asarray(
            candidate_set.expert_gonion_context, dtype=np.float32
        ).reshape(len(candidate_set), -1)
        oof_states = {
            (mode, l2): np.zeros_like(target, dtype=np.float32)
            for mode in requested
            for l2 in l2_values
        }
        coverage = np.zeros(len(candidate_set), dtype=np.int64)
        for train_indices, validation_indices in splits:
            train_indices = np.asarray(train_indices, dtype=np.int64)
            validation_indices = np.asarray(validation_indices, dtype=np.int64)
            if np.intersect1d(train_indices, validation_indices).size:
                raise ValueError("Mixture selector train/validation folds overlap")
            if (
                np.any(train_indices < 0)
                or np.any(validation_indices < 0)
                or np.any(train_indices >= len(candidate_set))
                or np.any(validation_indices >= len(candidate_set))
            ):
                raise ValueError("Mixture selector fold index is out of range")
            coverage[validation_indices] += 1
            for mode in requested:
                family = _ridge_family(
                    families[mode][train_indices], target[train_indices], l2_values
                )
                for l2, model in family.items():
                    oof_states[(mode, l2)][validation_indices] = model.predict(
                        families[mode][validation_indices]
                    )
        if not np.all(coverage == 1):
            raise ValueError(
                "Mixture selector requires exactly one OOF prediction per sample"
            )

        rows = []
        for mode in requested:
            for l2 in l2_values:
                state = oof_states[(mode, l2)].reshape(-1, 2, 3)
                for alpha in config.mixture_alpha_grid:
                    for decoder in (
                        "continuous",
                        "surface",
                        "soft_surface",
                        "hybrid",
                    ):
                        coordinate = _decode_state(
                            candidate_set,
                            proposal_sources,
                            state,
                            alpha,
                            decoder,
                            config.mixture_shortlist,
                        )
                        rows.append(
                            {
                                "feature_mode": mode,
                                "l2": float(l2),
                                "alpha": float(alpha),
                                "decoder": decoder,
                                **_metrics(coordinate, candidate_set.expert[:, 1:3]),
                            }
                        )
        selected = min(
            rows,
            key=lambda row: (row["ale"], row["p95"], -row["sdr_at_2mm"]),
        )
        selected_state = oof_states[(selected["feature_mode"], selected["l2"])].reshape(
            -1, 2, 3
        )
        oof_prediction = _decode_state(
            candidate_set,
            proposal_sources,
            selected_state,
            selected["alpha"],
            selected["decoder"],
            config.mixture_shortlist,
        )

        mode = selected["feature_mode"]
        l2 = float(selected["l2"])
        member_models = []
        for train_indices, _ in splits:
            train_indices = np.asarray(train_indices, dtype=np.int64)
            member_models.append(
                _ridge_family(
                    families[mode][train_indices], target[train_indices], (l2,)
                )[l2]
            )
        final_model = _ridge_family(families[mode], target, (l2,))[l2]
        policy = {
            **selected,
            "shortlist": int(config.mixture_shortlist),
        }
        report = {
            "version": cls.version,
            "method": (
                "joint bilateral ridge stacker over nested-OOF proposal-coordinate "
                "distributions and Core20 shape context"
            ),
            "uses_outer_validation_labels": False,
            "uses_test_labels": False,
            "selected": policy,
            "policy_sweep": rows,
            "feature_dims": {
                name: int(values.shape[-1]) for name, values in families.items()
            },
            "parameter_count": int(final_model.weights.size),
            "source_count": int(proposal_sources.shape[2]),
        }
        selector = cls(final_model, member_models, policy, categories, report)
        selector.oof_prediction = oof_prediction
        return selector

    def _state(self, candidate_set, proposal_sources, model):
        features = _mixture_features(candidate_set, proposal_sources, self.categories)
        return model.predict(features[self.policy["feature_mode"]]).reshape(-1, 2, 3)

    def _coordinate(self, candidate_set, proposal_sources, state, decoder=None):
        return _decode_state(
            candidate_set,
            proposal_sources,
            state,
            self.policy["alpha"],
            self.policy["decoder"] if decoder is None else decoder,
            self.policy["shortlist"],
        )

    def predict(self, candidate_set, proposal_sources):
        member_states = np.stack(
            [
                self._state(candidate_set, proposal_sources, model)
                for model in self.member_models
            ]
        )
        ensemble_state = member_states.mean(axis=0)
        refit_state = self._state(candidate_set, proposal_sources, self.final_model)
        member_coordinate = np.stack(
            [
                self._coordinate(candidate_set, proposal_sources, state)
                for state in member_states
            ]
        )
        return {
            self.primary_key: self._coordinate(
                candidate_set, proposal_sources, ensemble_state
            ),
            "mixture_state_refit": self._coordinate(
                candidate_set, proposal_sources, refit_state
            ),
            "mixture_state_continuous": self._coordinate(
                candidate_set, proposal_sources, ensemble_state, "continuous"
            ),
            "mixture_state_surface": self._coordinate(
                candidate_set, proposal_sources, ensemble_state, "surface"
            ),
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
            "final_model": self.final_model.state_dict(),
            "member_models": [model.state_dict() for model in self.member_models],
            "policy": self.policy,
            "categories": list(self.categories),
            "report": self.report,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != cls.version:
            raise ValueError(f"Unsupported mixture selector: {state.get('version')}")
        return cls(
            _RidgeModel.from_state_dict(state["final_model"]),
            [_RidgeModel.from_state_dict(values) for values in state["member_models"]],
            state["policy"],
            state["categories"],
            state["report"],
        )
