"""Cross-fitted, low-capacity Gonion candidate calibration.

The neural proposal has high surface recall but its high-capacity pair decoder
does not generalize reliably from a small outer-training fold.  This module
fits two regularized models only on out-of-fold proposal evidence:

* a pointwise contour ranker over center-invariant candidate features; and
* a Core20-conditioned bilateral Gonion state regressor.

Their relative weights and coordinate decoder are selected on inner OOF
predictions.  Outer-validation and test labels are never used by ``fit``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from all23_rgb_geodesic_cascade.anatomy import CORE20


@dataclass
class _RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray

    def predict(self, features):
        values = np.asarray(features, dtype=np.float64)
        normalized = (values - self.mean) / self.scale
        return (normalized @ self.weights + self.target_mean).astype(np.float32)

    def state_dict(self):
        return {
            "mean": self.mean.astype(np.float32),
            "scale": self.scale.astype(np.float32),
            "target_mean": self.target_mean.astype(np.float32),
            "weights": self.weights.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            np.asarray(state["mean"], dtype=np.float64),
            np.asarray(state["scale"], dtype=np.float64),
            np.asarray(state["target_mean"], dtype=np.float64),
            np.asarray(state["weights"], dtype=np.float64),
        )


def _ridge_family(features, targets, l2_values, weights=None):
    features = np.nan_to_num(
        np.asarray(features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    targets = np.nan_to_num(
        np.asarray(targets, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    if targets.ndim == 1:
        targets = targets[:, None]
    row_weight = (
        np.ones(len(features), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    row_weight = np.maximum(row_weight, 1e-6)
    total_weight = float(row_weight.sum())
    mean = np.sum(features * row_weight[:, None], axis=0) / total_weight
    centered = features - mean
    variance = np.sum(np.square(centered) * row_weight[:, None], axis=0) / total_weight
    scale = np.maximum(np.sqrt(variance), 1e-4)
    normalized = centered / scale
    target_mean = np.sum(targets * row_weight[:, None], axis=0) / total_weight
    centered_target = targets - target_mean
    weighted = normalized * np.sqrt(row_weight[:, None])
    weighted_target = centered_target * np.sqrt(row_weight[:, None])
    gram = weighted.T @ weighted / total_weight
    rhs = weighted.T @ weighted_target / total_weight
    identity = np.eye(gram.shape[0], dtype=np.float64)
    family = {}
    for l2 in l2_values:
        regularized = gram + max(float(l2), 0.0) * identity
        try:
            coefficients = np.linalg.solve(regularized, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(regularized) @ rhs
        family[float(l2)] = _RidgeModel(mean, scale, target_mean, coefficients)
    return family


def _candidate_features(candidate_set, proposal_sources):
    """Build mirrored features that do not encode the noisy Hard3 ROI center."""
    canonical = np.asarray(candidate_set.canonical[:, 1:3], dtype=np.float32)
    sources = np.asarray(proposal_sources[:, 1:3], dtype=np.float32).transpose(
        0, 1, 3, 2
    )
    sources = np.nan_to_num(sources, nan=0.0, posinf=12.0, neginf=-12.0)

    # local XYZ (columns 0:3) is measured from the upstream Gonion estimate and
    # caused an in-sample-center shortcut in v7.  Global coordinates, distances
    # to LM10-12, normals and explicit contour channels remain center invariant.
    center_invariant = canonical[..., 3:]
    geometry = canonical[..., 3:18]
    quadratic = np.sign(geometry) * np.square(geometry)
    return np.nan_to_num(
        np.concatenate([sources, center_invariant, quadratic], axis=-1),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    ).astype(np.float32)


def _state_features(candidate_set, categories):
    context = np.asarray(candidate_set.shape_context, dtype=np.float32).reshape(
        len(candidate_set), 23, 3
    )
    core = context[:, list(CORE20)].reshape(len(candidate_set), -1)
    # Signed squares add low-capacity nonlinear morphology while ridge
    # regularization keeps the 192-sample fit stable.
    morphology = np.concatenate([core, np.sign(core) * np.square(core)], axis=1)
    one_hot = np.zeros((len(candidate_set), len(categories)), dtype=np.float32)
    lookup = {name: index for index, name in enumerate(categories)}
    for row, stratum in enumerate(candidate_set.strata):
        if stratum in lookup:
            one_hot[row, lookup[stratum]] = 1.0
    return np.concatenate([morphology, one_hot], axis=1).astype(np.float32)


def _fit_candidate_family(features, distance, mask, sample_indices, l2_values):
    selected_features = features[np.asarray(sample_indices)].reshape(
        -1, features.shape[-1]
    )
    selected_distance = np.asarray(distance[np.asarray(sample_indices)]).reshape(-1)
    selected_mask = np.asarray(mask[np.asarray(sample_indices)]).reshape(-1)
    selected_features = selected_features[selected_mask]
    selected_distance = selected_distance[selected_mask]
    clipped = np.minimum(selected_distance, 15.0)
    target = -clipped / 15.0
    # Give the clinically useful neighborhood enough influence without letting
    # one dense mesh region dominate the fit.
    row_weight = 0.25 + 4.0 * np.exp(-(clipped**2) / (2.0 * 4.0**2))
    return _ridge_family(selected_features, target, l2_values, row_weight)


def _candidate_scores(model, features, mask):
    scores = model.predict(features.reshape(-1, features.shape[-1])).reshape(
        features.shape[:-1]
    )
    return np.where(mask, scores, -np.inf).astype(np.float32)


def _state_scores(candidate_set, predicted_state):
    global_coordinate = np.asarray(
        candidate_set.canonical[:, 1:3, :, 3:6], dtype=np.float32
    )
    delta = (
        global_coordinate - np.asarray(predicted_state, dtype=np.float32)[:, :, None]
    )
    return -np.sum(np.square(delta), axis=-1)


def _standardize_per_sample(values, mask):
    values = np.asarray(values, dtype=np.float32)
    output = np.full_like(values, -np.inf)
    for sample in range(values.shape[0]):
        for side in range(values.shape[1]):
            valid = np.asarray(mask[sample, side], dtype=np.bool_)
            row = values[sample, side, valid]
            if not len(row):
                continue
            median = float(np.median(row))
            scale = max(float(np.percentile(row, 75) - np.percentile(row, 25)), 1e-4)
            output[sample, side, valid] = np.clip((row - median) / scale, -12.0, 12.0)
    return output


def _shortlist_mask(proposal, mask, count):
    proposal = np.where(mask, proposal, -np.inf)
    minimum_valid = max(1, int(np.asarray(mask).sum(axis=-1).min()))
    count = min(max(1, int(count)), proposal.shape[-1], minimum_valid)
    indices = np.argpartition(proposal, -count, axis=-1)[..., -count:]
    output = np.zeros_like(mask, dtype=np.bool_)
    np.put_along_axis(output, indices, True, axis=-1)
    return output & mask


def _combine_scores(proposal, contour, state, contour_weight, state_weight, mask):
    """Combine finite candidate terms without evaluating ``0 * -inf``."""
    combined = np.asarray(proposal, dtype=np.float32).copy()
    if float(contour_weight) != 0.0:
        combined += float(contour_weight) * np.where(mask, contour, 0.0)
    if float(state_weight) != 0.0:
        combined += float(state_weight) * np.where(mask, state, 0.0)
    return np.where(mask, combined, -np.inf).astype(np.float32)


def _decode(scores, points, mask, topk, temperature):
    safe = np.where(mask, scores, -np.inf)
    minimum_valid = max(1, int(np.asarray(mask).sum(axis=-1).min()))
    count = min(max(1, int(topk)), safe.shape[-1], minimum_valid)
    indices = np.argpartition(safe, -count, axis=-1)[..., -count:]
    selected_scores = np.take_along_axis(safe, indices, axis=-1)
    selected_points = np.take_along_axis(
        points,
        indices[..., None],
        axis=2,
    )
    if count == 1:
        return selected_points[..., 0, :].astype(np.float32)
    maximum = np.max(selected_scores, axis=-1, keepdims=True)
    weights = np.exp((selected_scores - maximum) / max(float(temperature), 1e-4))
    weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)
    return np.sum(weights[..., None] * selected_points, axis=2).astype(np.float32)


def _metrics(coordinate, expert):
    error = np.linalg.norm(np.asarray(coordinate) - np.asarray(expert), axis=-1)
    return {
        "ale": float(error.mean()),
        "lm21_ale": float(error[:, 0].mean()),
        "lm22_ale": float(error[:, 1].mean()),
        "median": float(np.median(error)),
        "p95": float(np.percentile(error, 95)),
        "sdr_at_2mm": float((error <= 2.0).mean()),
    }


def _policy_sweep(candidate_set, proposal, contour, state, config):
    rows = []
    mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
    points = np.asarray(candidate_set.points[:, 1:3], dtype=np.float32)
    expert = np.asarray(candidate_set.expert[:, 1:3], dtype=np.float32)
    for shortlist in config.statistical_shortlist_grid:
        selected_mask = _shortlist_mask(proposal, mask, shortlist)
        for contour_weight in config.statistical_contour_weight_grid:
            for state_weight in config.statistical_state_weight_grid:
                combined = _combine_scores(
                    proposal,
                    contour,
                    state,
                    contour_weight,
                    state_weight,
                    mask,
                )
                for topk, temperature in ((1, 1.0), (3, 0.25), (3, 0.5), (5, 0.5)):
                    coordinate = _decode(
                        combined, points, selected_mask, topk, temperature
                    )
                    rows.append(
                        {
                            "shortlist": int(shortlist),
                            "contour_weight": float(contour_weight),
                            "state_weight": float(state_weight),
                            "coordinate_topk": int(topk),
                            "temperature": float(temperature),
                            **_metrics(coordinate, expert),
                        }
                    )
    return rows


def _select_row(rows):
    return min(rows, key=lambda row: (row["ale"], row["p95"], -row["sdr_at_2mm"]))


class CrossFittedContourSelector:
    """Regularized OOF stacker for the high-recall Gonion surface proposal."""

    version = "H3-CFCS-v8"

    def __init__(self, candidate_model, state_model, policy, categories, report):
        self.candidate_model = candidate_model
        self.state_model = state_model
        self.policy = dict(policy)
        self.categories = tuple(categories)
        self.report = dict(report)

    @classmethod
    def fit(cls, candidate_set, proposal_sources, splits, config):
        l2_values = tuple(float(value) for value in config.statistical_l2_grid)
        features = _candidate_features(candidate_set, proposal_sources)
        mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
        distance = np.asarray(candidate_set.target_distance[:, 1:3], dtype=np.float32)
        categories = tuple(sorted(set(candidate_set.strata)))
        state_features = _state_features(candidate_set, categories)
        state_target = np.asarray(
            candidate_set.expert_gonion_context, dtype=np.float32
        ).reshape(len(candidate_set), -1)

        contour_oof = {
            l2: np.full(mask.shape, -np.inf, dtype=np.float32) for l2 in l2_values
        }
        state_oof = {
            l2: np.zeros((len(candidate_set), 2, 3), dtype=np.float32)
            for l2 in l2_values
        }
        for train_indices, validation_indices in splits:
            train_indices = np.asarray(train_indices, dtype=np.int64)
            validation_indices = np.asarray(validation_indices, dtype=np.int64)
            candidate_family = _fit_candidate_family(
                features, distance, mask, train_indices, l2_values
            )
            state_family = _ridge_family(
                state_features[train_indices], state_target[train_indices], l2_values
            )
            for l2 in l2_values:
                contour_oof[l2][validation_indices] = _candidate_scores(
                    candidate_family[l2],
                    features[validation_indices],
                    mask[validation_indices],
                )
                state_oof[l2][validation_indices] = (
                    state_family[l2]
                    .predict(state_features[validation_indices])
                    .reshape(-1, 2, 3)
                )

        proposal = _standardize_per_sample(
            np.asarray(proposal_sources[:, 1:3, 0], dtype=np.float32), mask
        )
        candidate_l2_rows = []
        for l2, scores in contour_oof.items():
            standardized = _standardize_per_sample(scores, mask)
            coordinate = _decode(
                standardized,
                candidate_set.points[:, 1:3],
                _shortlist_mask(proposal, mask, max(config.statistical_shortlist_grid)),
                1,
                1.0,
            )
            candidate_l2_rows.append(
                {"l2": l2, **_metrics(coordinate, candidate_set.expert[:, 1:3])}
            )
        selected_candidate_l2 = _select_row(candidate_l2_rows)["l2"]

        state_l2_rows = []
        for l2, predicted_state in state_oof.items():
            scores = _standardize_per_sample(
                _state_scores(candidate_set, predicted_state), mask
            )
            coordinate = _decode(
                scores,
                candidate_set.points[:, 1:3],
                _shortlist_mask(proposal, mask, max(config.statistical_shortlist_grid)),
                1,
                1.0,
            )
            state_l2_rows.append(
                {"l2": l2, **_metrics(coordinate, candidate_set.expert[:, 1:3])}
            )
        selected_state_l2 = _select_row(state_l2_rows)["l2"]

        contour = _standardize_per_sample(contour_oof[selected_candidate_l2], mask)
        state = _standardize_per_sample(
            _state_scores(candidate_set, state_oof[selected_state_l2]), mask
        )
        sweep = _policy_sweep(candidate_set, proposal, contour, state, config)
        selected_policy = _select_row(sweep)
        selected_policy.update(
            {
                "candidate_l2": float(selected_candidate_l2),
                "state_l2": float(selected_state_l2),
            }
        )
        selected_mask = _shortlist_mask(proposal, mask, selected_policy["shortlist"])
        selected_scores = _combine_scores(
            proposal,
            contour,
            state,
            selected_policy["contour_weight"],
            selected_policy["state_weight"],
            mask,
        )
        oof_prediction = _decode(
            selected_scores,
            candidate_set.points[:, 1:3],
            selected_mask,
            selected_policy["coordinate_topk"],
            selected_policy["temperature"],
        )

        all_indices = np.arange(len(candidate_set), dtype=np.int64)
        candidate_model = _fit_candidate_family(
            features,
            distance,
            mask,
            all_indices,
            (selected_candidate_l2,),
        )[selected_candidate_l2]
        state_model = _ridge_family(
            state_features,
            state_target,
            (selected_state_l2,),
        )[selected_state_l2]
        report = {
            "version": cls.version,
            "method": (
                "nested-OOF ridge contour ranking plus Core20-conditioned bilateral "
                "state regression"
            ),
            "uses_outer_validation_labels": False,
            "uses_test_labels": False,
            "candidate_feature_dim": int(features.shape[-1]),
            "state_feature_dim": int(state_features.shape[-1]),
            "selected": selected_policy,
            "candidate_l2_sweep": candidate_l2_rows,
            "state_l2_sweep": state_l2_rows,
            "policy_sweep": sweep,
        }
        selector = cls(
            candidate_model, state_model, selected_policy, categories, report
        )
        selector.oof_prediction = oof_prediction
        return selector

    def _components(self, candidate_set, proposal_sources):
        mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
        features = _candidate_features(candidate_set, proposal_sources)
        proposal = _standardize_per_sample(
            np.asarray(proposal_sources[:, 1:3, 0], dtype=np.float32), mask
        )
        contour = _standardize_per_sample(
            _candidate_scores(self.candidate_model, features, mask), mask
        )
        state_prediction = self.state_model.predict(
            _state_features(candidate_set, self.categories)
        ).reshape(-1, 2, 3)
        state = _standardize_per_sample(
            _state_scores(candidate_set, state_prediction), mask
        )
        return proposal, contour, state, state_prediction

    @staticmethod
    def _coordinate(candidate_set, scores, proposal, policy):
        mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
        selected_mask = _shortlist_mask(proposal, mask, policy["shortlist"])
        return _decode(
            scores,
            np.asarray(candidate_set.points[:, 1:3], dtype=np.float32),
            selected_mask,
            policy["coordinate_topk"],
            policy["temperature"],
        )

    def predict(self, candidate_set, proposal_sources):
        proposal, contour, state, state_prediction = self._components(
            candidate_set, proposal_sources
        )
        policy = self.policy
        mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
        combined = _combine_scores(
            proposal,
            contour,
            state,
            policy["contour_weight"],
            policy["state_weight"],
            mask,
        )
        contour_policy = {
            **policy,
            "contour_weight": 1.0,
            "state_weight": 0.0,
        }
        state_policy = {
            **policy,
            "contour_weight": 0.0,
            "state_weight": 1.0,
        }
        return {
            "crossfit_calibrated": self._coordinate(
                candidate_set, combined, proposal, policy
            ),
            "crossfit_contour_only": self._coordinate(
                candidate_set,
                _combine_scores(proposal, contour, state, 1.0, 0.0, mask),
                proposal,
                contour_policy,
            ),
            "crossfit_state_only": self._coordinate(
                candidate_set,
                _combine_scores(proposal, contour, state, 0.0, 1.0, mask),
                proposal,
                state_policy,
            ),
            "predicted_state": state_prediction.astype(np.float32),
        }

    def state_dict(self):
        return {
            "version": self.version,
            "candidate_model": self.candidate_model.state_dict(),
            "state_model": self.state_model.state_dict(),
            "policy": self.policy,
            "categories": list(self.categories),
            "report": self.report,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != cls.version:
            raise ValueError(
                f"Unsupported statistical selector: {state.get('version')}"
            )
        return cls(
            _RidgeModel.from_state_dict(state["candidate_model"]),
            _RidgeModel.from_state_dict(state["state_model"]),
            state["policy"],
            state["categories"],
            state["report"],
        )


def selector_validation_diagnostics(candidate_set, prediction, proposal_sources, topk):
    mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
    proposal = _standardize_per_sample(
        np.asarray(proposal_sources[:, 1:3, 0], dtype=np.float32), mask
    )
    shortlist = _shortlist_mask(proposal, mask, topk)
    oracle = np.where(
        shortlist,
        np.asarray(candidate_set.target_distance[:, 1:3], dtype=np.float32),
        np.inf,
    ).min(axis=-1)
    return {
        "selected": _metrics(prediction, candidate_set.expert[:, 1:3]),
        "shortlist_oracle": {
            "ale": float(oracle.mean()),
            "p95": float(np.percentile(oracle, 95)),
            "sdr_at_2mm": float((oracle <= 2.0).mean()),
            "both_within_2mm": float(np.all(oracle <= 2.0, axis=1).mean()),
        },
    }
