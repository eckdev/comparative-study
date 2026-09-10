"""Cross-fitted multi-scale surface selector for bilateral Gonion.

The broad H3 proposal already contains clinically close candidates.  This
selector therefore keeps proposal generation frozen and describes the local
surface around every candidate at three graph-diffusion scales.  Separate,
strongly regularized rankers are fitted for LM21 and LM22 so that left/right
acquisition asymmetry is not forced through a shared high-capacity model.
"""

from __future__ import annotations

import numpy as np

from .interaction_selector import (
    _gather_candidates,
    _query_standardize,
    _shortlist_indices,
)
from .statistical_selector import (
    _RidgeModel,
    _decode,
    _metrics,
    _ridge_family,
    _standardize_per_sample,
    selector_validation_diagnostics,
)


def _neighbor_mean(values, neighbor_index, neighbor_mask, valid_mask):
    """Diffuse point moments by one graph hop while retaining the center."""
    values = np.asarray(values, dtype=np.float32)
    neighbor_index = np.asarray(neighbor_index, dtype=np.int64)
    neighbor_index = np.clip(neighbor_index, 0, max(len(values) - 1, 0))
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    usable = np.asarray(neighbor_mask, dtype=np.bool_) & valid_mask[neighbor_index]
    gathered = values[neighbor_index]
    weight = usable[..., None].astype(np.float32)
    total = values + np.sum(gathered * weight, axis=1)
    count = 1.0 + np.sum(weight, axis=1)
    output = total / np.maximum(count, 1.0)
    output[~valid_mask] = 0.0
    return output.astype(np.float32)


def _moment_rows(canonical):
    xyz = canonical[:, 3:6]
    normal = canonical[:, 18:21] if canonical.shape[-1] >= 21 else np.zeros_like(xyz)
    scalar_indices = [
        index
        for index in (24, 25, 34, 35, 36, 37, 38, 39)
        if index < canonical.shape[-1]
    ]
    scalars = (
        canonical[:, scalar_indices]
        if scalar_indices
        else np.zeros((len(canonical), 0), dtype=np.float32)
    )
    second = np.stack(
        [
            xyz[:, 0] * xyz[:, 0],
            xyz[:, 1] * xyz[:, 1],
            xyz[:, 2] * xyz[:, 2],
            xyz[:, 0] * xyz[:, 1],
            xyz[:, 0] * xyz[:, 2],
            xyz[:, 1] * xyz[:, 2],
        ],
        axis=1,
    )
    return (
        np.concatenate(
            [xyz, second, normal, np.square(normal), scalars, np.square(scalars)],
            axis=1,
        ).astype(np.float32),
        xyz,
        normal,
        len(scalar_indices),
    )


def _surface_descriptor(moment, xyz, normal, scalar_count):
    mean_xyz = moment[:, :3]
    second = moment[:, 3:9]
    mean_normal = moment[:, 9:12]
    second_normal = moment[:, 12:15]
    scalar_mean = moment[:, 15 : 15 + scalar_count]
    scalar_second = moment[:, 15 + scalar_count : 15 + 2 * scalar_count]

    covariance = np.zeros((len(moment), 3, 3), dtype=np.float32)
    covariance[:, 0, 0] = second[:, 0] - np.square(mean_xyz[:, 0])
    covariance[:, 1, 1] = second[:, 1] - np.square(mean_xyz[:, 1])
    covariance[:, 2, 2] = second[:, 2] - np.square(mean_xyz[:, 2])
    covariance[:, 0, 1] = covariance[:, 1, 0] = (
        second[:, 3] - mean_xyz[:, 0] * mean_xyz[:, 1]
    )
    covariance[:, 0, 2] = covariance[:, 2, 0] = (
        second[:, 4] - mean_xyz[:, 0] * mean_xyz[:, 2]
    )
    covariance[:, 1, 2] = covariance[:, 2, 1] = (
        second[:, 5] - mean_xyz[:, 1] * mean_xyz[:, 2]
    )
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = np.maximum(eigenvalues.sum(axis=1, keepdims=True), 1e-8)
    normalized_eigenvalues = eigenvalues / trace
    small, middle, large = [eigenvalues[:, index] for index in range(3)]
    denominator = np.maximum(large, 1e-8)
    shape = np.stack(
        [
            (large - middle) / denominator,
            (middle - small) / denominator,
            small / denominator,
            small / np.maximum(trace[:, 0], 1e-8),
        ],
        axis=1,
    )
    principal_direction = np.abs(eigenvectors[:, :, 2])
    normal_std = np.sqrt(np.maximum(second_normal - np.square(mean_normal), 0.0))
    normal_dispersion = 1.0 - np.clip(
        np.linalg.norm(mean_normal, axis=1, keepdims=True), 0.0, 1.0
    )
    normal_alignment = np.sum(normal * mean_normal, axis=1, keepdims=True)
    scalar_std = np.sqrt(np.maximum(scalar_second - np.square(scalar_mean), 0.0))
    descriptor = np.concatenate(
        [
            mean_xyz - xyz,
            normalized_eigenvalues,
            shape,
            principal_direction,
            mean_normal,
            normal_dispersion,
            normal_std,
            normal_alignment,
            scalar_mean,
            scalar_std,
        ],
        axis=1,
    )
    return np.nan_to_num(descriptor, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


def multiscale_surface_descriptors(candidate_set, hops=(1, 2, 3)):
    """Return graph-moment descriptors for both Gonion candidate surfaces."""
    hops = tuple(sorted(set(max(1, int(value)) for value in hops)))
    if not hops:
        raise ValueError("At least one multi-scale graph hop is required")
    canonical = np.asarray(candidate_set.canonical[:, 1:3], dtype=np.float32)
    neighbor_index = np.asarray(candidate_set.neighbor_index[:, 1:3], dtype=np.int64)
    neighbor_mask = np.asarray(candidate_set.neighbor_mask[:, 1:3], dtype=np.bool_)
    valid_mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
    rows = None
    for sample in range(canonical.shape[0]):
        side_rows = []
        for side in range(2):
            moment, xyz, normal, scalar_count = _moment_rows(canonical[sample, side])
            scales = []
            current = moment
            for hop in range(1, max(hops) + 1):
                current = _neighbor_mean(
                    current,
                    neighbor_index[sample, side],
                    neighbor_mask[sample, side],
                    valid_mask[sample, side],
                )
                if hop in hops:
                    scales.append(
                        _surface_descriptor(current, xyz, normal, scalar_count)
                    )
            side_rows.append(np.concatenate(scales, axis=1))
        sample_row = np.stack(side_rows)
        if rows is None:
            rows = np.zeros((canonical.shape[0],) + sample_row.shape, dtype=np.float32)
        rows[sample] = sample_row
    rows[~valid_mask] = 0.0
    return rows


def _feature_families(candidate_set, proposal_sources, hops):
    canonical = np.nan_to_num(
        np.asarray(candidate_set.canonical[:, 1:3], dtype=np.float32),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    )
    mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
    sources = np.asarray(proposal_sources[:, 1:3], dtype=np.float32).transpose(
        0, 1, 3, 2
    )
    sources = _query_standardize(
        np.nan_to_num(sources, nan=0.0, posinf=8.0, neginf=-8.0), mask
    )
    descriptors = _query_standardize(
        multiscale_surface_descriptors(candidate_set, hops), mask
    )

    global_xyz = canonical[..., 3:6]
    anchor_distances = canonical[..., [9, 13, 17]]
    compact_surface = canonical[..., 18 : min(26, canonical.shape[-1])]
    invariant_geometry = canonical[..., 3 : min(26, canonical.shape[-1])]
    contour = (
        canonical[..., 34:]
        if canonical.shape[-1] > 34
        else np.zeros(canonical.shape[:-1] + (0,), dtype=np.float32)
    )
    appearance = (
        canonical[..., 26:34]
        if canonical.shape[-1] >= 34
        else np.zeros(canonical.shape[:-1] + (0,), dtype=np.float32)
    )
    local = _query_standardize(canonical[..., :3], mask)

    families = {
        "compact": np.concatenate(
            [sources, global_xyz, anchor_distances, compact_surface, descriptors],
            axis=-1,
        ),
        "geometry": np.concatenate([sources, invariant_geometry, descriptors], axis=-1),
        "contour": np.concatenate(
            [sources, invariant_geometry, contour, descriptors], axis=-1
        ),
        "full": np.concatenate(
            [
                sources,
                invariant_geometry,
                appearance,
                contour,
                local,
                descriptors,
            ],
            axis=-1,
        ),
    }
    output = {}
    for name, values in families.items():
        values = np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(
            np.float32
        )
        values[~mask] = 0.0
        output[name] = values
    return output


def _fit_side_family(features, distance, mask, sample_indices, l2_values):
    selected = np.asarray(sample_indices, dtype=np.int64)
    values = features[selected]
    distances = np.asarray(distance[selected], dtype=np.float32)
    valid = np.asarray(mask[selected], dtype=np.bool_)
    clipped = np.minimum(distances, 15.0)
    target = -clipped / 15.0
    weight = 0.25 + 4.0 * np.exp(-np.square(clipped) / (2.0 * 4.0**2))
    weight *= valid
    # Each subject contributes equal total mass regardless of local mesh density.
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-8)
    weight *= max(values.shape[1], 1)
    flat_valid = valid.reshape(-1)
    return _ridge_family(
        values.reshape(-1, values.shape[-1])[flat_valid],
        target.reshape(-1)[flat_valid],
        l2_values,
        weight.reshape(-1)[flat_valid],
    )


def _side_scores(model, features, mask):
    values = model.predict(features.reshape(-1, features.shape[-1])).reshape(
        features.shape[:-1]
    )
    values = _standardize_per_sample(values[:, None], mask[:, None])[:, 0]
    return np.where(mask, values, -np.inf).astype(np.float32)


def _side_metrics(coordinate, expert):
    error = np.linalg.norm(np.asarray(coordinate) - np.asarray(expert), axis=-1)
    return {
        "ale": float(error.mean()),
        "median": float(np.median(error)),
        "p95": float(np.percentile(error, 95)),
        "sdr_at_2mm": float((error <= 2.0).mean()),
    }


def _decode_side(scores, points, mask, topk, temperature):
    return _decode(
        scores[:, None],
        points[:, None],
        mask[:, None],
        topk,
        temperature,
    )[:, 0]


def _combined_score(descriptor, proposal, descriptor_weight, mask):
    values = np.asarray(proposal, dtype=np.float32).copy()
    values += float(descriptor_weight) * np.where(mask, descriptor, 0.0)
    return np.where(mask, values, -np.inf).astype(np.float32)


class CrossFittedMultiscaleContourSelector:
    """Side-specific low-capacity selector with multi-scale surface moments."""

    version = "H3-MSCSR-v11"
    primary_key = "crossfit_multiscale_contour"
    coordinate_keys = (
        "crossfit_multiscale_contour",
        "multiscale_contour_refit",
        "multiscale_contour_argmax",
        "multiscale_contour_descriptor_only",
    )

    def __init__(self, final_models, member_models, policy, report):
        self.final_models = tuple(final_models)
        self.member_models = tuple(tuple(row) for row in member_models)
        self.policy = dict(policy)
        self.report = dict(report)

    @staticmethod
    def _prepare(candidate_set, proposal_sources, config, shortlist=None):
        full_mask = np.asarray(candidate_set.mask[:, 1:3], dtype=np.bool_)
        proposal = _standardize_per_sample(
            np.asarray(proposal_sources[:, 1:3, 0], dtype=np.float32), full_mask
        )
        count = int(config.multiscale_shortlist if shortlist is None else shortlist)
        indices = _shortlist_indices(proposal, full_mask, count)
        families = _feature_families(
            candidate_set, proposal_sources, config.multiscale_hops
        )
        return {
            "indices": indices,
            "families": {
                name: _gather_candidates(values, indices)
                for name, values in families.items()
            },
            "proposal": _gather_candidates(proposal, indices).astype(np.float32),
            "mask": _gather_candidates(full_mask, indices).astype(np.bool_),
            "points": _gather_candidates(candidate_set.points[:, 1:3], indices).astype(
                np.float32
            ),
            "distance": _gather_candidates(
                candidate_set.target_distance[:, 1:3], indices
            ).astype(np.float32),
        }

    @classmethod
    def fit(cls, candidate_set, proposal_sources, splits, config):
        data = cls._prepare(candidate_set, proposal_sources, config)
        requested_modes = tuple(config.multiscale_feature_modes)
        unknown = sorted(set(requested_modes) - set(data["families"]))
        if unknown:
            raise ValueError(f"Unknown multi-scale feature modes: {unknown}")
        l2_values = tuple(float(value) for value in config.multiscale_l2_grid)
        if not l2_values:
            raise ValueError("multiscale_l2_grid cannot be empty")

        oof_scores = {
            (mode, l2): np.full(data["mask"].shape, -np.inf, dtype=np.float32)
            for mode in requested_modes
            for l2 in l2_values
        }
        for train_indices, validation_indices in splits:
            train_indices = np.asarray(train_indices, dtype=np.int64)
            validation_indices = np.asarray(validation_indices, dtype=np.int64)
            for mode in requested_modes:
                features = data["families"][mode]
                for side in range(2):
                    family = _fit_side_family(
                        features[:, side],
                        data["distance"][:, side],
                        data["mask"][:, side],
                        train_indices,
                        l2_values,
                    )
                    for l2, model in family.items():
                        oof_scores[(mode, l2)][validation_indices, side] = _side_scores(
                            model,
                            features[validation_indices, side],
                            data["mask"][validation_indices, side],
                        )
        for values in oof_scores.values():
            if not np.isfinite(values[data["mask"]]).all():
                raise RuntimeError("H3 multi-scale OOF scores are incomplete")

        side_policies = []
        feature_sweeps = []
        decoder_sweeps = []
        side_names = ("lm21", "lm22")
        descriptor_weights = tuple(
            float(value) for value in config.multiscale_descriptor_weight_grid
        )
        for side, side_name in enumerate(side_names):
            feature_rows = []
            for mode in requested_modes:
                for l2 in l2_values:
                    coordinate = _decode_side(
                        oof_scores[(mode, l2)][:, side],
                        data["points"][:, side],
                        data["mask"][:, side],
                        1,
                        1.0,
                    )
                    feature_rows.append(
                        {
                            "side": side_name,
                            "feature_mode": mode,
                            "l2": float(l2),
                            **_side_metrics(
                                coordinate, candidate_set.expert[:, side + 1]
                            ),
                        }
                    )
            selected_feature = min(
                feature_rows,
                key=lambda row: (row["ale"], row["p95"], -row["sdr_at_2mm"]),
            )
            descriptor = oof_scores[
                (selected_feature["feature_mode"], selected_feature["l2"])
            ][:, side]
            decoder_rows = []
            for descriptor_weight in descriptor_weights:
                score = _combined_score(
                    descriptor,
                    data["proposal"][:, side],
                    descriptor_weight,
                    data["mask"][:, side],
                )
                for topk, temperature in ((1, 1.0), (3, 0.25), (3, 0.5), (5, 0.5)):
                    coordinate = _decode_side(
                        score,
                        data["points"][:, side],
                        data["mask"][:, side],
                        topk,
                        temperature,
                    )
                    decoder_rows.append(
                        {
                            "side": side_name,
                            "feature_mode": selected_feature["feature_mode"],
                            "l2": float(selected_feature["l2"]),
                            "descriptor_weight": descriptor_weight,
                            "coordinate_topk": topk,
                            "temperature": temperature,
                            **_side_metrics(
                                coordinate, candidate_set.expert[:, side + 1]
                            ),
                        }
                    )
            selected_policy = min(
                decoder_rows,
                key=lambda row: (row["ale"], row["p95"], -row["sdr_at_2mm"]),
            )
            side_policies.append(selected_policy)
            feature_sweeps.extend(feature_rows)
            decoder_sweeps.extend(decoder_rows)

        def fit_models(sample_indices):
            models = []
            for side, policy in enumerate(side_policies):
                models.append(
                    _fit_side_family(
                        data["families"][policy["feature_mode"]][:, side],
                        data["distance"][:, side],
                        data["mask"][:, side],
                        sample_indices,
                        (policy["l2"],),
                    )[float(policy["l2"])]
                )
            return tuple(models)

        member_models = [fit_models(train_indices) for train_indices, _ in splits]
        final_models = fit_models(np.arange(len(candidate_set), dtype=np.int64))

        oof_prediction = np.zeros((len(candidate_set), 2, 3), dtype=np.float32)
        for side, policy in enumerate(side_policies):
            score = _combined_score(
                oof_scores[(policy["feature_mode"], policy["l2"])][:, side],
                data["proposal"][:, side],
                policy["descriptor_weight"],
                data["mask"][:, side],
            )
            oof_prediction[:, side] = _decode_side(
                score,
                data["points"][:, side],
                data["mask"][:, side],
                policy["coordinate_topk"],
                policy["temperature"],
            )
        selected_metrics = _metrics(oof_prediction, candidate_set.expert[:, 1:3])
        selected_metrics.update({"shortlist": int(data["points"].shape[2])})
        report = {
            "version": cls.version,
            "method": (
                "side-specific cross-fitted ridge ranking over multi-scale "
                "surface moments and explicit contour evidence"
            ),
            "uses_outer_validation_labels": False,
            "uses_test_labels": False,
            "selected": selected_metrics,
            "side_policies": side_policies,
            "feature_sweep": feature_sweeps,
            "decoder_sweep": decoder_sweeps,
            "feature_dims": {
                name: int(values.shape[-1]) for name, values in data["families"].items()
            },
            "parameter_count": int(sum(model.weights.size for model in final_models)),
            "graph_hops": list(config.multiscale_hops),
        }
        selector = cls(
            final_models,
            member_models,
            {"shortlist": int(data["points"].shape[2]), "sides": side_policies},
            report,
        )
        selector.oof_prediction = oof_prediction
        return selector

    def _scores(self, data, models):
        scores = np.full(data["mask"].shape, -np.inf, dtype=np.float32)
        descriptors = np.full_like(scores, -np.inf)
        for side, policy in enumerate(self.policy["sides"]):
            descriptor = _side_scores(
                models[side],
                data["families"][policy["feature_mode"]][:, side],
                data["mask"][:, side],
            )
            descriptors[:, side] = descriptor
            scores[:, side] = _combined_score(
                descriptor,
                data["proposal"][:, side],
                policy["descriptor_weight"],
                data["mask"][:, side],
            )
        return scores, descriptors

    def _decode_policy(self, data, scores, force_argmax=False):
        coordinate = np.zeros((len(scores), 2, 3), dtype=np.float32)
        for side, policy in enumerate(self.policy["sides"]):
            coordinate[:, side] = _decode_side(
                scores[:, side],
                data["points"][:, side],
                data["mask"][:, side],
                1 if force_argmax else policy["coordinate_topk"],
                1.0 if force_argmax else policy["temperature"],
            )
        return coordinate

    def predict(self, candidate_set, proposal_sources):
        class InferenceConfig:
            multiscale_shortlist = self.policy["shortlist"]
            multiscale_hops = tuple(self.report["graph_hops"])

        data = self._prepare(
            candidate_set,
            proposal_sources,
            InferenceConfig,
            shortlist=self.policy["shortlist"],
        )
        member_scores = []
        for models in self.member_models:
            scores, _ = self._scores(data, models)
            member_scores.append(scores)
        ensemble_scores = np.mean(np.stack(member_scores), axis=0)
        refit_scores, descriptor_scores = self._scores(data, self.final_models)
        member_coordinate = np.stack(
            [self._decode_policy(data, scores) for scores in member_scores]
        ).astype(np.float32)
        descriptor_only = np.where(data["mask"], descriptor_scores, -np.inf)
        return {
            self.primary_key: self._decode_policy(data, ensemble_scores),
            "multiscale_contour_refit": self._decode_policy(data, refit_scores),
            "multiscale_contour_argmax": self._decode_policy(
                data, ensemble_scores, force_argmax=True
            ),
            "multiscale_contour_descriptor_only": self._decode_policy(
                data, descriptor_only
            ),
            "member_coordinate": member_coordinate,
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
            "final_models": [model.state_dict() for model in self.final_models],
            "member_models": [
                [model.state_dict() for model in row] for row in self.member_models
            ],
            "policy": self.policy,
            "report": self.report,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != cls.version:
            raise ValueError(
                f"Unsupported multi-scale selector: {state.get('version')}"
            )
        return cls(
            [_RidgeModel.from_state_dict(values) for values in state["final_models"]],
            [
                [_RidgeModel.from_state_dict(values) for values in row]
                for row in state["member_models"]
            ],
            state["policy"],
            state["report"],
        )
