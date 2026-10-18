from __future__ import annotations

import numpy as np
import pytest
import torch

from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, NUM_LANDMARKS
from all23_mvsc_refinement.anatomy import (
    ALL23_INDICES,
    ANCHORS,
    LANDMARK_TYPES,
    mirror_landmark_index,
    prior_context_indices,
)
from all23_mvsc_refinement.gate import (
    DIAGNOSTIC_NAMES,
    GateEvidence,
    fit_oof_gate,
    predict_gate_alpha,
)
from all23_mvsc_refinement.model import All23MVSCNet
from all23_mvsc_refinement.patches import (
    All23CandidateSet,
    IMAGE_CHANNELS,
    VIEW_NAMES,
    context_radius,
    rasterize,
)
from all23_mvsc_refinement.refiner import (
    All23MVSCConfig,
    apply_policy,
    calibrate_policy,
    compute_loss,
    fit_network,
    predict_candidates,
)
from all23_mvsc_refinement.shape_prior import ProcrustesShapePrior, kabsch
from all23_mvsc_refinement.spatial_prior import (
    ConditionalSpatialPrior,
    SpatialPriorConfig,
)


def synthetic_batch(batch_size=6, candidates=32, geometry_dim=78, views=4):
    generator = torch.Generator().manual_seed(7)
    points = torch.randn(batch_size, candidates, 3, generator=generator)
    expert = points[:, 3].clone()
    distance = torch.linalg.norm(points - expert[:, None], dim=-1)
    anchors = torch.randn(batch_size, 4, 3, generator=generator)
    return {
        "images": torch.randn(batch_size, views, len(IMAGE_CHANNELS), 24, 24, generator=generator),
        "grids": torch.rand(batch_size, views, candidates, 2, generator=generator) * 2 - 1,
        "points": points,
        "features": torch.randn(batch_size, candidates, geometry_dim, generator=generator),
        "mask": torch.ones(batch_size, candidates, dtype=torch.bool),
        "heatmap_target": torch.exp(-distance.square() / 8.0),
        "distance": distance,
        "expert": expert,
        "base": expert + 1.0,
        "prior_mean": expert + 0.25,
        "prior_covariance": torch.eye(3).repeat(batch_size, 1, 1),
        "prior_score": -distance.square(),
        "anatomy_anchors": anchors,
        "anatomy_distances": torch.linalg.norm(expert[:, None] - anchors, dim=-1),
        "anatomy_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "landmark": torch.tensor([0, 2, 10, 13, 21, 22]),
        "group": torch.tensor([0, 1, 3, 4, 7, 7]),
        "type": torch.tensor([2, 0, 1, 0, 1, 1]),
    }


def synthetic_shapes(count=40, seed=3):
    rng = np.random.default_rng(seed)
    template = rng.normal(0.0, 30.0, size=(NUM_LANDMARKS, 3))
    shapes = template[None] + rng.normal(0.0, 1.5, size=(count, NUM_LANDMARKS, 3))
    return shapes.astype(np.float32)


def test_schema_covers_all_landmarks_once():
    flattened = sorted(index for values in LANDMARK_TYPES.values() for index in values)
    assert flattened == list(ALL23_INDICES)
    assert set(HARD3) <= set(LANDMARK_TYPES["contour"]) | set(LANDMARK_TYPES["texture_boundary"])
    assert all(landmark not in anchors for landmark, anchors in ANCHORS.items())
    assert mirror_landmark_index(21) == 22 and mirror_landmark_index(0) == 0


def test_prior_context_never_contains_target_or_hard3():
    for target in ALL23_INDICES:
        context = prior_context_indices(target)
        assert target not in context
        assert set(context).isdisjoint(HARD3)
        assert set(context) <= set(CORE20)


def test_context_view_covers_every_valid_candidate():
    rng = np.random.default_rng(0)
    coordinates = rng.uniform(-40.0, 40.0, size=(128, 3)).astype(np.float32)
    mask = np.ones(len(coordinates), dtype=bool)
    radius = context_radius(coordinates, mask, minimum=11.0)
    assert np.all(np.abs(coordinates[:, :2]) <= radius)
    image = rasterize(np.ones((128, 12), dtype=np.float32), coordinates, radius, 32)
    assert image.shape == (len(IMAGE_CHANNELS), 32, 32)
    assert np.isfinite(image).all()
    occupancy = IMAGE_CHANNELS.index("occupancy")
    assert image[occupancy].max() == 1.0


def test_model_outputs_are_finite_and_geometry_fallback_works():
    batch = synthetic_batch()
    model = All23MVSCNet(len(IMAGE_CHANNELS), 78, width=16, coordinate_topk=4)
    outputs = model(batch)
    for name in ("proposal", "entropy", "margin", "log_variance", "confidence"):
        assert torch.isfinite(outputs[name]).all(), name
    assert outputs["view_weights"].shape == (6, len(VIEW_NAMES))
    loss, components, _ = compute_loss(outputs, batch, All23MVSCConfig())
    assert torch.isfinite(loss)
    assert set(components) == {"heatmap_kl", "coordinate", "hard_negative", "anatomy", "confidence_nll"}
    loss.backward()
    geometry_only = All23MVSCNet(len(IMAGE_CHANNELS), 78, width=16, use_images=False)
    assert torch.isfinite(geometry_only(batch)["proposal"]).all()


def test_model_is_bfloat16_autocast_safe():
    batch = synthetic_batch()
    model = All23MVSCNet(len(IMAGE_CHANNELS), 78, width=16, coordinate_topk=4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = model(batch)
        loss, _, _ = compute_loss(outputs, batch, All23MVSCConfig())
    assert torch.isfinite(loss)


def test_spatial_prior_crossfit_excludes_target_samples():
    shapes = synthetic_shapes()
    ids = [f"S{index}" for index in range(len(shapes))]
    classes = ["Class1", "Class2"] * (len(shapes) // 2)
    genders = ["men", "women", "women", "men"] * (len(shapes) // 4)
    prior = ConditionalSpatialPrior(SpatialPriorConfig(folds=4)).fit(shapes, ids, classes, genders)
    oof = prior.oof_predict_from_shapes(shapes, ids, classes, genders)
    assert oof.shape == (len(shapes), NUM_LANDMARKS, 3)
    for landmark, rows in prior.crossfit_parameters.items():
        covered = sorted(sample for row in rows for sample in row["validation_sample_ids"])
        assert covered == sorted(ids), landmark
    mean, covariance = prior.predict(shapes[:3], classes[:3], genders[:3])
    assert mean.shape == (3, NUM_LANDMARKS, 3) and covariance.shape == (3, NUM_LANDMARKS, 3, 3)


def test_procrustes_prior_is_pose_invariant():
    shapes = synthetic_shapes(60).astype(np.float64)
    prior = ProcrustesShapePrior().fit(shapes)
    prior.config = {"components": 6, "l2": 10.0, "alpha_core": 0.3, "alpha_hard": 0.5}
    angle = np.deg2rad(12.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    sample = shapes[:5]
    moved = sample @ rotation + np.asarray([5.0, -3.0, 2.0])
    direct = prior.transform(sample)
    transformed = prior.transform(moved)
    np.testing.assert_allclose(transformed, direct @ rotation + np.asarray([5.0, -3.0, 2.0]), atol=1e-3)
    identity_prior = ProcrustesShapePrior().fit(shapes)
    identity_prior.config = {"components": 6, "l2": 10.0, "alpha_core": 0.0, "alpha_hard": 0.0}
    np.testing.assert_allclose(identity_prior.transform(sample), sample, atol=1e-4)
    rotation_found, translation = kabsch(sample[0], moved[0])
    np.testing.assert_allclose(sample[0] @ rotation_found + translation, moved[0], atol=1e-6)


def _evidence(count=30, seed=5):
    rng = np.random.default_rng(seed)
    expert = rng.normal(size=(count, NUM_LANDMARKS, 3)).astype(np.float32)
    base = expert + rng.normal(0.0, 2.0, size=expert.shape).astype(np.float32)
    good = rng.random((count, NUM_LANDMARKS)) < 0.5
    proposal = np.where(
        good[..., None],
        expert + rng.normal(0.0, 0.3, size=expert.shape),
        base + rng.normal(0.0, 3.0, size=expert.shape),
    ).astype(np.float32)
    diagnostics = rng.normal(size=(count, NUM_LANDMARKS, len(DIAGNOSTIC_NAMES))).astype(np.float32)
    diagnostics[..., 0] = np.where(good, -2.0, 2.0)
    return GateEvidence([f"S{index}" for index in range(count)], base, proposal, diagnostics, expert)


def test_oof_gate_learns_informative_diagnostics(tmp_path):
    evidence = _evidence()
    model, mean, std, report = fit_oof_gate(evidence, tmp_path, torch.device("cpu"), epochs=200)
    alpha = predict_gate_alpha(model, mean, std, evidence.diagnostics, torch.device("cpu"))
    informative = evidence.diagnostics[..., 0] < 0
    assert alpha[informative].mean() > alpha[~informative].mean() + 0.2
    assert (tmp_path / "oof_gate.pth").exists()
    assert report["best_epoch"] >= 1
    saved = tmp_path / "evidence.npz"
    evidence.save(saved)
    assert GateEvidence.load(saved).sample_ids == evidence.sample_ids


def test_policy_rejects_harmful_proposals_and_accepts_helpful_ones():
    config = All23MVSCConfig(bootstrap_iters=200)
    rng = np.random.default_rng(1)
    expert = rng.normal(size=(40, NUM_LANDMARKS, 3)).astype(np.float32)
    base = expert + rng.normal(0.0, 1.0, size=expert.shape).astype(np.float32)
    harmful = base + rng.normal(0.0, 3.0, size=expert.shape).astype(np.float32)
    helpful = expert + rng.normal(0.0, 0.2, size=expert.shape).astype(np.float32)
    alpha = np.ones((40, NUM_LANDMARKS), dtype=np.float32)

    def result(iteration_one, iteration_two):
        return {
            "sample_ids": [f"S{index}" for index in range(40)],
            "base": base,
            "iterations": {
                1: {"proposal": iteration_one, "gate_alpha": alpha, "confidence": alpha, "entropy": alpha},
                2: {"proposal": iteration_two, "gate_alpha": alpha, "confidence": alpha, "entropy": alpha},
            },
        }

    rejected = calibrate_policy(result(harmful, harmful), expert, config)
    assert set(rejected["proposed_alpha_caps"].values()) == {0.0}
    accepted = calibrate_policy(result(harmful, helpful), expert, config)
    assert accepted["accepted"] and accepted["iterations"] == 2
    applied = apply_policy(result(harmful, helpful), accepted)
    final_error = np.linalg.norm(applied["prediction"] - expert, axis=-1).mean()
    assert final_error < np.linalg.norm(base - expert, axis=-1).mean()
    assert applied["effective_alpha"].max() <= 1.0


def _candidate_set(samples=10, candidates=24, image_size=16):
    rng = np.random.default_rng(11)
    points = rng.normal(0.0, 5.0, size=(samples, NUM_LANDMARKS, candidates, 3)).astype(np.float32)
    expert = points[:, :, 0].copy()
    distance = np.linalg.norm(points - expert[:, :, None], axis=-1).astype(np.float32)
    anchors = rng.normal(size=(samples, NUM_LANDMARKS, 4, 3)).astype(np.float32)
    return All23CandidateSet(
        sample_ids=[f"S{index}" for index in range(samples)],
        classes=["Class1", "Class2"] * (samples // 2),
        genders=["men"] * (samples // 2) + ["women"] * (samples // 2),
        images=rng.normal(
            size=(samples, NUM_LANDMARKS, len(VIEW_NAMES), len(IMAGE_CHANNELS), image_size, image_size)
        ).astype(np.float16),
        grids=rng.uniform(-1, 1, size=(samples, NUM_LANDMARKS, len(VIEW_NAMES), candidates, 2)).astype(np.float32),
        points=points,
        features=rng.normal(size=(samples, NUM_LANDMARKS, candidates, 78)).astype(np.float16),
        mask=np.ones((samples, NUM_LANDMARKS, candidates), dtype=bool),
        heatmap_target=np.exp(-distance**2 / 8.0).astype(np.float32),
        target_distance=distance,
        expert=expert,
        expert_full=expert,
        base=expert + 1.0,
        prior_mean=expert + 0.5,
        prior_covariance=np.tile(np.eye(3, dtype=np.float32), (samples, NUM_LANDMARKS, 1, 1)),
        prior_score=-(distance**2),
        anatomy_anchors=anchors,
        anatomy_distances=np.linalg.norm(expert[:, :, None] - anchors, axis=-1).astype(np.float32),
        anatomy_mask=np.ones((samples, NUM_LANDMARKS, 4), dtype=bool),
        has_rgb=np.ones(samples, dtype=bool),
    )


def test_tiny_training_run_resumes_and_predicts(tmp_path):
    candidates = _candidate_set()
    config = All23MVSCConfig(epochs=2, min_epochs=1, patience=1, batch_size=16, width=16, coordinate_topk=4)
    model, report = fit_network(candidates, tmp_path, config, torch.device("cpu"), "sig")
    assert report["training_complete"] and (tmp_path / "best_model.pth").exists()
    prediction = predict_candidates(model, candidates, config, torch.device("cpu"))
    assert prediction["proposal"].shape == (10, NUM_LANDMARKS, 3)
    assert prediction["oracle"].shape == (10, NUM_LANDMARKS)
    # A completed run resumes without training further.
    _, resumed = fit_network(candidates, tmp_path, config, torch.device("cpu"), "sig")
    assert resumed["best_epoch"] == report["best_epoch"]


def test_candidate_subset_keeps_alignment():
    candidates = _candidate_set()
    subset = candidates.subset([3, 1])
    assert subset.sample_ids == ["S3", "S1"]
    np.testing.assert_array_equal(subset.expert[1], candidates.expert[1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
