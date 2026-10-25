from __future__ import annotations

from pathlib import Path

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
    anatomical_inconsistency,
    diagnostics_from_iterations,
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
    fit_refiner,
    predict_candidates,
    predict_members,
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
    assert accepted["accepted"]
    assert accepted["iterations"] == {name: 2 for name in LANDMARK_TYPES}
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


def test_policy_selects_iterations_per_landmark_type():
    config = All23MVSCConfig(bootstrap_iters=200)
    rng = np.random.default_rng(4)
    expert = rng.normal(size=(40, NUM_LANDMARKS, 3)).astype(np.float32)
    base = expert + rng.normal(0.0, 1.0, size=expert.shape).astype(np.float32)
    corner = list(LANDMARK_TYPES["corner"])
    hard = [index for index in ALL23_INDICES if index not in corner]
    first = base.copy()
    second = base.copy()
    # Iteration 1 is right for contour/texture landmarks, iteration 2 for corners.
    first[:, hard] = expert[:, hard] + 0.1
    second[:, corner] = expert[:, corner] + 0.1
    alpha = np.ones((40, NUM_LANDMARKS), dtype=np.float32)
    result = {
        "sample_ids": [f"S{index}" for index in range(40)],
        "base": base,
        "iterations": {
            1: {"proposal": first, "gate_alpha": alpha, "confidence": alpha, "entropy": alpha},
            2: {"proposal": second, "gate_alpha": alpha, "confidence": alpha, "entropy": alpha},
        },
    }
    policy = calibrate_policy(result, expert, config)
    assert policy["iterations"] == {"corner": 2, "contour": 1, "texture_boundary": 1}
    applied = apply_policy(result, policy)
    np.testing.assert_allclose(applied["prediction"], expert + 0.1, atol=1e-5)
    legacy = {**policy, "iterations": 2}
    np.testing.assert_allclose(
        apply_policy(result, legacy)["prediction"][:, corner], second[:, corner], atol=1e-5
    )


def test_anatomical_inconsistency_is_zero_for_a_symmetric_face():
    rng = np.random.default_rng(8)
    face = np.zeros((1, NUM_LANDMARKS, 3))
    face[0, :13, 1:] = rng.normal(0.0, 30.0, size=(13, 2))
    for left, right in ((13, 16), (14, 15), (17, 18), (19, 20), (21, 22)):
        point = rng.normal(0.0, 30.0, size=3)
        point[0] = abs(point[0]) + 5.0
        face[0, right] = point
        face[0, left] = point * np.asarray([-1.0, 1.0, 1.0])
    np.testing.assert_allclose(anatomical_inconsistency(face, face), 0.0, atol=1e-6)
    shifted = face.copy()
    shifted[0, 21, 1] += 4.0
    values = anatomical_inconsistency(shifted, face)
    assert values[0, 21] == pytest.approx(4.0) and values[0, 22] == pytest.approx(4.0)
    assert values[0, 3] == pytest.approx(0.0, abs=1e-6)


def test_gate_diagnostics_include_spread_and_anatomy():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(3, NUM_LANDMARKS, 3)) * 20
    iteration = {
        "proposal": base + 1.0,
        "entropy": np.zeros((3, NUM_LANDMARKS)),
        "margin": np.zeros((3, NUM_LANDMARKS)),
        "prior_disagreement": np.zeros((3, NUM_LANDMARKS)),
        "log_variance": np.zeros((3, NUM_LANDMARKS)),
        "mc_spread": np.full((3, NUM_LANDMARKS), 2.0),
    }
    diagnostics = diagnostics_from_iterations(base, [iteration])
    assert diagnostics.shape == (3, NUM_LANDMARKS, len(DIAGNOSTIC_NAMES))
    spread = DIAGNOSTIC_NAMES.index("mc_dropout_spread_over_type_scale")
    assert diagnostics[0, 1, spread] == pytest.approx(1.0)  # corner scale 2 mm
    assert diagnostics[0, 21, spread] == pytest.approx(0.5)  # contour scale 4 mm
    del iteration["mc_spread"]
    assert np.all(diagnostics_from_iterations(base, [iteration])[..., spread] == 0.0)


def test_members_are_averaged_and_mc_spread_comes_from_the_lead(tmp_path):
    candidates = _candidate_set()
    config = All23MVSCConfig(width=16, coordinate_topk=4, batch_size=64, dropout=0.3)
    torch.manual_seed(0)
    models = [
        All23MVSCNet(len(IMAGE_CHANNELS), 78, width=16, dropout=0.3, coordinate_topk=4)
        for _ in range(2)
    ]
    single = [predict_candidates(model, candidates, config, torch.device("cpu")) for model in models]
    merged = predict_members(models, candidates, config, torch.device("cpu"), mc_samples=4)
    np.testing.assert_allclose(
        merged["proposal"], (single[0]["proposal"] + single[1]["proposal"]) / 2, atol=1e-5
    )
    assert merged["members"] == 2 and merged["mc_spread"].shape == (10, NUM_LANDMARKS)
    assert np.all(merged["mc_spread"] >= 0) and merged["mc_spread"].max() > 0
    assert not models[0].training  # MC passes restore eval mode
    again = predict_members(models, candidates, config, torch.device("cpu"), mc_samples=4)
    np.testing.assert_array_equal(merged["mc_spread"], again["mc_spread"])


def test_reused_refiner_must_exist(tmp_path):
    class Sample:
        def __init__(self, sample_id):
            self.sample_id, self.class_name, self.gender = sample_id, "Class1", "men"

    class Dataset:
        samples = [Sample("A")]
        records = {}

    record = tmp_path / "A.npz"
    np.savez(record, landmarks=np.zeros((NUM_LANDMARKS, 3), dtype=np.float32))
    Dataset.records = {"A": record}
    with pytest.raises(RuntimeError, match="no compatible trained refiner"):
        fit_refiner(
            Dataset(),
            tmp_path / "missing",
            All23MVSCConfig(),
            torch.device("cpu"),
            require_cached=True,
        )


def test_trained_refiner_signature_is_stable():
    # Pinned to the signature of commit 9644570, whose trained refiners v2.1
    # reuses. Inference-only settings must not change it.
    from all23_mvsc_refinement.refiner import config_signature

    expected = "266488b5db0ed259366fd8c542fcae9fee53d80e4f2f7e297e01c9f8e15bda77"
    assert config_signature(All23MVSCConfig(), ["A", "B"]) == expected
    assert config_signature(All23MVSCConfig(ensemble=False, mc_samples=0), ["A", "B"]) == expected
    assert config_signature(All23MVSCConfig(use_gate=False), ["A", "B"]) == expected


def test_gate_free_policy_only_switches_refinement_on_or_off():
    rng = np.random.default_rng(6)
    expert = rng.normal(size=(30, NUM_LANDMARKS, 3)).astype(np.float32)
    base = expert + rng.normal(0.0, 1.0, size=expert.shape).astype(np.float32)
    proposal = expert + rng.normal(0.0, 0.3, size=expert.shape).astype(np.float32)
    ones = np.ones((30, NUM_LANDMARKS), dtype=np.float32)
    result = {
        "sample_ids": [f"S{index}" for index in range(30)],
        "base": base,
        "iterations": {
            1: {"proposal": proposal, "gate_alpha": ones, "confidence": ones, "entropy": ones}
        },
    }
    policy = calibrate_policy(result, expert, All23MVSCConfig(use_gate=False, bootstrap_iters=100))
    assert policy["use_gate"] is False
    assert {row["alpha_cap"] for row in policy["sweep"]} == {0.0, 1.0}
    np.testing.assert_allclose(apply_policy(result, policy)["prediction"], proposal, atol=1e-6)


def test_gate_free_refinement_builds_no_evidence_or_gates(tmp_path, monkeypatch):
    from all23_mvsc_refinement import refiner as module

    from types import SimpleNamespace

    class Fitted:
        def __init__(self, name):
            self.model, self.report = name, {"name": name}
            self.prior = SimpleNamespace(config=SimpleNamespace(covariates=("class", "sex")))

    loaded = []

    def fake_fit_refiner(dataset, output_dir, config, device, **kwargs):
        loaded.append((Path(output_dir).name, kwargs.get("require_cached")))
        return Fitted(Path(output_dir).name)

    monkeypatch.setattr(module, "fit_refiner", fake_fit_refiner)
    monkeypatch.setattr(
        module, "crossfit_splits", lambda dataset, config: [(1, ["A"], ["B"]), (2, ["B"], ["A"])]
    )
    monkeypatch.setattr(module, "subset_dataset", lambda dataset, ids: dataset)
    monkeypatch.setattr(
        module,
        "build_crossfit_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no evidence without gate")),
    )
    config = All23MVSCConfig(use_gate=False, mc_samples=8)
    refinement = module.fit_or_load_all23_refinement(
        object(), tmp_path / "out", config, torch.device("cpu"), model_dir=tmp_path / "source"
    )
    assert refinement.gates == {} and refinement.reports["use_gate"] is False
    assert refinement.ensemble.models == ["full_model", "inner_fold_1", "inner_fold_2"]
    assert refinement.ensemble.config.mc_samples == 0
    assert all(require_cached for _, require_cached in loaded)
    assert not (tmp_path / "out" / "crossfit").exists()


def test_spatial_prior_covariates_are_configurable(tmp_path):
    shapes = synthetic_shapes()
    ids = [f"S{index}" for index in range(len(shapes))]
    classes = ["Class1", "Class2", "Class3", "Class1"] * (len(shapes) // 4)
    genders = ["men", "women"] * (len(shapes) // 2)
    both = ConditionalSpatialPrior(SpatialPriorConfig(folds=4)).fit(shapes, ids, classes, genders)
    sex_only = ConditionalSpatialPrior(SpatialPriorConfig(folds=4, covariates=("sex",))).fit(
        shapes, ids, classes, genders
    )
    width_both = both._raw_design(shapes, 1, classes, genders)[0].shape[1]
    width_sex = sex_only._raw_design(shapes, 1, classes, genders)[0].shape[1]
    assert width_both - width_sex == len(set(classes))
    # Predictions must not depend on class once the covariate is removed.
    swapped = list(reversed(classes))
    np.testing.assert_allclose(
        sex_only.predict(shapes[:4], classes[:4], genders[:4])[0],
        sex_only.predict(shapes[:4], swapped[:4], genders[:4])[0],
    )
    path = tmp_path / "prior.json"
    sex_only.save(path)
    assert ConditionalSpatialPrior.load(path).config.covariates == ("sex",)
    # Files written before covariates were configurable load with class and sex.
    import json

    payload = json.loads(path.read_text())
    del payload["config"]["covariates"]
    path.write_text(json.dumps(payload))
    assert ConditionalSpatialPrior.load(path).config.covariates == ("class", "sex")


def test_inference_prior_is_refitted_only_when_covariates_change(tmp_path, monkeypatch):
    from all23_mvsc_refinement import refiner as module

    shapes = synthetic_shapes()
    ids = [f"S{index}" for index in range(len(shapes))]
    classes = ["Class1", "Class2"] * (len(shapes) // 2)
    genders = ["men", "women", "women", "men"] * (len(shapes) // 4)
    original = ConditionalSpatialPrior(SpatialPriorConfig(folds=4)).fit(shapes, ids, classes, genders)

    class Full:
        prior = original

    monkeypatch.setattr(module, "dataset_shapes", lambda dataset: (shapes, ids, classes, genders))
    same = module.inference_prior(Full(), object(), All23MVSCConfig(), tmp_path)
    assert same is original
    refitted = module.inference_prior(
        Full(), object(), All23MVSCConfig(prior_covariates=("sex",), prior_folds=4), tmp_path
    )
    assert refitted is not original and refitted.config.covariates == ("sex",)
    assert (tmp_path / "spatial_prior_sex.json").exists()
    from all23_mvsc_refinement.refiner import config_signature

    assert config_signature(All23MVSCConfig(prior_covariates=("sex",)), ["A"]) == config_signature(
        All23MVSCConfig(), ["A"]
    )
