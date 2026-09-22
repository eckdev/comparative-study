from __future__ import annotations

import json

import numpy as np
import torch

from all23_rgb_geodesic_cascade.anatomy import HARD3
from core20_mvsc_refinement.anatomy import (
    CORE20_GROUPS,
    CORE20_INDICES,
    mirror_landmark_index,
)
from core20_mvsc_refinement.model import Core20MVSCNet
from core20_mvsc_refinement.refiner import (
    Core20MVSCConfig,
    _fit_network,
    _loss,
    apply_core20_refinement,
    calibrate_core20_policy,
    write_core20_split_report,
)
from core20_mvsc_refinement.patches import Core20CandidateSet
from core20_mvsc_refinement.spatial_prior import (
    ConditionalSpatialPrior,
    SpatialPriorConfig,
)


def synthetic_model_batch(batch_size=6, candidates=32):
    generator = torch.Generator().manual_seed(41)
    points = torch.randn(batch_size, candidates, 3, generator=generator)
    expert = points[:, 3].clone()
    distance = torch.linalg.norm(points - expert[:, None], dim=-1)
    mask = torch.ones(batch_size, candidates, dtype=torch.bool)
    target = torch.exp(-distance.square() / (2.0 * 2.5**2))
    anchors = torch.randn(batch_size, 4, 3, generator=generator)
    return {
        "images": torch.randn(batch_size, 3, 16, 32, 32, generator=generator),
        "grids": torch.rand(batch_size, 3, candidates, 2, generator=generator) * 2 - 1,
        "points": points,
        "features": torch.randn(batch_size, candidates, 28, generator=generator),
        "mask": mask,
        "heatmap_target": target,
        "distance": distance,
        "expert": expert,
        "base": expert + 1.0,
        "prior_mean": expert + 0.25,
        "prior_covariance": torch.eye(3).repeat(batch_size, 1, 1),
        "prior_score": -distance.square(),
        "anatomy_anchors": anchors,
        "anatomy_distances": torch.linalg.norm(expert[:, None] - anchors, dim=-1),
        "anatomy_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "local_landmark": torch.tensor([0, 2, 8, 12, 16, 19]),
        "actual_landmark": torch.tensor([1, 3, 9, 13, 17, 20]),
        "group": torch.tensor([0, 1, 1, 3, 4, 5]),
    }


def test_core20_schema_and_mirror_contract():
    flattened = [value for group in CORE20_GROUPS.values() for value in group]
    assert tuple(sorted(flattened)) == CORE20_INDICES
    assert set(flattened).isdisjoint(HARD3)
    assert mirror_landmark_index(13) == 16
    assert mirror_landmark_index(14) == 15
    assert mirror_landmark_index(17) == 18
    assert mirror_landmark_index(19) == 20
    assert mirror_landmark_index(7) == 7


def test_core20_model_and_full_loss_are_finite_without_batchnorm():
    batch = synthetic_model_batch()
    batch["mask"][:, -5:] = False
    batch["heatmap_target"][:, -5:] = 0.0
    model = Core20MVSCNet(16, 28, width=24, dropout=0.0)
    outputs = model(batch)
    loss, components, errors = _loss(
        outputs,
        batch,
        Core20MVSCConfig(epochs=1, min_epochs=1, patience=1),
    )
    assert torch.isfinite(loss)
    assert errors.shape == (6,)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert not any(
        isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()
    )


def test_core20_model_is_bfloat16_autocast_safe():
    batch = synthetic_model_batch()
    model = Core20MVSCNet(16, 28, width=24, dropout=0.0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = model(batch)
        loss, _, _ = _loss(
            outputs,
            batch,
            Core20MVSCConfig(epochs=1, min_epochs=1, patience=1),
        )
    assert outputs["logits"].dtype == torch.float32
    torch.testing.assert_close(
        outputs["gate_alpha"].float(),
        torch.sigmoid(outputs["gate_logit"].float()),
        rtol=2e-2,
        atol=2e-2,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_geometry_only_fallback_is_finite():
    batch = synthetic_model_batch()
    model = Core20MVSCNet(
        16,
        28,
        width=24,
        dropout=0.0,
        use_images=False,
        use_spatial_prior=False,
        use_confidence_gate=False,
    )
    outputs = model(batch)
    assert torch.isfinite(outputs["final"]).all()
    assert torch.all(outputs["gate_alpha"] == 1.0)


def test_spatial_prior_is_oof_and_target_is_excluded(tmp_path):
    rng = np.random.default_rng(17)
    samples = 72
    latent = rng.normal(size=(samples, 4))
    basis = rng.normal(size=(4, 23, 3))
    shapes = rng.normal(size=(23, 3))[None] * 10.0
    shapes = shapes + np.einsum("nf,flc->nlc", latent, basis)
    sample_ids = [f"sample_{index}" for index in range(samples)]
    classes = [f"Class{index % 3 + 1}" for index in range(samples)]
    genders = ["women" if index % 2 else "men" for index in range(samples)]
    prior = ConditionalSpatialPrior(
        SpatialPriorConfig(folds=5, l2_grid=(0.1, 1.0), seed=7)
    ).fit(shapes, sample_ids, classes, genders)
    assert set(prior.oof_means) == set(sample_ids)
    assert np.isfinite(prior.oof_prediction(sample_ids)).all()
    prediction, _ = prior.predict(shapes[:4], classes[:4], genders[:4])
    changed = shapes[:4].copy()
    changed[:, 7] += 1000.0
    changed_prediction, _ = prior.predict(changed, classes[:4], genders[:4])
    local_index = CORE20_INDICES.index(7)
    np.testing.assert_allclose(
        prediction[:, local_index], changed_prediction[:, local_index], atol=1e-5
    )
    assert prior.report()["uses_validation_labels"] is False
    assert prior.report()["uses_test_labels"] is False

    coarse = shapes.copy()
    coarse += rng.normal(0.0, 0.75, size=coarse.shape)
    contextual = prior.oof_predict_from_shapes(coarse, sample_ids, classes, genders)
    changed_coarse = coarse.copy()
    changed_coarse[:, 7] += 1000.0
    changed_contextual = prior.oof_predict_from_shapes(
        changed_coarse, sample_ids, classes, genders
    )
    np.testing.assert_allclose(
        contextual[:, local_index], changed_contextual[:, local_index], atol=1e-5
    )
    path = tmp_path / "prior.json"
    prior.save(path)
    loaded = ConditionalSpatialPrior.load(path)
    np.testing.assert_allclose(
        contextual,
        loaded.oof_predict_from_shapes(coarse, sample_ids, classes, genders),
        atol=1e-5,
    )


def test_core20_policy_improves_core_and_preserves_hard3_bitwise(tmp_path):
    rng = np.random.default_rng(23)
    expert = rng.normal(size=(20, 23, 3)).astype(np.float32)
    base = expert.astype(np.float64)
    base[:, list(CORE20_INDICES), 0] += 1.5
    base[:, list(HARD3), 0] += 4.0
    outputs = {
        "sample_ids": [f"sample_{index}" for index in range(len(base))],
        "prediction": base,
        "expert": expert,
    }
    candidate_result = {
        "sample_ids": list(outputs["sample_ids"]),
        "prediction": expert[:, list(CORE20_INDICES)],
        "oracle": np.full((len(base), 20), 0.5, dtype=np.float32),
        "confidence": np.ones((len(base), 20), dtype=np.float32),
        "gate_alpha": np.ones((len(base), 20), dtype=np.float32),
        "entropy": np.zeros((len(base), 20), dtype=np.float32),
        "margin": np.ones((len(base), 20), dtype=np.float32),
        "prior_disagreement": np.zeros((len(base), 20), dtype=np.float32),
    }
    config = Core20MVSCConfig(bootstrap_iters=100)
    policy = calibrate_core20_policy(outputs, candidate_result, config)
    refined = apply_core20_refinement(outputs, candidate_result, policy)
    assert policy["accepted"] is True
    np.testing.assert_array_equal(
        refined["prediction"][:, list(HARD3)], base[:, list(HARD3)]
    )
    assert refined["prediction"].dtype == base.dtype
    assert (
        np.linalg.norm(
            refined["prediction"][:, list(CORE20_INDICES)]
            - expert[:, list(CORE20_INDICES)],
            axis=-1,
        ).mean()
        < 1e-6
    )
    report = write_core20_split_report(
        tmp_path, "test", refined, candidate_result, bootstrap_iters=50
    )
    assert report["policy_locked_before_split_evaluation"] is True
    assert len(report["anatomical_groups"]) == 6
    assert (tmp_path / "anatomical_group_metrics_test.csv").exists()


def test_tiny_training_writes_and_reloads_best_checkpoint(tmp_path):
    rng = np.random.default_rng(31)
    samples, landmarks, candidates = 8, 20, 16
    expert = rng.normal(size=(samples, landmarks, 3)).astype(np.float32)
    points = expert[:, :, None] + rng.normal(
        0.0, 2.0, size=(samples, landmarks, candidates, 3)
    ).astype(np.float32)
    points[:, :, 0] = expert
    distance = np.linalg.norm(points - expert[:, :, None], axis=-1).astype(np.float32)
    mask = np.ones((samples, landmarks, candidates), dtype=np.bool_)
    mask[:, :, -2:] = False
    distance[~mask] = np.inf
    heatmap = np.exp(-np.square(distance) / (2.0 * 2.5**2)).astype(np.float32)
    heatmap[~mask] = 0.0
    expert_full = rng.normal(size=(samples, 23, 3)).astype(np.float32)
    expert_full[:, list(CORE20_INDICES)] = expert
    anchors = rng.normal(size=(samples, landmarks, 4, 3)).astype(np.float32)
    candidate_set = Core20CandidateSet(
        sample_ids=[f"sample_{index}" for index in range(samples)],
        classes=[f"Class{index % 3 + 1}" for index in range(samples)],
        genders=["women" if index % 2 else "men" for index in range(samples)],
        images=rng.normal(size=(samples, landmarks, 3, 16, 16, 16)).astype(np.float16),
        grids=rng.uniform(
            -1.0, 1.0, size=(samples, landmarks, 3, candidates, 2)
        ).astype(np.float32),
        points=points,
        features=rng.normal(size=(samples, landmarks, candidates, 26)).astype(
            np.float32
        ),
        mask=mask,
        heatmap_target=heatmap,
        target_distance=distance,
        expert=expert,
        expert_full=expert_full,
        base=expert + 1.0,
        prior_mean=expert + 0.25,
        prior_covariance=np.tile(
            np.eye(3, dtype=np.float32), (samples, landmarks, 1, 1)
        ),
        prior_score=-np.where(np.isfinite(distance), distance, 50.0),
        anatomy_anchors=anchors,
        anatomy_distances=np.linalg.norm(expert[:, :, None] - anchors, axis=-1),
        anatomy_mask=np.ones((samples, landmarks, 4), dtype=np.bool_),
        has_rgb=np.ones(samples, dtype=np.bool_),
    )
    config = Core20MVSCConfig(
        epochs=1,
        min_epochs=1,
        patience=1,
        batch_size=64,
        image_size=16,
        width=16,
        dropout=0.0,
        mixed_precision=False,
        bootstrap_iters=10,
    )
    model, report = _fit_network(candidate_set, tmp_path, config, torch.device("cpu"))
    assert (tmp_path / "best_model.pth").exists()
    assert (tmp_path / "history.json").exists()
    assert report["best_epoch"] == 1
    assert report["parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    _, resumed_report = _fit_network(
        candidate_set, tmp_path, config, torch.device("cpu")
    )
    assert resumed_report["best_epoch"] == report["best_epoch"]
    assert len(json.loads((tmp_path / "history.json").read_text())) == 1
