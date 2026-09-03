import json
from types import SimpleNamespace

import numpy as np
import torch

from agh_former_vnext_orthodontic_comparison.hard3_structured import (
    Hard3CandidateSet,
    Hard3StructuredConfig,
    StructuredCandidateRanker,
    _cache_signature,
    _decode_policy,
    _ranking_loss,
    _select_coordinate_policy,
    _tensor_batch,
    apply_hard3_blend,
    calibrate_hard3_blend,
)
from agh_former_vnext_orthodontic_comparison.model import AGHFormerVNext
from agh_former_vnext_orthodontic_comparison.run_aghformer_vnext import (
    build_stage3_decision,
    load_completed_fold,
    vnext_signature,
)
from agh_former_vnext_orthodontic_comparison.shape_prior import TrainOnlyShapePrior
from all23_rgb_geodesic_cascade.anatomy import CORE20, MIDLINE, SYMMETRY_PAIRS
from all23_rgb_geodesic_cascade.losses import LossWeights, compute_loss


def synthetic_batch(batch_size=2, vertices=48, roi_points=12):
    generator = torch.Generator().manual_seed(7)
    points = torch.randn(batch_size * vertices, 3, generator=generator) * 20.0
    features = torch.randn(batch_size * vertices, 14, generator=generator)
    graph_batch = torch.arange(batch_size).repeat_interleave(vertices)
    edges = []
    for batch_index in range(batch_size):
        offset = batch_index * vertices
        for index in range(vertices):
            next_index = (index + 1) % vertices
            edges.extend(
                [
                    (offset + index, offset + index),
                    (offset + index, offset + next_index),
                    (offset + next_index, offset + index),
                ]
            )
    roi = []
    for batch_index in range(batch_size):
        offset = batch_index * vertices
        roi.append(
            torch.stack(
                [
                    offset + ((torch.arange(roi_points) + landmark) % vertices)
                    for landmark in range(23)
                ]
            )
        )
    roi = torch.stack(roi)
    coarse = torch.randn(batch_size, 23, 3, generator=generator) * 5.0
    expert = coarse + torch.randn(batch_size, 23, 3, generator=generator)
    candidates = points[roi]
    distances = torch.linalg.norm(candidates - expert[:, :, None], dim=-1)
    return {
        "sample_id": [f"sample_{index}" for index in range(batch_size)],
        "points": points,
        "features": features,
        "batch": graph_batch,
        "edge_index": torch.tensor(edges, dtype=torch.long).T,
        "vertex_mask": torch.ones(batch_size * vertices, dtype=torch.bool),
        "coarse": coarse,
        "expert": expert,
        "roi_index": roi,
        "roi_mask": torch.ones(batch_size, 23, roi_points, dtype=torch.bool),
        "heatmap_target": torch.exp(-(distances**2) / 18.0),
        "region_target": (distances <= 3.5).float(),
        "oracle_error": distances.min(dim=-1).values,
        "sample_radius_scale": torch.ones(batch_size),
    }


def test_canonical_anatomy_contract():
    assert MIDLINE == tuple(range(13))
    assert SYMMETRY_PAIRS == ((13, 16), (14, 15), (17, 18), (19, 20), (21, 22))


def test_vnext_forward_and_loss_are_finite():
    batch = synthetic_batch()
    model = AGHFormerVNext(
        input_dim=14,
        width=32,
        global_blocks=1,
        heads=4,
        dropout=0.0,
        token_blocks=1,
        token_surface_points=256,
        coordinate_topk=8,
        gonion_pair_topk=8,
        use_anatomical_attention=True,
        use_specialized_heads=True,
        use_local_refiner=True,
        use_refinement_gate=True,
        use_hard_candidate_ranker=True,
        use_e10_rankers=True,
    )
    outputs = model(batch, coordinate_mode="topk")
    assert outputs["final_coordinates"].shape == (2, 23, 3)
    assert outputs["local_logits"].shape == (2, 23, 12)
    assert torch.all((outputs["fusion_alpha"] >= 0) & (outputs["fusion_alpha"] <= 1))
    loss, errors, _ = compute_loss(
        outputs,
        batch,
        LossWeights(hard_landmark=2.0, hard_rank=0.5),
        hard_rank_mode="soft_listwise",
    )
    assert torch.isfinite(loss)
    assert errors.shape == (2, 23)
    loss.backward()
    assert model.fusion_gate[-1].weight.grad is not None


def test_shape_prior_uses_train_fit_and_validation_selection():
    rng = np.random.default_rng(42)
    mean = rng.normal(size=(23, 3)) * 10.0
    basis = rng.normal(size=(5, 23, 3))
    train_latent = rng.normal(size=(60, 5))
    train = mean + np.einsum("nf,flc->nlc", train_latent, basis) * 0.5
    validation_latent = rng.normal(size=(16, 5))
    expert = mean + np.einsum("nf,flc->nlc", validation_latent, basis) * 0.5
    prediction = expert + rng.normal(0.0, 1.5, size=expert.shape)
    prior = TrainOnlyShapePrior(
        component_grid=(5, 10), l2_grid=(1.0, 10.0)
    ).fit(train, [f"train_{index}" for index in range(len(train))])
    prior.calibrate(
        prediction,
        expert,
        [f"val_{index}" for index in range(len(expert))],
    )
    refined = prior.transform(prediction)
    assert np.linalg.norm(refined - expert, axis=-1).mean() < np.linalg.norm(
        prediction - expert, axis=-1
    ).mean()
    report = prior.report()
    assert report["uses_test_labels"] is False
    assert len(report["fit_sample_ids"]) == len(train)


def synthetic_hard3_candidates(samples=8, candidates=16, feature_dim=24):
    rng = np.random.default_rng(19)
    points = rng.normal(0.0, 4.0, size=(samples, 3, candidates, 3)).astype(np.float32)
    target_index = rng.integers(0, candidates - 2, size=(samples, 3))
    expert = np.empty((samples, 3, 3), dtype=np.float32)
    for sample in range(samples):
        for landmark in range(3):
            expert[sample, landmark] = points[sample, landmark, target_index[sample, landmark]]
    distance = np.linalg.norm(points - expert[:, :, None], axis=-1).astype(np.float32)
    features = rng.normal(
        size=(samples, 3, candidates, feature_dim)
    ).astype(np.float32)
    # Give the test ranker a learnable signal tied to expert distance.
    features[..., 0] = -distance
    mask = np.ones((samples, 3, candidates), dtype=bool)
    mask[:, :, -2:] = False
    distance[~mask] = np.inf
    return Hard3CandidateSet(
        sample_ids=[f"hard_{index}" for index in range(samples)],
        strata=[f"Class{index % 2 + 1}|{'women' if index % 2 else 'men'}" for index in range(samples)],
        features=features,
        canonical=points / 100.0,
        points=points,
        mask=mask,
        expert=expert,
        target_distance=distance,
    )


def test_hard3_pair_ranker_has_finite_loss_and_pair_gradients():
    candidates = synthetic_hard3_candidates()
    config = Hard3StructuredConfig(batch_size=4, width=16, pair_topk=6)
    batch = _tensor_batch(
        candidates,
        np.arange(4),
        np.zeros(candidates.features.shape[-1], dtype=np.float32),
        np.ones(candidates.features.shape[-1], dtype=np.float32),
        torch.device("cpu"),
    )
    model = StructuredCandidateRanker(
        candidates.features.shape[-1], width=16, dropout=0.0, pair_topk=6
    )
    logits = model(batch["features"], batch["canonical"], batch["mask"])
    loss, components = _ranking_loss(logits, batch, config)
    assert logits.shape == (4, 3, 16)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert model.pair_score[-1].weight.grad is not None
    assert torch.isfinite(model.pair_score[-1].weight.grad).all()


def test_hard3_oof_policy_decodes_surface_candidates():
    candidates = synthetic_hard3_candidates()
    logits = -candidates.target_distance**2
    policy = _select_coordinate_policy(candidates, logits)
    prediction = _decode_policy(candidates, logits, policy)
    error = np.linalg.norm(prediction - candidates.expert, axis=-1)
    assert prediction.shape == candidates.expert.shape
    assert float(error.mean()) < 1e-5
    assert policy["lm0"]["topk"] == 1
    assert policy["gonion"]["topk"] == 1


def test_hard3_blend_is_validation_gated_and_never_changes_core20():
    rng = np.random.default_rng(23)
    expert = rng.normal(size=(12, 23, 3)).astype(np.float32)
    base = expert.copy()
    base[:, [0, 21, 22], 0] += 4.0
    outputs = {
        "sample_ids": [f"sample_{index}" for index in range(len(expert))],
        "prediction": base,
        "expert": expert,
    }
    candidate_result = {
        "sample_ids": list(reversed(outputs["sample_ids"])),
        "prediction": expert[::-1][:, [0, 21, 22]],
    }
    config = Hard3StructuredConfig(
        bootstrap_iters=50,
        minimum_overall_gain_mm=0.01,
        minimum_hard3_gain_mm=0.1,
    )
    policy = calibrate_hard3_blend(outputs, candidate_result, config)
    refined = apply_hard3_blend(outputs, candidate_result, policy)
    assert policy["accepted"] is True
    np.testing.assert_array_equal(refined["prediction"][:, CORE20], base[:, CORE20])
    assert np.linalg.norm(
        refined["prediction"][:, [0, 21, 22]] - expert[:, [0, 21, 22]], axis=-1
    ).mean() < 1e-6


def test_hard3_blend_caps_large_candidate_steps_and_applies_reliability():
    expert = np.zeros((2, 23, 3), dtype=np.float32)
    base = expert.copy()
    candidate = np.full((2, 3, 3), 100.0, dtype=np.float32)
    outputs = {
        "sample_ids": ["a", "b"],
        "prediction": base,
        "expert": expert,
    }
    candidate_result = {
        "sample_ids": ["a", "b"],
        "prediction": candidate,
        "reliability": np.full((2, 3), 0.5, dtype=np.float32),
        "ensemble_spread": np.ones((2, 3), dtype=np.float32),
    }
    policy = {
        "limits_mm": [4.0, 6.0, 6.0],
        "selected": {
            "confidence_mode": "ensemble",
            "alpha_lm0": 1.0,
            "alpha_gonion": 1.0,
        },
    }
    refined = apply_hard3_blend(outputs, candidate_result, policy)
    step = np.linalg.norm(
        refined["prediction"][:, [0, 21, 22]] - base[:, [0, 21, 22]], axis=-1
    )
    np.testing.assert_allclose(step[:, 0], 2.0, atol=1e-5)
    np.testing.assert_allclose(step[:, 1:3], 3.0, atol=1e-5)
    np.testing.assert_array_equal(refined["prediction"][:, CORE20], base[:, CORE20])


def test_hard3_cache_signature_tracks_training_inputs_not_acceptance_thresholds(tmp_path):
    record = tmp_path / "sample.npz"
    record.write_bytes(b"record")
    sample = SimpleNamespace(sample_id="sample")

    class DatasetStub:
        samples = [sample]
        records = {"sample": record}
        mean = np.zeros(14, dtype=np.float32)
        std = np.ones(14, dtype=np.float32)
        roi_points = 32
        roi_radius_scale = 1.5
        roi_mode = "hybrid"
        roi_euclidean_scale = 1.25
        roi_multi_seeds = 3
        coarse = np.zeros((23, 3), dtype=np.float32)

        def _coarse(self, _sample):
            return self.coarse

    dataset = DatasetStub()
    base = Hard3StructuredConfig(bootstrap_iters=10, minimum_hard3_gain_mm=0.1)
    recalibrated = Hard3StructuredConfig(
        bootstrap_iters=500, minimum_hard3_gain_mm=0.5
    )
    assert _cache_signature(dataset, base) == _cache_signature(dataset, recalibrated)
    original = _cache_signature(dataset, base)
    dataset.coarse = np.ones((23, 3), dtype=np.float32)
    assert _cache_signature(dataset, base) != original


def test_completed_fold_cache_requires_stage3_artifacts_when_enabled(tmp_path):
    args = SimpleNamespace(
        resume_stage2=True,
        force_stage2_retrain=False,
        validation_only=False,
        hard3_structured=True,
        output_dir="ignored",
    )
    splits = {"train": ["a"], "val": ["b"], "test": ["c"]}
    for name in ("metrics_val.json", "metrics_test.json", "predictions_test.csv"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    result = {
        "postprocess_version": 1,
        "stage2_signature": vnext_signature(args, splits),
        "hard3_structured": {"enabled": True},
    }
    (tmp_path / "run_summary.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    assert load_completed_fold(tmp_path, args, splits) is None

    hard3 = tmp_path / "hard3_structured"
    hard3.mkdir()
    for name in ("hard3_structured_model.pth", "metrics_val.json", "metrics_test.json"):
        (hard3 / name).write_bytes(b"artifact")
    assert load_completed_fold(tmp_path, args, splits) == result


def test_hard3_options_do_not_invalidate_expensive_stage2_signature():
    splits = {"train": ["a"], "val": ["b"], "test": ["c"]}
    first = SimpleNamespace(
        width=128,
        epochs=220,
        output_dir="first",
        hard3_structured=True,
        hard3_structured_width=16,
        hard3_stage3_full_cv_max_overall=2.25,
    )
    second = SimpleNamespace(
        width=128,
        epochs=220,
        output_dir="second",
        hard3_structured=False,
        hard3_structured_width=64,
        hard3_stage3_full_cv_max_overall=2.10,
    )
    assert vnext_signature(first, splits) == vnext_signature(second, splits)


def test_stage3_decision_exposes_two_mm_hard3_budget():
    args = SimpleNamespace(
        hard3_stage3_full_cv_max_overall=2.25,
        hard3_stage3_full_cv_max_hard3=4.5,
    )
    baseline = {
        "overall": {"ale": 2.2818, "p95": 6.38},
        "core20": {"ale": 1.8356},
        "hard3": {"ale": 5.2565},
    }
    final = {
        "overall": {"ale": 2.18, "p95": 6.20},
        "core20": {"ale": 1.8356},
        "hard3": {"ale": 4.476},
    }
    decision = build_stage3_decision(
        args, baseline, final, {"blend": {"accepted": True}}
    )
    expected_budget = (23.0 * 2.0 - 20.0 * 1.8356) / 3.0
    assert abs(
        decision["two_mm_budget"]["required_hard3_ale_at_current_core20"]
        - expected_budget
    ) < 1e-9
    assert decision["run_full_cv"] is True
    assert decision["test_labels_consumed"] is False
