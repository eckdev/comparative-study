import numpy as np
import torch

from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3
from hard3_anatomical_context_refinement.atlas import TrainOnlyLocalHard3Atlas
from hard3_anatomical_context_refinement.model import DualViewHard3Net
from hard3_anatomical_context_refinement.patches import (
    DualViewCandidateSet,
    render_item,
)
from hard3_anatomical_context_refinement.refiner import (
    Hard3DualViewConfig,
    _dual_blend_prediction,
    _loss,
    _train_model,
    apply_dual_view_blend,
    calibrate_dual_view_blend,
)


def test_dual_view_forward_candidate_logits_and_loss_are_finite():
    generator = torch.Generator().manual_seed(17)
    batch, candidates, size = 2, 20, 32
    images = torch.randn(batch, 3, 2, 20, size, size, generator=generator)
    targets = torch.sigmoid(torch.randn(batch, 3, 2, size, size, generator=generator))
    grids = torch.rand(batch, 3, 2, candidates, 2, generator=generator) * 2 - 1
    points = torch.randn(batch, 3, candidates, 3, generator=generator)
    canonical = torch.randn(batch, 3, candidates, 18, generator=generator)
    expert = points[:, :, 3].clone()
    distance = torch.linalg.norm(points - expert[:, :, None], dim=-1)
    mask = torch.ones(batch, 3, candidates, dtype=torch.bool)
    model = DualViewHard3Net(20, width=8, dropout=0.0)
    heatmaps, view_weights = model.forward_with_context(images)
    logits = model.candidate_logits(heatmaps, grids, mask, view_weights)
    pair = model.gonion_pair(
        logits,
        canonical,
        points,
        mask,
        temperature=0.5,
        target_distance=distance,
    )
    loss, components = _loss(
        heatmaps,
        logits,
        {
            "targets": targets,
            "points": points,
            "canonical": canonical,
            "expert": expert,
            "distance": distance,
            "mask": mask,
            "target_view_mask": torch.ones(batch, 3, 2, dtype=torch.bool),
        },
        Hard3DualViewConfig(width=8),
        pair,
    )
    assert heatmaps.shape == (batch, 3, 2, size, size)
    assert logits.shape == (batch, 3, candidates)
    assert pair["logits"].shape == (batch, candidates, candidates)
    assert pair["soft_coordinate"].shape == (batch, 2, 3)
    torch.testing.assert_close(view_weights.sum(dim=-1), torch.ones(batch, 3))
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert model.gonion.output.weight.grad is not None
    assert model.gonion_pair_ranker[-1].weight.grad is not None
    assert model.gonion_view_gate[-1].weight.grad is not None


def test_sparse_mesh_patch_renderer_produces_finite_dual_views():
    rng = np.random.default_rng(5)
    vertices, roi_points = 96, 24
    points = rng.normal(size=(vertices, 3)).astype(np.float32) * 10.0
    coarse = rng.normal(size=(23, 3)).astype(np.float32)
    coarse[13], coarse[16] = (-30, 25, 5), (30, 25, 5)
    coarse[14], coarse[15] = (-12, 20, 8), (12, 20, 8)
    coarse[17], coarse[18] = (-18, 0, 12), (18, 0, 12)
    coarse[19], coarse[20] = (-22, -15, 8), (22, -15, 8)
    coarse[1], coarse[2], coarse[5] = (0, 40, 8), (0, 30, 10), (0, 0, 12)
    coarse[10], coarse[11], coarse[12] = (0, -30, 8), (0, -40, 6), (0, -48, 4)
    expert = coarse + rng.normal(0.0, 0.5, size=coarse.shape).astype(np.float32)
    features = rng.normal(size=(vertices, 14)).astype(np.float32)
    features[:, :3] = points
    features[:, 3:6] = rng.random((vertices, 3))
    features[:, 9:12] /= np.maximum(
        np.linalg.norm(features[:, 9:12], axis=1, keepdims=True), 1e-6
    )
    features[:, 12] = 1.0
    features[:, 13] = np.abs(features[:, 13])
    roi = np.stack([rng.choice(vertices, roi_points, replace=False) for _ in range(23)])
    item = {
        "points": torch.from_numpy(points),
        "features": torch.from_numpy(features),
        "coarse": torch.from_numpy(coarse),
        "expert": torch.from_numpy(expert),
        "roi_index": torch.from_numpy(roi),
        "roi_mask": torch.ones(23, roi_points, dtype=torch.bool),
    }
    rendered = render_item(
        item,
        np.zeros(14, dtype=np.float32),
        np.ones(14, dtype=np.float32),
        image_size=32,
    )
    assert rendered[0].shape == (3, 2, 20, 32, 32)
    assert rendered[1].shape == (3, 2, 32, 32)
    assert rendered[2].shape == (3, 2, roi_points, 2)
    assert rendered[4].shape == (3, roi_points, 18)
    assert rendered[5].any(axis=-1).all()
    assert rendered[9].shape == (3, 2)
    assert all(np.isfinite(values).all() for values in rendered[:5])
    # CoordConv and contour/depth-gradient channels are present immediately
    # before the final occupancy channel.
    assert rendered[0][..., -5, :, :].min() >= -1.0
    assert rendered[0][..., -5, :, :].max() <= 1.0
    assert rendered[0][..., -2, :, :].min() >= 0.0


def test_joint_pair_decoder_uses_both_gonion_candidate_sets():
    torch.manual_seed(23)
    model = DualViewHard3Net(20, width=8, dropout=0.0, pair_topk=4)
    torch.nn.init.normal_(model.gonion_pair_ranker[-1].weight, std=0.1)
    logits = torch.randn(1, 3, 8)
    canonical = torch.randn(1, 3, 8, 18)
    points = torch.randn(1, 3, 8, 3)
    mask = torch.ones(1, 3, 8, dtype=torch.bool)
    first = model.gonion_pair(logits, canonical, points, mask)
    changed = canonical.clone()
    changed[:, 2, :, 1] += 3.0
    second = model.gonion_pair(logits, changed, points, mask)
    assert not torch.allclose(first["logits"], second["logits"])
    # Both coordinates are marginals of one normalized pair distribution.
    probability = torch.softmax(first["logits"].flatten(1) / 0.5, dim=-1)
    torch.testing.assert_close(probability.sum(dim=-1), torch.ones(1))


def test_joint_pair_decoder_is_finite_when_topk_contains_padding():
    torch.manual_seed(27)
    model = DualViewHard3Net(20, width=8, dropout=0.0, pair_topk=6)
    logits = torch.randn(2, 3, 8)
    canonical = torch.randn(2, 3, 8, 18)
    points = torch.randn(2, 3, 8, 3)
    mask = torch.zeros(2, 3, 8, dtype=torch.bool)
    mask[..., :2] = True
    pair = model.gonion_pair(logits, canonical, points, mask)
    assert torch.isfinite(pair["logits"][pair["mask"]]).all()
    assert torch.isfinite(pair["soft_coordinate"]).all()


def test_tiny_joint_pair_training_and_inference_complete():
    rng = np.random.default_rng(29)
    samples, candidates, size = 6, 8, 16
    points = rng.normal(size=(samples, 3, candidates, 3)).astype(np.float32)
    expert = points[:, :, 0].copy()
    distance = np.linalg.norm(points - expert[:, :, None], axis=-1).astype(np.float32)
    candidate_set = DualViewCandidateSet(
        sample_ids=[f"sample_{index}" for index in range(samples)],
        strata=["Class1|women", "Class1|men"] * 3,
        images=rng.normal(size=(samples, 3, 2, 20, size, size)).astype(np.float16),
        targets=rng.random(size=(samples, 3, 2, size, size)).astype(np.float16),
        grids=rng.uniform(-1, 1, size=(samples, 3, 2, candidates, 2)).astype(
            np.float32
        ),
        points=points,
        canonical=rng.normal(size=(samples, 3, candidates, 18)).astype(np.float32),
        mask=np.ones((samples, 3, candidates), dtype=bool),
        expert=expert,
        expert_full=rng.normal(size=(samples, 23, 3)).astype(np.float32),
        target_distance=distance,
        target_view_mask=np.ones((samples, 3, 2), dtype=bool),
    )
    outputs, best_epoch, score, history, _ = _train_model(
        candidate_set,
        np.arange(4),
        np.arange(4, 6),
        Hard3DualViewConfig(
            epochs=1,
            min_epochs=1,
            patience=1,
            batch_size=2,
            width=8,
            pair_topk=4,
            translation_pixels=0,
            color_noise=0.0,
            gonion_color_dropout=0.0,
        ),
        torch.device("cpu"),
        fold_number=1,
    )
    assert best_epoch == 1
    assert np.isfinite(score)
    assert len(history) == 1
    assert outputs["logits"].shape == (2, 3, candidates)
    assert outputs["pair_soft"].shape == (2, 2, 3)


def test_dual_blend_supports_independent_left_and_right_strengths():
    base = np.zeros((2, 23, 3), dtype=np.float32)
    candidate = np.ones((2, 3, 3), dtype=np.float32)
    reliability = np.ones((2, 3), dtype=np.float32)
    prediction, alpha = _dual_blend_prediction(
        base,
        candidate,
        reliability,
        {
            "confidence_mode": "none",
            "alpha_lm0": 0.25,
            "alpha_gonion": 0.5,
            "alpha_gonion_left": 0.5,
            "alpha_gonion_right": 1.0,
        },
    )
    np.testing.assert_allclose(alpha[0], [0.25, 0.5, 1.0])
    np.testing.assert_allclose(prediction[0, list(HARD3), 0], [0.25, 0.5, 1.0])


def test_train_only_atlas_excludes_matching_training_sample():
    rng = np.random.default_rng(9)
    shapes = rng.normal(size=(12, 23, 3)).astype(np.float32)
    ids = [f"sample_{index}" for index in range(len(shapes))]
    atlas = TrainOnlyLocalHard3Atlas(neighbors=3).fit(shapes, ids)
    result = atlas.predict(shapes[:2], ids[:2])
    assert result["prediction"].shape == (2, 3, 3)
    assert result["dispersion"].shape == (2, 3)
    assert 0 not in result["neighbor_indices"][0]
    assert 1 not in result["neighbor_indices"][1]


def test_dual_view_validation_policy_never_changes_core20():
    rng = np.random.default_rng(13)
    expert = rng.normal(size=(16, 23, 3)).astype(np.float32)
    base = expert.copy()
    base[:, list(HARD3), 0] += 5.0
    improved = expert[:, list(HARD3)].copy()
    outputs = {
        "sample_ids": [f"sample_{index}" for index in range(len(expert))],
        "prediction": base,
        "expert": expert,
    }
    candidate_result = {
        "sample_ids": list(outputs["sample_ids"]),
        "prediction": improved,
        "variant_predictions": {
            "neural_policy": improved,
            "atlas_direct": improved + 0.25,
        },
        "reliability": np.ones((len(expert), 3), dtype=np.float32),
    }
    policy = calibrate_dual_view_blend(
        outputs,
        candidate_result,
        Hard3DualViewConfig(
            bootstrap_iters=50,
            minimum_overall_gain_mm=0.01,
            minimum_hard3_gain_mm=0.1,
        ),
    )
    refined = apply_dual_view_blend(outputs, candidate_result, policy)
    assert policy["accepted"] is True
    np.testing.assert_array_equal(
        refined["prediction"][:, list(CORE20)], base[:, list(CORE20)]
    )
    assert (
        np.linalg.norm(
            refined["prediction"][:, list(HARD3)] - expert[:, list(HARD3)], axis=-1
        ).mean()
        < 1e-6
    )
