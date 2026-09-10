from types import SimpleNamespace

import numpy as np
import torch

from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3
from hard3_anatomical_context_refinement.atlas import TrainOnlyLocalHard3Atlas
from hard3_anatomical_context_refinement.model import (
    DualViewHard3Net,
    ShapeConditionedContourPairRanker,
    diverse_topk_indices,
)
from hard3_anatomical_context_refinement.patches import (
    DualViewCandidateSet,
    _rasterize,
    render_item,
)
from hard3_anatomical_context_refinement.refiner import (
    Hard3DualViewConfig,
    _cache_signature,
    _clinical_full_pair_loss,
    _contour_coordinate_loss,
    _dual_blend_prediction,
    _loss,
    _median_best_epoch,
    _proposal_diagnostics,
    _sharp_rerank_loss,
    _set_training_stage,
    _teacher_force_probability,
    _train_model,
    apply_dual_view_blend,
    calibrate_dual_view_blend,
)
from hard3_anatomical_context_refinement.interaction_selector import (
    CrossFittedInteractionSelector,
)
from hard3_anatomical_context_refinement.statistical_selector import (
    CrossFittedContourSelector,
)


def test_diverse_topk_keeps_the_best_candidate_from_each_proposal_source():
    sources = torch.full((1, 4, 12), -10.0)
    for source, candidate in enumerate((0, 3, 6, 9)):
        sources[0, source] = torch.linspace(-2.0, -3.1, 12)
        sources[0, source, candidate] = 10.0
    selected = diverse_topk_indices(
        sources, torch.ones(1, 12, dtype=torch.bool), topk=4
    )
    assert set(selected[0].tolist()) == {0, 3, 6, 9}


def test_geometry_proposal_scores_candidates_before_pair_pruning():
    torch.manual_seed(3)
    model = DualViewHard3Net(20, width=8, dropout=0.0, geometry_dim=34, pair_topk=4)
    images = torch.randn(2, 3, 2, 20, 16, 16)
    grids = torch.rand(2, 3, 2, 12, 2) * 2 - 1
    canonical = torch.randn(2, 3, 12, 34)
    mask = torch.ones(2, 3, 12, dtype=torch.bool)
    neighbors = torch.arange(12).view(1, 1, 12, 1).expand(2, 3, -1, -1)
    neighbor_mask = torch.ones_like(neighbors, dtype=torch.bool)
    heatmaps, weights = model.forward_with_context(images)
    evidence = model.candidate_logits(
        heatmaps,
        grids,
        mask,
        weights,
        canonical=canonical,
        neighbor_index=neighbors,
        neighbor_mask=neighbor_mask,
        return_evidence=True,
    )
    assert evidence["proposal_sources"].shape == (2, 3, 4, 12)
    evidence["logits"][:, 1:3].sum().backward()
    output = model.gonion_geometry_proposal.fusion[-1]
    assert output.weight.grad is not None
    assert torch.isfinite(output.weight.grad).all()


def test_surface_context_proposal_changes_when_local_neighbors_change():
    torch.manual_seed(7)
    model = DualViewHard3Net(20, width=8, dropout=0.0, geometry_dim=18, pair_topk=2)
    torch.nn.init.normal_(model.gonion_geometry_proposal.fusion[-1].weight, std=0.2)
    features = torch.randn(1, 2, 6, 23)
    geometry = torch.randn(1, 2, 6, 18)
    mask = torch.ones(1, 2, 6, dtype=torch.bool)
    first_neighbors = torch.arange(6).view(1, 1, 6, 1).expand(1, 2, -1, -1)
    second_neighbors = torch.flip(first_neighbors, dims=(2,))
    neighbor_mask = torch.ones_like(first_neighbors, dtype=torch.bool)
    first = model.gonion_geometry_proposal(
        features, geometry, first_neighbors, neighbor_mask, mask
    )
    second = model.gonion_geometry_proposal(
        features, geometry, second_neighbors, neighbor_mask, mask
    )
    assert not torch.allclose(first, second)


def test_pair_decoder_uses_learned_shortlist_instead_of_rank_union():
    model = DualViewHard3Net(20, width=8, dropout=0.0, pair_topk=2)
    logits = torch.zeros(1, 3, 8)
    logits[:, 1:3, 0] = 9.0
    logits[:, 1:3, 1] = 8.0
    sources = torch.zeros(1, 3, 4, 8)
    sources[:, 1:3, 1:, 7] = 20.0
    pair = model.gonion_pair(
        logits,
        torch.randn(1, 3, 8, 18),
        torch.randn(1, 3, 8, 3),
        torch.ones(1, 3, 8, dtype=torch.bool),
        proposal_sources=sources,
    )
    assert set(pair["left_indices"][0].tolist()) == {0, 1}
    assert set(pair["right_indices"][0].tolist()) == {0, 1}


def test_pair_stage_freezes_everything_except_bilateral_ranker():
    model = DualViewHard3Net(20, width=8, dropout=0.0, pair_topk=2)
    _set_training_stage(model, "pair")
    assert all(
        parameter.requires_grad for parameter in model.gonion_pair_ranker.parameters()
    )
    frozen = [
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("gonion_pair_ranker.")
    ]
    assert not any(frozen)


def test_rerank_stage_freezes_everything_except_sharp_unary_ranker():
    model = DualViewHard3Net(
        20, width=8, dropout=0.0, pair_topk=2, enable_unary_reranker=True
    )
    _set_training_stage(model, "rerank")
    assert all(
        parameter.requires_grad
        for parameter in model.gonion_unary_reranker.parameters()
    )
    frozen = [
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("gonion_unary_reranker.")
    ]
    assert not any(frozen)


def test_sharp_reranker_preserves_broad_topk_then_reorders_candidates():
    model = DualViewHard3Net(
        20,
        width=8,
        dropout=0.0,
        geometry_dim=18,
        proposal_topk=4,
        pair_topk=2,
        enable_unary_reranker=True,
    )
    logits = torch.zeros(1, 3, 8)
    logits[:, 1:3] = torch.arange(8, dtype=torch.float32)
    canonical = torch.randn(1, 3, 8, 18)
    mask = torch.ones(1, 3, 8, dtype=torch.bool)
    sources = torch.zeros(1, 3, 4, 8)
    sources[:, 1:3, 0] = logits[:, 1:3]
    reranked = model.rerank_gonion(logits, canonical, mask, sources)
    assert set(reranked["proposal_indices"][0, 0].tolist()) == {4, 5, 6, 7}
    pair = model.gonion_pair(
        reranked["logits"],
        canonical,
        torch.randn(1, 3, 8, 3),
        mask,
    )
    assert set(pair["left_indices"][0].tolist()).issubset({4, 5, 6, 7})
    assert set(pair["right_indices"][0].tolist()).issubset({4, 5, 6, 7})


def test_pair_teacher_forcing_is_removed_after_warmup():
    model = DualViewHard3Net(20, width=8, dropout=0.0, pair_topk=4)
    logits = torch.arange(8, 0, -1, dtype=torch.float32).view(1, 1, 8)
    logits = logits.expand(1, 3, 8).clone()
    canonical = torch.randn(1, 3, 8, 18)
    points = torch.randn(1, 3, 8, 3)
    mask = torch.ones(1, 3, 8, dtype=torch.bool)
    distance = torch.ones(1, 3, 8)
    distance[:, 1:3, 7] = 0.0
    unforced = model.gonion_pair(
        logits, canonical, points, mask, target_distance=distance
    )
    forced = model.gonion_pair(
        logits,
        canonical,
        points,
        mask,
        target_distance=distance,
        teacher_force_probability=1.0,
    )
    assert 7 not in unforced["left_indices"][0]
    assert 7 not in unforced["right_indices"][0]
    assert 7 in forced["left_indices"][0]
    assert 7 in forced["right_indices"][0]
    assert _teacher_force_probability(1, 5) == 1.0
    assert _teacher_force_probability(6, 5) == 0.0


def test_pair_teacher_forcing_does_not_duplicate_an_existing_nearest_candidate():
    model = DualViewHard3Net(20, width=8, dropout=0.0, pair_topk=4)
    logits = torch.arange(8, 0, -1, dtype=torch.float32).view(1, 1, 8)
    logits = logits.expand(1, 3, 8).clone()
    canonical = torch.randn(1, 3, 8, 18)
    points = torch.randn(1, 3, 8, 3)
    mask = torch.ones(1, 3, 8, dtype=torch.bool)
    distance = torch.ones(1, 3, 8)
    distance[:, 1:3, 0] = 0.0
    forced = model.gonion_pair(
        logits,
        canonical,
        points,
        mask,
        target_distance=distance,
        teacher_force_probability=1.0,
    )
    assert torch.unique(forced["left_indices"][0]).numel() == 4
    assert torch.unique(forced["right_indices"][0]).numel() == 4


def test_final_refit_epoch_uses_inner_fold_best_epoch_without_minimum_floor():
    assert _median_best_epoch([10, 10, 18, 17, 9], 90) == 10


def test_rasterizer_uses_outer_zbuffer_instead_of_depth_averaging():
    features = np.asarray([[1.0], [9.0]], dtype=np.float32)
    image = _rasterize(
        features,
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.asarray([-2.0, 3.0], dtype=np.float32),
        radius=10.0,
        image_size=5,
        view_code=1.0,
    )
    assert image[0, 2, 2] == 9.0


def test_proposal_diagnostics_reports_joint_recall_and_shortlist_oracle():
    distances = np.full((2, 3, 8), 10.0, dtype=np.float32)
    distances[:, 1:3, 7] = 0.5
    sources = np.zeros((2, 3, 4, 8), dtype=np.float32)
    sources[:, 1:3, 0, 7] = 20.0
    candidate_set = SimpleNamespace(
        target_distance=distances,
        mask=np.ones((2, 3, 8), dtype=np.bool_),
    )
    report = _proposal_diagnostics(candidate_set, sources, (1, 4))
    assert report["at_k"]["4"]["lm21_recall"] == 1.0
    assert report["at_k"]["4"]["lm22_recall"] == 1.0
    assert report["at_k"]["4"]["both_recall"] == 1.0
    assert report["at_k"]["4"]["gonion_oracle_ale"] == 0.5
    assert report["at_k"]["4"]["gonion_oracle_sdr_at_2mm"] == 1.0


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
    assert model.gonion_pair_ranker.unary_score[-1].weight.grad is not None
    assert model.gonion_view_gate[-1].weight.grad is not None


def test_sharp_rerank_loss_is_finite_and_updates_only_reranker():
    generator = torch.Generator().manual_seed(31)
    batch_size, candidates = 2, 12
    model = DualViewHard3Net(
        20,
        width=8,
        dropout=0.0,
        geometry_dim=18,
        proposal_topk=8,
        pair_topk=4,
        enable_unary_reranker=True,
    )
    _set_training_stage(model, "rerank")
    logits = torch.randn(batch_size, 3, candidates)
    canonical = torch.randn(batch_size, 3, candidates, 18)
    mask = torch.ones(batch_size, 3, candidates, dtype=torch.bool)
    sources = torch.randn(batch_size, 3, 4, candidates)
    reranked = model.rerank_gonion(logits, canonical, mask, sources)
    points = torch.randn(batch_size, 3, candidates, 3, generator=generator)
    expert = points[:, :, 0].clone()
    distance = torch.linalg.norm(points - expert[:, :, None], dim=-1)
    loss, components = _sharp_rerank_loss(
        reranked,
        {"points": points, "expert": expert, "distance": distance},
        Hard3DualViewConfig(width=8, proposal_topk=8, pair_topk=4),
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert model.gonion_unary_reranker.score[-1].weight.grad is not None
    assert model.gonion_pair_ranker.unary_score[-1].weight.grad is None


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
        include_contour_features=True,
    )
    assert rendered[0].shape == (3, 2, 20, 32, 32)
    assert rendered[1].shape == (3, 2, 32, 32)
    assert rendered[2].shape == (3, 2, roi_points, 2)
    assert rendered[4].shape == (3, roi_points, 40)
    assert rendered[5].shape == (3, roi_points, 12)
    assert rendered[6].shape == (3, roi_points, 12)
    assert rendered[7].any(axis=-1).all()
    assert rendered[11].shape == (3, 2)
    assert rendered[12].shape == (23, 3)
    assert rendered[13].shape == (69,)
    assert rendered[14].shape == (2, 3)
    assert rendered[15].shape == (2, 3)
    assert all(np.isfinite(values).all() for values in rendered[:5])
    # CoordConv and contour/depth-gradient channels are present immediately
    # before the final occupancy channel.
    assert rendered[0][..., -5, :, :].min() >= -1.0
    assert rendered[0][..., -5, :, :].max() <= 1.0
    assert rendered[0][..., -2, :, :].min() >= 0.0
    legacy = render_item(
        item,
        np.zeros(14, dtype=np.float32),
        np.ones(14, dtype=np.float32),
        image_size=32,
    )
    assert legacy[4].shape == (3, roi_points, 34)


def test_contour_pair_ranker_uses_shape_context_and_true_bilateral_energy():
    torch.manual_seed(47)
    ranker = ShapeConditionedContourPairRanker(
        input_dim=44,
        geometry_dim=40,
        shape_context_dim=69,
        width=16,
        dropout=0.0,
    )
    torch.nn.init.normal_(ranker.state_head[-1].weight, std=0.05)
    left_features = torch.randn(2, 7, 44)
    right_features = torch.randn(2, 7, 44)
    left_geometry = torch.randn(2, 7, 40)
    right_geometry = torch.randn(2, 7, 40)
    mask = torch.ones(2, 7, dtype=torch.bool)
    base = torch.randn(2, 2, 3) * 0.1
    first = ranker(
        left_features,
        right_features,
        left_geometry,
        right_geometry,
        mask,
        mask,
        torch.zeros(2, 69),
        base,
    )
    second = ranker(
        left_features,
        right_features,
        left_geometry,
        right_geometry,
        mask,
        mask,
        torch.randn(2, 69),
        base,
    )
    assert not torch.allclose(first["predicted_mean"], second["predicted_mean"])

    pair = first["pair_correction"][0]
    interaction = pair[0, 0] + pair[1, 1] - pair[0, 1] - pair[1, 0]
    assert abs(float(interaction)) > 1e-6


def test_dual_view_cache_signature_tracks_cascade_center_overrides(tmp_path):
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

        def _coarse(self, _sample):
            return np.zeros((23, 3), dtype=np.float32)

    dataset = DatasetStub()
    config = Hard3DualViewConfig(width=8)
    first = {"sample": np.zeros((23, 3), dtype=np.float32)}
    second = {"sample": np.ones((23, 3), dtype=np.float32)}
    assert _cache_signature(dataset, config, first) != _cache_signature(
        dataset, config, second
    )


def test_contour_coordinate_loss_is_finite_and_updates_state_head():
    torch.manual_seed(53)
    batch_size, candidates = 2, 10
    model = DualViewHard3Net(
        20,
        width=8,
        dropout=0.0,
        geometry_dim=40,
        proposal_topk=candidates,
        pair_topk=candidates,
        decoder_mode="contour_coordinate",
    )
    _set_training_stage(model, "pair")
    logits = torch.randn(batch_size, 3, candidates)
    canonical = torch.randn(batch_size, 3, candidates, 40)
    points = torch.randn(batch_size, 3, candidates, 3)
    expert = points[:, :, 0].clone()
    distance = torch.linalg.norm(points - expert[:, :, None], dim=-1)
    mask = torch.ones(batch_size, 3, candidates, dtype=torch.bool)
    pair = model.gonion_pair(
        logits,
        canonical,
        points,
        mask,
        proposal_sources=torch.randn(batch_size, 3, 4, candidates),
        shape_context=torch.randn(batch_size, 69),
        base_gonion=torch.randn(batch_size, 2, 3) * 0.1,
    )
    loss, components = _contour_coordinate_loss(
        pair,
        {"distance": distance, "mask": mask},
        Hard3DualViewConfig(
            width=8,
            proposal_topk=candidates,
            pair_topk=candidates,
            decoder_mode="contour_coordinate",
        ),
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(components["contour_state"])
    assert torch.isfinite(components["contour_moment"])
    loss.backward()
    gradient = model.gonion_pair_ranker.state_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_joint_pair_decoder_uses_both_gonion_candidate_sets():
    torch.manual_seed(23)
    model = DualViewHard3Net(20, width=8, dropout=0.0, pair_topk=4)
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


def test_full_pair_decoder_preserves_all_broad_top96_candidates():
    torch.manual_seed(25)
    candidates = 128
    model = DualViewHard3Net(
        20,
        width=8,
        dropout=0.0,
        geometry_dim=18,
        proposal_topk=96,
        pair_topk=96,
    )
    logits = torch.randn(1, 3, candidates)
    canonical = torch.randn(1, 3, candidates, 18)
    points = torch.randn(1, 3, candidates, 3)
    mask = torch.ones(1, 3, candidates, dtype=torch.bool)
    sources = torch.randn(1, 3, 4, candidates)
    pair = model.gonion_pair(
        logits,
        canonical,
        points,
        mask,
        proposal_sources=sources,
    )
    expected_left = set(torch.topk(logits[0, 1], 96).indices.tolist())
    expected_right = set(torch.topk(logits[0, 2], 96).indices.tolist())
    assert set(pair["left_indices"][0].tolist()) == expected_left
    assert set(pair["right_indices"][0].tolist()) == expected_right
    assert pair["logits"].shape == (1, 96, 96)


def test_joint_pair_decoder_is_finite_when_topk_contains_padding():
    torch.manual_seed(27)
    model = DualViewHard3Net(20, width=8, dropout=0.0, pair_topk=6)
    logits = torch.randn(2, 3, 8, requires_grad=True)
    canonical = torch.randn(2, 3, 8, 18)
    points = torch.randn(2, 3, 8, 3)
    mask = torch.zeros(2, 3, 8, dtype=torch.bool)
    mask[..., :2] = True
    pair = model.gonion_pair(logits, canonical, points, mask)
    assert torch.isfinite(pair["logits"][pair["mask"]]).all()
    assert torch.isfinite(pair["soft_coordinate"]).all()
    objective = (
        pair["logits"][pair["mask"]].mean() + pair["soft_coordinate"].square().mean()
    )
    objective.backward()
    assert torch.isfinite(logits.grad).all()


def test_clinical_full_pair_loss_is_finite_and_updates_pair_decoder():
    torch.manual_seed(41)
    batch_size, candidates = 2, 12
    model = DualViewHard3Net(
        20,
        width=8,
        dropout=0.0,
        geometry_dim=18,
        proposal_topk=candidates,
        pair_topk=candidates,
    )
    _set_training_stage(model, "pair")
    logits = torch.randn(batch_size, 3, candidates)
    canonical = torch.randn(batch_size, 3, candidates, 18)
    points = torch.randn(batch_size, 3, candidates, 3)
    expert = points[:, :, 0].clone()
    distance = torch.linalg.norm(points - expert[:, :, None], dim=-1)
    mask = torch.ones(batch_size, 3, candidates, dtype=torch.bool)
    sources = torch.randn(batch_size, 3, 4, candidates)
    pair = model.gonion_pair(
        logits,
        canonical,
        points,
        mask,
        proposal_sources=sources,
    )
    loss, components = _clinical_full_pair_loss(
        pair,
        {"distance": distance, "mask": mask},
        Hard3DualViewConfig(
            width=8,
            proposal_topk=candidates,
            pair_topk=candidates,
        ),
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    trainable_gradients = [
        parameter.grad
        for parameter in model.gonion_pair_ranker.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None for gradient in trainable_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in trainable_gradients
        if gradient is not None
    )


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
        neighbor_index=np.broadcast_to(
            np.arange(candidates)[None, None, :, None],
            (samples, 3, candidates, 1),
        ).copy(),
        neighbor_mask=np.ones((samples, 3, candidates, 1), dtype=bool),
        mask=np.ones((samples, 3, candidates), dtype=bool),
        expert=expert,
        expert_full=rng.normal(size=(samples, 23, 3)).astype(np.float32),
        target_distance=distance,
        target_view_mask=np.ones((samples, 3, 2), dtype=bool),
        expert_gonion_context=rng.normal(size=(samples, 2, 3)).astype(np.float32),
    )
    outputs, best_epoch, score, history, _ = _train_model(
        candidate_set,
        np.arange(4),
        np.arange(4, 6),
        Hard3DualViewConfig(
            decoder_mode="contour_coordinate",
            epochs=1,
            min_epochs=1,
            patience=1,
            batch_size=2,
            width=8,
            pair_topk=4,
            proposal_topk=8,
            rerank_stage_epochs=1,
            rerank_stage_min_epochs=1,
            rerank_stage_patience=1,
            pair_stage_epochs=1,
            pair_stage_min_epochs=1,
            pair_stage_patience=1,
            translation_pixels=0,
            color_noise=0.0,
            gonion_color_dropout=0.0,
        ),
        torch.device("cpu"),
        fold_number=1,
    )
    assert best_epoch == {"proposal": 1, "rerank": 0, "pair": 1}
    assert all(np.isfinite(value) for value in score.values())
    assert len(history) == 2
    assert outputs["logits"].shape == (2, 3, candidates)
    assert outputs["pair_soft"].shape == (2, 2, 3)


def test_crossfit_selector_fits_oof_policy_and_roundtrips_state():
    rng = np.random.default_rng(41)
    samples, candidates = 10, 12
    points = rng.normal(size=(samples, 3, candidates, 3)).astype(np.float32)
    expert = points[:, :, 0].copy()
    distance = np.linalg.norm(points - expert[:, :, None], axis=-1).astype(np.float32)
    canonical = rng.normal(size=(samples, 3, candidates, 40)).astype(np.float32)
    canonical[:, :, :, 3:6] = points
    sources = rng.normal(size=(samples, 3, 4, candidates)).astype(np.float32)
    sources[:, 1:3, 0, 0] += 8.0
    candidate_set = DualViewCandidateSet(
        sample_ids=[f"sample_{index}" for index in range(samples)],
        strata=["Class1|women", "Class1|men"] * 5,
        images=np.zeros((samples, 3, 2, 20, 8, 8), dtype=np.float16),
        targets=np.zeros((samples, 3, 2, 8, 8), dtype=np.float16),
        grids=np.zeros((samples, 3, 2, candidates, 2), dtype=np.float32),
        points=points,
        canonical=canonical,
        neighbor_index=np.zeros((samples, 3, candidates, 1), dtype=np.int64),
        neighbor_mask=np.ones((samples, 3, candidates, 1), dtype=bool),
        mask=np.ones((samples, 3, candidates), dtype=bool),
        expert=expert,
        expert_full=rng.normal(size=(samples, 23, 3)).astype(np.float32),
        target_distance=distance,
        target_view_mask=np.ones((samples, 3, 2), dtype=bool),
        shape_context=rng.normal(size=(samples, 69)).astype(np.float32),
        base_gonion=rng.normal(size=(samples, 2, 3)).astype(np.float32),
        expert_gonion_context=points[:, 1:3, 0].copy(),
    )
    splits = [
        (np.arange(5, 10), np.arange(0, 5)),
        (np.arange(0, 5), np.arange(5, 10)),
    ]
    config = Hard3DualViewConfig(
        decoder_mode="crossfit_calibrated",
        statistical_l2_grid=(0.01, 0.1),
        statistical_shortlist_grid=(4, 8),
        statistical_contour_weight_grid=(0.0, 1.0),
        statistical_state_weight_grid=(0.0, 1.0),
    )
    selector = CrossFittedContourSelector.fit(candidate_set, sources, splits, config)
    restored = CrossFittedContourSelector.from_state_dict(selector.state_dict())
    result = restored.predict(candidate_set, sources)
    assert result["crossfit_calibrated"].shape == (samples, 2, 3)
    assert np.isfinite(result["crossfit_calibrated"]).all()
    assert selector.report["uses_outer_validation_labels"] is False
    assert selector.report["selected"]["shortlist"] in (4, 8)


def test_query_interaction_selector_trains_oof_and_roundtrips_state():
    rng = np.random.default_rng(43)
    samples, candidates = 10, 12
    points = rng.normal(size=(samples, 3, candidates, 3)).astype(np.float32)
    expert = points[:, :, 0].copy()
    distance = np.linalg.norm(points - expert[:, :, None], axis=-1).astype(np.float32)
    canonical = rng.normal(size=(samples, 3, candidates, 40)).astype(np.float32)
    canonical[:, :, :, 3:6] = points
    sources = rng.normal(size=(samples, 3, 4, candidates)).astype(np.float32)
    sources[:, 1:3, 0, 0] += 8.0
    candidate_set = DualViewCandidateSet(
        sample_ids=[f"sample_{index}" for index in range(samples)],
        strata=["Class1|women", "Class1|men"] * 5,
        images=np.zeros((samples, 3, 2, 20, 8, 8), dtype=np.float16),
        targets=np.zeros((samples, 3, 2, 8, 8), dtype=np.float16),
        grids=np.zeros((samples, 3, 2, candidates, 2), dtype=np.float32),
        points=points,
        canonical=canonical,
        neighbor_index=np.zeros((samples, 3, candidates, 1), dtype=np.int64),
        neighbor_mask=np.ones((samples, 3, candidates, 1), dtype=bool),
        mask=np.ones((samples, 3, candidates), dtype=bool),
        expert=expert,
        expert_full=rng.normal(size=(samples, 23, 3)).astype(np.float32),
        target_distance=distance,
        target_view_mask=np.ones((samples, 3, 2), dtype=bool),
        shape_context=rng.normal(size=(samples, 69)).astype(np.float32),
        base_gonion=rng.normal(size=(samples, 2, 3)).astype(np.float32),
        expert_gonion_context=points[:, 1:3, 0].copy(),
    )
    splits = [
        (np.arange(5, 10), np.arange(0, 5)),
        (np.arange(0, 5), np.arange(5, 10)),
    ]
    config = Hard3DualViewConfig(
        decoder_mode="crossfit_interaction",
        batch_size=4,
        statistical_l2_grid=(0.01, 0.1),
        interaction_shortlist=8,
        interaction_width=8,
        interaction_epochs=2,
        interaction_min_epochs=1,
        interaction_patience=1,
    )
    selector = CrossFittedInteractionSelector.fit(
        candidate_set, sources, splits, config, torch.device("cpu")
    )
    restored = CrossFittedInteractionSelector.from_state_dict(selector.state_dict())
    result = restored.predict(candidate_set, sources)
    assert result["crossfit_interaction"].shape == (samples, 2, 3)
    assert result["member_coordinate"].shape == (2, samples, 2, 3)
    assert np.isfinite(result["crossfit_interaction"]).all()
    assert selector.report["uses_outer_validation_labels"] is False
    assert selector.report["parameter_count"] < 10_000


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
