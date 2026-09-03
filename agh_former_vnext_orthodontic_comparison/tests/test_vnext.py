import numpy as np
import torch

from agh_former_vnext_orthodontic_comparison.model import AGHFormerVNext
from agh_former_vnext_orthodontic_comparison.shape_prior import TrainOnlyShapePrior
from all23_rgb_geodesic_cascade.anatomy import MIDLINE, SYMMETRY_PAIRS
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
