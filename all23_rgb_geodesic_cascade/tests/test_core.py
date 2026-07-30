import numpy as np
import torch

from all23_rgb_geodesic_cascade.alignment import apply_transform
from all23_rgb_geodesic_cascade.anatomy import MIDLINE, SYMMETRY_PAIRS, graph_attention_mask, mirror_permutation
from all23_rgb_geodesic_cascade.data import assert_disjoint_splits, collate_graphs
from all23_rgb_geodesic_cascade.losses import (
    LossWeights, adaptive_wing_loss, compute_loss, region_loss,
)
from all23_rgb_geodesic_cascade.model import All23RGBGeodesicCascade, segment_softmax
from all23_rgb_geodesic_cascade.train import amp_torch_dtype


def test_anatomical_contract():
    assert MIDLINE == tuple(range(13))
    assert SYMMETRY_PAIRS == ((13, 16), (14, 15), (17, 18), (19, 20), (21, 22))
    permutation = mirror_permutation()
    assert permutation[13] == 16 and permutation[16] == 13
    assert all(permutation[permutation[index]] == index for index in range(23))
    mask = graph_attention_mask()
    assert not mask[21, 22]
    assert mask[21, 13]


def test_rigid_alignment_preserves_mm_distances():
    points = np.asarray([[0, 0, 0], [3, 4, 0], [1, 2, 7]], dtype=np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    matrix[:3, 3] = [10, -4, 2]
    transformed = apply_transform(points, matrix)
    assert np.allclose(np.linalg.norm(points[0] - points[1]), np.linalg.norm(transformed[0] - transformed[1]))


def test_split_overlap_is_rejected():
    try:
        assert_disjoint_splits({"train": ["A"], "val": ["A"], "test": ["B"]})
    except ValueError:
        pass
    else:
        raise AssertionError("split leakage was not rejected")


def synthetic_item(vertex_count=48, roi_points=8):
    points = torch.randn(vertex_count, 3)
    features = torch.randn(vertex_count, 14)
    nodes = torch.arange(vertex_count)
    edge_index = torch.stack(
        [torch.cat([nodes, nodes]), torch.cat([torch.roll(nodes, 1), nodes])]
    )
    roi = torch.stack([torch.arange(roi_points) + (index % 4) for index in range(23)])
    target = torch.zeros(23, roi_points)
    target[:, 0] = 1.0
    item = {
        "sample_id": "Class1_F1",
        "class": "Class1",
        "gender": "women",
        "subject_id": 1,
        "points": points,
        "features": features,
        "edge_index": edge_index,
        "vertex_mask": torch.ones(vertex_count, dtype=torch.bool),
        "coarse": points[roi[:, 0]],
        "expert": points[roi[:, 0]],
        "roi_index": roi,
        "roi_mask": torch.ones(23, roi_points, dtype=torch.bool),
        "heatmap_target": target,
        "region_target": target,
        "oracle_error": torch.zeros(23),
    }
    item["vertex_mask"][[3, 17]] = False
    return item


def test_dropout_vertices_are_removed_from_edges():
    batch = collate_graphs([synthetic_item()])
    assert not torch.any(batch["edge_index"] == 17)
    dropped_roi_positions = batch["roi_index"] == 3
    assert torch.any(dropped_roi_positions)
    assert not torch.any(batch["roi_mask"][dropped_roi_positions])


def test_model_forward_and_backward_are_finite():
    batch = collate_graphs([synthetic_item()])
    model = All23RGBGeodesicCascade(input_dim=14, width=32, global_blocks=1, heads=4, dropout=0.0)
    outputs = model(batch, coordinate_mode="topk")
    assert outputs["final_coordinates"].shape == (1, 23, 3)
    loss, errors, _ = compute_loss(outputs, batch, LossWeights())
    assert torch.isfinite(loss)
    assert torch.isfinite(errors).all()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    mse_outputs = model(batch, coordinate_mode="mse_over_mesh")
    assert torch.isfinite(mse_outputs["final_coordinates"]).all()


def test_global_only_ablation_uses_all_landmarks():
    batch = collate_graphs([synthetic_item()])
    model = All23RGBGeodesicCascade(
        input_dim=14, width=32, global_blocks=1, heads=4, dropout=0.0, use_local_refiner=False
    )
    outputs = model(batch)
    assert outputs["local_logits"].shape == (1, 23, 8)
    assert outputs["final_coordinates"].shape == (1, 23, 3)


def test_segment_softmax_supports_mixed_precision():
    scores = torch.tensor([[1.0, 0.0], [2.0, 1.0], [0.5, 0.5]], dtype=torch.float16)
    groups = torch.tensor([0, 0, 1], dtype=torch.long)
    weights = segment_softmax(scores, groups, segment_count=2)
    assert weights.dtype == torch.float16
    assert torch.allclose(weights[:2].float().sum(dim=0), torch.ones(2), atol=1e-3)
    assert torch.allclose(weights[2].float(), torch.ones(2), atol=1e-3)


def test_heatmap_losses_promote_half_precision_inputs_to_float32():
    logits = torch.tensor(
        [[[80.0, -80.0, 0.0], [-40.0, 40.0, 0.0]]], dtype=torch.float16
    )
    target = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]], dtype=torch.float16
    )
    mask = torch.ones_like(target, dtype=torch.bool)
    heatmap = adaptive_wing_loss(logits, target, mask)
    region = region_loss(logits, target, mask)
    assert heatmap.dtype == torch.float32
    assert region.dtype == torch.float32
    assert torch.isfinite(heatmap)
    assert torch.isfinite(region)


def test_roi_radius_scale_is_applied_consistently_in_model():
    model = All23RGBGeodesicCascade(
        input_dim=14,
        width=32,
        global_blocks=1,
        heads=4,
        dropout=0.0,
        roi_radius_scale=1.5,
    )
    assert torch.isclose(model.roi_radii[0], torch.tensor(52.5))
    assert torch.isclose(model.roi_radii[21], torch.tensor(67.5))


def test_amp_dtype_mapping():
    assert amp_torch_dtype("float16") == torch.float16
    assert amp_torch_dtype("bfloat16") == torch.bfloat16
