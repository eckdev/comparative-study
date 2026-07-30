import numpy as np
import torch
from types import SimpleNamespace

from all23_rgb_geodesic_cascade.alignment import apply_transform, build_robust_atlas
from all23_rgb_geodesic_cascade.anatomy import MIDLINE, SYMMETRY_PAIRS, graph_attention_mask, mirror_permutation
from all23_rgb_geodesic_cascade.data import (
    assert_disjoint_splits, collate_graphs, collate_surface_graphs,
)
from all23_rgb_geodesic_cascade.losses import (
    LossWeights, adaptive_wing_loss, compute_loss, region_loss,
)
from all23_rgb_geodesic_cascade.model import (
    All23RGBGeodesicCascade, GlobalCoarseNetwork, segment_softmax,
)
from all23_rgb_geodesic_cascade.stage1 import _inner_partitions, stage1_loss
from all23_rgb_geodesic_cascade.train import amp_torch_dtype
from all23_rgb_geodesic_cascade.run_all23_rgb_geodesic import oracle_gate_status


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


def test_global_stage1_forward_and_loss_are_finite():
    item = synthetic_item()
    surface_item = {
        key: item[key]
        for key in (
            "sample_id", "class", "gender", "subject_id", "points", "features",
            "edge_index", "vertex_mask", "expert",
        )
    }
    batch = collate_surface_graphs([surface_item])
    model = GlobalCoarseNetwork(
        input_dim=14, width=32, global_blocks=1, heads=4, dropout=0.0, coordinate_topk=5
    )
    outputs = model(batch)
    loss, errors, _ = stage1_loss(outputs, batch)
    assert outputs["coordinates"].shape == (1, 23, 3)
    assert torch.isfinite(loss)
    assert torch.isfinite(errors).all()
    loss.backward()


def test_nested_oof_partitions_exclude_holdout_samples():
    samples = []
    for index in range(12):
        samples.append(
            SimpleNamespace(
                sample_id=f"S{index}",
                class_name=f"Class{index % 2 + 1}",
                gender="women" if index % 2 else "men",
            )
        )
    by_id = {sample.sample_id: sample for sample in samples}
    partitions = _inner_partitions(by_id, list(by_id), 2, 0.25, 42)
    held_out = set()
    for partition in partitions:
        assert not (set(partition["train"]) & set(partition["holdout"]))
        assert not (set(partition["val"]) & set(partition["holdout"]))
        held_out.update(partition["holdout"])
    assert held_out == set(by_id)


def test_robust_atlas_uses_multiple_train_meshes():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(40, 3)).astype(np.float32) * [20.0, 30.0, 10.0]
    normals = base / np.clip(np.linalg.norm(base, axis=1, keepdims=True), 1e-6, None)
    meshes = {
        "a": SimpleNamespace(vertices=base, vertex_normals=normals),
        "b": SimpleNamespace(vertices=base + [2.0, -1.0, 0.5], vertex_normals=normals),
        "c": SimpleNamespace(vertices=base + [-1.0, 1.0, -0.5], vertex_normals=normals),
    }
    samples = [SimpleNamespace(sample_id=name, mesh_path=name) for name in meshes]
    atlas, atlas_normals, _, selected, _ = build_robust_atlas(
        samples,
        lambda path: meshes[path],
        sample_points=32,
        atlas_size=3,
        atlas_iterations=1,
        seed=7,
    )
    assert atlas.shape == (32, 3)
    assert atlas_normals.shape == (32, 3)
    assert len(selected) == 3
    assert np.isfinite(atlas).all()


def test_oracle_gate_rejects_sample_level_outlier():
    args = SimpleNamespace(
        max_val_oracle_ale=1.5,
        max_val_hard3_oracle_ale=2.5,
        max_val_oracle_p95=2.0,
        max_val_oracle_max=15.0,
        max_val_sample_oracle_ale=2.0,
    )
    validation = {
        "ale": 1.0,
        "hard3_ale": 1.0,
        "p95": 1.5,
        "max": 10.0,
        "sample_ale_max": 3.0,
    }
    status = oracle_gate_status(validation, args)
    assert not status["sample_max"]
    assert not status["passed"]
