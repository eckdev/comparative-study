import json
from types import SimpleNamespace

import numpy as np
import torch

from agh_former_vnext_orthodontic_comparison.run_aghformer_vnext import (
    build_parser,
    configure_vnext,
    hard3_artifact_contract,
    hard3_curve_config_from_args,
)
from curve_supervised_hard3_refinement.annotations import CurveAnnotationStore
from curve_supervised_hard3_refinement.model import CurveFirstHard3Net
from curve_supervised_hard3_refinement.refiner import (
    CurveHard3Config,
    _batch,
    _loss,
)
from curve_supervised_hard3_refinement.targets import (
    build_curve_targets,
    point_to_polyline_distance,
)


def synthetic_tensors(batch_size=2, candidates=16, feature_dim=40):
    generator = torch.Generator().manual_seed(17)
    mask = torch.ones(batch_size, 3, candidates, dtype=torch.bool)
    mask[:, :, -2:] = False
    points = torch.randn(batch_size, 3, candidates, 3, generator=generator)
    expert = points[:, :, 0].clone()
    point_distance = torch.linalg.norm(points - expert[:, :, None], dim=-1)
    curve_distance = point_distance.clone()
    point_distance[~mask] = torch.inf
    curve_distance[~mask] = torch.inf
    return {
        "images": torch.randn(batch_size, 3, 2, 7, 32, 32, generator=generator),
        "canonical": torch.randn(
            batch_size, 3, candidates, feature_dim, generator=generator
        ),
        "neighbor_index": torch.randint(
            0, candidates, (batch_size, 3, candidates, 4), generator=generator
        ),
        "neighbor_mask": torch.ones(batch_size, 3, candidates, 4, dtype=torch.bool),
        "mask": mask,
        "points": points,
        "expert": expert,
        "shape_context": torch.randn(batch_size, 69, generator=generator),
        "curve_distance": curve_distance,
        "point_distance": point_distance,
        "curve_source": torch.tensor([[1, 1, 1], [0, 0, 0]]),
    }


def test_curve_first_model_and_loss_are_finite():
    batch = synthetic_tensors()
    model = CurveFirstHard3Net(
        input_channels=7,
        feature_dim=40,
        width=24,
        blocks=1,
        dropout=0.0,
    )
    output = model(
        batch["images"],
        batch["canonical"],
        batch["neighbor_index"],
        batch["neighbor_mask"],
        batch["mask"],
        batch["shape_context"],
    )
    assert output["final_logits"].shape == (2, 3, 16)
    assert output["log_variance"].shape == (2, 3)
    loss, components = _loss(output, batch, CurveHard3Config())
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert model.curve_head[-1].weight.grad is not None
    assert torch.isfinite(model.curve_head[-1].weight.grad).all()


def test_inference_batch_does_not_carry_expert_labels():
    tensors = synthetic_tensors(batch_size=1)
    candidate_set = SimpleNamespace(
        images=tensors["images"].numpy(),
        canonical=tensors["canonical"].numpy(),
        neighbor_index=tensors["neighbor_index"].numpy(),
        neighbor_mask=tensors["neighbor_mask"].numpy(),
        mask=tensors["mask"].numpy(),
        points=tensors["points"].numpy(),
        expert=tensors["expert"].numpy(),
        shape_context=tensors["shape_context"].numpy(),
    )
    batch = _batch(candidate_set, None, [0], torch.device("cpu"))
    assert "expert" not in batch
    assert "point_distance" not in batch
    assert "curve_distance" not in batch


def test_polyline_distance_and_manifest_transform(tmp_path):
    manifest = {
        "version": 1,
        "coordinate_space": "raw_mesh_mm",
        "samples": {
            "patient_1": {
                "curves": {
                    "hairline": [[0, 0, 0], [2, 0, 0]],
                    "jaw_left": [[0, 1, 0], [2, 1, 0]],
                    "jaw_right": [[0, -1, 0], [2, -1, 0]],
                },
                "repeat_landmarks": {"0": [[1, 0, 0]]},
            }
        },
    }
    path = tmp_path / "curves.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    store = CurveAnnotationStore.load(path)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = [10, 20, 30]
    transformed = store.transformed("patient_1", matrix)
    np.testing.assert_allclose(transformed.curves["hairline"][0], [10, 20, 30])
    train_only = store.subset(["patient_1"])
    assert set(train_only.samples) == {"patient_1"}
    assert train_only.source_hash != store.source_hash
    distance = point_to_polyline_distance(
        np.asarray([[1, 2, 0], [3, 0, 0]], dtype=np.float32),
        np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_allclose(distance, [2.0, 1.0])


def test_curve_targets_distinguish_real_and_pseudo_supervision(tmp_path):
    manifest = {
        "version": 1,
        "coordinate_space": "raw_mesh_mm",
        "samples": {
            "annotated": {
                "curves": {
                    "hairline": [[-2, 0, 0], [2, 0, 0]],
                    "jaw_left": [[-2, 1, 0], [2, 1, 0]],
                    "jaw_right": [[-2, -1, 0], [2, -1, 0]],
                }
            }
        },
    }
    path = tmp_path / "curves.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    points = np.zeros((2, 3, 12, 3), dtype=np.float32)
    points[..., 0] = np.linspace(-3, 3, 12)
    points[:, 1, :, 1] = 1.0
    points[:, 2, :, 1] = -1.0
    expert = points[:, :, 5].copy()
    candidate_set = SimpleNamespace(
        sample_ids=["annotated", "pseudo"],
        points=points,
        mask=np.ones((2, 3, 12), dtype=bool),
        expert=expert,
    )
    dataset = SimpleNamespace(
        samples=[
            SimpleNamespace(sample_id="annotated"),
            SimpleNamespace(sample_id="pseudo"),
        ],
        transforms={
            "annotated": np.eye(4, dtype=np.float32),
            "pseudo": np.eye(4, dtype=np.float32),
        },
    )
    targets = build_curve_targets(
        dataset, candidate_set, CurveAnnotationStore.load(path)
    )
    assert targets.curve_distance.shape == (2, 3, 12)
    np.testing.assert_array_equal(targets.source[0], [1, 1, 1])
    np.testing.assert_array_equal(targets.source[1], [0, 0, 0])
    assert targets.annotation_report["fully_curve_annotated"] == 1


def test_vnext_curve_mode_has_distinct_checkpoint_contract():
    args = configure_vnext(
        build_parser().parse_args(
            [
                "--data-root",
                "/tmp/data",
                "--output-dir",
                "/tmp/output",
                "--hard3-refiner-mode",
                "curve_supervised",
                "--hard3-curve-min-annotated-samples",
                "60",
                "--hard3-curve-annotation-manifest",
                "/tmp/curves.json",
            ]
        )
    )
    config = hard3_curve_config_from_args(args)
    assert config.minimum_annotated_samples == 60
    assert config.annotation_manifest == "/tmp/curves.json"
    assert hard3_artifact_contract(args) == (
        "curve_supervised_hard3_v1",
        15,
        "curve_hard3_model.pth",
    )
