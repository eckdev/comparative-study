import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

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
    _balanced_batches,
    _batch,
    _fit_one,
    _loss,
    _set_training_stage,
    _source_aware_splits,
    fit_or_load_curve_hard3_refiner,
)
from curve_supervised_hard3_refinement.targets import (
    build_curve_targets,
    point_to_polyline_distance,
)
from curve_supervised_hard3_refinement.validate_annotations import validate_manifest


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


def test_source_balanced_batches_and_folds_keep_real_curves_visible():
    source = np.zeros((30, 3), dtype=np.int8)
    source[:10] = 1
    strata = [
        f"Class{index % 3 + 1}|{'men' if index % 2 else 'women'}" for index in range(30)
    ]
    splits, report = _source_aware_splits(strata, source, folds=5, seed=42)
    assert report["fully_annotated_samples"] == 10
    assert all(source[val].all(axis=1).sum() > 0 for _, val in splits)
    batches = _balanced_batches(
        np.arange(30),
        source,
        batch_size=6,
        real_fraction=0.5,
        rng=np.random.default_rng(9),
    )
    assert all(source[batch].any(axis=1).sum() == 3 for batch in batches)
    assert set(np.concatenate(batches)) == set(range(30))


def test_sparse_curve_pilot_balances_source_and_class_gender():
    strata, source = [], []
    for group in range(6):
        for _ in range(4):
            strata.append(f"group_{group}")
            source.append([1, 1, 1])
        for _ in range(28):
            strata.append(f"group_{group}")
            source.append([0, 0, 0])
    source = np.asarray(source, dtype=np.int8)
    splits, report = _source_aware_splits(strata, source, folds=5, seed=42)
    validation_sizes = [len(validation) for _, validation in splits]
    annotated_sizes = [
        int(source[validation].all(axis=1).sum()) for _, validation in splits
    ]
    assert report["strategy"] == "greedy_class_gender_and_curve_source"
    assert max(validation_sizes) - min(validation_sizes) <= 1
    assert max(annotated_sizes) - min(annotated_sizes) <= 1
    assert min(annotated_sizes) >= 4


def test_real_curve_pretraining_and_checkpoint_selection_run_end_to_end():
    tensors = synthetic_tensors(batch_size=10, candidates=12)
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
    source = np.zeros((10, 3), dtype=np.int8)
    source[:4] = 1
    targets = SimpleNamespace(
        curve_distance=tensors["curve_distance"].numpy(),
        point_distance=tensors["point_distance"].numpy(),
        source=source,
    )
    result = _fit_one(
        candidate_set,
        targets,
        train_indices=np.arange(2, 10),
        val_indices=np.arange(2),
        config=CurveHard3Config(
            folds=2,
            epochs=1,
            min_epochs=1,
            patience=1,
            batch_size=4,
            width=24,
            blocks=1,
            dropout=0.0,
            curve_pretrain_epochs=1,
        ),
        device=torch.device("cpu"),
        fold=1,
    )
    history = result[-2]
    assert [row["stage"] for row in history] == ["curve_pretrain", "joint"]
    assert history[-1]["val_real_curve_expected_distance_mm"] is not None
    assert np.isfinite(result[3])
    assert np.isfinite(result[4])


def test_curve_pretraining_freezes_landmark_decoder_only():
    model = CurveFirstHard3Net(7, 40, width=24, blocks=1)
    _set_training_stage(model, "curve")
    assert all(parameter.requires_grad for parameter in model.curve_head.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.landmark_head.parameters()
    )
    _set_training_stage(model, "joint")
    assert all(parameter.requires_grad for parameter in model.parameters())


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


def test_annotation_qa_detects_raw_coordinate_mismatch(tmp_path):
    mesh_dir = tmp_path / "data" / "Class1" / "men"
    landmark_dir = tmp_path / "data" / "Class1" / "Class1-Landmark" / "men"
    mesh_dir.mkdir(parents=True)
    landmark_dir.mkdir(parents=True)
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=10.0)
    mesh.export(mesh_dir / "1.ply")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    landmarks = vertices[np.arange(23) % len(vertices)]
    (landmark_dir / "Class1_M1.txt").write_text(
        "\n".join(
            f"Point #{index}, {point[0]}, {point[1]}, {point[2]}"
            for index, point in enumerate(landmarks)
        ),
        encoding="utf-8",
    )

    def write_manifest(path, offset):
        curves = {}
        for key, landmark in (("hairline", 0), ("jaw_left", 21), ("jaw_right", 22)):
            curves[key] = [
                (landmarks[landmark] + offset).tolist(),
                (vertices[(landmark + 1) % len(vertices)] + offset).tolist(),
            ]
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "coordinate_space": "raw_mesh_mm",
                    "samples": {"Class1_M1": {"curves": curves}},
                }
            ),
            encoding="utf-8",
        )

    valid_path = tmp_path / "valid.json"
    write_manifest(valid_path, 0.0)
    valid = validate_manifest(tmp_path / "data", valid_path, 1)
    assert valid["passed"]
    assert valid["fully_annotated_samples"] == 1

    invalid_path = tmp_path / "invalid.json"
    write_manifest(invalid_path, 100.0)
    invalid = validate_manifest(tmp_path / "data", invalid_path, 1)
    assert not invalid["passed"]
    assert any("curve-to-mesh" in failure for failure in invalid["failures"])


def test_publication_mode_cannot_bypass_minimum_annotation_gate(tmp_path):
    manifest = tmp_path / "blank.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_space": "raw_mesh_mm",
                "samples": {"sample_1": {"curves": {}}},
            }
        ),
        encoding="utf-8",
    )
    dataset = SimpleNamespace(samples=[SimpleNamespace(sample_id="sample_1")])
    output_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="at least 60"):
        fit_or_load_curve_hard3_refiner(
            dataset,
            output_dir,
            CurveHard3Config(
                run_mode="publication",
                annotation_manifest=str(manifest),
                minimum_annotated_samples=0,
                publication_minimum_annotated_samples=60,
            ),
            torch.device("cpu"),
        )
    report = json.loads((output_dir / "annotation_preflight.json").read_text())
    assert report["required_fully_annotated_training_samples"] == 60
    assert not report["passed"]


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
                "--hard3-curve-run-mode",
                "publication",
            ]
        )
    )
    config = hard3_curve_config_from_args(args)
    assert config.minimum_annotated_samples == 60
    assert config.run_mode == "publication"
    assert config.publication_minimum_annotated_samples == 60
    assert config.curve_checkpoint_weight == 0.05
    assert config.annotation_manifest == "/tmp/curves.json"
    assert hard3_artifact_contract(args) == (
        "curve_supervised_hard3_v2",
        16,
        "curve_hard3_model.pth",
    )
