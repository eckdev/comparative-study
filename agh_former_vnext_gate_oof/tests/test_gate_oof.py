import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from agh_former_vnext_gate_oof.analysis import paired_comparison
from agh_former_vnext_gate_oof.evidence import collect_evidence
from agh_former_vnext_gate_oof.gate import DiagnosticGate, Evidence, fit_gate, gate_predictions, load_gate
from agh_former_vnext_gate_oof.run_gate_oof import (
    assert_archived_stage1_parity, assert_archived_validation_parity,
    inner_partitions, pilot_decision,
)
from all23_rgb_geodesic_cascade.data import RGBGeodesicDataset, Sample, build_roi_cache
from all23_rgb_geodesic_cascade.train import collect_outputs


class DummyStage2:
    use_refinement_gate = True
    roi_radii = torch.full((23,), 10.0)

    def eval(self):
        return self

    def coordinates_from_logits(self, logits, candidates, mask, mode):
        probabilities = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        return (probabilities[..., None] * candidates).sum(dim=-2)

    def __call__(self, batch, coordinate_mode):
        count, landmark_count, points = batch["roi_index"].shape
        assert landmark_count == 23
        logits = torch.tensor([0.0, 1.0, 2.0])[None, None].expand(count, 23, points)
        return {
            "local_logits": logits,
            "log_var": torch.zeros(count, 23),
            "refinement_alpha": torch.full((count, 23), 0.5),
            "fusion_alpha": torch.zeros(count, 23),
            "gate_coarse": batch["coarse"],
            "global_heatmap_coordinates": batch["coarse"] + 1.0,
        }


class GateOOFTests(unittest.TestCase):
    def test_geometry_only_roi_without_landmark_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
            edges = np.asarray([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=np.int64)
            source = root / "mesh.npz"
            np.savez_compressed(source, points=points, edge_index=edges)
            path = build_roi_cache(
                source, np.zeros((23, 3), dtype=np.float32), 4,
                root / "cache", "sample", 42, geometry_only=True,
            )
            with np.load(path) as record:
                self.assertEqual(record["roi_index"].shape, (23, 4))
                self.assertTrue(np.all(record["heatmap_target"] == 0))
                self.assertTrue(np.all(record["oracle_error"] == 0))
                geometry_indices = record["roi_index"].copy()
            labelled = build_roi_cache(
                source, np.zeros((23, 3), dtype=np.float32), 4,
                root / "cache", "sample", 42,
                landmarks=np.zeros((23, 3), dtype=np.float32),
            )
            with np.load(labelled) as record:
                np.testing.assert_array_equal(geometry_indices, record["roi_index"])
            sample = Sample(root / "missing.ply", root / "missing.txt", "Class1", "men", 1)
            np.savez_compressed(
                source, points=points, edge_index=edges,
                features=np.zeros((4, 14), dtype=np.float32),
                landmarks=np.ones((23, 3), dtype=np.float32),
            )
            dataset = RGBGeodesicDataset(
                samples=[sample], sample_ids=[sample.sample_id],
                records={sample.sample_id: source},
                transforms={sample.sample_id: np.eye(4, dtype=np.float32)},
                coarse_predictions={sample.sample_id: np.zeros((23, 3), dtype=np.float32)},
                legacy_transform_dir=None,
                normalizer={"mean": [0.] * 14, "std": [1.] * 14},
                cache_dir=root / "inference_cache", roi_points=4,
                coarse_in_target_space=True, include_expert=False,
            )
            np.testing.assert_array_equal(dataset[0]["expert"].numpy(), np.zeros((23, 3)))

    def test_oof_partition_excludes_holdout_and_covers_train(self):
        samples = [
            Sample(Path(f"{group}-{gender}-{index}.ply"), Path("missing.txt"),
                   f"Class{group}", gender, index)
            for group in (1, 2, 3) for gender in ("men", "women")
            for index in range(1, 11)
        ]
        ids = [sample.sample_id for sample in samples]
        partitions = inner_partitions(samples, ids, 5, 0.2, 42)
        holdouts = []
        for split in partitions:
            self.assertFalse(set(split["test"]) & (set(split["train"]) | set(split["val"])))
            holdouts.extend(split["test"])
        self.assertEqual(sorted(holdouts), sorted(ids))

    def test_gate_controls_and_selection_threshold(self):
        sample_ids = [f"s{index}" for index in range(12)]
        expert = np.zeros((12, 23, 3), dtype=np.float32)
        local = np.ones_like(expert)
        center = np.zeros_like(expert)
        diagnostics = np.zeros((12, 23, 5), dtype=np.float32)
        evidence = Evidence(sample_ids, center, local, diagnostics, np.ones((12, 23)), expert).validate(True)
        alpha0, _ = gate_predictions(evidence, DiagnosticGate(), np.zeros((1, 1, 5)),
                                     np.ones((1, 1, 5)), torch.device("cpu"), accepted=False)
        self.assertTrue(np.allclose(alpha0, local))
        comparison = paired_comparison(expert, local, center, sample_ids, iterations=100)
        self.assertTrue(pilot_decision(comparison)["run_full_cv"])

    def test_gate_evidence_matches_existing_stage2_aggregation(self):
        model = DummyStage2()
        batch = {
            "sample_id": ["s1"], "class": ["Class1"], "gender": ["men"],
            "subject_id": torch.tensor([1]),
            "points": torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
            "features": torch.zeros((3, 14)),
            "coarse": torch.zeros((1, 23, 3)),
            "expert": torch.zeros((1, 23, 3)),
            "roi_index": torch.arange(3)[None, None].expand(1, 23, 3),
            "roi_mask": torch.ones((1, 23, 3), dtype=torch.bool),
            "heatmap_target": torch.zeros((1, 23, 3)),
            "region_target": torch.zeros((1, 23, 3)),
            "oracle_error": torch.zeros((1, 23)),
            "sample_radius_scale": torch.ones(1),
        }
        args = SimpleNamespace(mixed_precision=False, amp_dtype="float32", coordinate_mode="softmax", no_tqdm=True)
        normalizer = {"mean": [0.0] * 14, "std": [1.0] * 14}
        loader = [batch]
        original = collect_outputs(model, loader, torch.device("cpu"), args, normalizer)
        candidate = collect_evidence(model, loader, torch.device("cpu"), args, normalizer, use_tta=False)
        np.testing.assert_allclose(candidate.center, original["coarse"])
        np.testing.assert_allclose(candidate.local, original["refined"])
        np.testing.assert_allclose(candidate.token_alpha, original["refinement_alpha"])
        original_tta = collect_outputs(model, loader, torch.device("cpu"), args, normalizer, use_tta=True)
        candidate_tta = collect_evidence(model, loader, torch.device("cpu"), args, normalizer, use_tta=True)
        np.testing.assert_allclose(candidate_tta.center, original_tta["coarse"])
        np.testing.assert_allclose(candidate_tta.local, original_tta["refined"])
        np.testing.assert_allclose(candidate_tta.token_alpha, original_tta["refinement_alpha"])

    def test_small_gate_fit_and_reload(self):
        count = 30
        center = np.zeros((count, 23, 3), dtype=np.float32)
        local = np.ones_like(center)
        expert = np.full_like(center, 0.4)
        diagnostic = np.zeros((count, 23, 5), dtype=np.float32)
        diagnostic[:, :, 0] = np.linspace(-1, 1, count)[:, None]
        train = Evidence(
            [f"train{index}" for index in range(20)], center[:20], local[:20],
            diagnostic[:20], np.ones((20, 23), dtype=np.float32), expert[:20],
        )
        validation = Evidence(
            [f"val{index}" for index in range(10)], center[20:], local[20:],
            diagnostic[20:], np.ones((10, 23), dtype=np.float32), expert[20:],
        )
        with tempfile.TemporaryDirectory() as directory:
            model, mean, std, summary = fit_gate(
                train, validation, directory, torch.device("cpu"), epochs=3, patience=2,
            )
            self.assertIn("comparison", summary)
            reloaded, reloaded_mean, reloaded_std = load_gate(
                Path(directory) / "best_gate.pth", torch.device("cpu")
            )
            first, _ = gate_predictions(validation, model, mean, std, torch.device("cpu"))
            second, _ = gate_predictions(
                validation, reloaded, reloaded_mean, reloaded_std, torch.device("cpu")
            )
            np.testing.assert_allclose(first, second)
            path = Path(directory) / "evidence.npz"
            validation.save(path)
            self.assertEqual(Evidence.load(path).sample_ids, validation.sample_ids)

    def test_archived_validation_replay_accepts_sparse_vertex_flips(self):
        count = 48
        evidence = Evidence(
            [f"s{index}" for index in range(count)],
            np.zeros((count, 23, 3), dtype=np.float32),
            np.ones((count, 23, 3), dtype=np.float32),
            np.zeros((count, 23, 5), dtype=np.float32),
            np.full((count, 23), 0.5, dtype=np.float32),
            np.zeros((count, 23, 3), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions_val.csv"
            fields = ["sample_id", "landmark"] + [
                f"{prefix}_{axis}"
                for prefix in ("expert", "coarse", "refined", "neural_prediction", "stage1_coarse")
                for axis in "xyz"
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for sample_id in evidence.sample_ids:
                    for landmark in range(23):
                        row = {"sample_id": sample_id, "landmark": landmark}
                        for prefix, value in (("expert", 0.0), ("coarse", 0.0),
                                              ("refined", 1.0), ("neural_prediction", 0.5),
                                              ("stage1_coarse", 0.0)):
                            for axis in "xyz":
                                row[f"{prefix}_{axis}"] = value
                        writer.writerow(row)
            self.assertEqual(
                assert_archived_validation_parity(evidence, path)["status"], "exact"
            )
            stage1 = {
                sample_id: np.zeros((23, 3), dtype=np.float32)
                for sample_id in evidence.sample_ids
            }
            self.assertEqual(
                assert_archived_stage1_parity(stage1, path, evidence.sample_ids), 0.0
            )
            stage1["s0"][0, 0] = 0.1
            with self.assertRaisesRegex(ValueError, "Stage 1 centers differ"):
                assert_archived_stage1_parity(stage1, path, evidence.sample_ids)
            evidence.local[0, 21, 0] += 2.0
            evidence.center[1, 22, 0] += 0.25
            report = assert_archived_validation_parity(
                evidence, path, report_path=Path(directory) / "replay.json"
            )
            self.assertEqual(report["status"], "bounded_numeric_drift")
            self.assertEqual(report["coordinates"]["local"]["count_above_0p05mm"], 1)
            self.assertEqual(len(report["landmark_ale_delta_mm"]), 23)
            self.assertIn("p95_delta_mm", report["clinical"]["hard3"])
            self.assertTrue((Path(directory) / "replay.json").exists())
            evidence.local[2:14, 21, 0] += 2.0
            with self.assertRaisesRegex(ValueError, "hard3 ALE delta"):
                assert_archived_validation_parity(evidence, path)
            evidence.local[2:14, 21, 0] -= 2.0
            evidence.center[:, :, 0] += 0.1
            with self.assertRaisesRegex(ValueError, "beyond guardrails"):
                assert_archived_validation_parity(evidence, path)
            evidence.center[:, :, 0] -= 0.1
            evidence.expert[0, 0, 0] = 1.0
            with self.assertRaisesRegex(ValueError, "expert coordinates differ"):
                assert_archived_validation_parity(evidence, path)


if __name__ == "__main__":
    unittest.main()
