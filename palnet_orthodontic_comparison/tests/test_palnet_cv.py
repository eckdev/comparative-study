import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
PALNET_ROOT = ROOT / "palnet_orthodontic_comparison"
UPSTREAM = PALNET_ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM))

from src.datasets.orthodontic_dataset import OrthodonticDataset  # noqa: E402


def load_cv_runner():
    path = PALNET_ROOT / "colab_run_palnet_cv.py"
    spec = importlib.util.spec_from_file_location("palnet_cv_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PALNetCVTest(unittest.TestCase):
    def test_protocol_is_bound_to_shared_manifest(self):
        manifest_path = ROOT / "shared_splits" / "orthodontic_5fold_192_48_60_seed42.json"
        protocol_path = PALNET_ROOT / "publication_cv_protocol.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        payload = dict(manifest)
        expected_hash = payload.pop("manifest_sha256")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected_hash)
        self.assertEqual(protocol["manifest_sha256"], expected_hash)
        self.assertEqual(protocol["samples_per_fold"], {"train": 192, "validation": 48, "test": 60})
        self.assertEqual(protocol["model_configuration"]["checkpoint_metric"], "validation_snapped_ALE")

    def test_fold_selection(self):
        runner = load_cv_runner()
        self.assertEqual(runner.selected_folds(None, 5), [1, 2, 3, 4, 5])
        self.assertEqual(runner.selected_folds("5,2,2", 5), [2, 5])
        with self.assertRaises(ValueError):
            runner.selected_folds("0,6", 5)

    def test_cv_preset_matches_frozen_protocol(self):
        runner = load_cv_runner()
        settings = runner.preset_settings("cv")
        protocol = json.loads(
            (PALNET_ROOT / "publication_cv_protocol.json").read_text(encoding="utf-8")
        )["model_configuration"]
        self.assertEqual(settings["surface_points"], protocol["surface_points"])
        self.assertEqual(settings["patch_size"], protocol["patch_size"])
        self.assertEqual(settings["epochs"], protocol["epochs"])
        self.assertEqual(settings["min_epochs"], protocol["minimum_epochs"])
        self.assertEqual(settings["patience"], protocol["patience"])
        self.assertEqual(settings["batch_size"], protocol["batch_size"])

    def test_npz_transform_is_applied_to_mesh_and_landmarks_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            mesh_dir = root / "Class1" / "men"
            landmark_dir = root / "Class1" / "Class1-Landmark" / "men"
            mesh_dir.mkdir(parents=True)
            landmark_dir.mkdir(parents=True)

            vertices = np.asarray(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
            )
            faces = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_dir / "1.ply")
            landmark_lines = [f"Point #{index}, 0, 0, 0" for index in range(23)]
            (landmark_dir / "Class1_M1.txt").write_text(
                "\n".join(landmark_lines), encoding="utf-8"
            )

            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, 3] = [10.0, 20.0, 30.0]
            transform_path = Path(temporary) / "transforms.npz"
            np.savez_compressed(transform_path, Class1_M1=matrix)

            first = OrthodonticDataset(
                root,
                cache_dir=Path(temporary) / "cache_a",
                num_surface_points=32,
                transformation_npz=transform_path,
                seed=17,
            )
            second = OrthodonticDataset(
                root,
                cache_dir=Path(temporary) / "cache_b",
                num_surface_points=32,
                transformation_npz=transform_path,
                seed=17,
            )
            points_a, landmarks_a, vertices_a = first[0]
            points_b, landmarks_b, _ = second[0]

            np.testing.assert_allclose(landmarks_a.numpy(), np.tile([10.0, 20.0, 30.0], (23, 1)))
            self.assertGreaterEqual(float(vertices_a[:, 0].min()), 10.0)
            self.assertGreaterEqual(float(vertices_a[:, 1].min()), 20.0)
            self.assertGreaterEqual(float(vertices_a[:, 2].min()), 30.0)
            np.testing.assert_allclose(points_a.numpy(), points_b.numpy())

    def test_transform_archive_must_cover_every_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            mesh_dir = root / "Class1" / "men"
            landmark_dir = root / "Class1" / "Class1-Landmark" / "men"
            mesh_dir.mkdir(parents=True)
            landmark_dir.mkdir(parents=True)
            trimesh.creation.icosphere(subdivisions=1).export(mesh_dir / "1.ply")
            (landmark_dir / "Class1_M1.txt").write_text(
                "\n".join(f"Point #{index}, 0, 0, 0" for index in range(23)),
                encoding="utf-8",
            )
            transform_path = Path(temporary) / "missing.npz"
            np.savez_compressed(transform_path, DifferentSample=np.eye(4, dtype=np.float32))
            with self.assertRaises(KeyError):
                OrthodonticDataset(root, transformation_npz=transform_path)


if __name__ == "__main__":
    unittest.main()
