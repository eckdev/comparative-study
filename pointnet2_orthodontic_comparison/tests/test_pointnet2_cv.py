import hashlib
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh


ROOT = Path(__file__).resolve().parents[2]
POINTNET_ROOT = ROOT / "pointnet2_orthodontic_comparison"
sys.path.insert(0, str(POINTNET_ROOT))

from run_orthodontic_pointnet2 import (  # noqa: E402
    OrthodonticPointCloudDataset,
    capture_rng_state,
    restore_rng_state,
)


def load_cv_runner():
    path = POINTNET_ROOT / "colab_run_pointnet2_cv.py"
    spec = importlib.util.spec_from_file_location("pointnet2_cv_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PointNet2CVTest(unittest.TestCase):
    def test_protocol_is_bound_to_shared_manifest(self):
        manifest_path = ROOT / "shared_splits" / "orthodontic_5fold_192_48_60_seed42.json"
        protocol_path = POINTNET_ROOT / "publication_cv_protocol.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        payload = dict(manifest)
        expected_hash = payload.pop("manifest_sha256")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected_hash)
        self.assertEqual(protocol["manifest_sha256"], expected_hash)
        self.assertEqual(protocol["samples_per_fold"], {"train": 192, "validation": 48, "test": 60})
        self.assertEqual(protocol["model_configuration"]["checkpoint_metric"], "validation_ALE")

    def test_fold_selection(self):
        runner = load_cv_runner()
        self.assertEqual(runner.selected_folds(None, 5), [1, 2, 3, 4, 5])
        self.assertEqual(runner.selected_folds("5,2,2", 5), [2, 5])
        with self.assertRaises(ValueError):
            runner.selected_folds("0,6", 5)

    def test_manifest_hash_rejects_modified_content(self):
        runner = load_cv_runner()
        manifest = json.loads(
            (ROOT / "shared_splits" / "orthodontic_5fold_192_48_60_seed42.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(runner.verified_manifest_hash(manifest), manifest["manifest_sha256"])
        manifest["folds"][0]["train"][0] = "TamperedSample"
        with self.assertRaises(ValueError):
            runner.verified_manifest_hash(manifest)

    def test_cv_preset_matches_frozen_protocol(self):
        runner = load_cv_runner()
        settings = runner.preset_settings("cv")
        protocol = json.loads(
            (POINTNET_ROOT / "publication_cv_protocol.json").read_text(encoding="utf-8")
        )["model_configuration"]
        self.assertEqual(settings["surface_points"], protocol["surface_points"])
        self.assertEqual(settings["eval_surface_points"], protocol["eval_surface_points"])
        self.assertEqual(
            [settings["sa1_points"], settings["sa2_points"], settings["sa3_points"]],
            protocol["sa_points"],
        )
        self.assertEqual(settings["epochs"], protocol["epochs"])
        self.assertEqual(settings["min_epochs"], protocol["minimum_epochs"])
        self.assertEqual(settings["patience"], protocol["patience"])
        self.assertEqual(settings["batch_size"], protocol["batch_size"])

    def test_npz_transform_is_applied_to_mesh_and_landmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            mesh_dir = root / "Class1" / "men"
            landmark_dir = root / "Class1" / "Class1-Landmark" / "men"
            mesh_dir.mkdir(parents=True)
            landmark_dir.mkdir(parents=True)

            vertices = np.asarray(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
            )
            faces = np.asarray(
                [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
            )
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_dir / "1.ply")
            (landmark_dir / "Class1_M1.txt").write_text(
                "\n".join(f"Point #{index}, 0, 0, 0" for index in range(23)),
                encoding="utf-8",
            )

            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, 3] = [10.0, 20.0, 30.0]
            transform_path = Path(temporary) / "transforms.npz"
            np.savez_compressed(transform_path, Class1_M1=matrix)

            dataset = OrthodonticPointCloudDataset(
                root,
                cache_dir=Path(temporary) / "cache",
                num_points=32,
                transformation_npz=transform_path,
                seed=17,
            )
            item = dataset[0]
            np.testing.assert_allclose(
                item["landmarks_world"].numpy(),
                np.tile([10.0, 20.0, 30.0], (23, 1)),
            )
            self.assertGreaterEqual(float(item["points_world"][:, 0].min()), 10.0)
            self.assertGreaterEqual(float(item["points_world"][:, 1].min()), 20.0)
            self.assertGreaterEqual(float(item["points_world"][:, 2].min()), 30.0)

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
                OrthodonticPointCloudDataset(
                    root,
                    cache_dir=Path(temporary) / "cache",
                    transformation_npz=transform_path,
                )

    def test_rng_restore_coerces_loaded_state_to_cpu_byte_tensor(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        state = capture_rng_state()
        state["torch"] = state["torch"].to(dtype=torch.int64)
        restore_rng_state(state)
        restored = torch.get_rng_state()
        self.assertEqual(restored.device.type, "cpu")
        self.assertEqual(restored.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
