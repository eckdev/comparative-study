import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from pointnet2_orthodontic_comparison.colab_download_pointnet2_cv_results import main


class PointNet2DownloadBundleTest(unittest.TestCase):
    def test_analysis_bundle_uses_pointnet2_metric_and_excludes_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "pointnet2_publication_cv_seed42"
            fold_dir = run_dir / "fold_1"
            fold_dir.mkdir(parents=True)
            (fold_dir / "metrics.json").write_text(
                json.dumps({"pointnet2": {"ale": 1.5}}), encoding="utf-8"
            )
            (fold_dir / "best_model.pth").write_bytes(b"excluded-by-default")
            with open(
                fold_dir / "predictions_test.csv",
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "class",
                        "gender",
                        "subject_id",
                        "landmark",
                        "localization_error",
                    ],
                )
                writer.writeheader()
                for landmark in range(23):
                    writer.writerow(
                        {
                            "sample_id": "Class1_F1",
                            "class": "Class1",
                            "gender": "women",
                            "subject_id": 1,
                            "landmark": landmark,
                            "localization_error": 1.5,
                        }
                    )

            output = root / "pointnet2_bundle.zip"
            result = main(
                [
                    "--run-dir",
                    str(run_dir),
                    "--output",
                    str(output),
                    "--folds",
                    "1",
                    "--skip-preprocessing",
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(output.exists())

            bundle_root = f"{run_dir.name}_analysis"
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertNotIn(
                    f"{bundle_root}/run/fold_1/best_model.pth", names
                )
                readme = archive.read(f"{bundle_root}/README_TR.txt").decode(
                    "utf-8"
                )
                self.assertIn("PointNet++ 5-fold analysis bundle", readme)
                manifest = json.loads(
                    archive.read(f"{bundle_root}/MANIFEST.json")
                )
                self.assertEqual(manifest["model"], "PointNet++")
                self.assertEqual(manifest["metric_key"], "pointnet2")
                integrity = manifest["integrity_report"]
                self.assertTrue(integrity["integrity_passed"])
                self.assertEqual(integrity["prediction_rows"], 23)
                self.assertEqual(integrity["unique_outer_test_samples"], 1)


if __name__ == "__main__":
    unittest.main()
